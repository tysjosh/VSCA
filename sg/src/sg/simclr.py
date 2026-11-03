"""SimCLR extractor phi and class means mu_c (Section III-D "Extractor and label budget", D7)."""

from __future__ import annotations

import contextlib
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import budget_split, limit_per_class, load_dataset, normalize, split_indices
from .model import ResNet18
from .utils import (
    ComputeLog, atomic_torch_save, env_info, past_deadline, pick_device, rng_state, same_config_or_exit, set_rng_state,
    set_seed, stop_for_resume,
)


class TwoViews(torch.utils.data.Dataset):
    def __init__(self, x_u8_nhwc: np.ndarray, mean, std):
        from torchvision.transforms import v2

        self.x = torch.from_numpy(np.ascontiguousarray(x_u8_nhwc)).permute(0, 3, 1, 2)
        size = self.x.shape[-1]
        self.t = v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(list(mean), list(std)),
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        return self.t(img), self.t(img)


class SimCLR(nn.Module):
    def __init__(self, n_classes: int, proj_dim: int):
        super().__init__()
        self.backbone = ResNet18(n_classes)
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        return self.proj(self.backbone.features(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def extract_features(model: ResNet18, x_u8: np.ndarray, device, bs: int, mean, std) -> np.ndarray:
    """512-d GAP features of un-augmented images (NHWC uint8)."""
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x_u8[i:i + bs])).permute(0, 3, 1, 2).to(device)
        out.append(model.features(normalize(xb, mean, std)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512), np.float32)


def class_means(feats: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    f = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    mu = np.zeros((n_classes, f.shape[1]), np.float32)
    for c in range(n_classes):
        m = y == c
        if m.any():
            mu[c] = f[m].mean(0)
    return mu


def load_backbone(run_dir, device) -> ResNet18:
    ck = torch.load(f"{run_dir}/backbone.pt", map_location=device)
    model = ResNet18(ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def train_simclr(cfg) -> dict:
    run_dir = cfg.run_dir
    ckpt_path = run_dir / "ckpt.pt"
    if cfg.skip_if_done and (run_dir / "summary.json").exists() and (run_dir / "class_means.npy").exists():
        same_config_or_exit(run_dir, cfg)
        return json.loads((run_dir / "summary.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    resuming = cfg.skip_if_done and ckpt_path.exists()
    if resuming:
        same_config_or_exit(run_dir, cfg)
    else:
        for f in ("compute.json", "ckpt.pt", "ckpt.pt.tmp"):
            (run_dir / f).unlink(missing_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    device = pick_device(cfg.device)
    set_seed(cfg.seed)
    ds = load_dataset(cfg)
    if cfg.simclr_scope == "all":  # training + validation pools (audit pool excluded); smoke limits still apply
        tr, va = budget_split(ds.y_train, 1.0, cfg.val_frac, cfg.split_seed, ds.n_classes, cfg.audit_frac)
        base = np.sort(np.concatenate([tr, va]))
        idx = limit_per_class(base, ds.y_train, cfg.limit_train_per_class)
    else:
        idx, _ = split_indices(cfg, ds)
    x, y = ds.x_train[idx], ds.y_train[idx]

    clog = ComputeLog(run_dir / "compute.json")
    with clog.phase("simclr"):
        data = TwoViews(x, ds.mean, ds.std)
        bs = min(cfg.simclr_batch_size, len(data))
        g = torch.Generator().manual_seed(cfg.seed)
        loader = torch.utils.data.DataLoader(
            data, batch_size=bs, shuffle=True, drop_last=len(data) > bs, generator=g,
            num_workers=cfg.num_workers, pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0,
        )
        model = SimCLR(ds.n_classes, cfg.simclr_proj_dim).to(device)
        lr = cfg.simclr_lr_per256 * bs / 256
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.simclr_wd)
        total = max(cfg.simclr_epochs * len(loader), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
        use_amp = cfg.amp and device.type == "cuda"
        losses, start, resumes = [], 0, 0
        if resuming:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            g.set_state(ck["loader_gen"])  # the DataLoader draws its shuffle and worker seeds from g
            set_rng_state(ck["rng"])
            losses, start, resumes = ck["losses"], ck["epoch"], ck.get("resumes", 0) + 1
            print(f"[resume] {run_dir}: continuing after SimCLR epoch {start} (resume #{resumes})", flush=True)

        def save_ckpt(done_epochs):
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                               "loader_gen": g.get_state(), "rng": rng_state(), "losses": losses,
                               "epoch": done_epochs, "resumes": resumes}, ckpt_path)

        if resuming:
            save_ckpt(start)
        if start < cfg.simclr_epochs and past_deadline(cfg):
            stop_for_resume("session deadline already reached")
        for ep in range(start, cfg.simclr_epochs):
            model.train()
            tot, nb = torch.zeros((), device=device), 0
            for v1, v2_ in loader:
                v1, v2_ = v1.to(device, non_blocking=True), v2_.to(device, non_blocking=True)
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
                with ctx:
                    loss = nt_xent(model(v1), model(v2_), cfg.simclr_temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.detach()
                nb += 1
            losses.append(float(tot) / max(nb, 1))
            print(f"simclr epoch {ep + 1}/{cfg.simclr_epochs} loss {losses[-1]:.4f}", flush=True)
            done = ep + 1
            if done < cfg.simclr_epochs:
                debug_stop = cfg.debug_stop_after_epochs is not None and done >= cfg.debug_stop_after_epochs
                out_of_time = past_deadline(cfg)
                if debug_stop or out_of_time or done % max(cfg.ckpt_every_epochs, 1) == 0:
                    save_ckpt(done)
                if out_of_time:
                    stop_for_resume(f"session deadline reached after SimCLR epoch {done}")
                if debug_stop:
                    stop_for_resume(f"debug stop after SimCLR epoch {done}")
        backbone = model.backbone
        torch.save({"state_dict": backbone.state_dict(), "n_classes": ds.n_classes}, run_dir / "backbone.pt")
        feats = extract_features(backbone, x, device, cfg.eval_batch_size, ds.mean, ds.std)
        mu = class_means(feats, y, ds.n_classes)
        np.save(run_dir / "class_means.npy", mu)
    summary = {
        "dataset": cfg.dataset, "scope": cfg.simclr_scope, "budget": None if cfg.simclr_scope == "all" else cfg.budget,
        "n_images": int(len(idx)), "epochs": cfg.simclr_epochs, "batch_size": bs, "lr": lr,
        "temperature": cfg.simclr_temperature, "final_loss": losses[-1] if losses else None, "loss_per_epoch": losses,
        "class_means_shape": list(mu.shape), "resumes": resumes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not cfg.keep_ckpt:
        ckpt_path.unlink(missing_ok=True)
    return summary
