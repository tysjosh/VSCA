"""Training (Section III-E): Eq. 4 objective, gradient-norm-matched lambda_s (D4), wd / rho selection (D6, D5)."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import AUX_STAGES
from .data import gpu_augment, load_dataset, load_index_file, normalize, split_indices, test_subset
from .model import ResNet18
from .partitions import ari, load_partition, true_partition
from .utils import (
    ComputeLog, atomic_torch_save, env_info, past_deadline, pick_device, rng_state, same_config_or_exit, set_rng_state,
    set_seed, stop_for_resume, write_jsonl,
)


def grad_norms(model, x, y, aux_y, stages):
    """One forward pass; Frobenius norms of dL_fine/dh_s and dL_aux,s/dh_s at each stage s (D4).

    Returns (norms {s: (fine, aux)}, out) where out holds logits, loss_fine and aux_losses with
    the graph retained, so the caller can form Eq. 4 and call backward.
    """
    logits, aux, st = model(x, return_stages=True)
    loss_fine = F.cross_entropy(logits.float(), y)
    aux_losses = {s: F.cross_entropy(aux[s].float(), aux_y) for s in stages}
    hs = [st[s] for s in stages]
    g_fine = torch.autograd.grad(loss_fine, hs, retain_graph=True)
    norms = {}
    for s, h, gf in zip(stages, hs, g_fine):
        (ga,) = torch.autograd.grad(aux_losses[s], h, retain_graph=True)
        norms[s] = (float(gf.float().norm()), float(ga.float().norm()))
    return norms, {"logits": logits, "loss_fine": loss_fine, "aux_losses": aux_losses}


class Calibrator:
    """Running means of ||g_fine|| and ||g_aux|| per stage; lambda_s = rho * mean_fine / mean_aux."""

    def __init__(self, stages, rho: float):
        self.rho = rho
        self.fine = {s: 0.0 for s in stages}
        self.aux = {s: 0.0 for s in stages}
        self.n = 0

    def update(self, norms: dict) -> None:
        for s, (f, a) in norms.items():
            self.fine[s] += f
            self.aux[s] += a
        self.n += 1

    def lambdas(self) -> dict:
        return {s: self.rho * self.fine[s] / max(self.aux[s], 1e-12) for s in self.fine}

    def means(self) -> dict:
        return {s: {"fine": self.fine[s] / max(self.n, 1), "aux": self.aux[s] / max(self.n, 1)} for s in self.fine}


def _aux_assign(cfg, ds):
    """Aux label map (class -> aux target) and the run's own partition dict (None for ce / flat_aux)."""
    pct = int(round(cfg.budget * 100))
    if cfg.condition in ("true_hier", "final_hier"):
        a = true_partition(ds.coarse_of_fine)
        part = {"id": f"{cfg.dataset}_b{pct:03d}_true", "name": "true", "kind": "true", "dataset": cfg.dataset,
                "budget": cfg.budget, "assign": a, "W_train": None, "ari_true": 1.0}
        return a, part
    if cfg.condition in ("random_hier", "spectrum_hier"):
        part = load_partition(cfg.partition)
        if len(part["assign"]) != ds.n_classes:
            raise ValueError(f"partition {cfg.partition} has {len(part['assign'])} classes, dataset has {ds.n_classes}")
        part.setdefault("ari_true", ari(part["assign"], ds.coarse_of_fine))
        return part["assign"], part
    if cfg.condition == "flat_aux":
        return np.arange(ds.n_classes), None
    return None, None


@torch.no_grad()
def _predict(model, x_u8_nhwc, device, bs, mean, std, use_amp):
    model.eval()
    out = []
    for i in range(0, len(x_u8_nhwc), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x_u8_nhwc[i:i + bs])).permute(0, 3, 1, 2).to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            logits, _, _ = model(normalize(xb, mean, std))
        out.append(logits.float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def _rows(ids, y, logits):
    p = torch.softmax(logits, 1)
    pred = logits.argmax(1)
    yt = torch.as_tensor(y)
    p_true = p[torch.arange(len(yt)), yt]
    return [
        {"id": int(i), "y": int(t), "pred": int(pr), "p_true": float(pt), "correct": bool(pr == t)}
        for i, t, pr, pt in zip(ids, yt.tolist(), pred.tolist(), p_true.tolist())
    ]


def _example_stats(ids, y, logits_v0, logits_v1, keep_logits: bool = False) -> dict:
    """Per-example outputs for the privacy study, views [image, horizontal flip] stacked on axis 1 ([n, 2]).

    q = log p_y / (1 - p_y) computed stably as z_y - logsumexp_{y' != y} z_y'; loss = -log p_y;
    margin = z_y - max_{y' != y} z_y'; conf = max softmax; pred = argmax (int16).
    """
    z = torch.stack([torch.as_tensor(logits_v0).float(), torch.as_tensor(logits_v1).float()], 1).double()  # [n, 2, C]
    yt = torch.as_tensor(np.asarray(y, np.int64))
    n = z.shape[0]
    zy = z[torch.arange(n), :, yt] if n else z[:, :, 0]
    others = z.clone()
    if n:
        others[torch.arange(n), :, yt] = -math.inf
    lse_all = torch.logsumexp(z, 2)
    out = {
        "ids": np.asarray(ids, np.int64), "y": np.asarray(y, np.int64),
        "q": (zy - torch.logsumexp(others, 2)).float().numpy(),
        "loss": (lse_all - zy).float().numpy(),
        "p_true": torch.exp(zy - lse_all).float().numpy(),
        "pred": z.argmax(2).numpy().astype(np.int16),
        "margin": (zy - others.max(2).values).float().numpy(),
        "conf": torch.exp(z.max(2).values - lse_all).float().numpy(),
    }
    if keep_logits:
        out["logits"] = z.float().numpy().astype(np.float16)
    return out


def train(cfg) -> dict:
    run_dir = Path(cfg.run_dir)
    ckpt_path = run_dir / "ckpt.pt"
    if cfg.skip_if_done and (run_dir / "train_summary.json").exists():
        # Run names do not encode weight decay (or other settings), so refuse to reuse a run with a different config.
        same_config_or_exit(run_dir, cfg)
        return json.loads((run_dir / "train_summary.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    resuming = cfg.skip_if_done and ckpt_path.exists()
    if resuming:
        same_config_or_exit(run_dir, cfg)
    else:
        for f in ("train_log.jsonl", "compute.json", "ckpt.pt", "ckpt.pt.tmp"):
            (run_dir / f).unlink(missing_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    clog = ComputeLog(run_dir / "compute.json")

    device = pick_device(cfg.device)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    set_seed(cfg.seed)
    ds = load_dataset(cfg)
    tr, va = split_indices(cfg, ds)
    if cfg.train_indices:  # explicit training rows (privacy model D_m); per-class limits do not apply
        tr = load_index_file(cfg.train_indices)
        if len(tr) == 0 or len(np.unique(tr)) != len(tr):
            raise ValueError(f"train_indices {cfg.train_indices}: empty or contains duplicates")
        if tr.min() < 0 or tr.max() >= len(ds.y_train):
            raise ValueError(f"train_indices {cfg.train_indices}: rows outside [0, {len(ds.y_train)})")
        if np.intersect1d(tr, va).size:
            raise ValueError(f"train_indices {cfg.train_indices}: overlaps the validation split")
    te = test_subset(len(ds.y_test), cfg.limit_test)
    stages = AUX_STAGES[cfg.condition]
    assign, part = _aux_assign(cfg, ds)
    if part is not None:
        (run_dir / "partition.json").write_text(json.dumps({**part, "assign": [int(v) for v in part["assign"]]}))
    n_aux = ds.n_classes if cfg.condition == "flat_aux" else ds.n_groups
    model = ResNet18(ds.n_classes, stages, n_aux).to(device)

    xtr = torch.from_numpy(ds.x_train[tr]).permute(0, 3, 1, 2).contiguous().to(device)
    ytr = torch.from_numpy(ds.y_train[tr]).long().to(device)
    atr = torch.as_tensor(assign, dtype=torch.long, device=device)[ytr] if stages else None
    n, bs = len(tr), cfg.batch_size
    spe = math.ceil(n / bs)
    if n % bs == 1 and spe > 1:  # never feed a batch of one to BatchNorm
        spe -= 1
    total = cfg.total_steps or cfg.epochs * spe
    n_epochs = math.ceil(total / spe)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    order_gen = torch.Generator().manual_seed(cfg.seed)
    aug_gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    cal = Calibrator(stages, cfg.rho)
    lam = {s: 0.0 for s in stages}
    log, step = [], 0
    xva, yva = ds.x_train[va], ds.y_train[va]

    start_ep, resumes = 1, 0
    if resuming:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        order_gen.set_state(ck["order_gen"])
        aug_gen.set_state(ck["aug_gen"])
        cal.__dict__.update(ck["cal"])
        lam, log, step = ck["lam"], ck["log"], ck["step"]
        set_rng_state(ck["rng"])
        start_ep, resumes = ck["epoch"] + 1, ck.get("resumes", 0) + 1
        write_jsonl(run_dir / "train_log.jsonl", log)
        print(f"[resume] {run_dir}: continuing after epoch {ck['epoch']} of {n_epochs} (resume #{resumes})", flush=True)

    def save_ckpt(ep):
        atomic_torch_save({
            "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "order_gen": order_gen.get_state(), "aug_gen": aug_gen.get_state(), "cal": dict(cal.__dict__),
            "lam": dict(lam), "log": log, "step": step, "epoch": ep, "rng": rng_state(), "resumes": resumes,
        }, ckpt_path)

    if resuming:
        save_ckpt(start_ep - 1)  # record this session even if it stops at once
    if start_ep <= n_epochs and past_deadline(cfg):
        stop_for_resume("session deadline already reached")

    for ep in range(start_ep, n_epochs + 1):
        phase = "calibrate" if (ep == 1 and stages) else "train"
        with clog.phase(phase):
            model.train()
            perm = torch.randperm(n, generator=order_gen).to(device)
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device, dtype=torch.long)
            seen, ratio = 0, None
            for bi in range(spe):
                if step >= total:
                    break
                idx = perm[bi * bs:(bi + 1) * bs]
                x = gpu_augment(xtr[idx], cfg.crop_pad, aug_gen, ds.mean, ds.std)
                y = ytr[idx]
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
                with ctx:
                    if stages and (ep == 1 or bi == 0):
                        norms, out = grad_norms(model, x, y, atr[idx], stages)
                        if ep == 1:
                            cal.update(norms)
                            lam = cal.lambdas()
                        elif bi == 0:
                            ratio = {s: lam[s] * norms[s][1] / max(norms[s][0], 1e-12) for s in stages}
                        logits = out["logits"]
                        loss = out["loss_fine"] + sum(lam[s] * out["aux_losses"][s] for s in stages)
                    else:
                        logits, aux, _ = model(x)
                        loss = F.cross_entropy(logits.float(), y)
                        for s in stages:
                            loss = loss + lam[s] * F.cross_entropy(aux[s].float(), atr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                loss_sum += loss.detach() * len(idx)
                correct += (logits.detach().argmax(1) == y).sum()
                seen += len(idx)
            val_logits = _predict(model, xva, device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
            val_acc = float((val_logits.argmax(1).numpy() == yva).mean()) if len(yva) else None
        row = {"epoch": ep, "step": step, "loss": float(loss_sum) / max(seen, 1), "train_acc": float(correct) / max(seen, 1),
               "val_acc": val_acc, "lambdas": dict(lam), "realized_ratio": ratio, "lr": sched.get_last_lr()[0]}
        log.append(row)
        print(json.dumps(row), flush=True)
        write_jsonl(run_dir / "train_log.jsonl", log)
        if ep < n_epochs:
            debug_stop = cfg.debug_stop_after_epochs is not None and ep >= cfg.debug_stop_after_epochs
            out_of_time = past_deadline(cfg)
            if debug_stop or out_of_time or ep == 1 or ep % max(cfg.ckpt_every_epochs, 1) == 0:
                save_ckpt(ep)
            if out_of_time:
                stop_for_resume(f"session deadline reached after epoch {ep}")
            if debug_stop:
                stop_for_resume(f"debug stop after epoch {ep}")

    with clog.phase("eval"):
        val_logits = _predict(model, xva, device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
        test_logits = _predict(model, ds.x_test[te], device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
        val_rows = _rows(va, yva, val_logits)
        test_rows = _rows(te, ds.y_test[te], test_logits)
        write_jsonl(run_dir / "val.jsonl", val_rows)
        write_jsonl(run_dir / "test.jsonl", test_rows)
        np.save(run_dir / "test_logits.npy", test_logits.numpy().astype(np.float16))
    if cfg.example_sets:
        with clog.phase("examples"):
            sets = json.loads(Path(cfg.example_sets).read_text())
            for name, rows in sets.items():
                ids = np.asarray(rows, np.int64)
                x = ds.x_train[ids]
                l0 = _predict(model, x, device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
                l1 = _predict(model, x[:, :, ::-1], device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
                np.savez(run_dir / f"examples_{name}.npz",
                         **_example_stats(ids, ds.y_train[ids], l0, l1, cfg.example_logits))
    if cfg.save_model:
        torch.save({"state_dict": model.state_dict(), "n_classes": ds.n_classes, "aux_stages": list(stages), "n_aux": n_aux},
                   run_dir / "model.pt")
    summary = {
        "final_val_acc": float(np.mean([r["correct"] for r in val_rows])) if val_rows else None,
        "test_acc": float(np.mean([r["correct"] for r in test_rows])) if test_rows else None,
        "lambdas": dict(lam),
        "grad_norm_epoch1": cal.means() if stages else {},
        "n_train": int(n), "n_val": int(len(va)), "n_test": int(len(te)),
        "condition": cfg.condition, "partition_id": part["id"] if part is not None else None,
        "dataset": cfg.dataset, "budget": cfg.budget, "rho": cfg.rho, "seed": cfg.seed,
        "weight_decay": cfg.weight_decay, "total_steps": cfg.total_steps, "steps": step, "epochs": n_epochs,
        "resumes": resumes,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    if not cfg.keep_ckpt:
        ckpt_path.unlink(missing_ok=True)
    return summary


def _scan(root):
    for s in sorted(Path(root).rglob("train_summary.json")):
        c = s.parent / "config.json"
        if c.exists():
            yield json.loads(c.read_text()), json.loads(s.read_text())


def select_wd(root, out) -> dict:
    """D6: per budget, the weight decay with the highest final validation accuracy (CE, seed 0)."""
    table: dict = {}
    for cfg, summ in _scan(root):
        if cfg["condition"] != "ce" or cfg["seed"] != 0 or cfg.get("total_steps") is not None:
            continue
        table.setdefault(str(float(cfg["budget"])), {})[cfg["weight_decay"]] = summ["final_val_acc"]
    if not table:
        raise SystemExit(f"select_wd: no CE seed-0 runs under {root}")
    best = {b: float(max(sorted(t), key=lambda wd: t[wd])) for b, t in sorted(table.items())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(best, indent=2))
    return best


def select_rho(root, out, budget: float = 1.0) -> dict:
    """D5: rho with the highest mean final validation accuracy over True-Hier, Random-Hier random_00, Flat-Aux."""
    acc: dict = {}
    for cfg, summ in _scan(root):
        cond = cfg["condition"]
        if cond not in ("true_hier", "random_hier", "flat_aux") or cfg["seed"] != 0 or float(cfg["budget"]) != budget:
            continue
        if cfg.get("total_steps") is not None:
            continue
        if cond == "random_hier" and not str(summ.get("partition_id") or "").endswith("_random_00"):
            continue
        acc.setdefault(float(cfg["rho"]), {}).setdefault(cond, []).append(summ["final_val_acc"])
    if not acc:
        raise SystemExit(f"select_rho: no pilot runs under {root}")
    table = {}
    for r, conds in sorted(acc.items()):
        row = {c: float(np.mean(v)) for c, v in conds.items()}
        row["mean"] = float(np.mean(list(row.values())))
        row["n_conditions"] = len(conds)
        table[str(r)] = row
    best = max(sorted(acc), key=lambda r: table[str(r)]["mean"])
    res = {"rho": float(best), "table": table}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
