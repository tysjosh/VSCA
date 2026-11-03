"""Representation probes (Section III-G, D12): cosine k-NN and linear probes at the 8 residual blocks,
neighbour purity, prediction depth (Eq. 6) and manipulation-check flags."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import AUX_STAGES
from .data import load_dataset, normalize, split_indices, test_subset
from .model import BLOCK_STAGE, ResNet18
from .utils import ComputeLog, pick_device, write_jsonl

# Fields taken from the probed run's own config.json (data subset and condition must match the run).
RUN_FIELDS = ("dataset", "data_dir", "split_seed", "val_frac", "limit_train_per_class", "limit_val_per_class",
              "limit_test", "condition")


def knn_neighbors(ref: torch.Tensor, query: torch.Tensor, k: int, chunk: int = 2048):
    """Indices and cosine similarities of the k nearest reference rows (rows already L2-normalized)."""
    k = min(k, ref.shape[0])
    idx, sim = [], []
    for i in range(0, query.shape[0], chunk):
        s = query[i:i + chunk] @ ref.T
        v, j = s.topk(k, dim=1)
        idx.append(j)
        sim.append(v)
    return torch.cat(idx), torch.cat(sim)


def knn_vote(nbr_labels: torch.Tensor, nbr_sim: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Majority vote; ties broken by the larger summed similarity of the tied labels."""
    B, k = nbr_labels.shape
    counts = torch.zeros(B, n_classes, dtype=torch.float64, device=nbr_labels.device)
    counts.scatter_add_(1, nbr_labels, torch.ones_like(nbr_sim, dtype=torch.float64))
    sims = torch.zeros(B, n_classes, dtype=torch.float64, device=nbr_labels.device)
    sims.scatter_add_(1, nbr_labels, nbr_sim.double())
    # counts are integers; (sims + k) / (2k + 1) lies in [0, 1), so it only orders labels with equal counts.
    return (counts + (sims + k) / (2 * k + 1)).argmax(1)


def knn_predict(ref_feats, ref_y, query_feats, k: int = 20, n_classes: int | None = None):
    """Cosine k-NN predictions. Returns (pred [N], neighbour index [N, k])."""
    ref = F.normalize(torch.as_tensor(ref_feats).float(), dim=1)
    q = F.normalize(torch.as_tensor(query_feats).float(), dim=1)
    ry = torch.as_tensor(ref_y).long().to(ref.device)
    n_classes = n_classes or int(ry.max()) + 1
    idx, sim = knn_neighbors(ref, q, k)
    return knn_vote(ry[idx], sim, n_classes), idx


def neighbor_purity(nbr_labels: torch.Tensor, y) -> float:
    """Mean fraction of the k neighbours whose fine label equals the query's ground-truth label."""
    y = torch.as_tensor(y).long().to(nbr_labels.device)
    return float((nbr_labels == y[:, None]).double().mean())


def linear_probe(train_x, train_y, test_x, n_classes: int, iters: int = 500, l2: float = 1e-4, device="cpu"):
    """Standardized features, nn.Linear, full-batch L-BFGS with L2 penalty on the weights. Returns test preds."""
    xtr = torch.as_tensor(train_x).float().to(device)
    xte = torch.as_tensor(test_x).float().to(device)
    ytr = torch.as_tensor(train_y).long().to(device)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    lin = torch.nn.Linear(xtr.shape[1], n_classes).to(device)
    with torch.no_grad():  # convex objective: start from zero so the probe is deterministic
        lin.weight.zero_()
        lin.bias.zero_()
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(xtr), ytr) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return lin(xte).argmax(1).cpu()


def prediction_depth(preds) -> np.ndarray:
    """Eq. 6: smallest l in 1..9 with preds[j] == preds[9] for all j >= l (1-based). preds: [N, 9]."""
    P = np.asarray(preds)
    agree = P == P[:, -1:]
    trailing = np.cumprod(agree[:, ::-1], axis=1).sum(1)  # >= 1, because column 9 agrees with itself
    return (P.shape[1] + 1 - trailing).astype(np.int64)


@torch.no_grad()
def _block_feats(model, x_u8_nhwc, device, bs, mean, std, use_amp):
    """L2-normalized GAP features of the 8 blocks (CPU float32) and the classifier's predictions."""
    model.eval()
    feats, preds = [[] for _ in range(8)], []
    for i in range(0, len(x_u8_nhwc), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x_u8_nhwc[i:i + bs])).permute(0, 3, 1, 2).to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            bf = model.block_features(normalize(xb, mean, std))
            logits = model.fc(bf[-1])
        for j, f in enumerate(bf):
            feats[j].append(F.normalize(f.float(), dim=1).cpu())
        preds.append(logits.float().argmax(1).cpu())
    return [torch.cat(f) for f in feats], torch.cat(preds)


def run_probes(cfg, run_dir) -> dict:
    """Probe a trained run (needs model.pt). Writes probes.json and pd.jsonl into ``run_dir``."""
    run_dir = Path(run_dir)
    if not (run_dir / "model.pt").exists():
        raise SystemExit(f"probes: {run_dir}/model.pt not found (train the run with save_model=true)")
    run_cfg = json.loads((run_dir / "config.json").read_text())
    cfg = dataclasses.replace(cfg, **{k: run_cfg[k] for k in RUN_FIELDS if k in run_cfg})
    device = pick_device(cfg.device)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    clog = ComputeLog(run_dir / "compute.json")

    with clog.phase("probes"):
        ds = load_dataset(cfg)
        ck = torch.load(run_dir / "model.pt", map_location=device)
        model = ResNet18(ck["n_classes"], tuple(ck["aux_stages"]), ck["n_aux"])
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()

        ref = split_indices(cfg, ds, budget=cfg.probe_ref_budget)[0]  # fixed reference set (D12)
        te = test_subset(len(ds.y_test), cfg.limit_test)
        ref_f, _ = _block_feats(model, ds.x_train[ref], device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
        te_f, final_pred = _block_feats(model, ds.x_test[te], device, cfg.eval_batch_size, ds.mean, ds.std, use_amp)
        ref_y = torch.from_numpy(ds.y_train[ref]).long()
        te_y = torch.from_numpy(ds.y_test[te]).long()

        a_true = torch.as_tensor(ds.coarse_of_fine).long()
        a_own = None
        if cfg.condition not in ("ce", "flat_aux") and (run_dir / "partition.json").exists():
            a_own = torch.as_tensor(json.loads((run_dir / "partition.json").read_text())["assign"]).long()
        head_stages = {int(s[1]) for s in AUX_STAGES[cfg.condition]}

        blocks, knn_preds = [], []
        for b in range(8):
            r, q = ref_f[b].to(device), te_f[b].to(device)
            idx, sim = knn_neighbors(r, q, cfg.knn_k)
            idx, sim = idx.cpu(), sim.cpu()
            nbr = ref_y[idx]
            fine = knn_vote(nbr, sim, ds.n_classes)
            knn_preds.append(fine)
            row = {"block": b + 1, "stage": BLOCK_STAGE[b], "manipulation_check": BLOCK_STAGE[b] in head_stages,
                   "knn_fine": float((fine == te_y).double().mean())}
            for frame, a in (("true", a_true), ("own", a_own)):
                if a is None:
                    row[f"knn_coarse_{frame}"] = None
                    continue
                g = knn_vote(a[nbr], sim, int(a.max()) + 1)
                row[f"knn_coarse_{frame}"] = float((g == a[te_y]).double().mean())
            row["purity"] = neighbor_purity(nbr, te_y)
            lin = linear_probe(r, ref_y, q, ds.n_classes, cfg.probe_linear_iters, cfg.probe_l2, device)
            row["linear_fine"] = float((lin == te_y).double().mean())
            for frame, a in (("true", a_true), ("own", a_own)):
                if a is None:
                    row[f"linear_coarse_{frame}"] = None
                    continue
                g = linear_probe(r, a[ref_y], q, int(a.max()) + 1, cfg.probe_linear_iters, cfg.probe_l2, device)
                row[f"linear_coarse_{frame}"] = float((g == a[te_y]).double().mean())
            blocks.append(row)
            print(json.dumps(row), flush=True)

        preds = torch.stack(knn_preds + [final_pred], 1).numpy()
        pd = prediction_depth(preds)
        hist = {d: int((pd == d).sum()) for d in range(1, 10)}
        n = len(pd)
        out = {"blocks": blocks, "pd_hist": hist, "pd_frac": {d: (c / n if n else None) for d, c in hist.items()},
               "n_ref": int(len(ref)), "n_test": int(n), "condition": cfg.condition, "knn_k": cfg.knn_k}
        (run_dir / "probes.json").write_text(json.dumps(out, indent=2))
        write_jsonl(run_dir / "pd.jsonl", [{"id": int(i), "pd": int(d), "preds": [int(v) for v in p]}
                                           for i, d, p in zip(te, pd, preds)])
    return out
