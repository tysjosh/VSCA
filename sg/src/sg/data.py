"""CIFAR-100 / iNat-100 loading, Table II nested budgets (D1), GPU augmentation."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BUDGETS = (0.1, 0.25, 0.5, 1.0)
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


@dataclass
class Dataset:
    name: str
    x_train: np.ndarray  # uint8 [N, H, W, 3] (the official training pool)
    y_train: np.ndarray  # int64 [N]
    x_test: np.ndarray
    y_test: np.ndarray
    coarse_of_fine: np.ndarray  # int64 [n_classes]
    fine_names: list
    coarse_names: list
    n_classes: int
    n_groups: int
    pool_per_class: int
    mean: tuple
    std: tuple


# --- CIFAR-100 ------------------------------------------------------------

def _cifar_dir(data_dir) -> Path:
    return Path(data_dir) / "cifar-100-python"


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh, encoding="bytes")


def _cifar_split(data_dir, split: str):
    d = _load_pickle(_cifar_dir(data_dir) / split)
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(d[b"fine_labels"], np.int64), np.asarray(d[b"coarse_labels"], np.int64)


def _coarse_map(fine: np.ndarray, coarse: np.ndarray, n_classes: int) -> np.ndarray:
    cof = np.full(n_classes, -1, np.int64)
    for f in range(n_classes):
        vals = np.unique(coarse[fine == f])
        assert len(vals) == 1, f"fine class {f} maps to coarse labels {vals.tolist()}"
        cof[f] = vals[0]
    return cof


def load_cifar100(data_dir) -> Dataset:
    xtr, ytr, ctr = _cifar_split(data_dir, "train")
    xte, yte, cte = _cifar_split(data_dir, "test")
    meta = _load_pickle(_cifar_dir(data_dir) / "meta")
    cof = _coarse_map(ytr, ctr, 100)
    assert (cof[yte] == cte).all(), "test coarse labels disagree with the train fine->coarse map"
    return Dataset(
        name="cifar100", x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, coarse_of_fine=cof,
        fine_names=[s.decode() for s in meta[b"fine_label_names"]],
        coarse_names=[s.decode() for s in meta[b"coarse_label_names"]],
        n_classes=100, n_groups=20, pool_per_class=int(np.bincount(ytr).min()),
        mean=CIFAR_MEAN, std=CIFAR_STD,
    )


def load_inat100(data_dir) -> Dataset:
    d = Path(data_dir) / "inat100"
    z = np.load(d / "inat100.npz")
    meta = json.loads((d / "meta.json").read_text())
    xtr = z["x_train"]
    mean = tuple(float(v) for v in (xtr.reshape(-1, 3).mean(0) / 255.0))
    std = tuple(float(v) for v in (xtr.reshape(-1, 3).std(0) / 255.0))
    ytr = z["y_train"].astype(np.int64)
    cof = z["coarse_of_fine"].astype(np.int64)
    return Dataset(
        name="inat100", x_train=xtr, y_train=ytr, x_test=z["x_test"], y_test=z["y_test"].astype(np.int64),
        coarse_of_fine=cof, fine_names=meta["fine_names"], coarse_names=meta["coarse_names"],
        n_classes=len(cof), n_groups=int(cof.max()) + 1, pool_per_class=int(np.bincount(ytr).min()),
        mean=mean, std=std,
    )


def load_dataset(cfg) -> Dataset:
    if cfg.dataset == "cifar100":
        return load_cifar100(cfg.data_dir)
    if cfg.dataset == "inat100":
        return load_inat100(cfg.data_dir)
    raise ValueError(cfg.dataset)


# --- budgets (D1) -----------------------------------------------------------

def _allocate(exact: list) -> list:
    """Integer counts per class summing to round(sum(exact)): floor everywhere, +1 for the first classes
    with a fractional part (deterministic, so allocations are nested across budgets)."""
    out = [int(math.floor(v + 1e-9)) for v in exact]
    extra = int(round(sum(exact))) - sum(out)
    for c in range(len(exact)):
        if extra <= 0:
            break
        if exact[c] - out[c] > 1e-9:
            out[c] += 1
            extra -= 1
    return out


def _segments(y: np.ndarray, val_frac: float, audit_frac: float, split_seed: int, n_classes: int):
    """Shared split (D1): one fixed permutation per class, cut into training pool | validation pool | audit pool.

    CIFAR-100 with val_frac = audit_frac = 0.1: 400 | 50 | 50 images per class, i.e. 40,000 / 5,000 / 5,000.
    This is the split shared with the privacy-redistribution study; the audit pool is never trained on here.
    """
    rng = np.random.default_rng(split_seed)
    segs = []
    for c in range(n_classes):
        p = rng.permutation(np.flatnonzero(y == c))
        n_val, n_aud = int(round(val_frac * len(p))), int(round(audit_frac * len(p)))
        n_tr = len(p) - n_val - n_aud
        segs.append((p[:n_tr], p[n_tr:n_tr + n_val], p[n_tr + n_val:]))
    return segs


def budget_split(y: np.ndarray, budget: float, val_frac: float, split_seed: int, n_classes: int, audit_frac: float = 0.0):
    """Nested, stratified train/val indices for a budget (Table II, D1).

    Per class, the budget takes the first round(budget * |training pool|) images of the training pool and the
    first budget * |validation pool| images of the validation pool (fractional counts allocated so totals
    match). Validation is therefore the same share of the labeled budget at every budget (CIFAR-100: 1/9,
    e.g. 4,000 + 500 at 10%) and counts toward it.
    Indices are returned class by class in permutation order.
    """
    segs = _segments(y, val_frac, audit_frac, split_seed, n_classes)
    t_c = _allocate([budget * len(tr) for tr, _, _ in segs])
    v_c = _allocate([budget * len(va) for _, va, _ in segs])
    train = [tr[:t] for (tr, _, _), t in zip(segs, t_c)]
    val = [va[:v] for (_, va, _), v in zip(segs, v_c)]
    return np.concatenate(train).astype(np.int64), np.concatenate(val).astype(np.int64)


def audit_split(y: np.ndarray, val_frac: float, audit_frac: float, split_seed: int, n_classes: int) -> np.ndarray:
    """Nonmember audit pool (5,000 CIFAR-100 images); used only by the privacy study, never trained on."""
    segs = _segments(y, val_frac, audit_frac, split_seed, n_classes)
    return np.concatenate([au for _, _, au in segs]).astype(np.int64)


def limit_per_class(idx: np.ndarray, y: np.ndarray, k) -> np.ndarray:
    """Keep the first k indices of each class, in budget order. k=None keeps all."""
    if k is None:
        return idx
    counts: dict[int, int] = {}
    keep = []
    for i in idx:
        c = int(y[i])
        if counts.get(c, 0) < k:
            counts[c] = counts.get(c, 0) + 1
            keep.append(i)
    return np.asarray(keep, np.int64)


def load_index_file(path) -> np.ndarray:
    """Row indices of the training pool from a ``.npy`` file or a JSON list (privacy study, explicit D_m)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.asarray(np.load(p), np.int64).ravel()
    return np.asarray(json.loads(p.read_text()), np.int64).ravel()


def test_subset(n: int, limit) -> np.ndarray:
    return np.arange(n if limit is None else min(n, limit), dtype=np.int64)


def split_indices(cfg, ds: Dataset, budget=None):
    """Budget train/val indices with the config's per-class limits applied."""
    tr, va = budget_split(ds.y_train, cfg.budget if budget is None else budget, cfg.val_frac, cfg.split_seed,
                          ds.n_classes, cfg.audit_frac)
    return limit_per_class(tr, ds.y_train, cfg.limit_train_per_class), limit_per_class(va, ds.y_train, cfg.limit_val_per_class)


def write_splits(ds: Dataset, val_frac: float, split_seed: int, path, audit_frac: float = 0.0) -> dict:
    out = {"dataset": ds.name, "split_seed": split_seed, "val_frac": val_frac, "audit_frac": audit_frac, "budgets": {}}
    stats = {}
    for b in BUDGETS:
        tr, va = budget_split(ds.y_train, b, val_frac, split_seed, ds.n_classes, audit_frac)
        out["budgets"][str(b)] = {"n_train": len(tr), "n_val": len(va), "train": tr.tolist(), "val": va.tolist()}
        stats[str(b)] = {"n_train": len(tr), "n_val": len(va)}
    au = audit_split(ds.y_train, val_frac, audit_frac, split_seed, ds.n_classes)
    out["audit"] = au.tolist()
    stats["audit"] = {"n": len(au)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))
    return stats


def prepare_cifar100(data_dir, val_frac: float = 0.1, split_seed: int = 1234, audit_frac: float = 0.1) -> dict:
    import torchvision

    for train in (True, False):
        torchvision.datasets.CIFAR100(root=str(data_dir), train=train, download=True)
    ds = load_cifar100(data_dir)  # asserts fine -> coarse is a function
    budgets = write_splits(ds, val_frac, split_seed, Path(data_dir) / "cifar100_splits.json", audit_frac)
    return {"dataset": "cifar100", "n_pool": len(ds.y_train), "n_test": len(ds.y_test), "budgets": budgets}


# --- augmentation -------------------------------------------------------------

def normalize(x_u8: torch.Tensor, mean=CIFAR_MEAN, std=CIFAR_STD) -> torch.Tensor:
    """uint8 [B, 3, H, W] -> normalized float32."""
    m = torch.tensor(mean, device=x_u8.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x_u8.device).view(1, 3, 1, 1)
    return (x_u8.float() / 255.0 - m) / s


def gpu_augment(x_u8_nchw: torch.Tensor, pad: int, gen: torch.Generator, mean=CIFAR_MEAN, std=CIFAR_STD) -> torch.Tensor:
    """Zero-pad random crop + horizontal flip per sample (one gather), then normalize."""
    B, C, H, W = x_u8_nchw.shape
    dev = x_u8_nchw.device
    xp = torch.nn.functional.pad(x_u8_nchw, (pad, pad, pad, pad)) if pad > 0 else x_u8_nchw
    oy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    rows = oy[:, None] + torch.arange(H, device=dev)[None]
    cols = ox[:, None] + torch.arange(W, device=dev)[None]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(C, device=dev)[None, :, None, None]
    out = xp[b, c, rows[:, None, :, None], cols[:, None, None, :]]
    return normalize(out, mean, std)
