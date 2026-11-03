"""Index universe (rows of ds.x_train), output layout and small file helpers.

D_beta = sg.data.split_indices(cfg, ds, beta)[0]; A = audit_split limited per class. privred never re-implements
the split. Outputs: $OUT/<dataset>/b{pct}/{sets/<name>/, difficulty.npz, ...}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sg.data import audit_split, limit_per_class, split_indices
from sg.plans import pct


def pool_indices(cfg, ds, budget) -> np.ndarray:
    """D_beta (per-class limits applied), sorted."""
    return np.sort(split_indices(cfg, ds, budget=budget)[0])


def val_indices(cfg, ds, budget) -> np.ndarray:
    return np.sort(split_indices(cfg, ds, budget=budget)[1])


def audit_indices(cfg, ds) -> np.ndarray:
    """Nonmember audit pool A (never trained on), first limit_audit_per_class per class, sorted."""
    au = audit_split(ds.y_train, cfg.val_frac, cfg.audit_frac, cfg.split_seed, ds.n_classes)
    return np.sort(limit_per_class(au, ds.y_train, cfg.limit_audit_per_class))


def budget_dir(cfg, budget) -> Path:
    return Path(cfg.out_dir) / cfg.dataset / f"b{pct(budget):03d}"


def set_dir(cfg, budget, name) -> Path:
    return budget_dir(cfg, budget) / "sets" / name


def member_dir(sdir, m) -> Path:
    return Path(sdir) / f"m{int(m):02d}"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_npz(path, **arrays) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arrays)


def load_npz(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def align(ids_from, ids_to) -> np.ndarray:
    """Positions p with ids_from[p] == ids_to (every id of ids_to must occur in ids_from)."""
    ids_from = np.asarray(ids_from)
    order = np.argsort(ids_from, kind="stable")
    pos = np.searchsorted(ids_from[order], ids_to)
    pos = np.clip(pos, 0, len(order) - 1)
    out = order[pos]
    if len(ids_to) and not (ids_from[out] == ids_to).all():
        raise ValueError("align: ids missing from the source array")
    return out
