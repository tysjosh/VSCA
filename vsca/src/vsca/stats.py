"""Paired statistics over items and seeds, Holm correction, and seed-count power analysis (Section 4.6)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .utils import read_jsonl


def load_runs(paths: list[str]) -> tuple[list[str], np.ndarray]:
    """Per-item correctness matrix [n_seeds, n_items] over items common to all runs."""
    runs = [{r["id"]: bool(r["correct"]) for r in read_jsonl(p)} for p in paths]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    return ids, np.array([[r[i] for i in ids] for r in runs], dtype=float)


def paired_bootstrap(a_paths: list[str], b_paths: list[str], n_boot: int = 10000, seed: int = 0) -> dict:
    ids_a, A = load_runs(a_paths)
    ids_b, B = load_runs(b_paths)
    common = sorted(set(ids_a) & set(ids_b))
    ia = [ids_a.index(i) for i in common]
    ib = [ids_b.index(i) for i in common]
    A, B = A[:, ia], B[:, ib]
    rng = np.random.default_rng(seed)
    n = len(common)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        items = rng.integers(0, n, n)
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = rng.integers(0, B.shape[0], B.shape[0])
        diffs[k] = A[sa][:, items].mean() - B[sb][:, items].mean()
    obs = A.mean() - B.mean()
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "n_items": n,
        "seeds_a": A.shape[0],
        "seeds_b": B.shape[0],
        "acc_a": float(A.mean()),
        "acc_b": float(B.mean()),
        "diff": float(obs),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_two_sided": float(min(1.0, p)),
    }


def mcnemar_exact(a_path: str, b_path: str) -> dict:
    ids_a, A = load_runs([a_path])
    ids_b, B = load_runs([b_path])
    common = sorted(set(ids_a) & set(ids_b))
    a = A[0, [ids_a.index(i) for i in common]].astype(bool)
    b = B[0, [ids_b.index(i) for i in common]].astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "discordant_rate": n / len(common), "p": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def seeds_needed(seed_accs: list[float], discordant_rate: float, n_items: int, mde: float = 0.015,
                 alpha: float = 0.05, power: float = 0.8, minimum: int = 5) -> dict:
    """Seeds per method so that a paired difference of ``mde`` is detected.

    Var(mean diff over n seed pairs) ≈ (2 σ_seed² + discordant_rate / n_items) / n.
    """
    sd = float(np.std(seed_accs, ddof=1)) if len(seed_accs) > 1 else 0.0
    var1 = 2 * sd**2 + discordant_rate / n_items
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    n = math.ceil(z**2 * var1 / mde**2) if var1 > 0 else 1
    return {"seed_sd": sd, "per_pair_var": var1, "required": n, "use": max(n, minimum)}
