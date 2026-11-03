"""Outcome factorization (Eqs. 1-2), cross-fitted difficulty strata (Section III-F, Eq. 5, D3),
continuous CE-A probability curve and label-error exclusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRICS = ("acc", "C", "S", "F")


def factorize(y, pred, assign) -> dict:
    """Eqs. 1-2 for partition ``assign`` (class -> group).

    C = P(pred in G(y)), S = P(pred == y | pred in G(y)), acc = C * S,
    F = C(1-S)/(1-CS) and F_direct = (#errors with assign[pred] == assign[y]) / #errors.
    S is None when no prediction falls in the true group; F / F_direct are None without errors.
    """
    y = np.asarray(y, np.int64)
    pred = np.asarray(pred, np.int64)
    assign = np.asarray(assign, np.int64)
    n = int(len(y))
    in_group = assign[pred] == assign[y]
    correct = pred == y
    n_in, n_corr = int(in_group.sum()), int(correct.sum())
    n_err = n - n_corr
    acc = n_corr / n if n else None
    C = n_in / n if n else None
    S = n_corr / n_in if n_in else None
    F = C * (1 - S) / (1 - C * S) if S is not None and n_err > 0 else None
    F_direct = int((in_group & ~correct).sum()) / n_err if n_err > 0 else None
    return {"n": n, "n_in_group": n_in, "n_correct": n_corr, "acc": acc, "C": C, "S": S, "F": F, "F_direct": F_direct}


def run_outcomes(rows, assign_true, assign_own=None) -> dict:
    """acc and C/S/F in the true frame and (if ``assign_own`` is given) the own frame for one run's test rows."""
    y = np.array([r["y"] for r in rows], np.int64)
    pred = np.array([r["pred"] for r in rows], np.int64)
    return outcomes_from_preds(y, pred, assign_true, assign_own)


def outcomes_from_preds(y, pred, assign_true, assign_own=None) -> dict:
    """Array form of ``run_outcomes``: keys acc, C_true, S_true, F_true, C_own, S_own, F_own."""
    t = factorize(y, pred, assign_true)
    out = {"acc": t["acc"], "C_true": t["C"], "S_true": t["S"], "F_true": t["F"]}
    o = factorize(y, pred, assign_own) if assign_own is not None else None
    for m in ("C", "S", "F"):
        out[f"{m}_own"] = o[m] if o is not None else None
    return out


def metric_value(y, pred, idx, assign, name: str):
    """Metric ``name`` (acc | C | S | F) of predictions ``pred`` on the item subset ``idx``."""
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        return None
    return factorize(np.asarray(y)[idx], np.asarray(pred)[idx], assign)[name]


def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _sub(a, b):
    return None if a is None or b is None else a - b


def crossfit_folds(ce_ptrue, hard_frac: float = 0.25, exclude=None) -> list:
    """Fold f uses CE seed f as CE-A; its hard stratum is the round(hard_frac*N) items with the lowest
    CE-A true-class probability (stable argsort). Items at positions in ``exclude`` are then removed.

    ce_ptrue: [K, N]. Returns [{"ref": f, "others": [b != f], "hard": positions}] for f = 0..K-1.
    """
    P = np.asarray(ce_ptrue, np.float64)
    K, N = P.shape
    n_hard = int(round(hard_frac * N))
    excl = np.zeros(N, bool)
    if exclude is not None:
        e = np.asarray(sorted(exclude), np.int64)
        excl[e[(e >= 0) & (e < N)]] = True
    folds = []
    for f in range(K):
        hard = np.argsort(P[f], kind="stable")[:n_hard]
        hard = np.sort(hard[~excl[hard]])
        folds.append({"ref": f, "others": [b for b in range(K) if b != f], "hard": hard})
    return folds


def stratum_deltas(metric_fn, ce_preds_by_seed, model_preds_by_seed, folds) -> dict:
    """Eq. 5 with cross-fitting (D3), averaged over folds.

    Delta_null = mean_{b != f} M(CE_b) - M(CE_f); Delta_G = mean_s M(model_s) - mean_{b != f} M(CE_b),
    both on fold f's hard stratum. ``metric_fn(pred, idx)`` returns the metric or None.
    ``model_preds_by_seed=None`` gives Delta_null only.
    """
    rows = []
    for fd in folds:
        h = fd["hard"]
        m_a = metric_fn(ce_preds_by_seed[fd["ref"]], h)
        m_b = _mean([metric_fn(ce_preds_by_seed[b], h) for b in fd["others"]])
        m_g = _mean([metric_fn(p, h) for p in model_preds_by_seed]) if model_preds_by_seed is not None else None
        rows.append({"ref": fd["ref"], "n_hard": int(len(h)), "ce_a": m_a, "ce_b": m_b, "model": m_g,
                     "delta_null": _sub(m_b, m_a), "delta_G": _sub(m_g, m_b)})
    return {"delta_null": _mean([r["delta_null"] for r in rows]), "delta_G": _mean([r["delta_G"] for r in rows]),
            "folds": rows}


def probability_curve(metric_fn, ce_ptrue, ce_preds_by_seed, model_preds_by_seed, n_bins: int = 10) -> list:
    """Outcomes as a function of CE-A true-class probability: per fold, n_bins equal-count quantile bins of
    CE-A p_true; in each bin M(CE-A), mean M(CE-B), mean M(model) and the two deltas; averaged over folds."""
    P = np.asarray(ce_ptrue, np.float64)
    K, N = P.shape
    per_bin = [[] for _ in range(n_bins)]
    for f in range(K):
        order = np.argsort(P[f], kind="stable")
        bins = np.empty(N, np.int64)
        bins[order] = np.arange(N) * n_bins // max(N, 1)
        others = [b for b in range(K) if b != f]
        for k in range(n_bins):
            idx = np.flatnonzero(bins == k)
            if len(idx) == 0:
                continue
            m_a = metric_fn(ce_preds_by_seed[f], idx)
            m_b = _mean([metric_fn(ce_preds_by_seed[b], idx) for b in others])
            m_g = _mean([metric_fn(p, idx) for p in model_preds_by_seed]) if model_preds_by_seed is not None else None
            per_bin[k].append({"p_lo": float(P[f][idx].min()), "p_hi": float(P[f][idx].max()), "n": int(len(idx)),
                               "ce_a": m_a, "ce_b": m_b, "model": m_g,
                               "delta_null": _sub(m_b, m_a), "delta_G": _sub(m_g, m_b)})
    out = []
    for k, rows in enumerate(per_bin):
        if not rows:
            continue
        out.append({"bin": k, **{key: _mean([r[key] for r in rows]) for key in
                                 ("p_lo", "p_hi", "n", "ce_a", "ce_b", "model", "delta_null", "delta_G")}})
    return out


def load_label_errors(path) -> set:
    """Test ids flagged as mislabeled: a json list of ints, or one int per line."""
    text = Path(path).read_text().strip()
    if not text:
        return set()
    if text.startswith("["):
        return {int(v) for v in json.loads(text)}
    return {int(line.split()[0]) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
