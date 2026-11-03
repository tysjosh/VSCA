"""LiRA with leave-one-model-out scoring (Sections 3.4-3.6).

Q[i, m]: Eq. q averaged over the image and its flip. For target m and record i the IN / OUT Gaussians of Eq. lira are
fitted on in_i \\ {m} and out_i \\ {m} (running sums, O(N M)). Variance rule: per-record variances iff every
(record, target) keeps >= min_ref_per_side reference models on both sides (with exactly M/2 IN per record this is
M/2 - 1 >= min_ref_per_side); otherwise one IN and one OUT variance per target, pooled over records (mean of the
LOMO per-record variances, ddof=1, over the records where that side has >= 2 models).
"""

from __future__ import annotations

import numpy as np

VAR_FLOOR = 1e-8


def q_from_logits(logits, y) -> np.ndarray:
    """Eq. q, stable: z_y - logsumexp_{y' != y} z_y'. logits [n, C] (or [n, V, C]), y [n]."""
    z = np.asarray(logits, np.float64)
    y = np.asarray(y, np.int64)
    idx = y.reshape((-1,) + (1,) * (z.ndim - 1))
    zy = np.take_along_axis(z, np.broadcast_to(idx, z.shape[:-1] + (1,)), axis=-1)[..., 0]
    o = np.where(np.arange(z.shape[-1]) == idx, -np.inf, z)
    mx = o.max(-1, keepdims=True)
    lse = mx[..., 0] + np.log(np.exp(o - mx).sum(-1))
    return zy - lse


def lomo_fits(Q, U, min_ref_per_side: int = 15, var_floor: float = VAR_FLOOR) -> dict:
    """LOMO IN / OUT Gaussian parameters for every (record i, target m): arrays [n, M]."""
    Q = np.asarray(Q, np.float64)
    U = np.asarray(U, bool)
    n, M = Q.shape
    shift = Q.mean(1, keepdims=True)  # Gaussians are shift-equivariant; centering keeps the sums well conditioned
    X = Q - shift
    Uf = U.astype(np.float64)
    out = {}
    for side, W in (("in", Uf), ("out", 1.0 - Uf)):
        k = W.sum(1, keepdims=True) - W            # models on this side, excluding target m
        s1 = (W * X).sum(1, keepdims=True) - W * X
        s2 = (W * X * X).sum(1, keepdims=True) - W * X * X
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s1 / k
            var = (s2 - s1 * s1 / k) / (k - 1)
        var = np.where(k >= 2, np.maximum(var, 0.0), np.nan)
        out[f"mu_{side}"] = mu + shift
        out[f"var_{side}"] = var
        out[f"k_{side}"] = k.astype(np.int64)
    per_record = bool(min(out["k_in"].min(), out["k_out"].min()) >= min_ref_per_side) if n else True
    for side in ("in", "out"):
        v = out[f"var_{side}"]
        if not per_record:
            with np.errstate(invalid="ignore"):
                pooled = np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=0)  # [M]
            pooled = np.where(np.isfinite(pooled), pooled, var_floor)
            v = np.broadcast_to(pooled[None, :], (n, M)).copy()
        out[f"var_{side}"] = np.maximum(np.nan_to_num(v, nan=var_floor), var_floor)
    out["per_record"] = per_record
    return out


def lira_scores(Q, fits) -> np.ndarray:
    """Eq. lira: Lambda[i, m] = log N(q; mu_in, var_in) - log N(q; mu_out, var_out)."""
    Q = np.asarray(Q, np.float64)
    vi, vo = fits["var_in"], fits["var_out"]
    return (-0.5 * np.log(vi / vo) - (Q - fits["mu_in"]) ** 2 / (2 * vi) + (Q - fits["mu_out"]) ** 2 / (2 * vo))


def threshold(scores_nonmember, alpha: float) -> float:
    """tau with a fraction <= alpha (exactly alpha without ties) of nonmember scores strictly above it."""
    s = np.sort(np.asarray(scores_nonmember, np.float64).ravel())[::-1]
    if len(s) == 0:
        return float("inf")
    k = int(np.floor(alpha * len(s) + 1e-9))
    return float(s[k]) if k < len(s) else -float("inf")


def attack_metrics(L, U, alphas, min_expected_fp: int = 100) -> dict:
    """Thresholds pooled over targets, TPR / realized FPR at each alpha, reportable flag, global AUC."""
    from sklearn.metrics import roc_auc_score

    L = np.asarray(L, np.float64)
    U = np.asarray(U, bool)
    mem, non = L[U], L[~U]
    res = {"n_member_pairs": int(mem.size), "n_nonmember_pairs": int(non.size), "alphas": {}}
    for a in alphas:
        tau = threshold(non, a)
        res["alphas"][f"{a:g}"] = {
            "tau": tau, "tpr": float((mem > tau).mean()) if mem.size else None,
            "fpr": float((non > tau).mean()) if non.size else None,
            "expected_fp": float(a * non.size), "reportable": bool(a * non.size >= min_expected_fp)}
    res["auc"] = float(roc_auc_score(U.ravel(), L.ravel())) if mem.size and non.size else None
    return res


def vulnerability(L, U, tau: float) -> np.ndarray:
    """Eq. vuln: r_i = fraction of IN models m with Lambda_i^(m) > tau."""
    U = np.asarray(U, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((np.asarray(L) > tau) & U).sum(1) / U.sum(1)


def per_example_auc(L, U) -> np.ndarray:
    """P(Lambda_i^(m) > Lambda_i^(m')) for m in in_i, m' in out_i, ties count 1/2."""
    L = np.asarray(L, np.float64)
    U = np.asarray(U, bool)
    n_in = U.sum(1)
    out = np.full(len(L), np.nan)
    if len(L) and (n_in == n_in[0]).all() and 0 < n_in[0] < U.shape[1]:
        h = int(n_in[0])
        order = np.argsort(~U, axis=1, kind="stable")  # IN columns first
        Li = np.take_along_axis(L, order[:, :h], 1)
        Lo = np.take_along_axis(L, order[:, h:], 1)
        d = Li[:, :, None] - Lo[:, None, :]
        return ((d > 0) + 0.5 * (d == 0)).mean((1, 2))
    for i in range(len(L)):
        a, b = L[i, U[i]], L[i, ~U[i]]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out[i] = float(((d > 0) + 0.5 * (d == 0)).mean())
    return out


def _out_only_fits(Qa, per_record: bool, var_floor: float = VAR_FLOOR):
    """LOMO OUT Gaussian for records that are OUT in every model (audit pool): mean / var over models != m."""
    Qa = np.asarray(Qa, np.float64)
    n, M = Qa.shape
    s1 = Qa.sum(1, keepdims=True) - Qa
    s2 = (Qa * Qa).sum(1, keepdims=True) - Qa * Qa
    k = M - 1
    mu = s1 / k
    var = np.maximum((s2 - s1 * s1 / k) / (k - 1), 0.0)
    if not per_record and n:
        var = np.broadcast_to(var.mean(0, keepdims=True), (n, M)).copy()
    return mu, np.maximum(var, var_floor)


def offline_attack(Q, U, fits, Q_audit, alphas) -> dict:
    """Offline LiRA (OUT Gaussian only): z = (q - mu_out^{-m}) / sigma_out^{-m}. tau calibrated on audit pairs
    (OUT in all models, LOMO excluding m); TPR on pool member pairs, FPR on pool nonmember pairs."""
    from sklearn.metrics import roc_auc_score

    U = np.asarray(U, bool)
    z = (np.asarray(Q, np.float64) - fits["mu_out"]) / np.sqrt(fits["var_out"])
    mu_a, var_a = _out_only_fits(Q_audit, fits["per_record"])
    za = (np.asarray(Q_audit, np.float64) - mu_a) / np.sqrt(var_a)
    res = {"n_audit_pairs": int(za.size), "alphas": {}}
    for a in alphas:
        tau = threshold(za, a)
        res["alphas"][f"{a:g}"] = {"tau": tau, "tpr": float((z[U] > tau).mean()) if U.any() else None,
                                   "fpr_pool": float((z[~U] > tau).mean()) if (~U).any() else None,
                                   "fpr_audit": float((za > tau).mean()) if za.size else None}
    res["auc"] = float(roc_auc_score(U.ravel(), z.ravel())) if U.any() and (~U).any() else None
    return res, z, za
