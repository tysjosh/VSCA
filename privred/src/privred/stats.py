"""Statistics for the privacy studies (Section 4.3): bootstrap p-values and intervals, the similarity-adjusted
residual with covariates Z(G) (sensitivity of Eq. residual), leave-one-partition-out residuals, the extrapolation
flag, and the P1 two-way bootstrap (records x target models)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def p_two_sided(boot) -> float | None:
    """p = min(1, 2 min((1 + #{b <= 0}) / (B + 1), (1 + #{b >= 0}) / (B + 1)))."""
    b = np.asarray(boot, np.float64)
    b = b[np.isfinite(b)]
    if not len(b):
        return None
    B = len(b)
    return float(min(1.0, 2 * min((1 + (b <= 0).sum()) / (B + 1), (1 + (b >= 0).sum()) / (B + 1))))


def percentile_ci(boot, level: float = 0.95):
    b = np.asarray(boot, np.float64)
    b = b[np.isfinite(b)]
    if not len(b):
        return None
    lo = (1 - level) / 2
    return [float(np.quantile(b, lo)), float(np.quantile(b, 1 - lo))]


def two_sided_from_one(p_one) -> float | None:
    """Two-sided p from sg.stats.residual's one-sided p: 2 min(p, 1 - p)."""
    return None if p_one is None else float(min(1.0, 2 * min(p_one, 1 - p_one)))


# --- residual with covariates ------------------------------------------------------------------

def _standardize(cols, center=None, scale=None):
    X = np.column_stack(cols).astype(np.float64)
    if center is None:
        center = X.mean(0)
        scale = X.std(0)
        scale = np.where(scale > 0, scale, 1.0)
    return (X - center) / scale, center, scale


def residual_cov(W, T, W_true, T_true, Z=None, Z_true=None, level: float = 0.95) -> dict:
    """OLS T = a + b W + omega' Z over the non-curated partitions; delta = T_true - fit(W_true, Z_true), with a
    ``level`` prediction interval and one- / two-sided t p-values. Without Z this is sg.stats.residual (degree 1).
    Output keys mirror sg.stats.residual plus p_two_sided and n_covariates."""
    W = np.asarray(W, np.float64)
    T = np.asarray(T, np.float64)
    cols = [W]
    x0cols = [float(W_true)]
    if Z is not None:
        Z = np.asarray(Z, np.float64).reshape(len(W), -1)
        cols += [Z[:, j] for j in range(Z.shape[1])]
        x0cols += list(np.asarray(Z_true, np.float64).ravel())
    Xs, center, scale = _standardize(cols)
    X = np.column_stack([np.ones(len(W)), Xs])
    x0 = np.concatenate([[1.0], (np.asarray(x0cols) - center) / scale])
    n, p = X.shape
    df = n - p
    if df < 0:
        raise ValueError(f"residual_cov needs at least {p} partitions, got {n}")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ T
    pred = float(x0 @ beta)
    delta = float(T_true) - pred
    raw = np.concatenate([[beta[0] - float(np.sum(beta[1:] * center / scale))], beta[1:] / scale])
    out = {"delta": delta, "pred_true": pred, "se_pred": None, "df": int(df), "level": level, "pi_low": None,
           "pi_high": None, "p_one_sided": None, "p_two_sided": None, "coef": [float(c) for c in raw], "degree": 1,
           "n": int(n), "resid_sd": None, "n_covariates": int(p - 2)}
    if df == 0:
        out["note"] = "df = 0: interval and p-value undefined"
        return out
    res = T - X @ beta
    s2 = float(res @ res) / df
    se = float(np.sqrt(s2 * (1.0 + x0 @ XtX_inv @ x0)))
    tq = float(sps.t.ppf(0.5 + level / 2, df))
    p_one = float(sps.t.sf(delta / se, df)) if se > 0 else (0.0 if delta > 0 else (1.0 if delta < 0 else 0.5))
    out.update({"se_pred": se, "pi_low": delta - tq * se, "pi_high": delta + tq * se, "p_one_sided": p_one,
                "p_two_sided": two_sided_from_one(p_one), "resid_sd": float(np.sqrt(s2))})
    return out


def lopo_residuals(W, T, Z=None) -> list:
    """Leave-one-partition-out residuals: for each non-curated partition k, T_k minus the fit on the others."""
    W = np.asarray(W, np.float64)
    T = np.asarray(T, np.float64)
    out = []
    for k in range(len(W)):
        keep = np.arange(len(W)) != k
        zk = None if Z is None else np.asarray(Z, np.float64).reshape(len(W), -1)
        try:
            r = residual_cov(W[keep], T[keep], W[k], T[k], None if zk is None else zk[keep],
                             None if zk is None else zk[k])
            out.append(r["delta"])
        except ValueError:
            out.append(None)
    return out


def extrapolation(W, W_true) -> bool:
    """True iff W(G_true) lies outside the non-curated range of W."""
    W = np.asarray(W, np.float64)
    return bool(W_true < W.min() or W_true > W.max())


# --- P1 two-way bootstrap -------------------------------------------------------------------------

def _tau(non, alpha: float) -> float:
    """lira.threshold via np.partition: the floor(alpha N)-th largest nonmember score (0-based)."""
    N = non.size
    k = int(np.floor(alpha * N + 1e-9))
    if N == 0:
        return float("inf")
    if k >= N:
        return -float("inf")
    return float(np.partition(non, N - 1 - k)[N - 1 - k])


def condition_tpr(L, U, alpha: float) -> float:
    """TPR over member pairs at the threshold with nonmember-pair FPR alpha (= mean r_i when every record has the
    same number of IN models)."""
    U = np.asarray(U, bool)
    tau = _tau(np.asarray(L, np.float64)[~U], alpha)
    return float((L[U] > tau).mean()) if U.any() else float("nan")


def p1_bootstrap(L_a, U_a, L_b, U_b, alpha: float, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Difference TPR(b) - TPR(a) at FPR alpha, two-way bootstrap: records resampled with replacement and paired
    across the two sets (same pool order), target-model columns resampled within each set; tau and the TPR are
    recomputed per draw from the stored Lambda matrices (reference Gaussians are not refitted: approximation)."""
    L_a, L_b = np.asarray(L_a, np.float64), np.asarray(L_b, np.float64)
    U_a, U_b = np.asarray(U_a, bool), np.asarray(U_b, bool)
    if L_a.shape[0] != L_b.shape[0]:
        raise ValueError("p1_bootstrap: the two sets must score the same records")
    n = L_a.shape[0]
    obs = condition_tpr(L_b, U_b, alpha) - condition_tpr(L_a, U_a, alpha)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for t in range(n_boot):
        i = rng.integers(0, n, n)
        ma = rng.integers(0, L_a.shape[1], L_a.shape[1])
        mb = rng.integers(0, L_b.shape[1], L_b.shape[1])
        boot[t] = (condition_tpr(L_b[np.ix_(i, mb)], U_b[np.ix_(i, mb)], alpha)
                   - condition_tpr(L_a[np.ix_(i, ma)], U_a[np.ix_(i, ma)], alpha))
    return {"estimate": float(obs), "ci95": percentile_ci(boot, level), "p_two_sided": p_two_sided(boot),
            "n_boot": int(n_boot), "n_records": int(n), "M": [int(L_a.shape[1]), int(L_b.shape[1])], "alpha": alpha,
            "approximation": "target-model columns of the stored Lambda resampled; reference fits not refitted"}


# --- partition-level interval for delta^priv ---------------------------------------------------------

def _boot_mean_r(L, U, i, m, alpha: float) -> np.ndarray:
    """Per-record r_i (Eq. vuln) on records i and target columns m, tau re-pooled over the drawn nonmember pairs."""
    Lb, Ub = L[np.ix_(i, m)], U[np.ix_(i, m)]
    tau = _tau(Lb[~Ub], alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((Lb > tau) & Ub).sum(1) / Ub.sum(1)


def partition_bootstrap(ce, true, noncur, W_nc, W_true, alpha: float, n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> dict:
    """Interval for delta^priv (Eq. residual) that adds resampling over partitions (sec:stats, units of inference).
    ``ce``, ``true`` and each entry of ``noncur`` are (Lambda, u) restricted to the analysed records, same pool
    order. Per draw: records resampled with replacement and paired across sets, target-model columns resampled
    within each set, T^priv(G) = mean_i Delta^sup_i(G) recomputed for G_true and every non-curated G, then the
    non-curated partitions resampled with replacement and delta^priv refitted (degree 1). Draws whose partitions
    have fewer than 3 distinct W are skipped. Reference Gaussians are not refitted (as p1_bootstrap)."""
    from sg.stats import residual

    L_ce, U_ce = np.asarray(ce[0], np.float64), np.asarray(ce[1], bool)
    sets = [(np.asarray(L, np.float64), np.asarray(U, bool)) for L, U in [true] + list(noncur)]
    W_nc = np.asarray(W_nc, np.float64)
    n, P = L_ce.shape[0], len(W_nc)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r_ce = _boot_mean_r(L_ce, U_ce, i, rng.integers(0, L_ce.shape[1], L_ce.shape[1]), alpha)
        T = []
        for L, U in sets:
            d = _boot_mean_r(L, U, i, rng.integers(0, L.shape[1], L.shape[1]), alpha) - r_ce
            d = d[np.isfinite(d)]
            T.append(float(d.mean()) if len(d) else np.nan)
        k = rng.integers(0, P, P)
        Tv = np.asarray(T[1:])[k]
        if len(np.unique(W_nc[k])) < 3 or not (np.isfinite(T[0]) and np.isfinite(Tv).all()):
            continue
        boot.append(residual(W_nc[k], Tv, W_true, T[0], degree=1)["delta"])
    boot = np.asarray(boot)
    return {"available": bool(len(boot)), "ci95": percentile_ci(boot, level) if len(boot) else None,
            "p_two_sided": p_two_sided(boot) if len(boot) else None, "n_boot": int(n_boot),
            "n_valid": int(len(boot)), "n_partitions": int(P), "n_records": int(n), "alpha": alpha,
            "resampling": "records (paired across sets), target models within each set, non-curated partitions",
            "approximation": "target-model columns of the stored Lambda resampled; reference fits not refitted",
            "exploratory": True}


# --- P3 / P4: cohort-mean differences ---------------------------------------------------------------

def cohort_record_bootstrap(per_cohort, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean over cohorts of the cohort-mean per-record differences; bootstrap resamples cohorts, then records
    within each drawn cohort. Two-sided p (p_two_sided)."""
    arrs = [np.asarray(a, np.float64)[np.isfinite(a)] for a in per_cohort]
    arrs = [a for a in arrs if len(a)]
    if not arrs:
        return {"available": False, "note": "no cohort with finite differences"}
    means = np.array([a.mean() for a in arrs])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for t in range(n_boot):
        ks = rng.integers(0, len(arrs), len(arrs))
        boot[t] = np.mean([arrs[k][rng.integers(0, len(arrs[k]), len(arrs[k]))].mean() for k in ks])
    return {"available": True, "estimate": float(means.mean()), "cohort_means": means.tolist(),
            "n_cohorts": int(len(arrs)), "ci95": percentile_ci(boot, level), "p_two_sided": p_two_sided(boot),
            "n_boot": int(n_boot)}


def sign_flip_p(cohort_means) -> float | None:
    """Exact two-sided sign-flip p over cohorts: fraction of the 2^Q sign vectors s with |mean(s c)| >= |mean(c)|."""
    c = np.asarray(cohort_means, np.float64)
    c = c[np.isfinite(c)]
    if not len(c):
        return None
    Q = len(c)
    signs = 1 - 2 * ((np.arange(2 ** Q)[:, None] >> np.arange(Q)[None, :]) & 1)
    stat = np.abs((signs * c).mean(1))
    return float(np.mean(stat >= abs(c.mean()) - 1e-12))


# --- P5: held-out MAE reduction over cohorts ------------------------------------------------------

def p5_bootstrap(groups, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """MAE_base - MAE_full on pooled held-out rows; one-sided bootstrap over held-out cohorts (groups = per test
    cohort {n_test, abs_err_base, abs_err_full}). p = (1 + #{b <= 0}) / (B + 1)."""
    if not groups:
        return {"available": False, "p_one_sided": None}
    n = np.array([g["n_test"] for g in groups], np.float64)
    eb = np.array([g["abs_err_base"] for g in groups], np.float64)
    ef = np.array([g["abs_err_full"] for g in groups], np.float64)
    est = float((eb.sum() - ef.sum()) / n.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for t in range(n_boot):
        k = rng.integers(0, len(n), len(n))
        boot[t] = (eb[k].sum() - ef[k].sum()) / n[k].sum()
    return {"available": True, "estimate": est, "n_cohorts": int(len(n)), "ci95": percentile_ci(boot, level),
            "p_one_sided": float((1 + (boot <= 0).sum()) / (n_boot + 1)), "n_boot": int(n_boot)}


# --- primary family (P1-P5) with Holm -------------------------------------------------------------

PRIMARY = ("P1", "P2", "P3", "P4", "P5")


def holm_family(p_raw: dict, alpha_family: float = 0.05) -> tuple:
    """sg.stats.holm over the available p-values (None stays None); reject iff p_holm <= alpha_family."""
    from sg.stats import holm

    avail = {k: float(v) for k, v in p_raw.items() if v is not None}
    adj = holm(avail) if avail else {}
    p_holm = {k: adj.get(k) for k in p_raw}
    reject = {k: (None if v is None else bool(v <= alpha_family)) for k, v in p_holm.items()}
    return p_holm, reject


def primary(cfg, out=None) -> dict:
    """P1 (True-Hier - CE TPR at alpha, Study A at the first privacy budget), P2 (delta^priv), P3 / P4 (CE cohort-mean
    sem - vis / sem - rand on T_q), P5 (held-out MAE reduction, leave-one-cohort-out, ridge) -> analysis/primary.json."""
    from pathlib import Path

    from sg.plans import pct

    from .io import budget_dir, load_npz, read_json, write_json

    prim = f"{float(cfg.alpha):g}"
    bA = float(cfg.privacy_budgets[0])
    bB = float(cfg.studyB_budget) if cfg.studyB_budget is not None else None
    tests = {k: {"available": False} for k in PRIMARY}
    pA = budget_dir(cfg, bA) / "analysis" / f"studyA_b{pct(bA):03d}.json"
    if pA.exists():
        A = read_json(pA)
        p1 = A.get("P1", {})
        if p1.get("estimate") is not None:
            tests["P1"] = {"available": True, "estimate": p1["estimate"], "ci95": p1.get("ci95"),
                           "p": p1.get("p_two_sided"), "sided": "two", "reportable": bool(p1.get("reportable")),
                           "source": str(pA)}
        p2 = A.get("P2", {})
        if p2.get("available") and p2.get("delta") is not None:
            rep = all(v["reportable"].get(prim, False) for v in A["conditions"].values()
                      if v["condition"] in ("true_hier", "random_hier", "spectrum_hier"))
            tests["P2"] = {"available": True, "estimate": p2["delta"], "pi": [p2.get("pi_low"), p2.get("pi_high")],
                           "p": p2.get("p_two_sided"), "sided": "two", "extrapolation": p2.get("extrapolation"),
                           "reportable": bool(rep), "source": str(pA)}
    if bB is not None:
        an = budget_dir(cfg, bB) / "analysis"
        if (an / "studyB.json").exists() and (an / "studyB_contrasts.npz").exists():
            B = read_json(an / "studyB.json")
            C = load_npz(an / "studyB_contrasts.npz")
            rep_ce = B["reportable"]
            for key, other, seed in (("P3", "vis", 3), ("P4", "rand", 4)):
                arrs = [C[k] for k in sorted((k for k in C if k.startswith(f"{key}_q")), key=lambda s: int(s[4:]))]
                r = cohort_record_bootstrap(arrs, cfg.n_boot, seed)
                if r["available"]:
                    sets = ["baseB_ce"] + [s for s in B["sets"] if s.startswith("del_ce_")
                                           and (s.endswith("_sem") or s.endswith(f"_{other}"))]
                    tests[key] = {"available": True, "estimate": r["estimate"], "ci95": r["ci95"],
                                  "p": r["p_two_sided"], "sided": "two", "cohort_means": r["cohort_means"],
                                  "n_cohorts": r["n_cohorts"], "sign_flip_p": sign_flip_p(r["cohort_means"]),
                                  "reportable": all(rep_ce.get(s, False) for s in sets), "contrast": f"sem - {other}"}
        if (an / "prediction.json").exists():
            Pj = read_json(an / "prediction.json")
            inp = Pj.get("P5_inputs", {})
            if inp.get("available"):
                r = p5_bootstrap(inp["groups"], cfg.n_boot, seed=5)
                rep = (all(v for k, v in B["reportable"].items() if k == "baseB_ce" or k.startswith("del_ce_"))
                       if (an / "studyB.json").exists() else False)
                tests["P5"] = {"available": True, "estimate": r["estimate"], "ci95": r["ci95"], "p": r["p_one_sided"],
                               "sided": "one (improvement)", "n_cohorts": r["n_cohorts"], "reportable": bool(rep)}
    p_raw = {k: tests[k].get("p") for k in PRIMARY}
    p_holm, reject = holm_family(p_raw, cfg.alpha_family)
    n_tests = sum(v is not None for v in p_raw.values())
    warning = None
    if n_tests < len(PRIMARY):
        missing = [k for k in PRIMARY if p_raw[k] is None]
        warning = (f"Holm adjusted over {n_tests} of {len(PRIMARY)} primary tests; missing {missing}: the adjusted "
                   f"p-values are smaller than with the full family")
        import sys

        print(f"primary: WARNING {warning}", file=sys.stderr)
    res = {"alpha_family": cfg.alpha_family, "alpha": float(cfg.alpha), "studyA_budget": bA, "studyB_budget": bB,
           "tests": tests, "p_raw": p_raw, "p_holm": p_holm, "reject": reject, "n_tests": int(n_tests),
           "n_family": len(PRIMARY), "holm_warning": warning,
           "reportable": {k: tests[k].get("reportable", False) for k in PRIMARY},
           "sensitivity": {k: {"sign_flip_p": tests[k].get("sign_flip_p")} for k in ("P3", "P4")}}
    path = Path(out) if out else budget_dir(cfg, bB if bB is not None else bA) / "analysis" / "primary.json"
    write_json(path, res)
    res["out"] = str(path)
    return res
