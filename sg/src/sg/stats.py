"""Inference (Section IV-D): Eq. 7 rank p-value, Eq. 8 similarity-adjusted residual with a prediction interval,
H2 / H3 bootstraps (D11), Holm, and the variance pilot with Monte Carlo power (Section IV-B.3, D16)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

NON_CURATED = ("random_hier", "spectrum_hier")


# --- outcomes-file helpers ----------------------------------------------------------

def load_outcomes(path) -> list:
    from .utils import read_jsonl

    return read_jsonl(path)


def true_row(rows) -> dict:
    t = [r for r in rows if r["condition"] == "true_hier"]
    if len(t) != 1:
        raise SystemExit(f"expected one true_hier row in outcomes, found {len(t)}")
    return t[0]


def _w_source(rows) -> str:
    """D9: W_analysis enters the fit; fall back to W_train only if some row lacks W_analysis."""
    if all(r.get("W_analysis") is not None for r in rows):
        return "W_analysis"
    if all(r.get("W_train") is not None for r in rows):
        return "W_train"
    raise SystemExit("outcomes rows lack W_analysis and W_train; run `sg diagnose geometry` and pass --geometry")


def fit_inputs(rows, metric: str):
    """(W [R], T [R], W_true, T_true, w_source, nc_rows) over non-curated partitions and True-Hier."""
    nc = [r for r in rows if r["condition"] in NON_CURATED and r.get(metric) is not None]
    tr = true_row(rows)
    if tr.get(metric) is None:
        raise SystemExit(f"true_hier {metric} is null")
    src = _w_source(nc + [tr])
    return np.array([r[src] for r in nc], float), np.array([r[metric] for r in nc], float), float(tr[src]), \
        float(tr[metric]), src, nc


# --- Eq. 7 ------------------------------------------------------------------------------

def rank_pvalue(t_true: float, t_random) -> float:
    """p = (1 + #{r: T(G_r) >= T(G_true)}) / (R + 1)."""
    t_random = np.asarray(t_random, float)
    return float((1 + int((t_random >= t_true).sum())) / (len(t_random) + 1))


# --- Eq. 8 ------------------------------------------------------------------------------

def _design(W, center, scale, degree):
    z = (np.asarray(W, float) - center) / scale
    return np.vander(np.atleast_1d(z), degree + 1, increasing=True)


def _fit(W, T, degree):
    W = np.asarray(W, float)
    center = float(W.mean())
    scale = float(W.std()) or 1.0
    X = _design(W, center, scale, degree)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ np.asarray(T, float)
    return X, XtX_inv, beta, center, scale


def residual(W, T, W_true, T_true, degree: int = 1, level: float = 0.95) -> dict:
    """OLS fit T = poly_degree(W) over non-curated partitions; delta = T_true - fit(W_true) (Eq. 8) with a
    ``level`` prediction interval and the one-sided p = 1 - t_cdf(delta / se_pred, df) (D11)."""
    T = np.asarray(T, float)
    n, p = len(T), degree + 1
    df = n - p
    if df < 0:
        raise ValueError(f"residual fit of degree {degree} needs at least {p} partitions, got {n}")
    X, XtX_inv, beta, center, scale = _fit(W, T, degree)
    x0 = _design([W_true], center, scale, degree)[0]
    pred = float(x0 @ beta)
    delta = float(T_true) - pred
    raw = np.polynomial.Polynomial(beta)(np.polynomial.Polynomial([-center / scale, 1.0 / scale])).coef
    out = {"delta": delta, "pred_true": pred, "se_pred": None, "df": int(df), "level": level, "pi_low": None,
           "pi_high": None, "p_one_sided": None, "coef": [float(c) for c in raw], "degree": degree, "n": int(n),
           "resid_sd": None}
    if df == 0:  # exact fit: no residual variance, so no prediction interval
        out["note"] = "df = 0: interval and p-value undefined"
        return out
    res = T - X @ beta
    s2 = float(res @ res) / df
    se = float(np.sqrt(s2 * (1.0 + x0 @ XtX_inv @ x0)))
    tq = float(sps.t.ppf(0.5 + level / 2, df))
    if se > 0:
        p_one = float(sps.t.sf(delta / se, df))
    else:
        p_one = 0.0 if delta > 0 else (1.0 if delta < 0 else 0.5)
    out.update({"se_pred": se, "pi_low": delta - tq * se, "pi_high": delta + tq * se, "p_one_sided": p_one,
                "resid_sd": float(np.sqrt(s2))})
    return out


def _delta_only(W, T, W_true, T_true, degree=1) -> float:
    _, _, beta, center, scale = _fit(W, T, degree)
    return float(T_true - _design([W_true], center, scale, degree)[0] @ beta)


# --- H2 ---------------------------------------------------------------------------------

def h2_bootstrap(m_correct, m_in, c_correct, c_in, n_boot: int = 10000, seed: int = 0) -> dict:
    """D11 H2: paired bootstrap over test items and seed pairs of own-frame S (model - CE-B).

    Inputs [P, N] (seed pair x item): correct and in-own-group indicators for the model and its paired CE.
    p = (1 + #{boot >= 0}) / (B + 1) (one-sided test of T_S < 0).
    """
    mc, mi, cc, ci = (np.asarray(a, float) for a in (m_correct, m_in, c_correct, c_in))
    P, N = mc.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = float(np.mean(mc.sum(1) / mi.sum(1) - cc.sum(1) / ci.sum(1)))
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            s = rng.integers(0, P, P)
            boot[b] = np.mean(mc[s] @ w / (mi[s] @ w) - cc[s] @ w / (ci[s] @ w))
    boot = boot[np.isfinite(boot)]
    p = (1 + int((boot >= 0).sum())) / (len(boot) + 1)
    return {"T_S_own": obs, "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else None,
            "p_one_sided": float(p), "n_items": int(N), "n_pairs": int(P), "n_boot": int(n_boot), "n_valid": int(len(boot))}


def h2_from_runs(runs_root, dataset, budget, rho, assign_true, n_boot, seed=0, total_steps=None) -> dict:
    from .analysis import paired_preds, scan_runs

    y, mp, cp, seeds, part = paired_preds(scan_runs(runs_root), dataset, budget, rho, total_steps, "true_hier")
    a = np.asarray(part["assign"] if part is not None else assign_true, np.int64)
    out = h2_bootstrap(mp == y, a[mp] == a[y], cp == y, a[cp] == a[y], n_boot, seed)
    return {"budget": budget, "seeds": seeds, **out}


# --- H3 ---------------------------------------------------------------------------------

def h3_bootstrap(low_rows, high_rows, metric: str = "T_acc", n_boot: int = 10000, seed: int = 0, degree: int = 1) -> dict:
    """D11 H3: delta_low - delta_high of the True-Hier residual on ``metric``; bootstrap resamples non-curated
    partitions with replacement, seeds within each unit, and True-Hier seeds, refitting per budget.
    p = (1 + #{boot <= 0}) / (B + 1)."""
    rng = np.random.default_rng(seed)

    def prep(rows):
        W, T, Wt, Tt, src, nc = fit_inputs(rows, metric)
        D = [np.asarray(r["seed_diffs"][metric], float) for r in nc]
        Dt = np.asarray(true_row(rows)["seed_diffs"][metric], float)
        return W, T, Wt, Tt, src, D, Dt

    lo, hi = prep(low_rows), prep(high_rows)
    d_lo, d_hi = _delta_only(lo[0], lo[1], lo[2], lo[3], degree), _delta_only(hi[0], hi[1], hi[2], hi[3], degree)

    def one(b):
        W, _, Wt, _, _, D, Dt = b
        r = rng.integers(0, len(D), len(D))
        T = np.array([D[i][rng.integers(0, len(D[i]), len(D[i]))].mean() for i in r])
        tt = Dt[rng.integers(0, len(Dt), len(Dt))].mean()
        return _delta_only(W[r], T, Wt, tt, degree)

    boot = np.array([one(lo) - one(hi) for _ in range(n_boot)])
    boot = boot[np.isfinite(boot)]
    p = (1 + int((boot <= 0).sum())) / (len(boot) + 1)
    return {"metric": metric, "delta_low": d_lo, "delta_high": d_hi, "diff": d_lo - d_hi,
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_one_sided": float(p), "n_boot": int(n_boot), "W_source": [lo[4], hi[4]],
            "n_partitions": [len(lo[5]), len(hi[5])]}


# --- Holm ---------------------------------------------------------------------------------

ALPHA_ONE_SIDED = 0.025  # family-wise, one-sided; equals the paper's "95% interval excludes zero" rule


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


# --- variance pilot and power (D16) ----------------------------------------------------------

def variance_pilot(diffs_by_partition) -> dict:
    """One-way random effects on per-seed paired accuracy differences, one list per partition."""
    groups = [np.asarray(d, float) for d in diffs_by_partition if len(d) > 0]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        raise SystemExit("variance pilot needs >= 2 partitions with >= 2 seeds each")
    dof = sum(len(g) - 1 for g in groups)
    s2_seed = float(sum(((g - g.mean()) ** 2).sum() for g in groups) / dof)
    means = np.array([g.mean() for g in groups])
    n = float(np.mean([len(g) for g in groups]))
    s2_part = max(0.0, float(means.var(ddof=1)) - s2_seed / n)
    return {"n_partitions": len(groups), "n_seeds_mean": n, "sigma2_seed": s2_seed, "sigma2_part": s2_part,
            "partition_means": [float(m) for m in means]}


def power_sim(sigma2_seed: float, sigma2_part: float, effect: float = 0.01, alpha: float = ALPHA_ONE_SIDED,
              Rs=(19, 29, 39, 49, 99), seeds=(2, 3, 5), n_sim: int = 2000, seed: int = 0, n_tests: int = 3) -> dict:
    """Monte Carlo power of the residual test (true slope 0) and the rank test for R partitions x n seeds.

    Matches the confirmatory H1 test: the residual fit uses all 2R non-curated partitions (R random + R
    spectrum), and both tests are judged at alpha / n_tests, the strictest Holm step. This makes the reported
    power a lower bound on power under Holm. The rank test uses the R random partitions (Eq. 7).

    Each partition (the curated one included) gets a partition effect ~ N(0, sigma2_part) plus the mean of n
    seed noises ~ N(0, sigma2_seed); the curated partition adds ``effect``.
    """
    rng = np.random.default_rng(seed)
    sp, ss = np.sqrt(max(sigma2_part, 0.0)), np.sqrt(max(sigma2_seed, 0.0))
    a_test = alpha / n_tests
    table = []
    for R in Rs:
        n_fit = 2 * R
        W = np.linspace(0.0, 1.0, n_fit)
        X = np.stack([np.ones(n_fit), W], 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        H = XtX_inv @ X.T
        x0 = np.array([1.0, 0.5])
        v0 = float(x0 @ XtX_inv @ x0)
        for n in seeds:
            T = rng.normal(0, sp, (n_sim, n_fit)) + rng.normal(0, ss / np.sqrt(n), (n_sim, n_fit))
            Tt = effect + rng.normal(0, sp, n_sim) + rng.normal(0, ss / np.sqrt(n), n_sim)
            beta = T @ H.T
            res = T - beta @ X.T
            s2 = (res ** 2).sum(1) / (n_fit - 2)
            se = np.sqrt(s2 * (1 + v0))
            delta = Tt - beta @ x0
            with np.errstate(divide="ignore", invalid="ignore"):
                p_res = np.where(se > 0, sps.t.sf(delta / np.where(se > 0, se, 1.0), n_fit - 2), np.where(delta > 0, 0.0, 1.0))
            T_rand = T[:, ::2][:, :R]  # R of the simulated partitions stand in for the random family
            p_rank = (1 + (T_rand >= Tt[:, None]).sum(1)) / (R + 1)
            table.append({"R": int(R), "fit_points": int(n_fit), "seeds": int(n),
                          "power_residual": float((p_res <= a_test).mean()),
                          "power_rank": float((p_rank <= a_test).mean())})

    def smallest(key):
        ok = [r["R"] for r in table if r["seeds"] == 3 and r[key] >= 0.8]
        return min(ok) if ok else None

    return {"effect": effect, "alpha_familywise_one_sided": alpha, "alpha_per_test": a_test, "n_sim": n_sim,
            "table": table,
            "recommend_R_at_3_seeds": {"residual": smallest("power_residual"), "rank": smallest("power_rank")}}


def pilot(runs_root, dataset, budget, rho, assign_true, **power_kw) -> dict:
    from .analysis import partition_outcomes, scan_runs

    rows, _ = partition_outcomes(scan_runs(runs_root), dataset, budget, rho, None, None, None,
                                 assign_true=assign_true, strata=False)
    rand = [r for r in rows if r["condition"] == "random_hier"]
    vp = variance_pilot([r["seed_diffs"]["T_acc"] for r in rand])
    vp["partitions"] = [r["partition_id"] for r in rand]
    return {"budget": budget, "variance": vp, "power": power_sim(vp["sigma2_seed"], vp["sigma2_part"], **power_kw)}


# --- primary tests (D10, D11) ----------------------------------------------------------------

def primary(outcomes_primary, runs_root, dataset, budget, rho, low, high, n_boot, assign_true, out=None, seed=0,
            alpha: float = ALPHA_ONE_SIDED) -> dict:
    """H1 (residual on T_C_own), H2 (bootstrap on own-frame S), H3 (bootstrap on T_acc residual difference),
    raw one-sided p-values and Holm-adjusted p-values, with decisions at family-wise one-sided ``alpha``.

    For a single test, p_one_sided < 0.025 is the same event as the 95% prediction interval for delta lying
    above zero, so with alpha = 0.025 the Holm decision is never looser than the paper's interval rule."""
    rows = load_outcomes(outcomes_primary)
    W, T, Wt, Tt, src, _ = fit_inputs(rows, "T_C_own")
    h1 = residual(W, T, Wt, Tt, degree=1)
    h1["W_source"] = src
    h2 = h2_from_runs(runs_root, dataset, budget, rho, assign_true, n_boot, seed)
    h3 = h3_bootstrap(load_outcomes(low), load_outcomes(high), "T_acc", n_boot, seed)
    raw = {"H1": h1["p_one_sided"], "H2": h2["p_one_sided"], "H3": h3["p_one_sided"]}
    adj = holm(raw)
    h1["pi_above_zero"] = h1["pi_low"] is not None and h1["pi_low"] > 0
    res = {"budget": budget, "H1": h1, "H2": h2, "H3": h3, "p_raw": raw, "p_holm": adj,
           "alpha_one_sided_familywise": alpha,
           "reject": {k: (v is not None and v < alpha) for k, v in adj.items()}}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2))
    return res
