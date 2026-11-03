"""Linking difficulty and redistribution (Section 3.11). ``privred linking`` writes analysis/linking.json:

Study A (per budget and non-CE condition, record bootstrap):  dsup ~ dPD + d0 + dPD*d0 + PD0 + dPD*PD0
Study B (per condition, cohort bootstrap; all rows and T_q rows): ddel ~ dK + d0 + dK*d0 + PD0 + dK*PD0
Key test: g_base vs g_full (Eqs. gbase, gfull) with PD0 added to both feature sets, cohort split, ridge.
d0 = OOF difficulty, PD0 = OOF prediction depth (difficulty.npz). The associations are observational.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stats as pstats
from .io import align, budget_dir, load_npz, write_json


def ols(y, X) -> tuple:
    """OLS with intercept via lstsq; returns (coef [1 + p], rank)."""
    A = np.column_stack([np.ones(len(y)), X])
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    return coef, int(rank)


def interaction_fit(y, x, d0, pd0, names, n_boot: int, seed: int, clusters=None) -> dict:
    """y ~ x + d0 + x*d0 + PD0 + x*PD0 with bootstrap CIs (records, or whole clusters if given)."""
    y, x, d0, pd0 = (np.asarray(a, np.float64) for a in (y, x, d0, pd0))
    X = np.column_stack([x, d0, x * d0, pd0, x * pd0])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    cl = None if clusters is None else np.asarray(clusters)[ok]
    terms = ["intercept"] + list(names)
    if len(y) < X.shape[1] + 2:
        return {"available": False, "n": int(len(y)), "note": "too few complete rows"}
    coef, rank = ols(y, X)
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(coef)), np.nan)
    if cl is None:
        for t in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            boot[t] = ols(y[i], X[i])[0]
    else:
        groups = [np.flatnonzero(cl == g) for g in np.unique(cl)]
        for t in range(n_boot):
            i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            boot[t] = ols(y[i], X[i])[0]
    return {"available": True, "n": int(len(y)), "rank": rank, "n_clusters": None if cl is None else int(len(np.unique(cl))),
            "bootstrap": "records" if cl is None else "clusters", "n_boot": int(n_boot),
            "coef": {k: float(c) for k, c in zip(terms, coef)},
            "ci95": {k: pstats.percentile_ci(boot[:, j]) for j, k in enumerate(terms)},
            "p_two_sided": {k: pstats.p_two_sided(boot[:, j]) for j, k in enumerate(terms)}}


def link(cfg, out=None) -> dict:
    from sg.plans import pct

    from .plans import STUDYB_CONDITIONS
    from .prediction import evaluate, load_rows

    prim = f"{float(cfg.alpha):g}"
    res = {"alpha": float(cfg.alpha), "studyA": {}, "studyB": {}, "key_test": {}}
    for b in cfg.privacy_budgets:
        bdir = budget_dir(cfg, b)
        rec_p = bdir / "analysis" / f"studyA_b{pct(b):03d}_records.npz"
        if not rec_p.exists() or not (bdir / "difficulty.npz").exists():
            res["studyA"][str(float(b))] = {"available": False, "note": "studyA records or difficulty.npz missing"}
            continue
        rec = load_npz(rec_p)
        dz = load_npz(bdir / "difficulty.npz")
        dp = align(dz["ids"], rec["ids"])
        d0, pd0 = dz["d0"][dp], dz["pd0"][dp].astype(np.float64)
        row = {}
        for k in rec:
            if k.startswith("dsup_") and k.endswith(f"_a{prim}"):
                name = k[len("dsup_"):-len(f"_a{prim}")]
                row[name] = interaction_fit(rec[k], rec[f"dpd_{name}"], d0, pd0,
                                            ["dPD", "d0", "dPD_x_d0", "PD0", "dPD_x_PD0"], cfg.n_boot, seed=0)
        res["studyA"][str(float(b))] = row
    if cfg.studyB_budget is not None:
        for cond in STUDYB_CONDITIONS:
            rows = load_rows(cfg, cond)
            if rows is None:
                res["studyB"][cond] = {"available": False, "note": f"rows_{cond}.npz missing"}
                continue
            sem = list(rows["sem_names"])
            base = list(rows["base_names"])
            dK = rows["X_sem"][:, sem.index("dK")]
            d0 = rows["X_base"][:, base.index("d0")]
            names = ["dK", "d0", "dK_x_d0", "PD0", "dK_x_PD0"]
            T = rows["in_T"].astype(bool)
            res["studyB"][cond] = {
                "all_rows": interaction_fit(rows["target"], dK, d0, rows["PD0"], names, cfg.n_boot, 1, rows["cohort"]),
                "T_rows": interaction_fit(rows["target"][T], dK[T], d0[T], rows["PD0"][T], names, cfg.n_boot, 2,
                                          rows["cohort"][T])}
            kt = evaluate(rows, "cohort", "ridge", cfg, add_pd0=True)
            if kt.get("available"):
                kt["p_one_sided"] = pstats.p5_bootstrap(kt["groups"], cfg.n_boot, seed=6)["p_one_sided"]
            res["key_test"][cond] = kt
    path = Path(out) if out else budget_dir(cfg, cfg.studyB_budget if cfg.studyB_budget is not None
                                            else cfg.privacy_budgets[0]) / "analysis" / "linking.json"
    write_json(path, res)
    res["out"] = str(path)
    return res
