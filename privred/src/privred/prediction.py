"""Predicting privacy redistribution (Section 3.10). ``privred predict`` reads analysis/rows_<cond>.npz (written
by ``privred studyB``) and writes analysis/prediction.json.

g_base (Eq. gbase) uses X_base, g_full (Eq. gfull) uses [X_base, X_sem]; both from the same family: ridge on
standardized features (train-median imputation, zero-sd columns kept with sd = 1), penalty from ridge_alphas by
inner GroupKFold over cohorts (experiments if the training rows hold fewer than 2 cohorts) minimizing MSE;
HistGradientBoostingRegressor(random_state=0) is the secondary family. Splits never share an experiment id or a
cohort between train and test:
  cohort      leave one cohort out (P5)
  experiment  test = one deletion experiment, train = rows of the other cohorts
  class       test = rows of one held-out cohort in one class fold, train = other cohorts and other class folds
  parent      test = rows of one held-out cohort in one curated parent, train = other cohorts and other parents
Metrics on the pooled held-out predictions: MAE, R^2, precision of the predicted top top_frac increases.
The class and parent splits are record-disjoint by construction (a record's class and parent are fixed). In the
cohort and experiment splits a record retained under every removal set appears in train and test rows, sharing the
-r_i(baseB) term of the target; ``record_disjoint`` (secondary variant, not used by P5) drops the training rows
whose record id occurs in the test fold.
"""

from __future__ import annotations

import numpy as np

from .io import budget_dir, load_npz, write_json

SPLITS = ("cohort", "experiment", "class", "parent")
FAMILIES = ("ridge", "gbt")
CLASS_FOLDS = 5
CLASS_FOLD_SEED = 11


# --- splits ------------------------------------------------------------------------------------

def class_folds(n_classes: int, seed: int, k: int = CLASS_FOLDS) -> np.ndarray:
    """Fold id per fine class: position in default_rng([seed, CLASS_FOLD_SEED]).permutation(n_classes) mod k."""
    perm = np.random.default_rng([int(seed), CLASS_FOLD_SEED]).permutation(n_classes)
    f = np.empty(n_classes, np.int64)
    f[perm] = np.arange(n_classes) % k
    return f


RECORD_DISJOINT_SPLITS = ("cohort", "experiment")


def make_splits(kind: str, cohort, experiment, y=None, parent=None, cls_fold=None, ids=None,
                record_disjoint: bool = False) -> list:
    """[(train idx, test idx)] for one split kind; each pair is checked to share no experiment and no cohort.
    With ``record_disjoint`` (needs ``ids``), training rows whose record id is in the test fold are dropped and each
    pair is checked to share no record."""
    cohort, experiment = np.asarray(cohort), np.asarray(experiment)
    out = []
    qs = np.unique(cohort)
    if kind == "cohort":
        out = [(np.flatnonzero(cohort != q), np.flatnonzero(cohort == q)) for q in qs]
    elif kind == "experiment":
        for e in np.unique(experiment):
            q = cohort[experiment == e][0]
            out.append((np.flatnonzero(cohort != q), np.flatnonzero(experiment == e)))
    elif kind == "class":
        f = np.asarray(cls_fold)[np.asarray(y)]
        for q in qs:
            for k in np.unique(f[cohort == q]):
                out.append((np.flatnonzero((cohort != q) & (f != k)), np.flatnonzero((cohort == q) & (f == k))))
    elif kind == "parent":
        parent = np.asarray(parent)
        for q in qs:
            for p in np.unique(parent[cohort == q]):
                out.append((np.flatnonzero((cohort != q) & (parent != p)),
                            np.flatnonzero((cohort == q) & (parent == p))))
    else:
        raise ValueError(f"unknown split kind {kind!r}")
    if record_disjoint:
        ids = np.asarray(ids)
        out = [(tr[~np.isin(ids[tr], ids[te])], te) for tr, te in out]
    out = [(tr, te) for tr, te in out if len(tr) and len(te)]
    for tr, te in out:
        if np.intersect1d(experiment[tr], experiment[te]).size or np.intersect1d(cohort[tr], cohort[te]).size:
            raise AssertionError(f"split {kind}: an experiment or cohort is in both train and test")
        if record_disjoint and np.intersect1d(ids[tr], ids[te]).size:
            raise AssertionError(f"split {kind}: a record is in both train and test")
    return out


# --- models ------------------------------------------------------------------------------------

def _prep(Xtr, Xte):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> median 0
        med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), 0) if len(Xtr) else np.zeros(Xtr.shape[1])
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_ridge(X, y, groups, alphas, inner_folds: int):
    """Ridge with the penalty chosen by inner GroupKFold (MSE). Returns (model, alpha, X transform)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    ng = len(np.unique(groups))
    alphas = [float(a) for a in alphas]
    if ng >= 2:
        gkf = GroupKFold(n_splits=min(int(inner_folds), ng))
        mse = np.zeros(len(alphas))
        for itr, iva in gkf.split(X, y, groups):
            for j, a in enumerate(alphas):
                m = Ridge(alpha=a).fit(X[itr], y[itr])
                mse[j] += float(np.sum((m.predict(X[iva]) - y[iva]) ** 2))
        best = alphas[int(np.argmin(mse))]
    else:
        best = alphas[len(alphas) // 2]
    return Ridge(alpha=best).fit(X, y), best


def predict_split(family, Xtr, ytr, Xte, groups, alphas, inner_folds):
    if family == "ridge":
        a, b = _prep(Xtr, Xte)
        m, alpha = fit_ridge(a, ytr, groups, alphas, inner_folds)
        return m.predict(b), alpha
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    return m.predict(Xte), None


def precision_top(y, yhat, frac: float):
    """Precision of the predicted top ``frac`` increases: among the k = ceil(frac n) rows with the largest
    prediction (ties: smaller row), the fraction whose true value is >= the k-th largest true value."""
    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    n = len(y)
    if not n:
        return None
    k = max(1, int(np.ceil(frac * n)))
    top = np.lexsort((np.arange(n), -yhat))[:k]
    thr = np.sort(y)[::-1][k - 1]
    return float(np.mean(y[top] >= thr))


def metrics(y, yhat, frac: float) -> dict:
    from sklearn.metrics import r2_score

    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    r2 = float(r2_score(y, yhat)) if len(y) > 1 and np.std(y) > 0 else None
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(y - yhat))) if len(y) else None, "r2": r2,
            "precision_top": precision_top(y, yhat, frac)}


# --- evaluation ----------------------------------------------------------------------------------

def load_rows(cfg, cond, budget=None):
    b = float(cfg.studyB_budget if budget is None else budget)
    p = budget_dir(cfg, b) / "analysis" / f"rows_{cond}.npz"
    return load_npz(p) if p.exists() else None


def evaluate(rows, kind: str, family: str, cfg, add_pd0: bool = False, record_disjoint: bool = False) -> dict:
    """g_base vs g_full for one split kind and family on pooled held-out predictions; per test group the sums of
    absolute errors (P5 inputs for the cohort split). ``record_disjoint``: see make_splits."""
    y = rows["target"].astype(np.float64)
    ok = np.isfinite(y)
    Xb = rows["X_base"].astype(np.float64)
    Xf = np.column_stack([Xb, rows["X_sem"].astype(np.float64)])
    if add_pd0:
        pd0 = rows["PD0"].astype(np.float64)[:, None]
        Xb, Xf = np.column_stack([Xb, pd0]), np.column_stack([Xf, pd0])
    idx = np.flatnonzero(ok)
    coh, exp = rows["cohort"][idx], rows["experiment"][idx]
    cf = class_folds(int(rows["n_classes"]), cfg.cohort_seed)
    splits = make_splits(kind, coh, exp, rows["y"][idx], rows["parent"][idx], cf, rows["ids"][idx], record_disjoint)
    if not splits:
        return {"available": False, "note": f"no {kind} split with non-empty train and test rows"}
    yy = y[idx]
    pred = {"base": np.full(len(idx), np.nan), "full": np.full(len(idx), np.nan)}
    chosen = {"base": [], "full": []}
    groups = []
    for tr, te in splits:
        g = coh[tr] if len(np.unique(coh[tr])) >= 2 else exp[tr]
        for name, X in (("base", Xb[idx]), ("full", Xf[idx])):
            p, a = predict_split(family, X[tr], yy[tr], X[te], g, cfg.ridge_alphas, cfg.inner_folds)
            pred[name][te] = p
            chosen[name].append(a)
        groups.append({"test_cohorts": np.unique(coh[te]).tolist(), "n_test": int(len(te)), "n_train": int(len(tr)),
                       "abs_err_base": float(np.abs(yy[te] - pred["base"][te]).sum()),
                       "abs_err_full": float(np.abs(yy[te] - pred["full"][te]).sum())})
    done = np.isfinite(pred["base"])
    out = {"available": True, "n_splits": len(splits), "n_rows": int(done.sum()),
           "base": metrics(yy[done], pred["base"][done], cfg.top_frac),
           "full": metrics(yy[done], pred["full"][done], cfg.top_frac), "groups": groups}
    out["mae_reduction"] = out["base"]["mae"] - out["full"]["mae"]
    if family == "ridge":
        out["alphas_chosen"] = chosen
    return out


def run_prediction(cfg, budget=None, add_pd0: bool = False) -> dict:
    from .plans import STUDYB_CONDITIONS

    b = float(cfg.studyB_budget if budget is None else budget)
    res = {"budget": b, "add_pd0": add_pd0, "features_base": None, "features_sem": None, "conditions": {},
           "secondary_dataset": {"available": False,
                                 "note": "the paper runs only Study A on iNat-100; no Study B rows exist there"}}
    for cond in STUDYB_CONDITIONS:
        rows = load_rows(cfg, cond, b)
        if rows is None:
            res["conditions"][cond] = {"available": False, "note": f"rows_{cond}.npz missing; run `privred studyB`"}
            continue
        res["features_base"] = rows["base_names"].tolist() + (["PD0"] if add_pd0 else [])
        res["features_sem"] = rows["sem_names"].tolist()
        res["conditions"][cond] = {"n_rows": int(len(rows["target"])),
                                   "n_cohorts": int(len(np.unique(rows["cohort"]))),
                                   "n_experiments": int(len(np.unique(rows["experiment"]))),
                                   **{k: {f: evaluate(rows, k, f, cfg, add_pd0) for f in FAMILIES} for k in SPLITS}}
        res["conditions"][cond]["record_disjoint"] = {
            k: {f: evaluate(rows, k, f, cfg, add_pd0, record_disjoint=True) for f in FAMILIES}
            for k in RECORD_DISJOINT_SPLITS}
    c = res["conditions"].get("ce", {}).get("cohort", {}).get("ridge", {})
    res["P5_inputs"] = ({"available": True, "groups": c["groups"], "mae_reduction": c["mae_reduction"]}
                        if c.get("available") else {"available": False, "note": "CE cohort split unavailable"})
    return res


def predict(cfg, budget=None, out=None) -> dict:
    from pathlib import Path

    b = float(cfg.studyB_budget if budget is None else budget)
    res = run_prediction(cfg, b)
    path = Path(out) if out else budget_dir(cfg, b) / "analysis" / "prediction.json"
    write_json(path, res)
    res["out"] = str(path)
    return res
