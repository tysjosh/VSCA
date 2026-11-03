"""Study B: deletion and privacy externalities (Section 3.9). ``privred studyB`` reads the scored baseline sets
baseB_<cond>, the deletion sets del_<cond>_q{q}_<type>, the removal sets (deletion/), the preliminary audit
(prelim), difficulty.npz, support.npz and features.npz, and writes <budget dir>/analysis/studyB.json,
studyB_contrasts.npz (per-record P3/P4 inputs) and rows_<cond>.npz (prediction rows).

Delta^del_i(R; gamma) = r_i(gamma, D_beta minus R; alpha) - r_i(gamma, D_beta; alpha)          (Eq. ddel)
The first term comes from the deletion set for R, the second from the baseline set baseB_gamma. Primary records:
the cohort T_q; secondary: D_beta minus the union of the cohort's removal sets (all types).
Prediction rows (Eq. target): one row per (retained record of D_beta minus union R_q, deletion set), with
X_base (Eq. xbase) = d0, r0 (prelim, alpha = 0.1%), Lbar0, Mem0, class phi-bar dispersion, class OOF accuracy,
V, samecls, retained n_y, dV, |R|; X_sem (Eq. xsem) = N, dN, U, dU, K, dK under G_true; PD0 = OOF PD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stats as pstats
from .deletion import TYPES
from .io import align, budget_dir, load_npz, pool_indices, read_json, save_npz, set_dir, write_json
from .plans import DEL_TYPES, STUDYB_CONDITIONS, _deletion_units, removal_path

BASE_NAMES = ("d0", "r0", "Lbar0", "Mem0", "cls_disp", "cls_oof_acc", "V", "samecls", "n_y", "dV", "R_size")
SEM_NAMES = ("N", "dN", "U", "dU", "K", "dK")
CONTRASTS = (("sem_minus_vis", "sem", "vis"), ("sem_minus_rand", "sem", "rand"), ("near_minus_far", "near", "far"),
             ("hv_minus_rand", "hv", "rand"))
# bootstrap seeds of the contrast intervals (sem - vis / sem - rand match P3 / P4 in stats.primary)
CONTRAST_SEEDS = {"sem_minus_vis": 3, "sem_minus_rand": 4, "near_minus_far": 6, "hv_minus_rand": 7}
INTERACTION_SEED = 8


def _ci(arrs: dict, n_boot: int, seed: int) -> dict:
    """stats.cohort_record_bootstrap over the per-cohort per-record arrays (cohorts, then records within)."""
    r = pstats.cohort_record_bootstrap([arrs[q] for q in sorted(arrs)], n_boot, seed)
    return {k: r.get(k) for k in ("available", "estimate", "ci95", "p_two_sided", "n_boot", "note") if k in r}


def _nanmean(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def class_dispersion(e, y, n_classes: int) -> np.ndarray:
    """Per class: mean over its records of 1 - cos(e_i, mean e of the class) (NaN for an empty class)."""
    e = np.asarray(e, np.float64)
    y = np.asarray(y, np.int64)
    out = np.full(n_classes, np.nan)
    for c in np.unique(y):
        E = e[y == c]
        mu = E.mean(0)
        mu /= max(np.linalg.norm(mu), 1e-12)
        out[c] = float(np.mean(1.0 - E @ mu))
    return out


def _contrast(dd, a, b, keep_by_q):
    """Per cohort: mean over T_q (restricted by keep) of Delta^del(a) - Delta^del(b) and the per-record arrays."""
    per, arrs = {}, {}
    for q, keep in keep_by_q.items():
        if (q, a) in dd and (q, b) in dd:
            x = dd[(q, a)] - dd[(q, b)]
            x = x[keep]
            per[q] = _nanmean(x)
            arrs[q] = x
    vals = [v for v in per.values() if v is not None]
    return {"per_cohort": {str(q): v for q, v in per.items()}, "mean": float(np.mean(vals)) if vals else None,
            "n_cohorts": len(vals)}, arrs


def analyze(cfg, budget=None, out=None) -> dict:
    from sg.data import load_dataset
    from sg.partitions import true_partition

    from .support import load_features

    b = float(cfg.studyB_budget if budget is None else budget)
    ds = load_dataset(cfg)
    assign = true_partition(ds.coarse_of_fine)
    bdir = budget_dir(cfg, b)
    coh = read_json(bdir / "deletion" / "cohorts.json")
    pool = pool_indices(cfg, ds, b)
    y_pool = ds.y_train[pool]
    prim = f"{float(cfg.alpha):g}"
    alphas = [f"{a:g}" for a in sorted(set([float(a) for a in cfg.alphas] + [float(cfg.alpha)]))]

    # baseline features on D_beta (prelim audit, OOF difficulty, support on G_true, class properties)
    P = load_npz(set_dir(cfg, b, "prelim") / "scores.npz")
    pp = align(P["ids"], pool)
    feat = {"r0": P["r_0.001"][pp].astype(np.float64), "Lbar0": P["Lbar_in"][pp].astype(np.float64),
            "Mem0": P["mem"][pp].astype(np.float64)}
    nan = np.full(len(pool), np.nan)
    flags = np.zeros(len(pool), bool)
    pd0 = nan.copy()
    cls_acc = np.full(ds.n_classes, np.nan)
    if (bdir / "difficulty.npz").exists():
        dz = load_npz(bdir / "difficulty.npz")
        dp = align(dz["ids"], pool)
        feat["d0"] = dz["d0"][dp].astype(np.float64)
        pd0 = dz["pd0"][dp].astype(np.float64)
        flags = dz["flag_label_error"][dp].astype(bool)
        oc = dz["oof_correct"][dp].astype(np.float64)
        for c in np.unique(y_pool):
            cls_acc[c] = oc[y_pool == c].mean()
    else:
        feat["d0"] = nan.copy()
    sup = load_npz(bdir / "support.npz")
    sp = align(sup["ids"], pool)
    feat["V"], feat["samecls"] = sup["V"][sp].astype(np.float64), sup["samecls"][sp].astype(np.float64)
    F = load_features(cfg)
    disp = class_dispersion(F["e"][align(F["ids"], pool)], y_pool, ds.n_classes)
    feat["cls_disp"], feat["cls_oof_acc"] = disp[y_pool], cls_acc[y_pool]

    # union of the removal sets per cohort (secondary records), cohort classes
    union = {}
    for c in coh["cohorts"]:
        q = c["q"]
        Rs = [np.load(removal_path(cfg, b, q, t)) for t in TYPES if removal_path(cfg, b, q, t).exists()]
        union[q] = np.unique(np.concatenate(Rs)) if Rs else np.zeros(0, np.int64)
    classes = {c["q"]: c["classes"] for c in coh["cohorts"]}

    res = {"budget": b, "alpha": float(cfg.alpha), "alphas": alphas, "R_size": coh["R_size"],
           "cohorts": {str(c["q"]): {"classes": c["classes"], "siblings": c["siblings"],
                                     "vis_classes": c["vis_classes"], "n_T": c["n_T"], "match_vis": c["match_vis"]}
                       for c in coh["cohorts"]},
           "sets": [], "missing": [], "reportable": {}, "cohort_means": {}, "type_means": {}, "contrasts": {},
           "label_error_repeat": {}, "rows": {}}
    contrast_arrays = {}
    dd_by_cond = {}
    for cond in STUDYB_CONDITIONS:
        bset = set_dir(cfg, b, f"baseB_{cond}")
        if not (bset / "scores.npz").exists():
            res["missing"].append(f"baseB_{cond}")
            continue
        B = load_npz(bset / "scores.npz")
        res["reportable"][f"baseB_{cond}"] = bool(read_json(bset / "summary.json")["attack"]["alphas"][prim]["reportable"])
        dd, dd_le, cm = {}, {}, {}
        rows = {k: [] for k in ("target", "ids", "cohort", "experiment", "type", "in_T", "flag") + BASE_NAMES
                + SEM_NAMES}
        exp_names = []
        keepT, keepT_le = {}, {}
        for name, c2, q, t in _deletion_units(cfg):
            if c2 != cond:
                continue
            sd = set_dir(cfg, b, name)
            if not (sd / "scores.npz").exists():
                res["missing"].append(name)
                continue
            res["sets"].append(name)
            res["reportable"][name] = bool(read_json(sd / "summary.json")["attack"]["alphas"][prim]["reportable"])
            S = load_npz(sd / "scores.npz")
            ids = S["ids"]
            pb = align(B["ids"], ids)
            dpos = align(pool, ids)
            ddel = {a: S[f"r_{a}"].astype(np.float64) - B[f"r_{a}"][pb].astype(np.float64) for a in alphas}
            Dz = load_npz(removal_path(cfg, b, q, t).parent / "deltas.npz")
            zp = align(Dz["ids"], ids)
            inT = np.isin(y_pool[dpos], classes[q])
            sec = ~np.isin(ids, union[q])
            # records T_q in pool order (identical across the cohort's sets)
            tpos = np.flatnonzero(inT)
            dd[(q, t)] = ddel[prim][tpos]
            keepT[q] = np.ones(len(tpos), bool)
            keepT_le[q] = ~flags[dpos][tpos]
            row = cm.setdefault(str(q), {})
            row[t] = {"T": _nanmean(ddel[prim][inT]), "secondary": _nanmean(ddel[prim][sec]),
                      "T_by_alpha": {a: _nanmean(v[inT]) for a, v in ddel.items()},
                      "T_label_error_repeat": _nanmean(ddel[prim][inT & ~flags[dpos]]),
                      "n_T": int(inT.sum()), "n_secondary": int(sec.sum()),
                      "mean_dK_T": _nanmean(Dz["dK"][zp][inT]), "mean_dV_T": _nanmean(Dz["dV"][zp][inT]),
                      "mean_dK_secondary": _nanmean(Dz["dK"][zp][sec])}
            # prediction rows on the secondary records (T_q included)
            e_id = len(exp_names)
            exp_names.append(name)
            sp_ = dpos[sec]
            n_y = np.bincount(y_pool[dpos], minlength=ds.n_classes)  # retained count per class in D minus R
            Rn = len(pool) - len(ids)
            rows["target"].append(ddel[prim][sec])
            rows["ids"].append(ids[sec])
            rows["cohort"].append(np.full(sec.sum(), q))
            rows["experiment"].append(np.full(sec.sum(), e_id))
            rows["type"].append(np.full(sec.sum(), t))
            rows["in_T"].append(inT[sec])
            rows["flag"].append(flags[sp_])
            for k in ("d0", "r0", "Lbar0", "Mem0", "cls_disp", "cls_oof_acc", "V", "samecls"):
                rows[k].append(feat[k][sp_])
            rows["n_y"].append(n_y[y_pool[sp_]].astype(np.float64))
            rows["dV"].append(Dz["dV"][zp][sec])
            rows["R_size"].append(np.full(sec.sum(), float(Rn)))
            for k, src in (("N", "N_before"), ("dN", "dN"), ("U", "U_before"), ("dU", "dU"), ("K", "K_before"),
                           ("dK", "dK")):
                rows[k].append(Dz[src][zp][sec].astype(np.float64))
            rows.setdefault("PD0", []).append(pd0[sp_])
        if not exp_names:
            continue
        dd_by_cond[cond] = dd
        res["cohort_means"][cond] = cm
        res["type_means"][cond] = {t: _nanmean([cm[q][t]["T"] for q in cm if t in cm[q] and cm[q][t]["T"] is not None])
                                   for t in DEL_TYPES[cond]}
        con, con_le = {}, {}
        for key, a, bb in CONTRASTS:
            con[key], arrs = _contrast(dd, a, bb, keepT)
            con[key]["bootstrap"] = _ci(arrs, cfg.n_boot, CONTRAST_SEEDS[key])
            con_le[key], _ = _contrast(dd, a, bb, keepT_le)
            if cond == "ce" and key in ("sem_minus_vis", "sem_minus_rand"):
                for q, x in arrs.items():
                    contrast_arrays[f"{'P3' if key == 'sem_minus_vis' else 'P4'}_q{q}"] = x
        res["contrasts"][cond] = con
        res["label_error_repeat"][cond] = {"n_flagged_T": {str(q): int((~k).sum()) for q, k in keepT_le.items()},
                                           "contrasts": con_le}
        # rows file for the prediction study
        cat = {k: np.concatenate(v) for k, v in rows.items()}
        rp = bdir / "analysis" / f"rows_{cond}.npz"
        save_npz(rp, target=cat["target"], ids=cat["ids"], y=ds.y_train[cat["ids"]],
                 parent=assign[ds.y_train[cat["ids"]]], cohort=cat["cohort"].astype(np.int64),
                 experiment=cat["experiment"].astype(np.int64), experiment_names=np.array(exp_names),
                 type=cat["type"].astype(str), in_T=cat["in_T"], flag_label_error=cat["flag"],
                 X_base=np.column_stack([cat[k] for k in BASE_NAMES]), base_names=np.array(BASE_NAMES),
                 X_sem=np.column_stack([cat[k] for k in SEM_NAMES]), sem_names=np.array(SEM_NAMES), PD0=cat["PD0"],
                 n_classes=np.int64(ds.n_classes))
        res["rows"][cond] = {"path": str(rp), "n_rows": int(len(cat["target"])), "n_experiments": len(exp_names)}

    # True-Hier arm: [TH(sem) - TH(vis)] - [CE(sem) - CE(vis)] on shared cohorts (T_q means)
    inter, inter_arrs = {}, {}
    if "ce" in dd_by_cond and "true_hier" in dd_by_cond:
        ce, th = dd_by_cond["ce"], dd_by_cond["true_hier"]
        for q in sorted({k[0] for k in th}):
            if all(k in d for d in (ce, th) for k in ((q, "sem"), (q, "vis"))):
                inter_arrs[q] = (th[(q, "sem")] - th[(q, "vis")]) - (ce[(q, "sem")] - ce[(q, "vis")])
                inter[str(q)] = _nanmean(inter_arrs[q])
    vals = [v for v in inter.values() if v is not None]
    res["true_hier_interaction"] = {"per_cohort": inter, "mean": float(np.mean(vals)) if vals else None,
                                    "n_cohorts": len(vals),
                                    "bootstrap": _ci(inter_arrs, cfg.n_boot, INTERACTION_SEED)}
    an = Path(out).parent if out else bdir / "analysis"
    cp = an / "studyB_contrasts.npz"
    save_npz(cp, **contrast_arrays)
    res["contrasts_npz"] = str(cp)
    path = Path(out) if out else an / "studyB.json"
    write_json(path, res)
    res["out"] = str(path)
    return res
