"""Study A: hierarchy supervision and privacy (Section 3.7, robustness of RQ2 and label errors from Section 4.3).

``privred studyA --budget B`` reads the scored Study A sets of budget B (scores.npz, summary.json), the OOF
difficulty (difficulty.npz) and the analysis features (features.npz), and writes
<budget dir>/analysis/studyA_b{pct}.json and studyA_b{pct}_records.npz (per-record Delta^sup and Delta^PD).

Delta^sup_i(gamma) = r_i(gamma, D_beta; alpha) - r_i(CE, D_beta; alpha)   (Eq. dsup), at every alpha
Delta^PD_i(gamma)  = mean PD over IN models of gamma - same for CE        (Eq. dpd)
T^priv(G) = mean_i Delta^sup_i(G); delta^priv = Eq. residual over the 8 non-curated partitions of the subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stats as pstats
from .config import ce_run, geometry_path, partitions_dir, rho_for, true_hier_run
from .io import (align, budget_dir, load_npz, member_dir, pool_indices, read_json, save_npz, set_dir, val_indices,
                 write_json)

PATTERNS = ("pd_down_v_down", "pd_down_v_up", "pd_down_v_same", "pd_notdown_v_change", "pd_notdown_v_same")
CS_KINDS = ("correct", "wrong_within", "wrong_across")
NON_CURATED = ("random_hier", "spectrum_hier")
TABLE_KEYS = ("acc", "C_true", "S_true", "F_true", "C_own", "S_own", "F_own", "gen_gap", "margin_members",
              "margin_audit", "ece_test", "ece_audit")


# --- example-level summaries -------------------------------------------------------------------

def pattern_codes(dpd, dsup, tol_pd: float = 0.0, tol_r: float = 0.0) -> np.ndarray:
    """Index into PATTERNS per record (-1 if either delta is NaN). lower = delta < -tol, higher = delta > tol,
    unchanged = |delta| <= tol. Patterns: (PD down, V down), (PD down, V up), (PD down, V same),
    (PD unchanged or up, V changed), and the remainder (PD unchanged or up, V same)."""
    dpd = np.asarray(dpd, np.float64)
    dsup = np.asarray(dsup, np.float64)
    pd_down = dpd < -tol_pd
    v_down, v_up = dsup < -tol_r, dsup > tol_r
    v_same = ~v_down & ~v_up
    code = np.select([pd_down & v_down, pd_down & v_up, pd_down & v_same, ~pd_down & ~v_same, ~pd_down & v_same],
                     [0, 1, 2, 3, 4], -1)
    code[~(np.isfinite(dpd) & np.isfinite(dsup))] = -1
    return code


def pattern_fractions(codes, dsup=None, dpd=None) -> dict:
    codes = np.asarray(codes)
    ok = codes >= 0
    n = int(ok.sum())
    out = {"n": n}
    for k, name in enumerate(PATTERNS):
        out[name] = float((codes[ok] == k).mean()) if n else None
    if dsup is not None:
        out["mean_dsup"] = _nanmean(np.asarray(dsup)[ok])
    if dpd is not None:
        out["mean_dpd"] = _nanmean(np.asarray(dpd)[ok])
    return out


def cs_outcome(pred, y, assign_true) -> np.ndarray:
    """0 correct, 1 wrong within the true group, 2 wrong across groups, -1 no prediction (pred < 0)."""
    pred = np.asarray(pred, np.int64)
    y = np.asarray(y, np.int64)
    a = np.asarray(assign_true, np.int64)
    out = np.where(pred == y, 0, np.where(a[np.maximum(pred, 0)] == a[y], 1, 2))
    out[pred < 0] = -1
    return out


def _nanmean(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def cs_split(dsup, outcome) -> dict:
    """Delta^sup split by the record's modal outcome in the CE OUT models."""
    dsup = np.asarray(dsup, np.float64)
    n_all = int((np.asarray(outcome) >= 0).sum())
    out = {}
    for k, name in enumerate(CS_KINDS):
        m = np.asarray(outcome) == k
        out[name] = {"n": int(m.sum()), "frac": float(m.sum() / n_all) if n_all else None,
                     "mean_dsup": _nanmean(dsup[m]),
                     "contribution_to_mean": float(np.nansum(dsup[m]) / n_all) if n_all else None}
    return out


def robustness(dsup, correct_ce, correct_g, valid) -> dict:
    """Delta^sup within records whose modal OUT correctness is the same under CE and gamma."""
    dsup = np.asarray(dsup, np.float64)
    ce, g, v = (np.asarray(a, bool) for a in (correct_ce, correct_g, valid))
    eq = v & (ce == g)
    return {"n_valid": int(v.sum()), "n_equal": int(eq.sum()), "mean_dsup_equal": _nanmean(dsup[eq]),
            "both_correct": {"n": int((eq & ce).sum()), "mean_dsup": _nanmean(dsup[eq & ce])},
            "both_wrong": {"n": int((eq & ~ce).sum()), "mean_dsup": _nanmean(dsup[eq & ~ce])},
            "changed": {"n": int((v & (ce != g)).sum()), "mean_dsup": _nanmean(dsup[v & (ce != g)])}}


# --- inputs ------------------------------------------------------------------------------------

def _units(cfg, budget):
    from .plans import _studyA_units

    return _studyA_units(cfg, budget)


def _w_table(cfg, budget, names) -> tuple:
    """{partition name: W} with W_analysis from the companion geometry.json if every partition has it, else
    W_train from index.json (as sg.stats._w_source)."""
    from sg.plans import pct

    idx = {p["name"]: p for p in read_json(partitions_dir(cfg, budget) / "index.json")["partitions"]}
    geo = read_json(geometry_path(cfg))["partitions"] if geometry_path(cfg).exists() else {}
    pid = lambda n: f"{cfg.dataset}_b{pct(budget):03d}_{n}"  # noqa: E731
    if all(geo.get(pid(n), {}).get("W_analysis") is not None for n in names):
        return {n: float(geo[pid(n)]["W_analysis"]) for n in names}, "W_analysis"
    return {n: float(idx[n]["W_train"]) for n in names}, "W_train"


def _probe_z(cfg, budget, ds, names) -> dict:
    """Z(G): accuracy of sg.probes.linear_probe for the 20-way task G(y) on frozen phi-bar features, trained on
    D_beta and tested on V_beta."""
    from sg.partitions import load_partition
    from sg.probes import linear_probe
    from sg.utils import pick_device

    from .support import load_features

    F = load_features(cfg)
    tr = pool_indices(cfg, ds, budget)
    va = val_indices(cfg, ds, budget)
    xtr, xva = F["e"][align(F["ids"], tr)], F["e"][align(F["ids"], va)]
    device = pick_device(cfg.device)
    out = {}
    for n in names:
        a = load_partition(partitions_dir(cfg, budget) / f"{n}.json")["assign"]
        gtr, gva = a[ds.y_train[tr]], a[ds.y_train[va]]
        pred = linear_probe(xtr, gtr, xva, int(a.max()) + 1, cfg.probe_linear_iters, cfg.probe_l2, device)
        out[n] = float((pred.numpy() == gva).mean()) if len(va) else None
    return out


def _reuse_check(cfg, budget, ds, summaries) -> dict:
    """Privacy CE / True-Hier vs the size-matched companion runs at beta / 2 on test acc, C_G and S_G (true frame)."""
    from sg.metrics import run_outcomes
    from sg.utils import read_jsonl

    half = float(budget) / 2
    out = {"companion_budget": half}
    try:
        rho = rho_for(cfg)
    except SystemExit:
        rho = None
    for name, runs in (("A_ce", [ce_run(cfg, half, s) for s in range(5)]),
                       ("A_true_hier", [true_hier_run(cfg, half, rho, s) for s in range(5)] if rho is not None else [])):
        if name not in summaries:
            continue
        comp = [run_outcomes(read_jsonl(r / "test.jsonl"), ds.coarse_of_fine) | {"run": str(r)}
                for r in runs if (r / "test.jsonl").exists()]
        priv = {k: summaries[name]["condition_metrics"].get(k) for k in ("acc", "C_true", "S_true")}
        row = {"privacy": priv, "companion_runs": comp, "available": bool(comp)}
        if comp:
            cm = {k: _nanmean([c[k] for c in comp if c[k] is not None]) for k in ("acc", "C_true", "S_true")}
            row["companion_mean"] = cm
            row["privacy_minus_companion"] = {k: (None if priv[k] is None or cm[k] is None else priv[k] - cm[k])
                                              for k in cm}
        else:
            row["note"] = "no size-matched companion runs found"
        out[name] = row
    return out


def _ce_own_frame(cfg, budget, units, summaries, assign_true) -> dict:
    """Own frame of Eq. factor: CE-B's per-model test outcomes under each condition's training partition G (from its
    member partition.json; conditions CE and Flat-Aux have no own frame), averaged over the CE models, and the
    condition's own-frame C_G / S_G / F_G minus CE's on the same G."""
    from sg.metrics import outcomes_from_preds
    from sg.utils import read_jsonl

    ce_dir = set_dir(cfg, budget, "A_ce")
    preds = []
    for m in range(int(summaries["A_ce"]["M"])):
        f = member_dir(ce_dir, m) / "test.jsonl"
        if f.exists():
            rows = read_jsonl(f)
            preds.append((np.array([r["y"] for r in rows], np.int64), np.array([r["pred"] for r in rows], np.int64)))
    out = {}
    for name, cond, _ in units:
        if cond in ("ce", "flat_aux") or name not in summaries:
            continue
        pf = member_dir(set_dir(cfg, budget, name), 0) / "partition.json"
        if not pf.exists() or not preds:
            out[name] = {"available": False, "note": f"{pf} or CE test.jsonl missing"}
            continue
        own = np.asarray(read_json(pf)["assign"], np.int64)
        per = [outcomes_from_preds(yt, pt, assign_true, own) for yt, pt in preds]
        ce = {k: _nanmean([r[f"{k}_own"] for r in per if r[f"{k}_own"] is not None]) for k in ("C", "S", "F")}
        g = summaries[name]["condition_metrics"]
        out[name] = {"available": True, "n_ce_models": len(per), "ce": ce,
                     "minus_ce": {k: (None if g.get(f"{k}_own") is None or ce[k] is None else g[f"{k}_own"] - ce[k])
                                  for k in ce}}
    return out


# --- analysis ----------------------------------------------------------------------------------

def _strata(codes, d, dpd, keep, quart) -> dict | None:
    if quart is None:
        return None
    return {str(q): pattern_fractions(codes[keep & (quart == q)], d[keep & (quart == q)], dpd[keep & (quart == q)])
            for q in range(4)}


def _contrasts(cfg, S, units, y, assign_true, quart, keep, pd_quart=None) -> dict:
    """Example-level summaries of every non-CE condition against CE on the records in ``keep``. ``strata`` uses the
    d0 quartiles, ``strata_pd`` the PD0-based sensitivity quartiles (PD0 desc, loss desc, purity asc)."""
    ce = S["A_ce"]
    alphas = [f"{float(a):g}" for a in sorted(set([float(a) for a in cfg.alphas] + [float(cfg.alpha)]))]
    prim = f"{float(cfg.alpha):g}"
    ce_out = cs_outcome(ce["pred_out_modal"], y, assign_true)
    ce_valid = np.asarray(ce["pred_out_modal"]) >= 0
    res = {k: {} for k in ("dsup_mean", "dpd_mean", "T_priv", "patterns", "strata", "strata_pd", "cs_split",
                           "robustness")}
    for name, cond, pname in units:
        if name == "A_ce" or name not in S:
            continue
        g = S[name]
        dsup = {a: (g[f"r_{a}"].astype(np.float64) - ce[f"r_{a}"].astype(np.float64)) for a in alphas}
        dpd = g["pd_in_mean"].astype(np.float64) - ce["pd_in_mean"].astype(np.float64)
        d = dsup[prim]
        res["dsup_mean"][name] = {a: _nanmean(v[keep]) for a, v in dsup.items()}
        res["T_priv"][name] = res["dsup_mean"][name][prim]
        res["dpd_mean"][name] = _nanmean(dpd[keep])
        codes = pattern_codes(dpd, d, cfg.pattern_tol_pd, cfg.pattern_tol_r)
        res["patterns"][name] = pattern_fractions(codes[keep], d[keep], dpd[keep])
        res["strata"][name] = _strata(codes, d, dpd, keep, quart)
        res["strata_pd"][name] = _strata(codes, d, dpd, keep, pd_quart)
        res["cs_split"][name] = cs_split(d[keep], ce_out[keep])
        valid = ce_valid & (np.asarray(g["pred_out_modal"]) >= 0)
        res["robustness"][name] = robustness(d[keep], ce["correct_out_modal"][keep], g["correct_out_modal"][keep],
                                             valid[keep])
    return res


def _delta_priv(T_priv, units, W, w_src, Z, S=None, keep=None, alpha=None, n_boot=0) -> dict:
    """sg.stats.residual over the non-curated partitions of the subset, two-sided p, and the sensitivity checks;
    the descriptive rank of G_true among the random partitions; with S, the partition-level bootstrap interval
    (records x models x partitions) of delta^priv."""
    from sg.stats import rank_pvalue, residual

    nc = [(n, p) for n, c, p in units if c in NON_CURATED and T_priv.get(n) is not None]
    if "A_true_hier" not in T_priv or T_priv["A_true_hier"] is None or len(nc) < 3:
        return {"available": False, "note": "needs True-Hier and >= 3 non-curated partitions with scores",
                "n_partitions": len(nc)}
    Wv = np.array([W[p] for _, p in nc])
    Tv = np.array([T_priv[n] for n, _ in nc])
    Wt, Tt = W["true"], T_priv["A_true_hier"]
    r = residual(Wv, Tv, Wt, Tt, degree=1)
    out = {"available": True, "W_source": w_src, "partitions": [p for _, p in nc], "W": Wv.tolist(), "T": Tv.tolist(),
           "W_true": Wt, "T_true": Tt, **r, "p_two_sided": pstats.two_sided_from_one(r["p_one_sided"]),
           "extrapolation": pstats.extrapolation(Wv, Wt)}
    lopo = pstats.lopo_residuals(Wv, Tv)
    vals = [v for v in lopo if v is not None]
    out["lopo"] = {"residuals": lopo, "frac_abs_ge_delta": (float(np.mean(np.abs(vals) >= abs(r["delta"])))
                                                              if vals else None)}
    if Z is not None and all(Z.get(p) is not None for _, p in nc) and Z.get("true") is not None:
        Zv = np.array([Z[p] for _, p in nc])
        try:
            out["covariate"] = {"Z": Zv.tolist(), "Z_true": Z["true"],
                                **pstats.residual_cov(Wv, Tv, Wt, Tt, Zv[:, None], [Z["true"]])}
        except ValueError as e:
            out["covariate"] = {"available": False, "note": str(e)}
    else:
        out["covariate"] = {"available": False, "note": "Z(G) unavailable"}
    # descriptive only (Eq. residual paragraph): rank 1 = largest T^priv among G_true and the random partitions
    tr = [T_priv[n] for n, c, _ in units if c == "random_hier" and T_priv.get(n) is not None]
    out["rank_random"] = {"n_random": len(tr), "rank": 1 + int(np.sum(np.asarray(tr) > Tt)),
                          "rank_p": rank_pvalue(Tt, tr) if tr else None, "descriptive": True}
    if S is not None:
        k = np.ones(len(S["A_ce"]["ids"]), bool) if keep is None else keep
        sub = lambda n: (S[n]["Lambda"][k], S[n]["u"][k])  # noqa: E731
        out["partition_bootstrap"] = pstats.partition_bootstrap(sub("A_ce"), sub("A_true_hier"),
                                                                [sub(n) for n, _ in nc], Wv, Wt, float(alpha),
                                                                n_boot, seed=9)
    return out


def analyze(cfg, budget, out=None) -> dict:
    from sg.data import load_dataset
    from sg.partitions import true_partition
    from sg.plans import pct

    ds = load_dataset(cfg)
    units = _units(cfg, budget)
    S, summ, missing = {}, {}, []
    for name, _, _ in units:
        d = set_dir(cfg, budget, name)
        if (d / "scores.npz").exists() and (d / "summary.json").exists():
            S[name] = load_npz(d / "scores.npz")
            summ[name] = read_json(d / "summary.json")
        else:
            missing.append(name)
    if "A_ce" not in S:
        raise SystemExit(f"studyA: {set_dir(cfg, budget, 'A_ce')} has no scores; run `privred score` on it first")
    ids = S["A_ce"]["ids"]
    for n, s in S.items():
        if not np.array_equal(s["ids"], ids):
            raise SystemExit(f"studyA: set {n} does not score the same pool as A_ce")
    y = ds.y_train[ids]
    assign_true = true_partition(ds.coarse_of_fine)
    bdir = budget_dir(cfg, budget)
    quart, pd_quart, flags, diff_note = None, None, np.zeros(len(ids), bool), None
    if (bdir / "difficulty.npz").exists():
        dz = load_npz(bdir / "difficulty.npz")
        pos = align(dz["ids"], ids)
        quart, flags = dz["quartile"][pos], dz["flag_label_error"][pos].astype(bool)
        pd_quart = dz["pd_quartile"][pos]
    else:
        diff_note = f"{bdir}/difficulty.npz missing: no difficulty strata and no label-error repeat"
    prim = f"{float(cfg.alpha):g}"
    alphas = sorted(set([float(a) for a in cfg.alphas] + [float(cfg.alpha)]))

    # condition table
    table = {}
    for name, cond, pname in units:
        if name not in summ:
            continue
        sm, sc = summ[name], S[name]
        row = {"condition": cond, "partition": pname, "M": sm["M"], "variance": sm["variance"]}
        row.update({k: sm["condition_metrics"].get(k) for k in TABLE_KEYS})
        row["tpr"] = {k: v["tpr"] for k, v in sm["attack"]["alphas"].items()}
        row["reportable"] = {k: v["reportable"] for k, v in sm["attack"]["alphas"].items()}
        row["auc"] = sm["attack"]["auc"]
        row["offline_auc"] = sm["offline_attack"]["auc"]
        row["mean_r"] = {f"{a:g}": _nanmean(sc[f"r_{a:g}"]) for a in alphas}
        row["mean_mem"] = _nanmean(sc["mem"])
        row["mean_pd_in"] = _nanmean(sc["pd_in_mean"])
        table[name] = row
    own_frame = _ce_own_frame(cfg, budget, units, summ, assign_true)
    for name, row in table.items():
        row["own_frame_vs_ce"] = own_frame.get(name)

    # partition-level inputs: W and Z(G)
    part_names = ["true"] + [p for _, c, p in units if c in NON_CURATED]
    W, w_src = _w_table(cfg, budget, part_names)
    try:
        Z = _probe_z(cfg, budget, ds, part_names)
        z_note = None
    except SystemExit as e:
        Z, z_note = None, str(e)

    def full(keep):
        c = _contrasts(cfg, S, units, y, assign_true, quart, keep, pd_quart)
        c["delta_priv"] = _delta_priv(c["T_priv"], units, W, w_src, Z, S, keep, cfg.alpha, cfg.n_boot)
        return c

    main = full(np.ones(len(ids), bool))
    res = {"budget": float(budget), "dataset": cfg.dataset, "primary_alpha": cfg.alpha,
           "alphas": [f"{a:g}" for a in alphas], "n_pool": int(len(ids)), "sets": [u[0] for u in units],
           "missing": missing, "conditions": table, **main, "W": W, "Z": Z, "Z_note": z_note,
           "difficulty_available": quart is not None, "difficulty_note": diff_note}

    # P1: True-Hier minus CE condition-level TPR at alpha (= mean Delta^sup(G_true)), two-way bootstrap
    if "A_true_hier" in S:
        a, b = S["A_ce"], S["A_true_hier"]
        p1 = pstats.p1_bootstrap(a["Lambda"], a["u"], b["Lambda"], b["u"], float(cfg.alpha), cfg.n_boot, seed=0)
        p1["mean_dsup"] = main["T_priv"].get("A_true_hier")
        p1["reportable"] = bool(summ["A_ce"]["attack"]["alphas"][prim]["reportable"]
                                and summ["A_true_hier"]["attack"]["alphas"][prim]["reportable"])
        p1["inputs"] = {"ce_scores": str(set_dir(cfg, budget, "A_ce") / "scores.npz"),
                        "true_hier_scores": str(set_dir(cfg, budget, "A_true_hier") / "scores.npz")}
        res["P1"] = p1
    else:
        res["P1"] = {"available": False, "note": "A_true_hier not scored"}
    # exploratory intervals: condition-level TPR difference (gamma - CE) at alpha for every other condition, with
    # the P1 two-way bootstrap (seed = position of the set in the Study A units)
    res["dsup_ci"] = {}
    for k, (name, _, _) in enumerate(units):
        if name == "A_ce" or name not in S:
            continue
        if name == "A_true_hier" and "estimate" in res["P1"]:
            r = {kk: res["P1"][kk] for kk in ("estimate", "ci95", "p_two_sided", "n_boot", "n_records", "M")}
        else:
            r = pstats.p1_bootstrap(S["A_ce"]["Lambda"], S["A_ce"]["u"], S[name]["Lambda"], S[name]["u"],
                                    float(cfg.alpha), cfg.n_boot, seed=k)
            r = {kk: r[kk] for kk in ("estimate", "ci95", "p_two_sided", "n_boot", "n_records", "M")}
        res["dsup_ci"][name] = {**r, "mean_dsup": main["T_priv"].get(name), "alpha": float(cfg.alpha),
                                "exploratory": True}
    res["P2"] = {k: main["delta_priv"].get(k) for k in ("delta", "p_two_sided", "pi_low", "pi_high", "extrapolation",
                                                         "available")}
    res["reuse_check"] = _reuse_check(cfg, budget, ds, summ)

    # label-error repeat: every summary without the records flagged by OOF confident-error screening
    if quart is not None:
        le = full(~flags)
        res["label_error_repeat"] = {"n_excluded": int(flags.sum()), "n_kept": int((~flags).sum()), **le}
    else:
        res["label_error_repeat"] = {"available": False, "note": diff_note}

    # per-record arrays for linking / later analyses
    bd = Path(out).parent if out else bdir / "analysis"
    rec = {"ids": ids, "y": y, "quartile": quart if quart is not None else np.full(len(ids), -1),
           "flag_label_error": flags}
    for name, _, _ in units:
        if name in S and name != "A_ce":
            for a in alphas:
                rec[f"dsup_{name}_a{a:g}"] = (S[name][f"r_{a:g}"].astype(np.float64)
                                              - S["A_ce"][f"r_{a:g}"].astype(np.float64)).astype(np.float32)
            rec[f"dpd_{name}"] = (S[name]["pd_in_mean"].astype(np.float64)
                                  - S["A_ce"]["pd_in_mean"].astype(np.float64)).astype(np.float32)
    rec_path = bd / f"studyA_b{pct(budget):03d}_records.npz"
    save_npz(rec_path, **rec)
    res["records"] = str(rec_path)
    path = Path(out) if out else bd / f"studyA_b{pct(budget):03d}.json"
    write_json(path, res)
    res["out"] = str(path)
    return res
