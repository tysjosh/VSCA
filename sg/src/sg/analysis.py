"""Run scanning, partition-level seed-paired outcomes T(G) in both frames (Section III-A/B, D2),
cross-fitted strata (D3), and the pre-registered diagnostics (Section IV-B.1 geometry, IV-B.2 baseline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .metrics import (
    crossfit_folds, factorize, metric_value, outcomes_from_preds, probability_curve, stratum_deltas,
)
from .partitions import cosine_matrix, load_partition, within_group_similarity
from .utils import read_jsonl

# outcomes-row key -> (metrics key, frame)
T_KEYS = {"T_acc": ("acc", "true"), "T_C_true": ("C", "true"), "T_S_true": ("S", "true"), "T_F_true": ("F", "true"),
          "T_C_own": ("C", "own"), "T_S_own": ("S", "own"), "T_F_own": ("F", "own")}
NON_CURATED = ("random_hier", "spectrum_hier")


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def scan_runs(root) -> list:
    """Every training run under ``root`` (config.json + train_summary.json), excluding weight-decay tuning runs."""
    runs = []
    for s in sorted(Path(root).rglob("train_summary.json")):
        d = s.parent
        if "tune_wd" in d.relative_to(root).parts or not (d / "config.json").exists():
            continue
        runs.append({"dir": d, "config": json.loads((d / "config.json").read_text()), "summary": json.loads(s.read_text())})
    return runs


def _same(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) < 1e-9)


def select_runs(runs, dataset, budget, rho=None, total_steps=None):
    """CE runs {seed: run} and aux units {(condition, partition_id): {seed: run}} for one budget / rho / total_steps."""
    ce, units, rhos = {}, {}, set()
    for r in runs:
        c = r["config"]
        if c["dataset"] != dataset or not _same(c["budget"], budget) or not _same(c.get("total_steps"), total_steps):
            continue
        cond, seed = c["condition"], int(c["seed"])
        if cond == "ce":
            if seed in ce:
                _warn(f"duplicate CE seed {seed}: keeping {ce[seed]['dir']}, ignoring {r['dir']}")
                continue
            ce[seed] = r
            continue
        if rho is not None and not _same(c["rho"], rho):
            continue
        rhos.add(float(c["rho"]))
        key = (cond, r["summary"].get("partition_id"))
        unit = units.setdefault(key, {})
        if seed in unit:
            _warn(f"duplicate run {key} seed {seed}: keeping {unit[seed]['dir']}, ignoring {r['dir']}")
            continue
        unit[seed] = r
    if rho is None and len(rhos) > 1:
        raise SystemExit(f"aux runs at budget {budget} use several rho values {sorted(rhos)}; pass --rho")
    return ce, units


def _test(run, cache):
    d = run["dir"]
    if d not in cache:
        rows = read_jsonl(d / "test.jsonl")
        cache[d] = {r["id"]: r for r in rows}
    return cache[d]


def _arrays(run_list, ids, cache):
    """y [N], pred [K, N], p_true [K, N] for runs aligned on ``ids``."""
    tabs = [_test(r, cache) for r in run_list]
    y = np.array([tabs[0][i]["y"] for i in ids], np.int64) if tabs else np.zeros(len(ids), np.int64)
    pred = np.array([[t[i]["pred"] for i in ids] for t in tabs], np.int64).reshape(len(tabs), len(ids))
    p = np.array([[t[i]["p_true"] for i in ids] for t in tabs], np.float64).reshape(len(tabs), len(ids))
    return y, pred, p


def _common_ids(run_list, cache):
    sets = [set(_test(r, cache)) for r in run_list]
    return sorted(set.intersection(*sets)) if sets else []


def _own_partition(unit_runs):
    r = next(iter(unit_runs.values()))
    if r["config"]["condition"] == "flat_aux" or not (r["dir"] / "partition.json").exists():
        return None
    return json.loads((r["dir"] / "partition.json").read_text())


def _diff(a, b):
    return None if a is None or b is None else a - b


def _unit_key(cond, pid):
    return cond if pid is None else f"{cond}:{pid}"


def partition_outcomes(runs, dataset, budget, rho, total_steps, geometry, label_errors, *, assign_true,
                       hard_frac: float = 0.25, strata: bool = True):
    """One outcomes row per partition unit (seed-paired with CE, D2) and the strata dict (D3).

    ``geometry``: dict from ``geometry()`` or None (W_analysis is then null). ``label_errors``: set of test ids or None.
    ``assign_true``: the dataset's curated class -> group map.
    """
    ce, units = select_runs(runs, dataset, budget, rho, total_steps)
    if not ce:
        raise SystemExit(f"no CE runs for {dataset} budget {budget} total_steps {total_steps}")
    cache: dict = {}
    all_runs = list(ce.values()) + [r for u in units.values() for r in u.values()]
    ids = _common_ids(all_runs, cache)
    ce_seeds = sorted(ce)
    y, ce_pred, ce_p = _arrays([ce[s] for s in ce_seeds], ids, cache)
    ce_pos = {s: i for i, s in enumerate(ce_seeds)}
    geo = (geometry or {}).get("partitions", {})
    a_true = np.asarray(assign_true, np.int64)

    pos_of = {i: k for k, i in enumerate(ids)}
    excl = sorted(pos_of[i] for i in (label_errors or ()) if i in pos_of)
    do_strata = strata and len(ce_seeds) >= 2
    if strata and not do_strata:
        _warn("fewer than 2 CE seeds: strata skipped")
    folds_all = crossfit_folds(ce_p, hard_frac) if do_strata else None
    folds_clean = crossfit_folds(ce_p, hard_frac, excl) if do_strata and label_errors is not None else None

    rows, strata_units = [], {}
    for (cond, pid), unit in sorted(units.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        seeds = sorted(set(unit) & set(ce))
        if not seeds:
            _warn(f"{cond} {pid}: no CE run shares its seeds {sorted(unit)}; skipped")
            continue
        part = _own_partition(unit)
        a_own = np.asarray(part["assign"], np.int64) if part is not None else None
        _, m_pred, _ = _arrays([unit[s] for s in seeds], ids, cache)
        diffs = {k: [] for k in T_KEYS}
        for j, s in enumerate(seeds):
            mo = outcomes_from_preds(y, m_pred[j], a_true, a_own)
            co = outcomes_from_preds(y, ce_pred[ce_pos[s]], a_true, a_own)
            for k, (m, frame) in T_KEYS.items():
                key = m if m == "acc" else f"{m}_{frame}"
                diffs[k].append(_diff(mo[key], co[key]))
        T = {k: (float(np.mean(v)) if all(x is not None for x in v) else None) for k, v in diffs.items()}
        g = geo.get(pid or "", {})
        w_train = part.get("W_train") if part is not None else None
        if w_train is None:
            w_train = g.get("W_train")
        rows.append({
            "dataset": dataset, "budget": budget, "rho": float(unit[seeds[0]]["config"]["rho"]),
            "total_steps": total_steps, "condition": cond, "kind": part.get("kind") if part is not None else None,
            "partition_id": pid, "W_train": w_train, "W_analysis": g.get("W_analysis"),
            "ari_true": part.get("ari_true") if part is not None else None, "n_seeds": len(seeds), **T,
            "seed_diffs": {k: (v if a_own is not None or T_KEYS[k][1] == "true" else None) for k, v in diffs.items()},
        })
        if not do_strata:
            continue
        model_preds = [m_pred[j] for j in range(len(seeds))]
        per_metric = {}
        for k, (m, frame) in T_KEYS.items():
            assign = a_true if frame == "true" else a_own
            if assign is None:
                per_metric[k[2:]] = None
                continue

            def fn(pred, idx, assign=assign, m=m):
                return metric_value(y, pred, idx, assign, m)

            res = stratum_deltas(fn, ce_pred, model_preds, folds_all)
            res["excl_label_errors"] = stratum_deltas(fn, ce_pred, model_preds, folds_clean) if folds_clean else None
            res["curve"] = probability_curve(fn, ce_p, ce_pred, model_preds)
            per_metric[k[2:]] = res
        strata_units[_unit_key(cond, pid)] = per_metric

    strata_out = {"dataset": dataset, "budget": budget, "rho": rho, "total_steps": total_steps, "hard_frac": hard_frac,
                  "n_test": len(ids), "ce_seeds": ce_seeds,
                  "n_label_errors_excluded": len(excl) if label_errors is not None else None, "units": strata_units}
    return rows, strata_out


def paired_preds(runs, dataset, budget, rho, total_steps, condition="true_hier", partition_id=None):
    """Seed-paired test predictions of one unit and CE: (y, model_pred [P, N], ce_pred [P, N], seeds, own partition)."""
    ce, units = select_runs(runs, dataset, budget, rho, total_steps)
    keys = [k for k in units if k[0] == condition and (partition_id is None or k[1] == partition_id)]
    if len(keys) != 1:
        raise SystemExit(f"expected one {condition} unit at budget {budget}, found {keys}")
    unit = units[keys[0]]
    seeds = sorted(set(unit) & set(ce))
    if not seeds:
        raise SystemExit(f"{condition}: no CE run shares its seeds")
    cache: dict = {}
    runs_m, runs_c = [unit[s] for s in seeds], [ce[s] for s in seeds]
    ids = _common_ids(runs_m + runs_c, cache)
    y, m_pred, _ = _arrays(runs_m, ids, cache)
    _, c_pred, _ = _arrays(runs_c, ids, cache)
    return y, m_pred, c_pred, seeds, _own_partition(unit)


def geometry(extractor_dir, partition_files) -> dict:
    """Section IV-B.1: W_analysis of every partition under one fixed extractor, ARI with the curated hierarchy,
    the most similar candidate by ARI, the curated W percentile among candidates, and the candidate W range."""
    mu = np.load(Path(extractor_dir) / "class_means.npy")
    S = cosine_matrix(mu)
    parts = {}
    for f in sorted(partition_files):
        f = Path(f)
        if f.name == "index.json":
            continue
        p = load_partition(f)
        parts[p["id"]] = {"name": p["name"], "kind": p["kind"], "dataset": p["dataset"], "budget": p["budget"],
                          "W_analysis": within_group_similarity(p["assign"], S), "W_train": p.get("W_train"),
                          "ari_true": p.get("ari_true")}
    by_budget = {}
    for key in sorted({(v["dataset"], float(v["budget"])) for v in parts.values()}):
        group = {pid: v for pid, v in parts.items() if (v["dataset"], float(v["budget"])) == key}
        true = [pid for pid, v in group.items() if v["kind"] == "true"]
        cands = {pid: v for pid, v in group.items() if v["kind"] != "true"}
        row = {"dataset": key[0], "budget": key[1], "n_candidates": len(cands)}
        if cands:
            w = np.array([v["W_analysis"] for v in cands.values()])
            row.update({"W_min": float(w.min()), "W_max": float(w.max())})
            best = max(cands, key=lambda pid: cands[pid]["ari_true"] if cands[pid]["ari_true"] is not None else -1)
            row["most_similar"] = {"id": best, "ari_true": cands[best]["ari_true"], "W_analysis": cands[best]["W_analysis"]}
        if true:
            wt = group[true[0]]["W_analysis"]
            row.update({"true_id": true[0], "W_true": wt})
            if cands:
                row["true_W_percentile"] = float(100.0 * (w <= wt).mean())
        by_budget[f"{key[0]}_b{int(round(key[1] * 100)):03d}"] = row
    return {"extractor": str(extractor_dir), "partitions": parts, "by_budget": by_budget}


def baseline(runs, budgets, dataset, assign_true) -> dict:
    """Section IV-B.2: true-frame C, S, F (and acc) of the CE seeds per budget: per seed, mean and SD."""
    cache: dict = {}
    out = {}
    for b in budgets:
        ce, _ = select_runs([r for r in runs if r["config"]["condition"] == "ce"], dataset, b, None, None)
        seeds = sorted(ce)
        per_seed = []
        for s in seeds:
            rows = list(_test(ce[s], cache).values())
            y = np.array([r["y"] for r in rows], np.int64)
            pred = np.array([r["pred"] for r in rows], np.int64)
            f = factorize(y, pred, assign_true)
            per_seed.append({"seed": s, "n": f["n"], **{m: f[m] for m in ("acc", "C", "S", "F")}})
        summ = {}
        for m in ("acc", "C", "S", "F"):
            v = [r[m] for r in per_seed if r[m] is not None]
            summ[m] = {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
        out[str(float(b))] = {"n_seeds": len(seeds), **summ, "per_seed": per_seed}
    return {"dataset": dataset, "frame": "true", "budgets": out}
