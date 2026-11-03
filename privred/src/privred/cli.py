"""Command-line entry point: ``python -m privred.cli <command> ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _cfg(args):
    return load_config(args.config or [], args.set or [])


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_check_companion(args):
    from .config import (ce_run, companion_root, geometry_path, partitions_dir, rho_for, simclr_dir, true_hier_run,
                         wd_for)
    from .plans import subset_partitions

    cfg = _cfg(args)
    missing, report = [], {"companion_root": str(companion_root(cfg)), "budgets": {}}
    phibar = simclr_dir(cfg, 1.0)
    for f in ("backbone.pt", "class_means.npy"):
        if not (phibar / f).exists():
            missing.append(str(phibar / f))
    needs_rho = any(c != "ce" for c in cfg.studyA_conditions) or cfg.studyB_budget is not None
    rho = None
    if needs_rho:
        try:
            rho = rho_for(cfg)
        except SystemExit as e:
            missing.append(str(e))
    budgets = args.budget or cfg.privacy_budgets
    for b in budgets:
        row = {}
        idx = partitions_dir(cfg, b) / "index.json"
        if not idx.exists():
            missing.append(str(idx))
        elif "subset" in cfg.studyA_conditions:
            row["subset"] = [f"{c}:{p}" for c, p in subset_partitions(cfg, b)]
        try:
            row["wd_privacy"], row["wd_oof"] = wd_for(cfg, b), wd_for(cfg, b, size_matched=False)
        except SystemExit as e:
            missing.append(str(e))
        row["ce_seeds_for_difficulty_check"] = [s for s in range(5) if (ce_run(cfg, b, s) / "test.jsonl").exists()]
        row["reuse_ce_seeds"] = [s for s in range(5) if (ce_run(cfg, float(b) / 2, s) / "test.jsonl").exists()]
        row["reuse_true_hier"] = bool(rho is not None and (true_hier_run(cfg, float(b) / 2, rho) / "test.jsonl").exists())
        report["budgets"][str(float(b))] = row
    report["geometry"] = geometry_path(cfg).exists()
    report["rho"] = rho
    report["missing"] = missing
    report["ok"] = not missing
    _print(report)
    if missing:
        raise SystemExit(1)


def cmd_plan(args):
    from .plans import count_runs, make_sets, plan_lines

    cfg = _cfg(args)
    if args.count:
        _print(count_runs(cfg))
        return
    if not args.stage or args.budget is None:
        raise SystemExit("plan needs --stage and --budget (or --count)")
    names = [n for n in args.names.split(",") if n] if args.names else None
    dirs = make_sets(args.stage, cfg, args.budget, names)
    lines = plan_lines(dirs, args.config, args.set)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(ln + "\n" for ln in lines))
        _print({"stage": args.stage, "budget": args.budget, "out": args.out, "sets": [str(d) for d in dirs],
                "n_commands": len(lines)})
    else:
        print("\n".join(lines))


def cmd_member(args):
    from .members import run_member

    _print(run_member(_cfg(args), args.set_dir, args.m))


def cmd_launch(args):
    from sg.plans import launch

    failures = launch(args.plan, args.gpus, args.jobs_per_gpu)
    _print({"plan": args.plan, "n_failed": len(failures), "failures": failures})
    if failures:
        raise SystemExit(1)


def cmd_score(args):
    from .vulnerability import score_set

    s = score_set(_cfg(args), args.set_dir)
    a = s["attack"]["alphas"]
    _print({"set": s["set"], "M": s["M"], "n_pool": s["n_pool"], "variance": s["variance"], "auc": s["attack"]["auc"],
            "tpr": {k: v["tpr"] for k, v in a.items()}, "reportable": {k: v["reportable"] for k, v in a.items()},
            "offline_auc": s["offline_attack"]["auc"], "test_acc": s["condition_metrics"].get("acc"),
            "pd_available": s["pd_available"], "out": str(Path(args.set_dir) / "scores.npz")})


def cmd_difficulty(args):
    from .difficulty import compute_difficulty

    _print(compute_difficulty(_cfg(args), args.budget))


def cmd_features(args):
    from .support import extract_features

    _print(extract_features(_cfg(args)))


def cmd_support(args):
    from .support import compute_support

    _print(compute_support(_cfg(args), args.budget))


def cmd_studyA(args):
    from .studyA import analyze

    r = analyze(_cfg(args), args.budget, args.out)
    dp = r["delta_priv"]
    _print({"out": r["out"], "records": r["records"], "budget": r["budget"], "n_pool": r["n_pool"],
            "conditions": list(r["conditions"]), "missing": r["missing"], "T_priv": r["T_priv"],
            "delta_priv": {k: dp.get(k) for k in ("available", "delta", "p_two_sided", "W_source", "extrapolation")},
            "P1": {k: r["P1"].get(k) for k in ("estimate", "ci95", "p_two_sided", "reportable")},
            "difficulty_available": r["difficulty_available"]})


def cmd_deletion_sets(args):
    from .deletion import build_deletion_sets

    r = build_deletion_sets(_cfg(args), args.budget)
    _print({"out": r["out"], "budget": r["budget"], "n_pool": r["n_pool"], "R_size": r["R_size"],
            "per_class": r["per_class"], "types": r["types"],
            "cohorts": [{k: c[k] for k in ("q", "classes", "siblings", "vis_classes", "n_T")}
                        | {"match_vis": {k: c["match_vis"][k] for k in ("gap", "tolerance", "reached", "proposals")}}
                        | {"sets": c["sets"]} for c in r["cohorts"]]})


def cmd_studyB(args):
    from .studyB import analyze

    r = analyze(_cfg(args), args.budget, args.out)
    _print({"out": r["out"], "budget": r["budget"], "R_size": r["R_size"], "sets": r["sets"], "missing": r["missing"],
            "type_means": r["type_means"],
            "contrasts": {c: {k: v["mean"] for k, v in d.items()} for c, d in r["contrasts"].items()},
            "true_hier_interaction": r["true_hier_interaction"]["mean"], "rows": r["rows"],
            "contrasts_npz": r["contrasts_npz"]})


def cmd_predict(args):
    from .prediction import FAMILIES, SPLITS, predict

    r = predict(_cfg(args), args.budget, args.out)
    summ = {}
    for c, d in r["conditions"].items():
        summ[c] = {k: {f: ({"mae_base": d[k][f]["base"]["mae"], "mae_full": d[k][f]["full"]["mae"],
                            "mae_reduction": d[k][f]["mae_reduction"]} if d[k][f].get("available") else None)
                       for f in FAMILIES} for k in SPLITS} if "cohort" in d else d
    _print({"out": r["out"], "conditions": summ, "P5_inputs": {"available": r["P5_inputs"]["available"]},
            "secondary_dataset": r["secondary_dataset"]})


def cmd_linking(args):
    from .linking import link

    r = link(_cfg(args), args.out)
    _print({"out": r["out"], "studyA": {b: (sorted(v) if isinstance(v, dict) and "available" not in v else v)
                                        for b, v in r["studyA"].items()},
            "studyB": {c: {s: v[s].get("available") for s in v if isinstance(v[s], dict)} for c, v in r["studyB"].items()},
            "key_test": {c: {k: v.get(k) for k in ("available", "mae_reduction", "p_one_sided")}
                         for c, v in r["key_test"].items()}})


def cmd_primary(args):
    from .stats import primary

    r = primary(_cfg(args), args.out)
    _print({"out": r["out"], "p_raw": r["p_raw"], "p_holm": r["p_holm"], "reject": r["reject"],
            "reportable": r["reportable"], "estimates": {k: v.get("estimate") for k, v in r["tests"].items()}})


def main(argv=None):
    p = argparse.ArgumentParser(prog="privred")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", action="append", help="YAML config (repeatable)")
        sp.add_argument("--set", action="append", help="override key=value (repeatable)")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("check_companion", cmd_check_companion)
    sp.add_argument("--budget", type=float, action="append", help="budget to check (default: privacy_budgets)")
    sp = add("plan", cmd_plan)
    sp.add_argument("--stage", choices=["oof", "studyA", "prelim", "baseB", "deletion"])
    sp.add_argument("--budget", type=float)
    sp.add_argument("--names", help="comma-separated subset of the stage's set names")
    sp.add_argument("--out", help="plan file (one command per line)")
    sp.add_argument("--count", action="store_true", help="print run counts and assumed A100-hours (no files)")
    sp = add("member", cmd_member)
    sp.add_argument("--set_dir", required=True)
    sp.add_argument("--m", type=int, required=True)
    sp = add("launch", cmd_launch)
    sp.add_argument("--plan", required=True)
    sp.add_argument("--gpus", default="none", help="comma-separated GPU ids, or none")
    sp.add_argument("--jobs_per_gpu", type=int, default=1)
    sp = add("score", cmd_score)
    sp.add_argument("--set_dir", required=True)
    sp = add("difficulty", cmd_difficulty)
    sp.add_argument("--budget", type=float, required=True)
    add("features", cmd_features)
    sp = add("support", cmd_support)
    sp.add_argument("--budget", type=float, required=True)
    sp = add("studyA", cmd_studyA)
    sp.add_argument("--budget", type=float, required=True)
    sp.add_argument("--out", help="output JSON (default <budget dir>/analysis/studyA_b{pct}.json)")
    sp = add("deletion_sets", cmd_deletion_sets)
    sp.add_argument("--budget", type=float, help="default: studyB_budget")
    for name, fn, default in (("studyB", cmd_studyB, "studyB.json"), ("predict", cmd_predict, "prediction.json")):
        sp = add(name, fn)
        sp.add_argument("--budget", type=float, help="default: studyB_budget")
        sp.add_argument("--out", help=f"output JSON (default <studyB budget dir>/analysis/{default})")
    sp = add("linking", cmd_linking)
    sp.add_argument("--out", help="output JSON (default <studyB budget dir>/analysis/linking.json)")
    sp = add("primary", cmd_primary)
    sp.add_argument("--out", help="output JSON (default <studyB budget dir>/analysis/primary.json)")
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
