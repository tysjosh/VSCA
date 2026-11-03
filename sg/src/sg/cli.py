"""Command-line entry point: ``python -m sg.cli <command> ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _cfg(args):
    return load_config(args.config or [], args.set or [])


def cmd_prepare(args):
    from .data import load_dataset, prepare_cifar100, write_splits

    cfg = _cfg(args)
    if cfg.dataset == "cifar100":
        stats = prepare_cifar100(cfg.data_dir, cfg.val_frac, cfg.split_seed, cfg.audit_frac)
    else:
        ds = load_dataset(cfg)
        stats = {"dataset": ds.name, "n_pool": len(ds.y_train), "n_test": len(ds.y_test),
                 "budgets": write_splits(ds, cfg.val_frac, cfg.split_seed, Path(cfg.data_dir) / f"{ds.name}_splits.json",
                                         cfg.audit_frac)}
    print(json.dumps(stats, indent=2))
    for b, s in stats["budgets"].items():
        if b == "audit":
            print(f"audit pool (never trained on): {s['n']}")
        else:
            print(f"budget {b}: train {s['n_train']} val {s['n_val']}")


def cmd_simclr(args):
    from .simclr import train_simclr

    print(json.dumps(train_simclr(_cfg(args)), indent=2))


def cmd_partitions(args):
    import numpy as np

    from .data import load_dataset
    from .partitions import build_all

    cfg = _cfg(args)
    ds = load_dataset(cfg)
    mu = np.load(Path(args.extractor) / "class_means.npy")
    index = build_all(cfg, mu, ds.coarse_of_fine, args.out_dir)
    kinds = {}
    for p in index["partitions"]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(json.dumps({"out_dir": args.out_dir, "counts": kinds, "W_min": index["W_min"], "W_max": index["W_max"]}, indent=2))


def cmd_train(args):
    from .train import train

    print(json.dumps(train(_cfg(args)), indent=2))


def cmd_select_wd(args):
    from .train import select_wd

    print(json.dumps(select_wd(args.root, args.out), indent=2))


def cmd_select_rho(args):
    from .train import select_rho

    print(json.dumps(select_rho(args.root, args.out), indent=2))


def _assign_true(cfg):
    from .data import load_dataset

    return load_dataset(cfg).coarse_of_fine


def _dump(obj, out=None):
    text = json.dumps(obj, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
    print(text)


def cmd_probes(args):
    from .probes import run_probes

    res = run_probes(_cfg(args), args.run)
    print(json.dumps({"run": args.run, "pd_hist": res["pd_hist"], "n_ref": res["n_ref"], "n_test": res["n_test"],
                      "manipulation_check_blocks": [b["block"] for b in res["blocks"] if b["manipulation_check"]]}, indent=2))


def cmd_outcomes(args):
    from .analysis import partition_outcomes, scan_runs
    from .metrics import load_label_errors
    from .utils import write_jsonl

    cfg = _cfg(args)
    geo = json.loads(Path(args.geometry).read_text()) if args.geometry else None
    le_path = args.label_errors or cfg.label_errors
    le = load_label_errors(le_path) if le_path else None
    rows, strata = partition_outcomes(scan_runs(args.runs_root), cfg.dataset, args.budget, args.rho, args.total_steps,
                                      geo, le, assign_true=_assign_true(cfg), hard_frac=cfg.hard_frac)
    out = Path(args.out)
    write_jsonl(out, rows)
    strata_path = Path(f"{out}.strata.json")  # contract: <out>.strata.json
    strata_path.write_text(json.dumps(strata, indent=2))
    print(json.dumps({"out": str(out), "strata": str(strata_path), "n_units": len(rows),
                      "units": [f"{r['condition']}:{r['partition_id']}" for r in rows]}, indent=2))


def cmd_diagnose(args):
    from .analysis import baseline, geometry, scan_runs

    cfg = _cfg(args)
    if args.which == "geometry":
        if not args.extractor or not args.partitions_dirs:
            raise SystemExit("diagnose geometry needs --extractor and --partitions_dirs")
        files = [f for d in args.partitions_dirs for f in sorted(Path(d).glob("*.json"))]
        res = geometry(args.extractor, files)
        _dump(res, args.out)
    else:
        if not args.runs_root or not args.budgets:
            raise SystemExit("diagnose baseline needs --runs_root and --budgets")
        _dump(baseline(scan_runs(args.runs_root), args.budgets, cfg.dataset, _assign_true(cfg)), args.out)


def cmd_stats(args):
    from . import stats as st

    cfg = _cfg(args)

    def need(*names):
        miss = [n for n in names if getattr(args, n) is None]
        if miss:
            raise SystemExit(f"stats {args.which} needs " + " ".join(f"--{n}" for n in miss))

    w = args.which
    if w == "rank":
        need("outcomes", "metric")
        rows = st.load_outcomes(args.outcomes)
        tr = st.true_row(rows)
        rand = [r[args.metric] for r in rows if r["condition"] == "random_hier" and r.get(args.metric) is not None]
        res = {"metric": args.metric, "T_true": tr[args.metric], "R": len(rand),
               "p": st.rank_pvalue(tr[args.metric], rand) if tr[args.metric] is not None else None}
    elif w == "residual":
        need("outcomes", "metric")
        W, T, Wt, Tt, src, _ = st.fit_inputs(st.load_outcomes(args.outcomes), args.metric)
        res = {"metric": args.metric, "W_source": src, "W_true": Wt, "T_true": Tt,
               **st.residual(W, T, Wt, Tt, degree=args.degree)}
    elif w == "h2":
        need("runs_root", "budget")
        res = st.h2_from_runs(args.runs_root, cfg.dataset, args.budget, args.rho, _assign_true(cfg), args.n_boot,
                              total_steps=args.total_steps)
    elif w == "h3":
        need("low", "high")
        res = st.h3_bootstrap(st.load_outcomes(args.low), st.load_outcomes(args.high), args.metric or "T_acc", args.n_boot)
    elif w == "holm":
        need("pvals")
        res = st.holm(json.loads(Path(args.pvals).read_text()))
    elif w == "pilot":
        need("runs_root", "budget")
        res = st.pilot(args.runs_root, cfg.dataset, args.budget, args.rho, _assign_true(cfg))
    else:  # primary
        need("outcomes_primary", "runs_root", "budget", "low", "high")
        res = st.primary(args.outcomes_primary, args.runs_root, cfg.dataset, args.budget, args.rho, args.low, args.high,
                         args.n_boot, _assign_true(cfg))
    _dump(res, args.out)


def cmd_inat_build(args):
    from .data import load_inat100, write_splits
    from .inat import build_inat100

    cfg = _cfg(args)
    root = args.inat_root or cfg.inat_root
    if not root:
        raise SystemExit("inat_build needs --inat_root (or inat_root in the config)")
    out = Path(cfg.data_dir) / "inat100"
    res = build_inat100(root, out, cfg.inat_seed, cfg.inat_per_class, cfg.inat_test_per_class, cfg.inat_size)
    res["budgets"] = write_splits(load_inat100(cfg.data_dir), cfg.val_frac, cfg.split_seed,
                                  Path(cfg.data_dir) / "inat100_splits.json", cfg.audit_frac)
    print(json.dumps(res, indent=2))


def cmd_plan(args):
    from .plans import count, generate, to_lines

    cfg = _cfg(args)
    jobs = generate(args.exp, cfg, args.state_dir)
    lines = to_lines(jobs, args.config, args.set)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(ln + "\n" for ln in lines))
    if args.count or args.sec_per_epoch45k is not None:
        print(json.dumps({**count(args.exp, jobs, cfg, args.sec_per_epoch45k), "out": args.out}, indent=2))
    elif not args.out:
        print("\n".join(lines))
    else:
        print(json.dumps({"exp": args.exp, "out": args.out, "n_commands": len(lines)}))


def cmd_launch(args):
    from .plans import launch

    from .utils import DEADLINE_EXIT

    failures = launch(args.plan, args.gpus, args.jobs_per_gpu)
    stopped = [f for f in failures if f["rc"] == DEADLINE_EXIT]
    real = [f for f in failures if f["rc"] != DEADLINE_EXIT]
    print(json.dumps({"plan": args.plan, "n_failed": len(real), "n_stopped_for_deadline": len(stopped),
                      "failures": real}, indent=2))
    if real:
        raise SystemExit(1)
    if stopped:
        print(f"[resume] {len(stopped)} runs stopped or not started because the session deadline was reached; "
              "rerun the same command to continue.")
        raise SystemExit(DEADLINE_EXIT)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sg")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", action="append", help="YAML config (repeatable)")
        sp.add_argument("--set", action="append", help="override key=value (repeatable)")
        sp.set_defaults(fn=fn)
        return sp

    add("prepare", cmd_prepare)
    add("simclr", cmd_simclr)
    sp = add("partitions", cmd_partitions)
    sp.add_argument("--extractor", required=True, help="SimCLR run dir with class_means.npy")
    sp.add_argument("--out_dir", required=True)
    add("train", cmd_train)
    sp = add("select_wd", cmd_select_wd)
    sp.add_argument("--root", required=True)
    sp.add_argument("--out", required=True)
    sp = add("select_rho", cmd_select_rho)
    sp.add_argument("--root", required=True)
    sp.add_argument("--out", required=True)
    sp = add("probes", cmd_probes)
    sp.add_argument("--run", required=True, help="training run dir with model.pt")
    sp = add("outcomes", cmd_outcomes)
    sp.add_argument("--runs_root", required=True)
    sp.add_argument("--budget", type=float, required=True)
    sp.add_argument("--rho", type=float)
    sp.add_argument("--total_steps", type=int)
    sp.add_argument("--geometry", help="json from `diagnose geometry` (supplies W_analysis)")
    sp.add_argument("--label_errors", help="test ids flagged as mislabeled (json list or one id per line)")
    sp.add_argument("--out", required=True)
    sp = add("diagnose", cmd_diagnose)
    sp.add_argument("which", choices=["geometry", "baseline"])
    sp.add_argument("--extractor")
    sp.add_argument("--partitions_dirs", nargs="+")
    sp.add_argument("--runs_root")
    sp.add_argument("--budgets", type=float, nargs="+")
    sp.add_argument("--out")
    sp = add("stats", cmd_stats)
    sp.add_argument("which", choices=["rank", "residual", "h2", "h3", "holm", "pilot", "primary"])
    sp.add_argument("--outcomes")
    sp.add_argument("--metric")
    sp.add_argument("--degree", type=int, default=1, choices=[1, 2])
    sp.add_argument("--runs_root")
    sp.add_argument("--budget", type=float)
    sp.add_argument("--rho", type=float)
    sp.add_argument("--total_steps", type=int)
    sp.add_argument("--n_boot", type=int, default=10000)
    sp.add_argument("--low")
    sp.add_argument("--high")
    sp.add_argument("--pvals")
    sp.add_argument("--outcomes_primary")
    sp.add_argument("--out")
    sp = add("inat_build", cmd_inat_build)
    sp.add_argument("--inat_root", help="dir with train2018.json, categories.json and train_val2018/")
    sp = add("plan", cmd_plan)
    sp.add_argument("--exp", required=True)
    sp.add_argument("--out", help="plan file (one command per line)")
    sp.add_argument("--state_dir", help="dir with best_wd.json / best_rho.json (default <out_dir>/<dataset>)")
    sp.add_argument("--count", action="store_true", help="print run counts per budget and condition")
    sp.add_argument("--sec_per_epoch45k", type=float, help="assumed GPU seconds per 45k-image epoch, for a GPU-hour estimate")
    sp = add("launch", cmd_launch)
    sp.add_argument("--plan", required=True)
    sp.add_argument("--gpus", default="none", help="comma-separated GPU ids, or none")
    sp.add_argument("--jobs_per_gpu", type=int, default=1)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
