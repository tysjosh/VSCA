"""Command-line entry point: ``python -m vsca.cli <command> ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _cfg(args):
    return load_config(args.config or [], args.set or [])


def cmd_prepare(args):
    from .data import prepare_gsm8k, prepare_svamp

    cfg = _cfg(args)
    stats = prepare_gsm8k(cfg.data_dir, cfg.val_frac, cfg.split_seed)
    stats.update(prepare_svamp(cfg.data_dir))
    Path(cfg.data_dir, "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def _train_examples(cfg, tok, with_cf=False):
    from .data import load_split, tokenize_example

    recs = load_split(cfg.data_dir, "gsm8k_train", cfg.limit_train)
    exs = [tokenize_example(tok, r, cfg.max_len, with_cf=with_cf) for r in recs]
    return recs, [e for e in exs if e is not None]


def cmd_score(args):
    from .modeling import load_model, load_tokenizer
    from .scoring import score_tokens

    cfg = _cfg(args)
    if cfg.resume and Path(args.out).exists():
        print(f"[resume] {args.out} exists; skipping")
        return
    tok = load_tokenizer(cfg.model_name)
    _, exs = _train_examples(cfg, tok)
    model, _ = load_model(cfg, adapter=args.adapter, attn_impl="eager" if args.attention else None)
    score_tokens(model, tok, exs, cfg.micro_bs, Path(args.out), with_attention=args.attention, attn_layer=cfg.attn_layer)
    print(f"wrote {args.out}")


def cmd_train(args):
    from .train import train

    res = train(_cfg(args))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


def cmd_certify(args):
    from .certify import certify
    from .modeling import load_model, load_tokenizer
    from .utils import ComputeLog

    cfg = _cfg(args)
    tok = load_tokenizer(cfg.model_name)
    _, exs = _train_examples(cfg, tok)
    model, _ = load_model(cfg, adapter=args.adapter)
    out = Path(args.out)
    with ComputeLog(out.with_suffix(".compute.json")).phase("certify"):
        certify(model, tok, exs, cfg, out)
    print(f"wrote {out}")


def cmd_onpolicy(args):
    """VSCA v2: sample -> verify -> localise -> certify the forks into one dir, shared by the arms of a seed."""
    from .data import load_split
    from .modeling import load_model, load_tokenizer
    from .onpolicy import prepare
    from .utils import ComputeLog

    cfg = _cfg(args)
    out = Path(args.out)
    tok = load_tokenizer(cfg.model_name)
    recs = load_split(cfg.data_dir, "gsm8k_train", cfg.limit_train)
    model, _ = load_model(cfg, adapter=cfg.init_adapter)
    stats = prepare(model, tok, recs, cfg, out, clog=ComputeLog(out / "compute.json"))
    print(json.dumps(stats, indent=2))


def cmd_eval(args):
    from .data import load_split
    from .evaluate import evaluate
    from .modeling import load_model, load_tokenizer
    from .utils import ComputeLog

    from .utils import check_deadline

    cfg = _cfg(args)
    out = Path(args.out)
    if cfg.resume and out.with_suffix(".summary.json").exists():
        print(f"[resume] {out} already evaluated; skipping")
        print(out.with_suffix(".summary.json").read_text())
        return
    tok = load_tokenizer(cfg.model_name)
    recs = load_split(cfg.data_dir, args.split, cfg.limit_test)
    model, _ = load_model(cfg, adapter=args.adapter)
    with ComputeLog(out.with_suffix(".compute.json")).phase(f"eval_{args.split}"):
        summ = evaluate(model, tok, recs, cfg.eval_bs, cfg.gen_max_new, out, desc=args.split,
                        should_stop=lambda: check_deadline(cfg, f"eval {args.split}"))
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def cmd_passk(args):
    from .data import load_split
    from .evaluate import evaluate_passk
    from .modeling import load_tokenizer
    from .utils import ComputeLog

    cfg = _cfg(args)
    out = Path(args.out)
    if cfg.resume and out.with_suffix(".summary.json").exists():
        print(f"[resume] {out} already evaluated; skipping")
        print(out.with_suffix(".summary.json").read_text())
        return
    tok = load_tokenizer(cfg.model_name)
    recs = load_split(cfg.data_dir, args.split, cfg.limit_test)
    with ComputeLog(out.with_suffix(".compute.json")).phase(f"passk_{args.split}"):
        summ = evaluate_passk(cfg, tok, recs, args.adapter, args.n, args.temperature, out, seed=cfg.seed)
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def cmd_diagnose(args):
    from . import diagnostics as D
    from .data import load_split
    from .modeling import load_model, load_tokenizer
    from .utils import read_jsonl

    cfg = _cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    if args.which == "taxonomy":
        if not args.eval_file:
            raise SystemExit("taxonomy needs --eval_file (output of `vsca eval --split gsm8k_val` with the base model)")
        res = D.error_taxonomy(read_jsonl(args.eval_file), load_split(cfg.data_dir, "gsm8k_val"), out_dir / "error_taxonomy.csv")
    elif args.which == "mass":
        res = D.certified_mass(args.cert)
    elif args.which == "audit":
        tok = load_tokenizer(cfg.model_name)
        _, exs = _train_examples(cfg, tok)
        res = D.audit_sample(args.cert, exs, tok, out_dir / "certificate_audit.csv")
    elif args.which == "earlier_score":
        tok = load_tokenizer(cfg.model_name)
        _, exs = _train_examples(cfg, tok, with_cf=True)
        res = D.earlier_score(exs)
        if args.with_model:
            model, _ = load_model(cfg)
            res["pairwise_saturation"] = D.pairwise_saturation(model, tok, exs, cfg.cf_tau)
    elif args.which == "grounding":
        res = D.grounding_coverage(args.csv, cfg)
    elif args.which == "vplus_trace":
        if not args.eval_file:
            raise SystemExit("vplus_trace needs --eval_file (output of `vsca eval`)")
        res = D.vplus_trace(read_jsonl(args.eval_file), load_split(cfg.data_dir, args.split), cfg)
    elif args.which == "recoverability":
        op_dir = args.onpolicy_dir
        if not op_dir:  # build the pipeline dir with --adapter (default init_adapter) first
            from .onpolicy import prepare
            from .utils import ComputeLog

            op_dir = out_dir / "onpolicy"
            tok = load_tokenizer(cfg.model_name)
            model, _ = load_model(cfg, adapter=args.adapter or cfg.init_adapter)
            prepare(model, tok, load_split(cfg.data_dir, "gsm8k_train", cfg.limit_train), cfg, op_dir,
                    clog=ComputeLog(Path(op_dir) / "compute.json"))
        res = D.fork_recoverability(op_dir)
    (out_dir / f"diag_{args.which}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_stats(args):
    from . import stats as S

    if args.which == "compare":
        res = S.paired_bootstrap(args.a, args.b, n_boot=args.n_boot)
    elif args.which == "mcnemar":
        res = S.mcnemar_exact(args.a[0], args.b[0])
    elif args.which == "power":
        import numpy as np

        _, M = S.load_runs(args.a)
        accs = M.mean(1).tolist()
        disc = float(np.mean([(M[i] != M[j]).mean() for i in range(len(M)) for j in range(i + 1, len(M))])) if len(M) > 1 else 0.1
        res = S.seeds_needed(accs, disc, M.shape[1], mde=args.mde)
        res.update({"pilot_accs": accs, "pilot_discordant_rate": disc})
    elif args.which == "holm":
        res = S.holm(json.loads(Path(args.pvals).read_text()))
    print(json.dumps(res, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="vsca")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--config", action="append", help="YAML config (repeatable)")
        sp.add_argument("--set", action="append", help="override key=value (repeatable)")
        sp.set_defaults(fn=fn)
        return sp

    add("prepare", cmd_prepare)
    sp = add("score", cmd_score)
    sp.add_argument("--adapter", default=None, help="LoRA adapter dir (reference model); omit for the base model")
    sp.add_argument("--attention", action="store_true", help="also compute ssToken attention scores")
    sp.add_argument("--out", required=True)
    add("train", cmd_train)
    sp = add("certify", cmd_certify)
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--out", required=True)
    sp = add("onpolicy", cmd_onpolicy)
    sp.add_argument("--out", required=True, help="on-policy pipeline dir (resumable by rerunning)")
    sp = add("eval", cmd_eval)
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--split", default="gsm8k_test", help="gsm8k_test | gsm8k_val | svamp_all")
    sp.add_argument("--out", required=True)
    sp = add("passk", cmd_passk)
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--split", default="gsm8k_test")
    sp.add_argument("--n", type=int, default=8)
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--out", required=True)
    sp = add("diagnose", cmd_diagnose)
    sp.add_argument("which", choices=["taxonomy", "mass", "audit", "earlier_score", "grounding", "vplus_trace",
                                      "recoverability"])
    sp.add_argument("--eval_file")
    sp.add_argument("--csv", default="../labeling/error_taxonomy_sft.csv", help="grounding: labelled taxonomy CSV")
    sp.add_argument("--split", default="gsm8k_test", help="vplus_trace: split the eval file was produced on")
    sp.add_argument("--cert")
    sp.add_argument("--onpolicy_dir", help="recoverability: existing on-policy dir (else built into OUT_DIR/onpolicy)")
    sp.add_argument("--adapter", default=None, help="recoverability without --onpolicy_dir: policy adapter "
                                                    "(default init_adapter)")
    sp.add_argument("--with_model", action="store_true")
    sp.add_argument("--out_dir", default="runs/diagnostics")
    sp = sub.add_parser("stats")
    sp.add_argument("which", choices=["compare", "mcnemar", "power", "holm"])
    sp.add_argument("--a", nargs="+", default=[])
    sp.add_argument("--b", nargs="+", default=[])
    sp.add_argument("--pvals")
    sp.add_argument("--mde", type=float, default=0.015)
    sp.add_argument("--n_boot", type=int, default=10000)
    sp.set_defaults(fn=cmd_stats)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
