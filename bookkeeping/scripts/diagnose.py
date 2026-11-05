"""Fit diagnostic (DIAGNOSTIC.md) for bookkeeping samples: first-error kinds, certified sets, fork tokens, and
FinBalance's BS_exact / BS_recon audit.

  python scripts/diagnose.py --data data --samples runs/sft_s0 --out runs/sft_s0/diagnostic.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookkeeping.check import audit, first_error, fork_token  # noqa: E402

SPLITS = ["test_id", "test_L3", "test_ind", "test_year"]
LOCATED = ("invalid", "no_match", "premature_end", "extra_entry", "bs_value", "bs_account", "bs_extra")


def summarise(rows):
    n = len(rows)
    fail = [r for r in rows if not r["ok"]]
    loc = [r for r in fail if r["kind"] in LOCATED]
    tok = [r for r in loc if r["n_cert_tokens"] is not None]
    sh = lambda a, b: a / b if b else 0.0
    return {
        "samples": n,
        "failure_rate": sh(len(fail), n),
        "parse_rate": sh(sum(r["kind"] != "format" for r in rows), n),
        "first_error_kinds": dict(Counter(r["kind"] for r in fail)),
        "no_match_sub": dict(Counter(r["sub"] for r in fail if r["kind"] == "no_match")),
        "located_share_of_failures": sh(len(loc), len(fail)),
        "certified_ge2_share": sh(sum(r["n_certified"] >= 2 for r in loc), len(loc)),
        "mean_certified": sh(sum(r["n_certified"] for r in loc), len(loc)),
        "gold_free_share_of_located": sh(sum(r["kind"] in ("invalid", "bs_value", "bs_account", "bs_extra")
                                             for r in loc), len(loc)),
        "token_fork_with_cert_alt_share": sh(sum(r["n_cert_tokens"] >= 1 for r in tok), len(tok)),
        "token_fork_ge2_share": sh(sum(r["n_cert_tokens"] >= 2 for r in tok), len(tok)),
        "bs_exact": sh(sum(r["audit"]["bs_exact"] for r in rows), n),
        "bs_recon": sh(sum(r["audit"]["bs_recon"] for r in rows), n),
        "self_consistent": sh(sum(r["audit"]["self_consistent"] for r in rows), n),
        "entries_exact": sh(sum(r["audit"]["entries_exact"] for r in rows), n),
    }


def go_rule(res):
    held = [s for s in SPLITS[1:] if s in res]
    f = [s for s in held if res[s]["sampled"]["failure_rate"] >= 0.30]
    c = [s for s in f if res[s]["sampled"]["parse_rate"] >= 0.90 and res[s]["sampled"]["located_share_of_failures"]
         >= 0.50 and res[s]["sampled"]["certified_ge2_share"] >= 0.50]
    return {"held_out_with_failure_ge_0.30": f, "held_out_passing_all": c, "decision": "go" if c else "no_go"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    res = {}
    for split in SPLITS:
        sp = Path(a.samples) / f"{split}.jsonl"
        if not sp.exists():
            continue
        recs = {json.loads(l)["id"]: json.loads(l)["record"] for l in open(Path(a.data) / f"{split}.jsonl")}
        greedy, sampled = [], []
        for line in open(sp):
            s = json.loads(line)
            rec = recs[s["id"]]
            for text, bucket in [(s["greedy"], greedy)] + [(t, sampled) for t in s["samples"]]:
                e = first_error(rec, text)
                row = {"ok": e["ok"], "kind": e["kind"], "sub": e.get("sub"), "n_certified": len(e["certified"]),
                       "n_cert_tokens": None, "audit": audit(rec, text)}
                if e["kind"] in LOCATED:
                    try:
                        row["n_cert_tokens"] = len(fork_token(tok, text, e)[2])
                    except StopIteration:
                        pass
                bucket.append(row)
        res[split] = {"greedy": summarise(greedy), "sampled": summarise(sampled)}
        g, sm = res[split]["greedy"], res[split]["sampled"]
        print(f"{split}: greedy fail {g['failure_rate']:.3f} bs_exact {g['bs_exact']:.3f} bs_recon {g['bs_recon']:.3f}"
              f" | sampled fail {sm['failure_rate']:.3f} parse {sm['parse_rate']:.3f} located "
              f"{sm['located_share_of_failures']:.3f} cert>=2 {sm['certified_ge2_share']:.3f} kinds "
              f"{sm['first_error_kinds']}")
    res["go_rule"] = go_rule(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["go_rule"]))


if __name__ == "__main__":
    main()
