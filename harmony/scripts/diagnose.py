"""Fit diagnostic (DIAGNOSTIC.md) on sampled responses: first-error kinds, certified sets and fork tokens.

  python scripts/diagnose.py --data data --samples runs/sft_s0 --out runs/sft_s0/diagnostic.json
Reads <samples>/<split>.jsonl for every test split present and applies the pre-registered go rule.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmony.diagnose import first_error, fork_token, summarise  # noqa: E402
from harmony.rules import Certifier, Exercise  # noqa: E402
from harmony.textfmt import prompt  # noqa: E402
from harmony.theory import MAJOR_KEYS, MINOR_KEYS, parse_note  # noqa: E402

SPLITS = ["test_id", "test_long", "test_minor", "test_keys"]
KEYS = {str(k): k for k in MAJOR_KEYS + MINOR_KEYS}


def exercise(r):
    return Exercise(KEYS[r["key"]], tuple(r["labels"]), tuple(parse_note(s) for s in r["soprano"]))


def go_rule(res: dict) -> dict:
    held = [s for s in SPLITS[1:] if s in res]
    fail_ok = [s for s in held if res[s]["sampled"]["failure_rate"] >= 0.30]
    cand = [s for s in fail_ok if res[s]["sampled"]["parse_rate"] >= 0.90
            and res[s]["sampled"]["located_share_of_failures"] >= 0.50
            and res[s]["sampled"]["certified_ge2_share"] >= 0.50]
    return {"held_out_with_failure_ge_0.30": fail_ok, "held_out_passing_all": cand,
            "decision": "go" if cand else "no_go"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--no_tokens", action="store_true")
    a = ap.parse_args()
    tok = None
    if not a.no_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    res = {}
    for split in SPLITS:
        sp = Path(a.samples) / f"{split}.jsonl"
        if not sp.exists():
            continue
        probs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl")}
        greedy, sampled = [], []
        for line in open(sp):
            s = json.loads(line)
            r = probs[s["id"]]
            ex = exercise(r)
            cert = Certifier(ex)
            for text, bucket in [(s["greedy"], greedy)] + [(t, sampled) for t in s["samples"]]:
                e = first_error(ex, text, cert)
                row = {"ok": e["ok"], "kind": e["kind"], "rules": e["rules"], "n_certified": len(e["certified"]),
                       "n_cert_tokens": None}
                if tok is not None and e["kind"] in ("rule", "dead_end"):
                    try:
                        row["n_cert_tokens"] = len(fork_token(tok, ex, prompt(ex), text, e)[2])
                    except StopIteration:
                        pass
                bucket.append(row)
        res[split] = {"greedy": summarise(greedy), "sampled": summarise(sampled)}
        g, sm = res[split]["greedy"], res[split]["sampled"]
        print(f"{split}: greedy fail {g['failure_rate']:.3f} | sampled fail {sm['failure_rate']:.3f} "
              f"parse {sm['parse_rate']:.3f} located {sm['located_share_of_failures']:.3f} "
              f"dead_end {sm['dead_end_share_of_located']:.3f} cert>=2 {sm['certified_ge2_share']:.3f} "
              f"tok_cert>=1 {sm['token_fork_with_cert_alt_share']:.3f}")
    res["go_rule"] = go_rule(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["go_rule"]))


if __name__ == "__main__":
    main()
