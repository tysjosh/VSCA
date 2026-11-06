"""Bookkeeping VSCA pilot (bookkeeping/PILOT.md): entries-only task, the ledger computes the balance sheet.

  python scripts/pilot.py pools --data data_p --samples runs/p_s0/train_samples.jsonl --out runs/p_s0/pools.json
  python scripts/pilot.py gate  --pools runs/p_s0/pools.json --out runs/p_s0/gate.json
  python scripts/pilot.py score --data data_p --split test_L3 --samples runs/p_s0/set/test_L3.jsonl --out ...
Training uses harmony/scripts/pilot.py train (domain-free: it reads the pools file).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookkeeping.check import first_error, fork_token  # noqa: E402
from bookkeeping.task import bs_lines, entry_key, gold_keys, replay  # noqa: E402

LOCATED = ("invalid", "no_match", "premature_end", "extra_entry")


def with_end(text: str) -> str:
    text = text.rstrip()
    return text if text.endswith("END") else text + "\nEND"


def cmd_pools(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.data) / f"{a.split}.jsonl")}
    verified, forks, refs, kinds, by_level = [], [], [], Counter(), Counter()
    for line in open(a.samples):
        s = json.loads(line)
        r = probs[s["id"]]
        rec, lvl = r["record"], r["record"]["difficulty_level"]
        refs.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"]})
        for j, text in enumerate(s["samples"]):
            text = with_end(text)
            e = first_error(rec, text, entries_only=True)
            kinds[e["kind"] or "ok"] += 1
            by_level[(lvl, "n")] += 1
            if e["ok"]:
                by_level[(lvl, "ok")] += 1
                verified.append({"id": f"{r['id']}-s{j}", "prompt": r["prompt"], "response": text})
            elif e["kind"] in LOCATED:
                try:
                    k, bad, alts = fork_token(tok, text, e)
                except StopIteration:
                    continue
                if not alts:
                    kinds["fork_without_certified_token"] += 1
                    continue
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                head = text[: enc["offset_mapping"][k][0]]
                votes = Counter()
                for c in e["certified"]:
                    full = text[: e["line_start"]] + c
                    if full.startswith(head):
                        e2 = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                        jj = next((i for i, (st, en) in enumerate(e2["offset_mapping"]) if st >= len(head)), None)
                        if jj is not None and e2["input_ids"][jj] in alts:
                            votes[e2["input_ids"][jj]] += 1
                a_star = min(votes, key=lambda t: (-votes[t], t)) if votes else alts[0]
                forks.append({"id": f"{r['id']}-s{j}-f", "prompt": r["prompt"], "response": text, "k": k, "e": bad,
                              "A": alts, "a_star": a_star, "kind": e["kind"], "level": lvl})
    lv = sorted({l for l, _ in by_level})
    stats = {"samples": sum(by_level[(l, "n")] for l in lv), "kinds": dict(kinds), "verified": len(verified),
             "forks": len(forks), "forks_A_ge2": sum(len(f["A"]) >= 2 for f in forks),
             "verified_share_by_level": {str(l): by_level[(l, "ok")] / by_level[(l, "n")] for l in lv}}
    Path(a.out).write_text(json.dumps({"verified": verified, "forks": forks, "refs": refs, "stats": stats}))
    print(json.dumps(stats))


def cmd_gate(a):
    """Learnability gate (PILOT.md): verified share >= 0.15 overall and >= 0.05 on level-3 training problems."""
    st = json.loads(Path(a.pools).read_text())["stats"]
    overall = st["verified"] / st["samples"]
    l3 = st["verified_share_by_level"].get("3", 0.0)
    ok = overall >= 0.15 and l3 >= 0.05
    out = {"verified_share": overall, "verified_share_L3": l3, "passed": ok}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    sys.exit(0 if ok else 3)


def cmd_score(a):
    probs = {json.loads(l)["id"]: json.loads(l)["record"] for l in open(Path(a.data) / f"{a.split}.jsonl")}
    rows = []
    for line in open(a.samples):
        s = json.loads(line)
        rec = probs[s["id"]]
        text = with_end(s["greedy"])
        e = first_error(rec, text, entries_only=True)
        rows.append({"id": s["id"], "correct": bool(e["ok"]), "kind": e["kind"]})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.out}: {sum(r['correct'] for r in rows) / len(rows):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pools")
    p.add_argument("--data", default="data_p")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--split", default="onpolicy", help="the records the samples were drawn for")
    g = sp.add_parser("gate")
    g.add_argument("--pools", required=True)
    g.add_argument("--out", required=True)
    s = sp.add_parser("score")
    s.add_argument("--data", default="data_p")
    s.add_argument("--split", required=True)
    s.add_argument("--samples", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pools": cmd_pools, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
