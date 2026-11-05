"""PROBE_V3.md experiment 3b: merge two pass@8 files of one adapter into 16 samples and score maj@16 and pmaj@16.

  python scripts/merge16.py RUN_DIR...   # reads test_pass8v.jsonl + test_pass8v_b.jsonl, writes test16.{maj,pmaj}.jsonl
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pal_check import OUT, majority  # noqa: E402

pal = {json.loads(l)["id"]: json.loads(l)["p_star"] for l in (OUT / "pal_programs.jsonl").read_text().splitlines()}
gold = {json.loads(l)["id"]: Fraction(json.loads(l)["gold"])
        for l in (Path(__file__).resolve().parents[1] / "data" / "gsm8k_test.jsonl").read_text().splitlines()}
for d in map(Path, sys.argv[1:]):
    load = lambda f: {r["id"]: r for r in map(json.loads, (d / f).read_text().splitlines())}
    a, b = load("test_pass8v.jsonl"), load("test_pass8v_b.jsonl")
    maj, pmaj = [], []
    for i in a:
        ans = [None if x is None else Fraction(x) for x in a[i]["answers"] + b[i]["answers"]]
        m = majority(ans)
        p = None if pal[i] is None else Fraction(pal[i])
        pick = p if p is not None and p in ans else m
        maj.append({"id": i, "correct": m is not None and m == gold[i]})
        pmaj.append({"id": i, "correct": pick is not None and pick == gold[i]})
    for name, rows in (("maj", maj), ("pmaj", pmaj)):
        (d / f"test16.{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(d.name, "maj@16 %.4f pmaj@16 %.4f" % (sum(r["correct"] for r in maj) / len(maj),
                                                sum(r["correct"] for r in pmaj) / len(pmaj)))
