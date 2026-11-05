"""Split a pass@n file into per-item `correct` files for `vsca.cli stats compare` (PROBE_V3.md, experiment 2).

  python scripts/vote_files.py runs/.../test_pass8v.jsonl
writes test_pass8v.maj.jsonl (correct = maj_correct) and test_pass8v.vmaj.jsonl (correct = vmaj_correct) next to
it, so verified voting and plain voting over the same samples can be compared with the paired bootstrap.
"""

import json
import sys
from pathlib import Path

for p in map(Path, sys.argv[1:]):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for key in ("maj", "vmaj"):
        out = p.with_suffix(f".{key}.jsonl")
        out.write_text("".join(json.dumps({"id": r["id"], "correct": bool(r[f"{key}_correct"])}) + "\n" for r in rows))
        print(f"{out}: {sum(r[f'{key}_correct'] for r in rows) / len(rows):.4f}")
