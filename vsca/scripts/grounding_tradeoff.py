"""Reproduce the V+ grounding trade-off table (README "VSCA v2", PILOT_V2.md) from committed code.

For each point, on the 200 rows of labeling/error_taxonomy_sft.csv:
  * ref. false rejection = share of the correct references with at least one ungrounded step;
  * flags / 163 = needs_manual SFT failures with at least one arithmetic or grounding failure.
Each point adds to the one above it:
  FEAT-001  verify_plus.py as of commit bc7a513 (read with `git show`), FEAT-001 whitelist
  C         current verify_plus.py, FEAT-001 whitelist
  D         + whitelist 1, 0.01
  E         + whitelist 30, 31, 365, 52 (the default grounding_whitelist)
  F         + any single-constant annotation <<N=N>> counts as grounded
  G         + the number word "fifth" (1/5, so also 5 and 20)
  H         + whitelist 3, 4, 5, 6, 8, 10
Usage (from vsca/): ../.venv/bin/python scripts/grounding_tradeoff.py [--csv PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import vsca.verify_plus as vp  # noqa: E402

FEAT001_COMMIT = "bc7a513"
WL_FEAT001 = "60,24,7,12,100,1000,2,0.5"
WL_D = WL_FEAT001 + ",1,0.01"
WL_E = WL_D + ",30,31,365,52"
WL_H = WL_E + ",3,4,5,6,8,10"


def load_feat001():
    """verify_plus.py from commit bc7a513, imported as a module of the vsca package (its relative imports work)."""
    src = subprocess.run(["git", "show", f"{FEAT001_COMMIT}:vsca/src/vsca/verify_plus.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("vsca._verify_plus_feat001", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "vsca"
    sys.modules[spec.name] = mod  # dataclasses look the module up while the class body runs
    exec(compile(src, f"{FEAT001_COMMIT}:verify_plus.py", "exec"), mod.__dict__)
    return mod


def with_fifth(fn):
    """Run fn with "fifth" added to the number words of the current verify_plus."""
    words, word_re = vp._WORDS, vp._WORD_RE
    vp._WORDS = {**words, "fifth": Fraction(1, 5)}
    vp._WORD_RE = re.compile(r"\b(" + "|".join(sorted(vp._WORDS, key=len, reverse=True)) + r")s?\b", re.IGNORECASE)
    try:
        return fn()
    finally:
        vp._WORDS, vp._WORD_RE = words, word_re


def measure(rows, check, whitelist: str, single_const: bool = False) -> dict:
    wl = vp.parse_whitelist(whitelist)

    def ungrounded(res) -> bool:
        for s in res.steps:
            if s.grounded:
                continue
            ops = vp.expr_operands(s.claim.expr)
            if single_const and ops is not None and len(ops) == 1:
                continue
            return True
        return False

    ref_n = ref_bad = fail_n = flagged = 0
    for r in rows:
        gold = Fraction(r["gold"])
        if r.get("reference"):
            ref_n += 1
            ref_bad += ungrounded(check(r["question"], r["reference"], gold, wl, True))
        if r["auto_hint"].startswith("needs_manual"):
            fail_n += 1
            res = check(r["question"], r["generation"], gold, wl, True)
            flagged += any(s.arithmetic is not True for s in res.steps) or ungrounded(res)
    return {"ref_n": ref_n, "ref_false_rejections": ref_bad, "ref_false_rejection_rate": ref_bad / ref_n,
            "failures_n": fail_n, "failures_flagged": flagged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT.parent / "labeling" / "error_taxonomy_sft.csv"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    with open(a.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    old = load_feat001().check_response
    cur = lambda q, t, g, wl, gr: vp.check_response(q, t, g, wl, gr, min_steps=0)
    table = {
        "FEAT-001": measure(rows, old, WL_FEAT001),
        "C": measure(rows, cur, WL_FEAT001),
        "D": measure(rows, cur, WL_D),
        "E": measure(rows, cur, WL_E),
        "F": measure(rows, cur, WL_E, single_const=True),
        "G": with_fifth(lambda: measure(rows, cur, WL_E, single_const=True)),
        "H": with_fifth(lambda: measure(rows, cur, WL_H, single_const=True)),
    }
    print(f"{'point':9} {'ref. false rejection':>24} {'flags':>10}")
    for k, v in table.items():
        print(f"{k:9} {100 * v['ref_false_rejection_rate']:6.1f} % ({v['ref_false_rejections']:>3}/{v['ref_n']})"
              f"      {v['failures_flagged']:>3}/{v['failures_n']}")
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
