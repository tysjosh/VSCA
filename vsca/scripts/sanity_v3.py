"""Checks for PROBE_V3.md: the gold-free step check and verified vote (experiment 2), and fork_mode=add training on
exactly rft_verified's SFT stream (experiment 1, CPU smoke model). Run from vsca/: python scripts/sanity_v3.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from vsca.config import Config
from vsca.evaluate import steps_valid, verified_vote
from vsca.verify_plus import vplus_from_cfg

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


WL, GR, MS = vplus_from_cfg(Config())
q = "Sam has 5 boxes with 4 pens each. He gives away 3 pens. How many pens does he have?"
good = "He has <<5*4=20>>20 pens.\nHe keeps <<20-3=17>>17.\n#### 17"
bad_arith = "He has <<5*4=21>>21 pens.\nHe keeps <<21-3=18>>18.\n#### 18"
invented = "He has <<5*4=20>>20 pens.\nHe keeps <<20-9=11>>11.\n#### 11"
nostep = "He has 17 pens.\n#### 17"
check("steps_valid: correct steps pass without the gold answer", steps_valid(q, good, WL, GR, MS))
check("steps_valid: an arithmetic error fails", not steps_valid(q, bad_arith, WL, GR, MS))
check("steps_valid: an invented operand fails", not steps_valid(q, invented, WL, GR, MS))
check("steps_valid: no step fails with min_steps=1", not steps_valid(q, nostep, WL, GR, MS))
F = Fraction
check("verified_vote: step-valid minority beats an invalid majority",
      verified_vote([F(18), F(18), F(18), F(17)], [False, False, False, True]) == F(17))
check("verified_vote: falls back to plain majority when nothing passes",
      verified_vote([F(18), F(18), F(17)], [False, False, False]) == F(18))
check("verified_vote: an answer of 0 is not treated as missing",
      verified_vote([F(0), F(5), F(5)], [True, False, False]) == F(0))
check("verified_vote: ties go to the first step-valid occurrence",
      verified_vote([F(3), F(4), F(4), F(3)], [True, True, True, True]) == F(3))
check("verified_vote: all answers missing -> None", verified_vote([None, None], [True, False]) is None)

# --- fork_mode=add trains on rft_verified's SFT stream ---------------------------------------------------------
PY = sys.executable
TMP = Path(tempfile.mkdtemp(prefix="sanity_v3_"))
base = ["--config", "configs/smoke.yaml", "--set", f"out_dir={TMP}", "--set", "onpolicy_k=0",
        "--set", "debug_seed_samples=true"]


def cli(*a):
    r = subprocess.run([PY, "-m", "vsca.cli", *a], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r.returncode


rc = cli("onpolicy", *base, "--out", str(TMP / "op"))
check("smoke pipeline dir built", rc == 0)
runs = {
    "rft": ["--set", "method=rft_verified"],
    "add": ["--set", "method=vsca2", "--set", "fork_mode=add", "--set", "fork_lambda=0.3"],
    "replace": ["--set", "method=vsca2"],
}
for name, a in runs.items():
    rc = cli("train", *base, *a, "--set", "refresh=once", "--set", f"onpolicy_dir={TMP / 'op'}", "--set", f"run_name={name}")
    check(f"train {name} ran", rc == 0)
st = {n: json.loads((TMP / n / "onpolicy_stats_epoch0.json").read_text()) for n in runs}
check("fork_mode=add: same SFT stream as rft_verified (main_order_sha256)",
      st["add"]["main_order_sha256"] == st["rft"]["main_order_sha256"])
check("fork_mode=replace: a different stream (forks displace samples)",
      st["replace"]["main_order_sha256"] != st["rft"]["main_order_sha256"])
logs = [json.loads(l) for l in (TMP / "add" / "train_log.jsonl").read_text().splitlines()]
check("fork_mode=add: fork loss logged and finite on every step",
      st["add"]["fork_draws"] > 0 and all("fork_add_loss" in r and r["fork_add_loss"] == r["fork_add_loss"] for r in logs))
check("fork_mode=add: no forks inside the SFT batches", all(r.get("n_forks", 0) == 0 for r in logs))
try:
    Config(method="rft_verified", fork_mode="add").validate()
    check("fork_mode=add refused for rft_verified", False)
except ValueError:
    check("fork_mode=add refused for rft_verified", True)
shutil.rmtree(TMP)
raise SystemExit(0 if ok else 1)
