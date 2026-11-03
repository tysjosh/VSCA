#!/usr/bin/env bash
# CPU test of resume: a True-Hier run and a SimCLR run interrupted after an epoch and by a past deadline must end
# with the same logs, predictions and outputs as uninterrupted runs. Run after `sg prepare` (uses runs_smoke).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
R=${R:-runs_smoke}
C="--config configs/smoke.yaml --set out_dir=$R --set budget=0.1"
T="$C --set condition=true_hier --set epochs=3 --set save_model=false"
S="$C --set simclr_epochs=3"
LOG=$R/resume_test.log
mkdir -p "$R"
rm -rf "$R/resume"
: > "$LOG"

expect() {  # expect <exit code> <description> <sg args...>
  local want=$1 what=$2
  shift 2
  "$PY" -m sg.cli "$@" >> "$LOG" 2>&1
  local got=$?
  if [ "$got" -ne "$want" ]; then
    echo "FAILED: $what (exit $got, expected $want)"; tail -n 30 "$LOG"; exit 1
  fi
  echo "ok: $what (exit $got)"
}

expect 0  "train: uninterrupted"            train $T --set run_name=resume/train_ref
expect 75 "train: stop after epoch 1"       train $T --set run_name=resume/train_res --set debug_stop_after_epochs=1
expect 75 "train: stop at a past deadline"  train $T --set run_name=resume/train_res --set deadline=1
expect 0  "train: resume to the end"        train $T --set run_name=resume/train_res
expect 1  "train: changed config refused"   train $T --set run_name=resume/train_res --set lr=0.05
expect 0  "simclr: uninterrupted"           simclr $S --set run_name=resume/simclr_ref
expect 75 "simclr: stop after epoch 1"      simclr $S --set run_name=resume/simclr_res --set debug_stop_after_epochs=1
expect 0  "simclr: resume to the end"       simclr $S --set run_name=resume/simclr_res

"$PY" - "$R/resume" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
r = Path(sys.argv[1])
rows = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
a, b = r / "train_ref", r / "train_res"
la, lb = rows(a / "train_log.jsonl"), rows(b / "train_log.jsonl")
assert [x["epoch"] for x in la] == [x["epoch"] for x in lb]
d = max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lb))
assert d < 1e-5, f"train loss differs by {d}"
assert la[-1]["lambdas"] == lb[-1]["lambdas"], "lambda_s differs"
assert [x["pred"] for x in rows(a / "test.jsonl")] == [x["pred"] for x in rows(b / "test.jsonl")], "test predictions differ"
sb = json.loads((b / "train_summary.json").read_text())
assert sb["resumes"] == 2 and not (b / "ckpt.pt").exists(), sb
ma, mb = np.load(r / "simclr_ref" / "class_means.npy"), np.load(r / "simclr_res" / "class_means.npy")
ds = float(np.abs(ma - mb).max())
assert ds < 1e-5, f"SimCLR class means differ by {ds}"
assert not (r / "simclr_res" / "ckpt.pt").exists()
print(f"resume matches: train loss diff {d:.1e}, SimCLR class-mean diff {ds:.1e}")
EOF
[ $? -eq 0 ] || { echo "FAILED: resumed runs differ from uninterrupted runs"; exit 1; }
rm -rf "$R/resume"
echo "resume OK"
