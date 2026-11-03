#!/usr/bin/env bash
# CPU test of resume: a VSCA run interrupted during certification, mid-epoch, and by a past deadline must end
# with exactly the same certification records, training losses and validation history as an uninterrupted run.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
R=${R:-runs_smoke}
C="--config configs/smoke.yaml --set out_dir=$R --set method=vsca --set epochs=2 --set refresh=epoch"
LOG=$R/resume_test.log
mkdir -p "$R"
rm -rf "$R/resume_ref" "$R/resume_test"
: > "$LOG"

expect() {  # expect <exit code> <description> <vsca train args...>
  local want=$1 what=$2
  shift 2
  "$PY" -m vsca.cli train "$@" >> "$LOG" 2>&1
  local got=$?
  if [ "$got" -ne "$want" ]; then
    echo "FAILED: $what (exit $got, expected $want)"; tail -n 30 "$LOG"; exit 1
  fi
  echo "ok: $what (exit $got)"
}

expect 0  "uninterrupted reference run" $C --set run_name=resume_ref
expect 75 "stop during epoch-0 certification" $C --set run_name=resume_test --set debug_stop_after_cert_examples=3
expect 75 "stop mid-epoch 1 (after optimizer step 3)" $C --set run_name=resume_test --set debug_stop_after_steps=3
expect 75 "stop at a deadline in the past" $C --set run_name=resume_test --set deadline=1
expect 0  "resume to the end" $C --set run_name=resume_test
expect 0  "rerun of a finished run is skipped" $C --set run_name=resume_test
expect 1  "changed config on an existing run is refused" $C --set run_name=resume_test --set lr=1e-3
grep -q "already finished; skipping" "$LOG" || { echo "FAILED: finished run was not skipped"; exit 1; }

"$PY" - "$R/resume_ref" "$R/resume_test" <<'EOF'
import json, sys
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
rows = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for e in (0, 1):
    ra, rb = rows(a / f"cert_epoch{e}.jsonl"), rows(b / f"cert_epoch{e}.jsonl")
    assert ra == rb, f"cert_epoch{e} differs"
    assert (b / f"cert_epoch{e}.jsonl.done").exists()
la, lb = rows(a / "train_log.jsonl"), rows(b / "train_log.jsonl")
assert [r["step"] for r in lb] == [r["step"] for r in la], ([r["step"] for r in la], [r["step"] for r in lb])
diff = max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lb))
assert diff < 1e-5, f"loss differs by {diff}"
sa, sb = json.loads((a / "train_summary.json").read_text()), json.loads((b / "train_summary.json").read_text())
assert sa["history"] == sb["history"], "validation history differs"
assert sb["resumes"] == 3, sb["resumes"]
assert not (b / "ckpt.pt").exists(), "checkpoint not cleaned up"
print(f"resume matches the uninterrupted run: {len(la)} steps, max loss difference {diff:.2e}, {sb['resumes']} resumes")
EOF
[ $? -eq 0 ] || { echo "FAILED: resumed run differs from the uninterrupted run"; exit 1; }
rm -rf "$R/resume_ref" "$R/resume_test"
echo "resume OK"
