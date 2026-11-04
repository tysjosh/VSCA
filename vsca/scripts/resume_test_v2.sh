#!/usr/bin/env bash
# CPU test of resume for VSCA v2: a dynamic vsca2 run (on-policy pipeline rebuilt every epoch) interrupted during
# on-policy sampling, during fork certification, mid-epoch and by a past deadline must end with exactly the same
# pipeline files, training losses and validation history as an uninterrupted run.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
R=${R:-runs_smoke_v2}
# onpolicy_k / mc_M below the defaults (8 / 4) keep the CPU run short
C="--config configs/smoke.yaml --set out_dir=$R --set method=vsca2 --set epochs=2 --set refresh=epoch
   --set debug_seed_samples=true --set onpolicy_k=2 --set mc_M=2"
LOG=$R/resume_test_v2.log
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
expect 75 "stop during epoch-0 on-policy sampling" $C --set run_name=resume_test --set debug_stop_after_sample_examples=3
grep -q "debug stop after .* problems in onpolicy sample" "$LOG" || { echo "FAILED: no stop during sampling"; exit 1; }
expect 75 "stop during epoch-0 fork certification" $C --set run_name=resume_test --set debug_stop_after_cert_examples=1
grep -q "debug stop after .* problems in onpolicy certify" "$LOG" || { echo "FAILED: no stop during fork certification"; exit 1; }
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
n_files = 0
for e in (0, 1):
    da, db = a / f"onpolicy_epoch{e}", b / f"onpolicy_epoch{e}"
    names = sorted(p.name for p in da.glob("*.jsonl"))
    assert names == sorted(p.name for p in db.glob("*.jsonl")), f"onpolicy_epoch{e}: different files"
    assert "forks.jsonl" in names, names
    for n in names:
        assert rows(da / n) == rows(db / n), f"onpolicy_epoch{e}/{n} differs"
        assert (db / f"{n}.done").exists(), f"onpolicy_epoch{e}/{n}.done missing"
        n_files += 1
    assert json.loads((da / "onpolicy_stats.json").read_text()) == json.loads((db / "onpolicy_stats.json").read_text())
    sa, sb = (json.loads((d / f"onpolicy_stats_epoch{e}.json").read_text()) for d in (a, b))
    sa.pop("onpolicy_dir"), sb.pop("onpolicy_dir")
    assert sa == sb, f"onpolicy_stats_epoch{e}.json differs"
la, lb = rows(a / "train_log.jsonl"), rows(b / "train_log.jsonl")
assert [r["step"] for r in lb] == [r["step"] for r in la], ([r["step"] for r in la], [r["step"] for r in lb])
diff = max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lb))
assert diff < 1e-5, f"loss differs by {diff}"
sa, sb = json.loads((a / "train_summary.json").read_text()), json.loads((b / "train_summary.json").read_text())
assert sa["history"] == sb["history"], "validation history differs"
assert sb["resumes"] == 4, sb["resumes"]
assert not (b / "ckpt.pt").exists(), "checkpoint not cleaned up"
print(f"resume matches the uninterrupted run: {n_files} pipeline files, {len(la)} steps, "
      f"max loss difference {diff:.2e}, {sb['resumes']} resumes")
EOF
[ $? -eq 0 ] || { echo "FAILED: resumed run differs from the uninterrupted run"; exit 1; }
rm -rf "$R/resume_ref" "$R/resume_test"
echo "resume v2 OK"
