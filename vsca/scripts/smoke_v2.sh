#!/usr/bin/env bash
# Exercise every VSCA v2 arm, the shared on-policy pipeline dirs, a dynamic vsca2 run, evaluations, the v2
# diagnostics, stats and the v2 resume test on a tiny model (CPU, hf backend). Checks that code paths run;
# the numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
R=runs_smoke_v2
# onpolicy_k / mc_M below the defaults (8 / 4) keep the CPU run short
C="--config configs/smoke.yaml --set out_dir=$R --set debug_seed_samples=true --set onpolicy_k=2 --set mc_M=2"
INIT="--set init_adapter=$R/sft/adapter_best"  # every v2 arm and both pipeline dirs start from the same SFT adapter
LOG=$R/smoke_v2.log
rm -rf "$R"
mkdir -p "$R"

run() {  # run one vsca.cli command; on failure print the end of its log and stop
  echo "=== $*"
  if ! "$PY" -m vsca.cli "$@" > "$LOG" 2>&1; then
    tail -n 40 "$LOG"; echo "FAILED: $*"; exit 1
  fi
  grep -E '^\[(onpolicy|val|resume)\]|"accuracy"' "$LOG" | cut -c1-240 || true
}

run train $C --set method=sft --set run_name=sft
run onpolicy $C $INIT --out $R/op
run onpolicy $C $INIT --set vplus_grounding=false --set onpolicy_samples=$R/op/samples.jsonl --out $R/op_nogr

run train $C $INIT --set method=rft_verified --set run_name=rft_verified --set onpolicy_dir=$R/op
run train $C $INIT --set method=stepdpo_fork --set run_name=stepdpo_fork --set onpolicy_dir=$R/op
run train $C $INIT --set method=vsca2 --set run_name=vsca2 --set onpolicy_dir=$R/op
run train $C $INIT --set method=vsca2 --set variant=uncertified --set run_name=vsca2_uncertified --set onpolicy_dir=$R/op
run train $C $INIT --set method=vsca2 --set vplus_grounding=false --set run_name=vsca2_nogrounding --set onpolicy_dir=$R/op_nogr
run train $C $INIT --set method=vsca2 --set run_name=vsca2_dynamic --set epochs=2 --set refresh=epoch

run eval $C --adapter $R/vsca2/adapter_best --split gsm8k_test --out $R/eval_vsca2_test.jsonl
run eval $C --adapter $R/rft_verified/adapter_best --split gsm8k_test --out $R/eval_rft_verified_test.jsonl
run eval $C --adapter $R/vsca2/adapter_best --split svamp_all --out $R/eval_vsca2_svamp.jsonl

run diagnose vplus_trace $C --eval_file $R/eval_vsca2_test.jsonl --split gsm8k_test --out_dir $R/diag
run diagnose grounding $C --out_dir $R/diag
run diagnose recoverability $C --onpolicy_dir $R/op --out_dir $R/diag

"$PY" -m vsca.cli stats compare --a $R/eval_vsca2_test.jsonl --b $R/eval_rft_verified_test.jsonl --n_boot 200
"$PY" -m vsca.cli stats mcnemar --a $R/eval_vsca2_test.jsonl --b $R/eval_rft_verified_test.jsonl

echo "=== checks"
"$PY" - "$R" <<'EOF'
import json, math, sys
from pathlib import Path
R = Path(sys.argv[1])
js = lambda p: json.loads(Path(p).read_text())
arms = ["rft_verified", "stepdpo_fork", "vsca2", "vsca2_uncertified", "vsca2_nogrounding"]
st = {a: js(R / a / "onpolicy_stats_epoch0.json") for a in arms}
shas = {a: s["samples_sha256"] for a, s in st.items()}
assert len(set(shas.values())) == 1, f"samples_sha256 differ across arms: {shas}"
assert shas["vsca2"] == js(R / "op" / "onpolicy_stats.json")["samples_sha256"]
same = [st[a]["verified_examples"] for a in ("rft_verified", "stepdpo_fork", "vsca2", "vsca2_uncertified")]
assert len(set(same)) == 1, f"verified examples differ across arms of one pipeline dir: {same}"
assert st["rft_verified"]["fork_examples"] == 0, st["rft_verified"]
assert st["vsca2"]["fork_examples"] > 0 and st["stepdpo_fork"]["fork_examples"] == st["vsca2"]["fork_examples"]
assert st["stepdpo_fork"]["forks_with_set"] == 0, st["stepdpo_fork"]
assert st["vsca2_uncertified"]["forks_with_set"] == st["vsca2"]["forks_with_set"]
steps_per_epoch = math.ceil(8 / 4)  # limit_train / eff_bs of configs/smoke.yaml (Decision 3)
for a in arms + ["vsca2_dynamic"]:
    cfg = js(R / a / "config.json")
    assert cfg["init_adapter"] == str(R / "sft" / "adapter_best"), (a, cfg["init_adapter"])
    log = [json.loads(l) for l in (R / a / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == list(range(1, steps_per_epoch * cfg["epochs"] + 1)), (a, len(log))
    assert (R / a / "train_summary.json").exists(), a
    assert "n_forks" in log[0], a
for d in ("op", "op_nogr"):
    assert js(R / d / "onpolicy_config.json")["init_adapter"] == str(R / "sft" / "adapter_best"), d
assert js(R / "op_nogr" / "onpolicy_config.json")["vplus_grounding"] is False
comp = js(R / "vsca2_dynamic" / "compute.json")
for ph in ("onpolicy_sample", "onpolicy_verify", "onpolicy_localise", "onpolicy_certify"):
    assert comp[ph]["calls"] == 2, (ph, comp.get(ph))  # refresh=epoch: one pipeline per epoch
for e in (0, 1):
    assert (R / "vsca2_dynamic" / f"onpolicy_epoch{e}" / "forks.jsonl.done").exists(), e
for w in ("vplus_trace", "grounding", "recoverability"):
    assert (R / "diag" / f"diag_{w}.json").exists(), w
print(f"checks OK: samples_sha256 {shas['vsca2'][:12]} in all five arms; "
      f"examples per arm {[st[a]['examples'] for a in arms]}; fork examples {st['vsca2']['fork_examples']}")
EOF

echo "=== resume test v2"
PY="$PY" R="$R" bash scripts/resume_test_v2.sh
rm -rf "$R"
echo "smoke v2 OK"
