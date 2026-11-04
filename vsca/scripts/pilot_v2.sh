#!/usr/bin/env bash
# VSCA v2 pilot (vsca/PILOT_V2.md): the two V+ diagnostics with the 0.40 grounding gate, one shared on-policy
# pipeline dir per seed (plus a no-grounding dir from the same samples), the five v2 arms x seeds 0-2 at lr 5e-5
# from the SFT adapter of the same seed, greedy GSM8K test + SVAMP + V+ trace evaluations, pass@8 with vLLM, and
# paired comparisons with decision.json. SFT and claim-span masking are reused from the go/no-go pilot, never
# retrained. Resumable: rerun the same command; finished trainings, pipelines and evaluations are skipped.
#   bash scripts/pilot_v2.sh [diag|onpolicy|train|passk|stats|all]   (default all; FORCE=1 overrides the gate)
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
OUT=runs/$(basename "$MODEL")
P=$OUT/v2pilot
C="--config configs/base.yaml --set model_name=$MODEL --set out_dir=$OUT"
V2_LR=5e-5
# Overrides for running the decision path first (run_v2_seedwise.sh): ARMS="vsca2 rft_verified" NOGR=0 SVAMP=0 EVAL_REUSED=0
ARMS=${ARMS:-rft_verified stepdpo_fork vsca2 vsca2_uncertified vsca2_nogrounding}
NOGR=${NOGR:-1}         # 0: skip the no-grounding pipeline dirs (only vsca2_nogrounding needs them)
SVAMP=${SVAMP:-1}       # 0: skip the SVAMP eval (secondary); a later run with SVAMP=1 adds it
EVAL_REUSED=${EVAL_REUSED:-1}  # 0: skip the extra evals of the reused SFT / claim-span masking runs (their test.jsonl exists)
REUSED="sft claim_mask"
SEEDS=${SEEDS:-0 1 2}  # e.g. SEEDS=0 to build and train one seed at a time; stats always needs all three
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
usage() { echo "usage: $0 [diag|onpolicy|train|passk|stats|all]"; exit 1; }
case "${1:-all}" in diag|onpolicy|train|passk|stats|all) ;; *) usage ;; esac
[ -f "$OUT/best_lr.json" ] || { echo "missing $OUT/best_lr.json: finish the tune stage first"; exit 1; }
lr_tag() {  # the learning rate as spelled in tune run names (best_lr.json stores floats, e.g. 0.0001 -> 1e-4)
  "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(next(s for s in sys.argv[3].split() if float(s)==v))' \
    "$OUT/best_lr.json" "$1" "5e-5 1e-4 2e-4 4e-4"
}
run_dir() {  # reused v1 runs: seed 0 is the tune run at the chosen learning rate (as pilot_gonogo.sh); v2: v2_<arm>_s<seed>
  case "$1" in
    sft|claim_mask) if [ "$2" = 0 ]; then echo "$OUT/tune_$1_lr$(lr_tag $1)"; else echo "$OUT/gonogo_$1_s$2"; fi ;;
    *) echo "$OUT/v2_$1_s$2" ;;
  esac
}
op_dir() { if [ "$1" = vsca2_nogrounding ]; then echo "$OUT/v2_onpolicy_nogr_s$2"; else echo "$OUT/v2_onpolicy_s$2"; fi; }
arm_args() {  # Decision 8: arm -> config
  case "$1" in
    rft_verified) echo "--set method=rft_verified" ;;
    stepdpo_fork) echo "--set method=stepdpo_fork" ;;
    vsca2) echo "--set method=vsca2 --set variant=full" ;;
    vsca2_uncertified) echo "--set method=vsca2 --set variant=uncertified" ;;
    vsca2_nogrounding) echo "--set method=vsca2 --set vplus_grounding=false" ;;
  esac
}
stop_if_deadline() { [ "$1" -eq 75 ] && { echo "[pilot_v2] session deadline reached; rerun to continue"; exit 75; }; }
check() {  # check <rc> <what>: stop at the deadline or on failure
  stop_if_deadline "$1"; [ "$1" -ne 0 ] && { echo "[pilot_v2] $2 failed ($1)"; exit "$1"; }; return 0
}
sft_adapter() {  # the init adapter of seed $1 (Decision 1); it must exist, it is never retrained here
  local a; a="$(run_dir sft $1)/adapter_best"
  [ -d "$a" ] || { echo "[pilot_v2] missing $a: finish the go/no-go pilot's SFT run for seed $1 first" >&2; exit 1; }
  echo "$a"
}
diag() {
  mkdir -p "$P/diag"
  local a; a=$(sft_adapter 0) || exit 1
  "$PY" -m vsca.cli diagnose grounding $C --out_dir "$P/diag"
  check $? "diagnose grounding"
  "$PY" -m vsca.cli diagnose recoverability $C --set seed=0 --set limit_train=500 --adapter "$a" --out_dir "$P/diag"
  check $? "diagnose recoverability"
  # Decision 11: the gate is evaluated on manual labels. It fails closed: an unlabelled CSV stops the pilot like a
  # failed gate, and FORCE=1 (to be reported as an ungated run) overrides both. The outcome is written to gate.json.
  "$PY" - "$P/diag/diag_grounding.json" "${FORCE:-0}" "$P/diag/gate.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
force = sys.argv[2] == "1"
def record(status):
    json.dump({"status": status, "force": force, "grouped_by": d["grouped_by"],
               "wrong_quantity_flag_share": d.get("wrong_quantity_flag_share")}, open(sys.argv[3], "w"), indent=2)
if "gate_0.40_passed" not in d:
    print(f"[pilot_v2] {d['csv']} has no manual labels (grouped by {d['grouped_by']}); the 0.40 grounding gate "
          "cannot be evaluated. Per-group shares:")
    for g, v in d["groups"].items():
        print(f"  {g}: n={v['n']} flagged={v['flagged_share']:.3f} grounding={v['grounding_share']:.3f}")
    if force:
        record("ungated_force")
        print("[pilot_v2] continuing UNGATED because FORCE=1; report this run as ungated")
        sys.exit(0)
    record("unlabelled_stopped")
    print("[pilot_v2] stopping: fill manual_label in the CSV and rerun, or rerun with FORCE=1 for an ungated pilot")
    sys.exit(2)
share = d["wrong_quantity_flag_share"]
print(f"[pilot_v2] wrong_quantity_or_operation flagged by V+: {share}")
if d["gate_0.40_passed"]:
    record("passed")
    print("[pilot_v2] grounding gate (>= 0.40) passed")
elif force:
    record("failed_force")
    print("[pilot_v2] grounding gate (>= 0.40) FAILED; continuing because FORCE=1")
else:
    record("failed_stopped")
    print("[pilot_v2] grounding gate (>= 0.40) FAILED: build a better verifier first (FORCE=1 overrides)")
    sys.exit(2)
EOF
  check $? "grounding gate"
}
onpolicy() {  # Decision 2: one pipeline dir per seed; the no-grounding dir reuses a copy of the same samples
  for s in $SEEDS; do
    local a; a=$(sft_adapter $s) || exit 1
    "$PY" -m vsca.cli onpolicy $C --set seed=$s --set init_adapter="$a" --out "$OUT/v2_onpolicy_s$s"
    check $? "onpolicy s$s"
    if [ "$NOGR" = 1 ]; then
      "$PY" -m vsca.cli onpolicy $C --set seed=$s --set init_adapter="$a" --set vplus_grounding=false \
        --set onpolicy_samples="$OUT/v2_onpolicy_s$s/samples.jsonl" --out "$OUT/v2_onpolicy_nogr_s$s"
      check $? "onpolicy nogrounding s$s"
    fi
  done
}
evals() {  # evals <run dir> <tag>: greedy test + SVAMP, then the V+ trace diagnostic (skips existing outputs)
  "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$1/adapter_best" --split gsm8k_test --out "$1/test.jsonl"
  check $? "eval test $2"
  if [ "$SVAMP" = 1 ]; then
    "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$1/adapter_best" --split svamp_all --out "$1/svamp.jsonl"
    check $? "eval svamp $2"
  fi
  if [ ! -f "$1/diag_vplus_trace.json" ]; then
    "$PY" -m vsca.cli diagnose vplus_trace $C --eval_file "$1/test.jsonl" --split gsm8k_test --out_dir "$1"
    check $? "vplus_trace $2"
  fi
}
train_and_eval() {
  for s in $SEEDS; do
    local a; a=$(sft_adapter $s) || exit 1
    for m in $ARMS; do
      r=$(run_dir $m $s)
      "$PY" -m vsca.cli train $C $(arm_args $m) --set lr=$V2_LR --set seed=$s --set refresh=once \
        --set init_adapter="$a" --set onpolicy_dir="$(op_dir $m $s)" --set run_name=$(basename "$r")
      check $? "train $m s$s"
      evals "$r" "$m s$s"
    done
    [ "$EVAL_REUSED" = 1 ] || continue
    for m in $REUSED; do
      r=$(run_dir $m $s)
      [ -d "$r/adapter_best" ] || { echo "[pilot_v2] missing $r/adapter_best: reused $m runs are not retrained"; exit 1; }
      evals "$r" "$m s$s"
    done
  done
}
passk() {
  for s in $SEEDS; do for m in $ARMS $REUSED; do
    r=$(run_dir $m $s)
    "$PY" -m vsca.cli passk $C --set limit_test=None --set vllm_gpu_mem=0.85 --set seed=$s --adapter "$r/adapter_best" \
      --split gsm8k_test --n 8 --temperature 0.7 --out "$r/test_pass8.jsonl"
    check $? "passk $m s$s"
  done; done
}
stats() {
  files() { for s in $SEEDS; do echo "$(run_dir $1 $s)/$2"; done; }
  mkdir -p "$P"
  for f in test.jsonl svamp.jsonl test_pass8.jsonl; do
    for b in rft_verified stepdpo_fork vsca2_uncertified vsca2_nogrounding $REUSED; do
      ok=1; for x in $(files vsca2 $f) $(files $b $f); do [ -f "$x" ] || ok=0; done
      [ $ok = 1 ] || { echo "[pilot_v2] skip vsca2 vs $b ($f): missing files"; continue; }
      "$PY" -m vsca.cli stats compare --a $(files vsca2 $f) --b $(files $b $f) > "$P/vsca2_vs_${b}_${f%.jsonl}.json"
      check $? "stats vsca2 vs $b ($f)"
      echo "== vsca2 vs $b ($f)"; cat "$P/vsca2_vs_${b}_${f%.jsonl}.json"
    done
  done
  vplus=""; for m in $ARMS $REUSED; do vplus="$vplus $m=$(files $m diag_vplus_trace.json | tr '\n' ',')"; done
  "$PY" - "$P" $vplus <<'EOF'
# Pre-registered rule (PILOT_V2.md), on greedy GSM8K test, observed paired-bootstrap difference in points
import json, sys
from pathlib import Path
P = Path(sys.argv[1])
def cmp(b):
    p = P / f"vsca2_vs_{b}_test.json"
    return json.loads(p.read_text()) if p.exists() else None
rft, cm = cmp("rft_verified"), cmp("claim_mask")
if rft is None or cm is None:
    sys.exit("[pilot_v2] decision needs vsca2_vs_rft_verified_test.json and vsca2_vs_claim_mask_test.json")
d = {"rft_verified": 100 * rft["diff"], "claim_mask": 100 * cm["diff"]}
if min(d.values()) >= 1.0:
    decision = "go"
elif min(d.values()) <= 0.5:
    decision = "no_go"
else:
    decision = "discuss"
vplus = {}
for kv in sys.argv[2:]:
    arm, paths = kv.split("=", 1)
    vals = [json.loads(Path(p).read_text()).get("vplus_verified_share") for p in paths.split(",") if p and Path(p).exists()]
    vplus[arm] = vals
out = {
    "rule": "vsca2 - b >= 1.0 point for both b in {rft_verified, claim_mask} -> go; <= 0.5 for either -> no_go; "
            "else discuss (greedy GSM8K test, paired bootstrap over items and seeds 0-2)",
    "diff_points": d,
    "ci95_points": {"rft_verified": [100 * x for x in rft["ci95"]], "claim_mask": [100 * x for x in cm["ci95"]]},
    "decision": decision,
    "filter_explains_gain": d["rft_verified"] <= 0.5,
    "secondary_files": sorted(p.name for p in P.glob("vsca2_vs_*.json")),
    "vplus_verified_share_per_seed": vplus,
    # the 0.40 grounding gate outcome from `diag` (status ungated_force / failed_force must be reported as such)
    "grounding_gate": json.loads((P / "diag" / "gate.json").read_text()) if (P / "diag" / "gate.json").exists() else None,
}
(P / "decision.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF
  check $? "decision"
}
case "${1:-all}" in
  diag) diag ;;
  onpolicy) onpolicy ;;
  train) train_and_eval ;;
  passk) passk ;;
  stats) stats ;;
  all) diag && onpolicy && train_and_eval && passk && stats ;;
esac
