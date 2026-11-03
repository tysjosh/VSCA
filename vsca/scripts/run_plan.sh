#!/usr/bin/env bash
# Full protocol of Section 4 on a remote GPU machine. Run stages in order:
#   ./scripts/run_plan.sh <stage>      stage in: prepare diag pilot scores tune main controls robust eval stats
# Environment:
#   MODEL   (default Qwen/Qwen2.5-1.5B; also run with meta-llama/Llama-3.2-1B, which needs HF access)
#   SEEDS   seeds for confirmatory runs (default "0 1 2 3 4"; raise to the number `stats power` reports)
#   OUT     output root (default runs)
#   SESSION_HOURS  length of the machine session (default 10)
#   MARGIN_MIN     stop this many minutes before the session ends (default 20)
#   VSCA_DEADLINE  unix time to stop; set once per session (see README "Sessions and resume")
# Every stage is resumable: rerun the same stage (or `all`) in the next session and finished work is skipped.
# A run that reaches the deadline saves its state and exits with code 75.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
TAG=$(basename "$MODEL")
SEEDS=${SEEDS:-"0 1 2 3 4"}
OUT=${OUT:-runs}/$TAG
C="--config configs/base.yaml --set model_name=$MODEL --set out_dir=$OUT"
LRS="5e-5 1e-4 2e-4 4e-4"
mkdir -p "$OUT"
SESSION_HOURS=${SESSION_HOURS:-10}
MARGIN_MIN=${MARGIN_MIN:-20}
if [ -z "${VSCA_DEADLINE:-}" ]; then
  # Fallback: count the session from now. Prefer `source scripts/session_start.sh` at login.
  VSCA_DEADLINE=$(( $(date +%s) + SESSION_HOURS * 3600 - MARGIN_MIN * 60 ))
fi
export VSCA_DEADLINE
echo "[session] stopping cleanly at $(date -d @"$VSCA_DEADLINE" 2>/dev/null || date -r "$VSCA_DEADLINE")"
trap 'rc=$?; if [ $rc -eq 75 ]; then echo "[session] deadline reached; rerun the same command in the next session"; fi' EXIT

lr_of() {  # best validation LR for a method, written by the `tune` stage
  "$PY" - "$OUT/best_lr.json" "$1" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], 2e-4))
EOF
}

method_args() {  # extra --set flags each method needs
  case "$1" in
    rho)            echo "--set ref_scores=$OUT/scores_ref.json" ;;
    token_cleaning) echo "--set ref_scores=$OUT/scores_ref.json --set base_scores=$OUT/scores_base.json" ;;
    sstoken)        echo "--set base_scores=$OUT/scores_base.json" ;;
    cf_select)      echo "--set ref_scores=$OUT/scores_ref.json" ;;
    unlikelihood)   echo "--set cert_path=$OUT/cert_base.jsonl" ;;
    rft)            echo "--set rft_from=$OUT/sft_rollouts_cert.jsonl" ;;
    *)              echo "" ;;
  esac
}

case "${1:-}" in
prepare)
  "$PY" -m vsca.cli prepare $C
  ;;

diag)  # Section 4.2: pre-registered diagnostics, before any comparative training
  "$PY" -m vsca.cli eval $C --set limit_test=None --split gsm8k_val --out "$OUT/diag/base_val.jsonl"
  "$PY" -m vsca.cli diagnose taxonomy $C --eval_file "$OUT/diag/base_val.jsonl" --out_dir "$OUT/diag"
  "$PY" -m vsca.cli certify $C --set limit_train=1000 --out "$OUT/diag/cert_base_1k.jsonl"
  "$PY" -m vsca.cli diagnose mass $C --cert "$OUT/diag/cert_base_1k.jsonl" --out_dir "$OUT/diag"
  "$PY" -m vsca.cli diagnose audit $C --set limit_train=1000 --cert "$OUT/diag/cert_base_1k.jsonl" --out_dir "$OUT/diag"
  "$PY" -m vsca.cli diagnose earlier_score $C --with_model --out_dir "$OUT/diag"
  echo "Label $OUT/diag/error_taxonomy.csv and $OUT/diag/certificate_audit.csv by hand before continuing."
  ;;

pilot)  # five SFT seeds -> seed variance -> number of seeds for confirmatory runs
  for s in 0 1 2 3 4; do
    "$PY" -m vsca.cli train $C --set method=sft --set seed=$s --set run_name=pilot_sft_s$s
    "$PY" -m vsca.cli eval $C --adapter "$OUT/pilot_sft_s$s/adapter_best" --split gsm8k_test --out "$OUT/pilot_sft_s$s/test.jsonl"
  done
  "$PY" -m vsca.cli stats power --a "$OUT"/pilot_sft_s*/test.jsonl --mde 0.015 | tee "$OUT/power.json"
  ;;

scores)  # reference model = SFT on the same split (its cost counts toward rho/token_cleaning/cf_select)
  "$PY" -m vsca.cli score $C --adapter "$OUT/pilot_sft_s0/adapter_best" --out "$OUT/scores_ref.json"
  "$PY" -m vsca.cli score $C --attention --out "$OUT/scores_base.json"
  "$PY" -m vsca.cli certify $C --out "$OUT/cert_base.jsonl"  # base-model candidates for unlikelihood
  ;;

tune)  # learning rate per method on the validation split (test never used)
  # VSCA last: its runs certify (expensive), so the cheap methods finish first.
  for m in sft random rho token_cleaning sstoken trimsft claim_mask claim_upweight unlikelihood cf_select vsca; do
    for lr in $LRS; do
      "$PY" -m vsca.cli train $C --set method=$m --set lr=$lr --set seed=0 --set run_name=tune_${m}_lr$lr $(method_args $m)
    done
  done
  "$PY" - "$OUT" <<'EOF'
import json, sys, glob, re
best = {}
for f in glob.glob(f"{sys.argv[1]}/tune_*_lr*/train_summary.json"):
    m, lr = re.search(r"tune_(.+)_lr([^/]+)/", f).groups()
    acc = json.load(open(f))["best_val_accuracy"]
    if acc > best.get(m, (-1, None))[0]:
        best[m] = (acc, float(lr))
json.dump({m: lr for m, (a, lr) in best.items()}, open(f"{sys.argv[1]}/best_lr.json", "w"), indent=2)
print(best)
EOF
  ;;

main)  # confirmatory comparisons (Section 4.6): VSCA vs SFT, vs claim-span upweighting, vs RFT
  for s in $SEEDS; do
    "$PY" -m vsca.cli train $C --set method=vsca --set lr=$(lr_of vsca) --set seed=$s --set run_name=vsca_s$s
    "$PY" -m vsca.cli train $C --set method=sft --set lr=$(lr_of sft) --set seed=$s --set run_name=sft_s$s
    "$PY" -m vsca.cli train $C --set method=claim_upweight --set lr=$(lr_of claim_upweight) --set seed=$s --set run_name=claim_upweight_s$s
    # RFT on the verified rollouts VSCA generated in the same seed (same rollout budget)
    cat "$OUT"/vsca_s$s/cert_epoch*.jsonl > "$OUT/sft_rollouts_cert.jsonl"
    "$PY" -m vsca.cli train $C --set method=rft --set lr=$(lr_of sft) --set seed=$s --set run_name=rft_s$s $(method_args rft)
    # compute-matched SFT: extra epochs so that GPU-hours match vsca_s$s (see README)
    "$PY" -m vsca.cli train $C --set method=sft --set lr=$(lr_of sft) --set seed=$s --set epochs=${SFT_MATCHED_EPOCHS:-6} --set select_best_on_val=true --set run_name=sft_matched_s$s
  done
  ;;

controls)  # baselines and VSCA controls (exploratory, Section 4.4)
  for s in $SEEDS; do
    for m in random rho token_cleaning sstoken trimsft claim_mask unlikelihood cf_select; do
      "$PY" -m vsca.cli train $C --set method=$m --set lr=$(lr_of $m) --set seed=$s --set run_name=${m}_s$s $(method_args $m)
    done
    for v in claim_only decision_only uncertified shuffled; do
      "$PY" -m vsca.cli train $C --set method=vsca --set variant=$v --set lr=$(lr_of vsca) --set seed=$s \
        --set cert_path="$OUT/vsca_s$s/cert_epoch0.jsonl" --set run_name=vsca_${v}_s$s
    done
  done
  ;;

robust)  # RQ4: false-certificate injection and certificate dropping
  for s in $SEEDS; do
    for eta in 0.05 0.10 0.20; do
      "$PY" -m vsca.cli train $C --set method=vsca --set lr=$(lr_of vsca) --set seed=$s --set inject_eta=$eta \
        --set cert_path="$OUT/vsca_s$s/cert_epoch0.jsonl" --set run_name=vsca_inject${eta}_s$s
      "$PY" -m vsca.cli train $C --set method=vsca --set lr=$(lr_of vsca) --set seed=$s --set drop_eta=$eta \
        --set cert_path="$OUT/vsca_s$s/cert_epoch0.jsonl" --set run_name=vsca_drop${eta}_s$s
    done
  done
  ;;

eval)  # GSM8K test (once) and SVAMP transfer for every finished run
  for d in "$OUT"/*/adapter_best; do
    r=$(dirname "$d")
    case "$(basename "$r")" in tune_*|pilot_*) continue ;; esac
    "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$d" --split gsm8k_test --out "$r/test.jsonl"
    "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$d" --split svamp_all --out "$r/svamp.jsonl"
  done
  ;;

stats)  # paired bootstrap over items and seeds; Holm over the confirmatory comparisons
  for b in sft claim_upweight rft sft_matched; do
    "$PY" -m vsca.cli stats compare --a "$OUT"/vsca_s*/test.jsonl --b "$OUT"/${b}_s*/test.jsonl | tee "$OUT/compare_vsca_vs_$b.json"
  done
  "$PY" - "$OUT" <<'EOF' > "$OUT/confirmatory_p.json"
import json, sys
o = sys.argv[1]
print(json.dumps({b: json.load(open(f"{o}/compare_vsca_vs_{b}.json"))["p_two_sided"] for b in ("sft", "claim_upweight", "rft")}))
EOF
  "$PY" -m vsca.cli stats holm --pvals "$OUT/confirmatory_p.json" | tee "$OUT/confirmatory_holm.json"
  ;;

all)  # every stage in order; rerun after each session until it prints "all stages done"
  for st in prepare diag pilot scores tune main controls robust eval stats; do
    echo "=== stage $st"
    "$0" "$st"
  done
  echo "all stages done"
  ;;

*)
  echo "usage: $0 {prepare|diag|pilot|scores|tune|main|controls|robust|eval|stats|all}"; exit 1 ;;
esac
