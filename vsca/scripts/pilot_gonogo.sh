#!/usr/bin/env bash
# Go/no-go pilot (vsca/PILOT_GONOGO.md): 5 seeds of VSCA (fixed base certificates), SFT and claim-span masking at
# their tuned learning rates; greedy test accuracy, then pass@8 / maj@8 with vLLM; paired comparisons.
# Resumable: rerun the same command; finished trainings and evaluations are skipped.
#   bash scripts/pilot_gonogo.sh [train|passk|stats|all]   (default all)
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
OUT=runs/$(basename "$MODEL")
C="--config configs/base.yaml --set model_name=$MODEL --set out_dir=$OUT"
METHODS="sft vsca claim_mask"
SEEDS="0 1 2 3 4"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$OUT/best_lr.json" ] || { echo "missing $OUT/best_lr.json: finish the tune stage first"; exit 1; }
lr_of() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$OUT/best_lr.json" "$1"; }
run_dir() {  # seed 0 reuses the tune run at the chosen learning rate
  if [ "$2" = 0 ]; then echo "$OUT/tune_$1_lr$(lr_of $1)"; else echo "$OUT/gonogo_$1_s$2"; fi
}
extra() { [ "$1" = vsca ] && echo "--set cert_path=$OUT/cert_base.jsonl"; }

stop_if_deadline() { [ "$1" -eq 75 ] && { echo "[pilot] session deadline reached; rerun to continue"; exit 75; }; }

train_and_eval() {
  for s in $SEEDS; do for m in $METHODS; do
    r=$(run_dir $m $s)
    if [ "$s" != 0 ]; then
      "$PY" -m vsca.cli train $C --set method=$m --set lr=$(lr_of $m) --set seed=$s --set run_name=$(basename $r) $(extra $m)
      rc=$?; stop_if_deadline $rc; [ $rc -ne 0 ] && { echo "[pilot] train $m s$s failed ($rc)"; exit $rc; }
    fi
    "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$r/adapter_best" --split gsm8k_test --out "$r/test.jsonl"
    rc=$?; stop_if_deadline $rc; [ $rc -ne 0 ] && { echo "[pilot] eval $m s$s failed ($rc)"; exit $rc; }
  done; done
}

passk() {
  for s in $SEEDS; do for m in $METHODS; do
    r=$(run_dir $m $s)
    "$PY" -m vsca.cli passk $C --set limit_test=None --set vllm_gpu_mem=0.85 --set seed=$s --adapter "$r/adapter_best" \
      --split gsm8k_test --n 8 --temperature 0.7 --out "$r/test_pass8.jsonl"
    rc=$?; stop_if_deadline $rc; [ $rc -ne 0 ] && { echo "[pilot] passk $m s$s failed ($rc)"; exit $rc; }
  done; done
}

stats() {
  files() { for s in $SEEDS; do echo "$(run_dir $1 $s)/$2"; done; }
  mkdir -p "$OUT/gonogo"
  for f in test.jsonl test_pass8.jsonl; do
    for b in sft claim_mask; do
      [ -f "$(run_dir vsca 4)/$f" ] || continue
      "$PY" -m vsca.cli stats compare --a $(files vsca $f) --b $(files $b $f) > "$OUT/gonogo/vsca_vs_${b}_${f%.jsonl}.json"
      echo "== VSCA vs $b ($f)"; cat "$OUT/gonogo/vsca_vs_${b}_${f%.jsonl}.json"
    done
  done
}

case "${1:-all}" in
  train) train_and_eval ;;
  passk) passk ;;
  stats) stats ;;
  all) train_and_eval && passk && stats ;;
  *) echo "usage: $0 [train|passk|stats|all]"; exit 1 ;;
esac
