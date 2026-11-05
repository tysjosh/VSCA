#!/usr/bin/env bash
# PROBE_V3.md on the GPU VM, seeds 0-1, in order of importance:
#   1. v3_vsca2add_l0.3 (experiment 1 primary)     2. pass@8 with texts for SFT / claim mask / rft_verified / vsca2
#   3. v3_vsca2add_l0.1, v3_stepdpoadd_l0.3        4. comparisons -> runs/<model>/v3probe/
# Resumable: finished trainings / evals are skipped. Stops at the session deadline (rc 75); rerun to continue.
#   screen -dmS v3 bash ~/VSCA/vsca/scripts/run_v3_probe.sh
cd ~/VSCA/vsca || exit 1
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
export SESSION_DEADLINE=$(cat ~/VSCA/deadline) VSCA_DEADLINE=$(cat ~/VSCA/deadline)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
OUT=runs/$(basename "$MODEL")
P=$OUT/v3probe
C="--config configs/base.yaml --set model_name=$MODEL --set out_dir=$OUT"
SEEDS=${SEEDS:-0 1}
mkdir -p "$P"
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/v3_progress.log; }
check() {  # check <rc> <what>
  [ "$1" -eq 75 ] && { log "deadline during $2; rerun to continue"; exit 75; }
  [ "$1" -ne 0 ] && { log "$2 FAILED ($1)"; exit "$1"; }
  return 0
}
reused() {  # reused <arm> <seed> -> run dir
  case "$1" in
    sft) [ "$2" = 0 ] && echo "$OUT/tune_sft_lr1e-4" || echo "$OUT/gonogo_sft_s$2" ;;
    claim_mask) [ "$2" = 0 ] && echo "$OUT/tune_claim_mask_lr5e-5" || echo "$OUT/gonogo_claim_mask_s$2" ;;
    *) echo "$OUT/v2_$1_s$2" ;;
  esac
}
train_arm() {  # train_arm <name> <seed> <config...>
  local name=$1 s=$2; shift 2
  local a; a="$(reused sft $s)/adapter_best"
  [ -d "$a" ] || { log "missing $a"; exit 1; }
  local r="$OUT/${name}_s$s"
  log "train ${name}_s$s"
  "$PY" -m vsca.cli train $C "$@" --set lr=5e-5 --set seed=$s --set refresh=once --set init_adapter="$a" \
    --set onpolicy_dir="$OUT/v2_onpolicy_s$s" --set run_name=$(basename "$r") > ~/VSCA/v3_${name}_s$s.log 2>&1
  check $? "train ${name}_s$s"
  "$PY" -m vsca.cli eval $C --set limit_test=None --adapter "$r/adapter_best" --split gsm8k_test --out "$r/test.jsonl" \
    >> ~/VSCA/v3_${name}_s$s.log 2>&1
  check $? "eval ${name}_s$s"
  if [ ! -f "$r/diag_vplus_trace.json" ]; then
    "$PY" -m vsca.cli diagnose vplus_trace $C --eval_file "$r/test.jsonl" --split gsm8k_test --out_dir "$r" \
      >> ~/VSCA/v3_${name}_s$s.log 2>&1
    check $? "vplus_trace ${name}_s$s"
  fi
  log "done ${name}_s$s: $(python3 -c "import json;print(round(100*json.load(open('$r/test.summary.json'))['accuracy'],2))")"
}
passk_arm() {  # passk_arm <arm> <seed>
  local r; r=$(reused $1 $2)
  [ -d "$r/adapter_best" ] || { log "missing $r/adapter_best"; exit 1; }
  log "passk $1 s$2"
  "$PY" -m vsca.cli passk $C --set limit_test=None --set vllm_gpu_mem=0.85 --set seed=$2 --adapter "$r/adapter_best" \
    --split gsm8k_test --n 8 --temperature 0.7 --out "$r/test_pass8v.jsonl" > ~/VSCA/v3_passk_$1_s$2.log 2>&1
  check $? "passk $1 s$2"
  log "done passk $1 s$2: $(python3 -c "import json;d=json.load(open('$r/test_pass8v.summary.json'));print({k:round(v,4) for k,v in d.items() if k[:3] in ('maj','vma','pas','ste')})")"
}
compare() {  # compare <out name> <files a...> -- <files b...>
  local name=$1; shift
  local a=() b=() side=a
  for x in "$@"; do if [ "$x" = -- ]; then side=b; elif [ $side = a ]; then a+=("$x"); else b+=("$x"); fi; done
  for f in "${a[@]}" "${b[@]}"; do [ -f "$f" ] || { log "skip $name: missing $f"; return 0; }; done
  "$PY" -m vsca.cli stats compare --a "${a[@]}" --b "${b[@]}" > "$P/$name.json"
  check $? "stats $name"
  log "$name: $(python3 -c "import json;d=json.load(open('$P/$name.json'));print('diff %+.2f pts, ci95 [%+.2f, %+.2f]'%(100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
files() { local arm=$1 f=$2; for s in $SEEDS; do case $arm in v3_*) echo "$OUT/${arm}_s$s/$f" ;; *) echo "$(reused $arm $s)/$f" ;; esac; done; }

log "probe start (seeds $SEEDS)"
for s in $SEEDS; do train_arm v3_vsca2add_l0.3 $s --set method=vsca2 --set variant=full --set fork_mode=add --set fork_lambda=0.3; done
compare exp1_primary_vsca2add_l0.3_vs_rft_verified $(files v3_vsca2add_l0.3 test.jsonl) -- $(files rft_verified test.jsonl)
for s in $SEEDS; do for arm in rft_verified sft claim_mask vsca2; do passk_arm $arm $s; done; done
for arm in rft_verified sft claim_mask vsca2; do
  for f in $(files $arm test_pass8v.jsonl); do "$PY" scripts/vote_files.py "$f" >/dev/null; done
  compare exp2_vmaj_vs_maj_$arm $(files $arm test_pass8v.vmaj.jsonl) -- $(files $arm test_pass8v.maj.jsonl)
done
for s in $SEEDS; do train_arm v3_vsca2add_l0.1 $s --set method=vsca2 --set variant=full --set fork_mode=add --set fork_lambda=0.1; done
for s in $SEEDS; do train_arm v3_stepdpoadd_l0.3 $s --set method=stepdpo_fork --set fork_mode=add --set fork_lambda=0.3; done
compare exp1_vsca2add_l0.1_vs_rft_verified $(files v3_vsca2add_l0.1 test.jsonl) -- $(files rft_verified test.jsonl)
compare exp1_stepdpoadd_l0.3_vs_rft_verified $(files v3_stepdpoadd_l0.3 test.jsonl) -- $(files rft_verified test.jsonl)
compare exp1_vsca2add_l0.3_vs_stepdpoadd_l0.3 $(files v3_vsca2add_l0.3 test.jsonl) -- $(files v3_stepdpoadd_l0.3 test.jsonl)
log "probe done"
cd ~/VSCA && .venv/bin/python hub_sync.py push >> ~/VSCA/hubsync.log 2>&1; log pushed
