#!/usr/bin/env bash
# PROBE_V3.md experiment 3b on the VM: 8 more chain-of-thought samples per test problem for rft_verified seeds 0-1,
# then maj@16 vs the program check. Log: ~/VSCA/v3_progress.log.   screen -dmS v3b bash scripts/run_v3b.sh
cd ~/VSCA/vsca || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline)
R=runs/Qwen2.5-1.5B
C="--config configs/base.yaml --set out_dir=$R"
log() { echo "$(date +%H:%M) $*" >> ~/VSCA/v3_progress.log; }
log "3b start"
for s in 0 1; do
  PYTHONPATH=src ../.venv-vllm/bin/python -m vsca.cli passk $C --set limit_test=None --set vllm_gpu_mem=0.85 \
    --set seed=$((s + 100)) --adapter $R/v2_rft_verified_s$s/adapter_best --split gsm8k_test --n 8 --temperature 0.7 \
    --out $R/v2_rft_verified_s$s/test_pass8v_b.jsonl > ~/VSCA/v3_passk_b_s$s.log 2>&1 || { log "3b passk s$s failed"; exit 1; }
  log "3b passk s$s done"
done
../.venv/bin/python scripts/merge16.py $R/v2_rft_verified_s0 $R/v2_rft_verified_s1 >> ~/VSCA/v3_progress.log 2>&1
A=$R/v2_rft_verified_s0
B=$R/v2_rft_verified_s1
cmp() {
  PYTHONPATH=src ../.venv/bin/python -m vsca.cli stats compare --a $1 $2 --b $3 $4 > $R/v3probe/$5.json || { log "$5 failed"; return; }
  log "$5: $(python3 -c "import json;d=json.load(open('$R/v3probe/$5.json'));print('%+.2f [%+.2f, %+.2f]'%(100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
cmp $A/test_pass8v.pmaj.jsonl $B/test_pass8v.pmaj.jsonl $A/test16.maj.jsonl $B/test16.maj.jsonl exp3b_pmaj8_vs_maj16_rft
cmp $A/test16.pmaj.jsonl $B/test16.pmaj.jsonl $A/test16.maj.jsonl $B/test16.maj.jsonl exp3b_pmaj16_vs_maj16_rft
cmp $A/test16.maj.jsonl $B/test16.maj.jsonl $A/test_pass8v.maj.jsonl $B/test_pass8v.maj.jsonl exp3b_maj16_vs_maj8_rft
log "3b done"
cd ~/VSCA && .venv/bin/python hub_sync.py push >> ~/VSCA/hubsync.log 2>&1
log "3b pushed"
