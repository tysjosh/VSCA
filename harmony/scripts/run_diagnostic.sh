#!/usr/bin/env bash
# DIAGNOSTIC.md on the GPU VM: data -> SFT (seed 0) -> greedy + 8 samples per test problem -> diagnostic + go rule.
# Resumable: every step skips finished outputs; SFT checkpoints and stops at the session deadline (rc 75).
#   screen -dmS harmony bash ~/VSCA/harmony/scripts/run_diagnostic.sh
cd ~/VSCA/harmony || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/harmony_progress.log; }
R=runs/sft_s0
HUB="--runs harmony/runs --repo olukotunjosh/harmony-runs"
log start
# restore earlier progress (a new VM starts empty); data is regenerated deterministically from seed 0
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || log "nothing to restore"
[ -f data/train.jsonl ] || ../.venv/bin/python scripts/make_data.py --out data > ~/VSCA/harmony_data.log 2>&1 || { log "data failed"; exit 1; }
log "data ready"
../.venv/bin/python scripts/sft.py --data data --out $R --seed 0 > ~/VSCA/harmony_sft.log 2>&1
rc=$?
if [ $rc -eq 75 ]; then
  log "deadline during SFT; pushing the checkpoint, rerun to continue"
  (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1)
  exit 75
fi
[ $rc -ne 0 ] && { log "SFT failed ($rc)"; exit $rc; }
log "sft done"
for s in test_id test_long test_minor test_keys; do
  ../.venv-vllm/bin/python scripts/sample.py --split data/$s.jsonl --adapter $R/adapter --out $R/$s.jsonl \
    > ~/VSCA/harmony_sample_$s.log 2>&1 || { log "sample $s failed"; exit 1; }
  log "sampled $s"
done
../.venv/bin/python scripts/diagnose.py --data data --samples $R --out $R/diagnostic.json > ~/VSCA/harmony_diag.log 2>&1 \
  || { log "diagnose failed"; exit 1; }
log "diagnostic: $(tail -1 ~/VSCA/harmony_diag.log)"
grep -E "^test_" ~/VSCA/harmony_diag.log | tee -a ~/VSCA/harmony_progress.log
cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1; log pushed
