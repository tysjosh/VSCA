#!/usr/bin/env bash
# VSCA v2 pilot, seed by seed on the GPU VM: diag (FORCE=1 -> ungated, recorded in gate.json), then for each seed the
# on-policy pipelines followed by training + evals of the five arms. Stops at the session deadline (rc 75) or on
# failure; rerun to resume. Progress: ~/VSCA/v2_progress.log. Ends with a Hugging Face push.
#   screen -dmS v2 bash ~/VSCA/vsca/scripts/run_v2_seedwise.sh
cd ~/VSCA/vsca || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline) VSCA_DEADLINE=$(cat ~/VSCA/deadline)
log() { echo "$(date +%H:%M) $*" >> ~/VSCA/v2_progress.log; }
log start
FORCE=1 bash scripts/pilot_v2.sh diag > ~/VSCA/v2_diag.log 2>&1; rc=$?; log "diag rc=$rc"
if [ $rc -eq 0 ]; then
  for s in ${SEED_LIST:-0 1 2}; do
    SEEDS=$s bash scripts/pilot_v2.sh onpolicy > ~/VSCA/v2_onpolicy_s$s.log 2>&1; rc=$?; log "onpolicy s$s rc=$rc"
    [ $rc -ne 0 ] && break
    SEEDS=$s bash scripts/pilot_v2.sh train > ~/VSCA/v2_train_s$s.log 2>&1; rc=$?; log "train s$s rc=$rc"
    [ $rc -ne 0 ] && break
  done
fi
cd ~/VSCA && .venv/bin/python hub_sync.py push >> ~/VSCA/hubsync.log 2>&1; log pushed
