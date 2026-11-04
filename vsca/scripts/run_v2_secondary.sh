#!/usr/bin/env bash
# VSCA v2 pilot, secondary arms on the seeds whose grounded pipelines exist (default 0 1), to find out whether
# vsca2's deficit to rft_verified comes from the fork negative (stepdpo_fork) or from the certified sets
# (vsca2_uncertified); then the no-grounding pipeline and arm for seed 0. Greedy test + V+ trace only (SVAMP=0);
# run_v2_seedwise.sh adds the rest later. Resumable; stops at the session deadline (rc 75).
#   screen -dmS v2 bash ~/VSCA/vsca/scripts/run_v2_secondary.sh
cd ~/VSCA/vsca || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline) VSCA_DEADLINE=$(cat ~/VSCA/deadline)
log() { echo "$(date +%H:%M) $*" >> ~/VSCA/v2_progress.log; }
step() {  # step <log name> [VAR=value ...] <pilot_v2 stage>
  local name=$1; shift
  local stage=${!#}
  env "${@:1:$#-1}" bash scripts/pilot_v2.sh "$stage" > ~/VSCA/v2_$name.log 2>&1; rc=$?; log "$name rc=$rc"
  return $rc
}
run() {
  log "secondary start"
  step train_secondary SEEDS="${SEC_SEEDS:-0 1}" ARMS="stepdpo_fork vsca2_uncertified" SVAMP=0 EVAL_REUSED=0 train || return
  step onpolicy_nogr_s0 SEEDS=0 onpolicy || return
  step train_nogr_s0 SEEDS=0 ARMS=vsca2_nogrounding SVAMP=0 EVAL_REUSED=0 train || return
}
run
cd ~/VSCA && .venv/bin/python hub_sync.py push >> ~/VSCA/hubsync.log 2>&1; log pushed
