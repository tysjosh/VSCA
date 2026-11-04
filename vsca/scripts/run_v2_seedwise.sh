#!/usr/bin/env bash
# VSCA v2 pilot on the GPU VM, decision path first (ungated: diag runs with FORCE=1, recorded in gate.json).
#   Phase 1, per seed: grounded pipeline, then vsca2 + rft_verified (train + greedy test + V+ trace);
#            then `stats` writes v2pilot/decision.json (the pre-registered rule needs only these arms).
#   Phase 2, per seed: no-grounding pipeline and the other three arms, SVAMP and the reused runs' evals; then
#            pass@8 and `stats` again with every secondary comparison.
# Every step skips finished work, so rerunning resumes. Stops at the session deadline (rc 75) or on failure.
# Progress: ~/VSCA/v2_progress.log. Ends with a Hugging Face push.
#   screen -dmS v2 bash ~/VSCA/vsca/scripts/run_v2_seedwise.sh
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
  log start
  step diag FORCE=1 diag || return
  for s in ${SEED_LIST:-0 1 2}; do
    step onpolicy_s$s SEEDS=$s NOGR=0 onpolicy || return
    step train_s$s SEEDS=$s ARMS="vsca2 rft_verified" SVAMP=0 EVAL_REUSED=0 train || return
  done
  step stats_decision stats || return
  log "decision ready: runs/Qwen2.5-1.5B/v2pilot/decision.json"
  for s in ${SEED_LIST:-0 1 2}; do
    step onpolicy_full_s$s SEEDS=$s onpolicy || return
    step train_full_s$s SEEDS=$s train || return
  done
  step passk passk || return
  step stats_full stats || return
}
run
cd ~/VSCA && .venv/bin/python hub_sync.py push >> ~/VSCA/hubsync.log 2>&1; log pushed
