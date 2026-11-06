#!/usr/bin/env bash
# PILOT.md on the GPU VM, seeds 0-2: entries-only data -> SFT -> on-policy samples -> pools -> learnability gate ->
# 3 arms -> greedy eval -> score -> comparisons. Resumable (finished steps are skipped; training checkpoints and
# stops at the deadline, rc 75). Backed up to olukotunjosh/bookkeeping-runs.
#   screen -dmS bpilot bash ~/VSCA/bookkeeping/scripts/run_pilot.sh
cd ~/VSCA/bookkeeping || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
PY=../.venv/bin/python
VPY=../.venv-vllm/bin/python
H=../harmony/scripts
D=data_p
HUB="--runs bookkeeping/runs --repo olukotunjosh/bookkeeping-runs"
ARMS="rft single set"
SPLITS="test_id test_L3 test_ind test_year"
SEEDS=${SEEDS:-0 1 2}
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/bpilot_progress.log; }
push() { (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1); }
chk() {
  [ "$1" -eq 75 ] && { log "deadline during $2; pushing, rerun to continue"; push; exit 75; }
  [ "$1" -ne 0 ] && { log "$2 FAILED ($1)"; push; exit "$1"; }
  return 0
}
log "bookkeeping pilot run 2 start (seeds $SEEDS)"
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || true
[ -d external/finbalance ] || { git clone -q https://github.com/Devansh1105/finbalance.git external/finbalance \
  && git -C external/finbalance checkout -q a1062b7b; chk $? "finbalance clone"; }
UV=$(command -v uv || echo ~/.local/bin/uv)
$PY -c "import requests, dotenv, pydantic" 2>/dev/null || "$UV" pip install --python $PY -q requests python-dotenv pydantic
MK="scripts/make_data.py --entries_only --train_levels 1,2,3 --onpolicy 3000"
[ -f $D/train.jsonl ] || { $PY $MK --out $D > ~/VSCA/bpilot_data.log 2>&1; chk $? data; }
if [ ! -f $D/onpolicy.jsonl ]; then  # data from run 1: regenerate and check every earlier split is unchanged
  rm -rf ${D}_new && $PY $MK --out ${D}_new > ~/VSCA/bpilot_data.log 2>&1; chk $? "data (onpolicy split)"
  for f in train test_id test_L3 test_ind test_year; do
    cmp -s $D/$f.jsonl ${D}_new/$f.jsonl || { log "regenerated $f differs from run 1; stopping"; exit 1; }
  done
  mv ${D}_new/onpolicy.jsonl ${D}_new/stats.json $D/ && rm -rf ${D}_new
fi
log "data: $(tr -d '\n ' < $D/stats.json | head -c 300)"
for s in $SEEDS; do
  S=runs/p_sft_s$s
  P=runs/pilotB_s$s  # run 2 (run 1, runs/pilot_s0, is invalid: memorised samples)
  mkdir -p $P
  $PY $H/sft.py --data $D --out $S --seed $s --micro_bs 2 --max_len 4400 --grad_ckpt > ~/VSCA/bpilot_sft_s$s.log 2>&1
  chk $? "sft s$s"
  $VPY $H/sample.py --split $D/onpolicy.jsonl --adapter $S/adapter --out $P/train_samples.jsonl --n 4 --seed $s \
    --max_tokens 800 --max_model_len 4400 > ~/VSCA/bpilot_sample_s$s.log 2>&1; chk $? "on-policy samples s$s"
  [ -f $P/pools.json ] || { $PY scripts/pilot.py pools --data $D --split onpolicy --samples $P/train_samples.jsonl --out $P/pools.json \
    > ~/VSCA/bpilot_pools_s$s.log 2>&1; chk $? "pools s$s"; }
  log "s$s pools: $(tail -1 ~/VSCA/bpilot_pools_s$s.log | cut -c1-400)"
  $PY scripts/pilot.py gate --pools $P/pools.json --out $P/gate.json > ~/VSCA/bpilot_gate_s$s.log 2>&1
  rc=$?
  log "s$s gate: $(cat $P/gate.json | tr -d '\n ')"
  if [ $rc -eq 3 ]; then log "learnability gate FAILED for s$s: stopping the pilot (PILOT.md)"; push; exit 3; fi
  chk $rc "gate s$s"
  for arm in $ARMS; do
    $PY $H/pilot.py train --arm $arm --seed $s --pools $P/pools.json --init $S/adapter --out $P/$arm --epochs 1 \
      --micro_bs 2 --accum 8 --fork_mb 1 --grad_ckpt >> ~/VSCA/bpilot_train_s$s.log 2>&1; chk $? "train $arm s$s"
    log "s$s $arm trained"
  done
  ads="$S/adapter"; for arm in $ARMS; do ads="$ads $P/$arm/adapter"; done
  $VPY $H/eval_many.py --data $D --splits $SPLITS --adapters $ads --max_tokens 800 --max_model_len 4400 \
    > ~/VSCA/bpilot_eval_s$s.log 2>&1; chk $? "eval s$s"
  for d in $S $(for arm in $ARMS; do echo $P/$arm; done); do
    for sp in $SPLITS; do
      [ -f $d/$sp.score.jsonl ] || $PY scripts/pilot.py score --data $D --split $sp --samples $d/$sp.jsonl \
        --out $d/$sp.score.jsonl >> ~/VSCA/bpilot_score.log 2>&1
    done
  done
  log "s$s scored (L3+ind): $(for arm in sft $ARMS; do d=$P/$arm; [ $arm = sft ] && d=$S; printf "%s=%s " $arm $(cat $d/test_L3.score.jsonl $d/test_ind.score.jsonl | python3 -c "import sys,json;r=[json.loads(l)['correct'] for l in sys.stdin];print(round(100*sum(r)/len(r),1))"); done)"
  push
done
mkdir -p runs/pilot_stats
pool() { for s in $SEEDS; do d=runs/pilotB_s$s/$1; [ $1 = sft ] && d=runs/p_sft_s$s; cat $d/test_L3.score.jsonl $d/test_ind.score.jsonl > $d/primary.score.jsonl; echo $d/primary.score.jsonl; done; }
cmp() {
  PYTHONPATH=../vsca/src $PY -m vsca.cli stats compare --a $(pool $1) --b $(pool $2) > runs/pilot_stats/$1_vs_$2.json
  log "$1 vs $2 (primary): $(python3 -c "import json;d=json.load(open('runs/pilot_stats/$1_vs_$2.json'));print('%.1f vs %.1f, diff %+.2f [%+.2f, %+.2f]'%(100*d['acc_a'],100*d['acc_b'],100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
cmp set rft; cmp set single; cmp set sft; cmp rft sft
python3 - <<'EOF' | tee -a ~/VSCA/bpilot_progress.log
import json
d = lambda b: 100 * json.load(open(f"runs/pilot_stats/set_vs_{b}.json"))["diff"]
r, s = d("rft"), d("single")
dec = "go" if r >= 2.0 and s >= 1.0 else ("no_go" if r <= 0.5 else "discuss")
json.dump({"set_minus_rft": r, "set_minus_single": s, "decision": dec}, open("runs/pilot_stats/decision.json", "w"), indent=2)
print("decision:", dec, r, s)
EOF
push
log "bookkeeping pilot done"
