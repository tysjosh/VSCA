#!/usr/bin/env bash
# PILOT.md on the GPU VM, seeds 0-2: SFT -> on-policy samples -> pools -> 6 arms -> greedy eval -> score ->
# comparisons. Resumable (every step skips finished work; training checkpoints and stops at the deadline, rc 75).
# Backed up to olukotunjosh/harmony-runs.   screen -dmS hpilot bash ~/VSCA/harmony/scripts/run_pilot.sh
cd ~/VSCA/harmony || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
PY=../.venv/bin/python
VPY=../.venv-vllm/bin/python
HUB="--runs harmony/runs --repo olukotunjosh/harmony-runs"
ARMS="sft_cont rft neg single set full"
SPLITS="test_id test_long test_minor test_keys"
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/hpilot_progress.log; }
push() { (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1); }
chk() { # chk <rc> <what>
  [ "$1" -eq 75 ] && { log "deadline during $2; pushing, rerun to continue"; push; exit 75; }
  [ "$1" -ne 0 ] && { log "$2 FAILED ($1)"; push; exit "$1"; }
  return 0
}
log "pilot start"
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || true
[ -f data/train.jsonl ] || { $PY scripts/make_data.py --out data > ~/VSCA/harmony_data.log 2>&1; chk $? data; }
[ -f data/train_pilot.jsonl ] || head -n 4000 data/train.jsonl > data/train_pilot.jsonl
for s in 0 1 2; do
  S=runs/sft_s$s
  P=runs/pilot_s$s
  mkdir -p $P
  $PY scripts/sft.py --data data --out $S --seed $s > ~/VSCA/hpilot_sft_s$s.log 2>&1; chk $? "sft s$s"
  $VPY scripts/sample.py --split data/train_pilot.jsonl --adapter $S/adapter --out $P/train_samples.jsonl --n 4 \
    --seed $s > ~/VSCA/hpilot_sample_s$s.log 2>&1; chk $? "on-policy samples s$s"
  [ -f $P/pools.json ] || { $PY scripts/pilot.py pools --data data --samples $P/train_samples.jsonl --out $P/pools.json \
    > ~/VSCA/hpilot_pools_s$s.log 2>&1; chk $? "pools s$s"; }
  log "s$s pools: $(tail -1 ~/VSCA/hpilot_pools_s$s.log | cut -c1-300)"
  for arm in $ARMS; do
    $PY scripts/pilot.py train --arm $arm --seed $s --pools $P/pools.json --init $S/adapter --out $P/$arm \
      >> ~/VSCA/hpilot_train_s$s.log 2>&1; chk $? "train $arm s$s"
  done
  log "s$s trained"
  ads="$S/adapter"; for arm in $ARMS; do ads="$ads $P/$arm/adapter"; done
  $VPY scripts/eval_many.py --data data --splits $SPLITS --adapters $ads > ~/VSCA/hpilot_eval_s$s.log 2>&1; chk $? "eval s$s"
  for d in $S $(for arm in $ARMS; do echo $P/$arm; done); do
    for sp in $SPLITS; do
      [ -f $d/$sp.score.jsonl ] || $PY scripts/pilot.py score --data data --split $sp --samples $d/$sp.jsonl \
        --out $d/$sp.score.jsonl >> ~/VSCA/hpilot_score.log 2>&1
    done
  done
  log "s$s scored: $(for arm in sft $ARMS; do d=$P/$arm; [ $arm = sft ] && d=$S; printf "%s=%s " $arm $(cat $d/test_id.score.jsonl $d/test_long.score.jsonl | python3 -c "import sys,json;r=[json.loads(l)['correct'] for l in sys.stdin];print(round(100*sum(r)/len(r),1))"); done)"
  push
done
# comparisons on the pooled primary splits (test_id + test_long) and per secondary split
mkdir -p runs/pilot_stats
pool() { for s in 0 1 2; do d=runs/pilot_s$s/$1; [ $1 = sft ] && d=runs/sft_s$s; cat $d/test_id.score.jsonl $d/test_long.score.jsonl > $d/primary.score.jsonl; echo $d/primary.score.jsonl; done; }
cmp() { # cmp <a> <b>
  PYTHONPATH=../vsca/src $PY -m vsca.cli stats compare --a $(pool $1) --b $(pool $2) > runs/pilot_stats/$1_vs_$2.json
  log "$1 vs $2 (primary): $(python3 -c "import json;d=json.load(open('runs/pilot_stats/$1_vs_$2.json'));print('%.1f vs %.1f, diff %+.2f [%+.2f, %+.2f]'%(100*d['acc_a'],100*d['acc_b'],100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
for b in rft single full neg sft_cont sft; do cmp set $b; done
cmp rft sft
python3 - <<'EOF' | tee -a ~/VSCA/hpilot_progress.log
import json
d = lambda b: 100 * json.load(open(f"runs/pilot_stats/set_vs_{b}.json"))["diff"]
r, s = d("rft"), d("single")
dec = "go" if r >= 2.0 and s >= 1.0 else ("no_go" if r <= 0.5 else "discuss")
out = {"set_minus_rft": r, "set_minus_single": s, "decision": dec}
json.dump(out, open("runs/pilot_stats/decision.json", "w"), indent=2)
print("decision:", out)
EOF
push
log "pilot done"
