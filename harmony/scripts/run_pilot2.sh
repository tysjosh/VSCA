#!/usr/bin/env bash
# PILOT2.md on the GPU VM, seeds 0-2: step-level pools from pilot 1's samples -> single_step, set_step -> greedy
# eval -> score -> comparisons. Needs pilot 1's runs (restored from olukotunjosh/harmony-runs). Resumable.
#   screen -dmS hpilot2 bash ~/VSCA/harmony/scripts/run_pilot2.sh
cd ~/VSCA/harmony || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
PY=../.venv/bin/python
VPY=../.venv-vllm/bin/python
HUB="--runs harmony/runs --repo olukotunjosh/harmony-runs"
ARMS="single_step set_step"
SPLITS="test_id test_long test_minor test_keys"
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/hpilot2_progress.log; }
push() { (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1); }
chk() {
  [ "$1" -eq 75 ] && { log "deadline during $2; pushing, rerun to continue"; push; exit 75; }
  [ "$1" -ne 0 ] && { log "$2 FAILED ($1)"; push; exit "$1"; }
  return 0
}
log "pilot2 start"
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || true
[ -f data/train.jsonl ] || { $PY scripts/make_data.py --out data > ~/VSCA/harmony_data.log 2>&1; chk $? data; }
for s in 0 1 2; do
  S=runs/sft_s$s
  P=runs/pilot_s$s
  [ -f $P/pools.json ] && [ -f $P/train_samples.jsonl ] && [ -f $P/rft/test_id.score.jsonl ] || { log "pilot 1 outputs missing for s$s"; exit 1; }
  [ -f $P/step_pools.json ] || { $PY scripts/pilot.py pools_step --data data --samples $P/train_samples.jsonl \
    --out $P/step_pools.json > ~/VSCA/hpilot2_pools_s$s.log 2>&1; chk $? "step pools s$s"; }
  log "s$s step pools: $(tail -1 ~/VSCA/hpilot2_pools_s$s.log | cut -c1-200)"
  for arm in $ARMS; do
    $PY scripts/pilot.py train --arm $arm --seed $s --pools $P/pools.json --step_pools $P/step_pools.json \
      --init $S/adapter --out $P/$arm >> ~/VSCA/hpilot2_train_s$s.log 2>&1; chk $? "train $arm s$s"
  done
  ads=""; for arm in $ARMS; do ads="$ads $P/$arm/adapter"; done
  $VPY scripts/eval_many.py --data data --splits $SPLITS --adapters $ads > ~/VSCA/hpilot2_eval_s$s.log 2>&1; chk $? "eval s$s"
  for arm in $ARMS; do for sp in $SPLITS; do
    [ -f $P/$arm/$sp.score.jsonl ] || $PY scripts/pilot.py score --data data --split $sp --samples $P/$arm/$sp.jsonl \
      --out $P/$arm/$sp.score.jsonl >> ~/VSCA/hpilot2_score.log 2>&1
  done; done
  log "s$s scored: $(for arm in rft single set $ARMS; do printf "%s=%s " $arm $(cat $P/$arm/test_id.score.jsonl $P/$arm/test_long.score.jsonl | python3 -c "import sys,json;r=[json.loads(l)['correct'] for l in sys.stdin];print(round(100*sum(r)/len(r),1))"); done)"
  push
done
mkdir -p runs/pilot2_stats
pool() { for s in 0 1 2; do d=runs/pilot_s$s/$1; cat $d/test_id.score.jsonl $d/test_long.score.jsonl > $d/primary.score.jsonl; echo $d/primary.score.jsonl; done; }
cmp() {
  PYTHONPATH=../vsca/src $PY -m vsca.cli stats compare --a $(pool $1) --b $(pool $2) > runs/pilot2_stats/$1_vs_$2.json
  log "$1 vs $2 (primary): $(python3 -c "import json;d=json.load(open('runs/pilot2_stats/$1_vs_$2.json'));print('%.1f vs %.1f, diff %+.2f [%+.2f, %+.2f]'%(100*d['acc_a'],100*d['acc_b'],100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
cmp set_step single_step; cmp set_step rft; cmp single_step rft; cmp set_step set
python3 - <<'EOF' | tee -a ~/VSCA/hpilot2_progress.log
import json
d = json.load(open("runs/pilot2_stats/set_step_vs_single_step.json"))
diff, lo = 100 * d["diff"], 100 * d["ci95"][0]
dec = "go" if diff >= 1.0 and lo > 0 else ("no_go" if diff <= 0.3 else "inconclusive")
json.dump({"set_step_minus_single_step": diff, "ci95_low": lo, "decision": dec},
          open("runs/pilot2_stats/decision.json", "w"), indent=2)
print("decision:", dec, round(diff, 2), round(lo, 2))
EOF
push
log "pilot2 done"
