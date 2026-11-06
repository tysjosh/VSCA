#!/usr/bin/env bash
# PILOT3.md on the GPU VM, seeds 0-2: unseen 'onpolicy' exercises -> samples from the pilot-1 SFT adapter -> pools ->
# 6 arms -> greedy eval -> score -> comparisons. Resumable; backed up to olukotunjosh/harmony-runs.
#   screen -dmS hpilot3 bash ~/VSCA/harmony/scripts/run_pilot3.sh
cd ~/VSCA/harmony || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
VPY=../.venv-vllm/bin/python
HUB="--runs harmony/runs --repo olukotunjosh/harmony-runs"
ARMS="rft neg single set single_step set_step"
SPLITS="test_id test_long test_minor test_keys"
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/hpilot3_progress.log; }
push() { (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1); }
chk() {
  [ "$1" -eq 75 ] && { log "deadline during $2; pushing, rerun to continue"; push; exit 75; }
  [ "$1" -ne 0 ] && { log "$2 FAILED ($1)"; push; exit "$1"; }
  return 0
}
log "pilot3 start"
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || true
MK="scripts/make_data.py --onpolicy 4000"
[ -f data/train.jsonl ] || { $PY $MK --out data > ~/VSCA/harmony_data.log 2>&1; chk $? data; }
if [ ! -f data/onpolicy.jsonl ]; then  # regenerate and check the pilot-1 splits are unchanged
  rm -rf data_new && $PY $MK --out data_new > ~/VSCA/harmony_data.log 2>&1; chk $? "data (onpolicy split)"
  for f in train test_id test_long test_minor test_keys; do
    cmp -s data/$f.jsonl data_new/$f.jsonl || { log "regenerated $f differs from pilot 1; stopping"; exit 1; }
  done
  mv data_new/onpolicy.jsonl data/ && rm -rf data_new
fi
log "onpolicy split ready ($(wc -l < data/onpolicy.jsonl) exercises)"
for s in 0 1 2; do
  S=runs/sft_s$s
  P=runs/pilot3_s$s
  mkdir -p $P
  [ -f $S/adapter/adapter_config.json ] || { log "missing $S/adapter (pilot 1)"; exit 1; }
  $VPY scripts/sample.py --split data/onpolicy.jsonl --adapter $S/adapter --out $P/train_samples.jsonl --n 4 \
    --seed $s > ~/VSCA/hpilot3_sample_s$s.log 2>&1; chk $? "samples s$s"
  [ -f $P/pools.json ] || { $PY scripts/pilot.py pools --data data --split onpolicy --samples $P/train_samples.jsonl \
    --out $P/pools.json > ~/VSCA/hpilot3_pools_s$s.log 2>&1; chk $? "pools s$s"; }
  [ -f $P/step_pools.json ] || { $PY scripts/pilot.py pools_step --data data --split onpolicy \
    --samples $P/train_samples.jsonl --out $P/step_pools.json > ~/VSCA/hpilot3_steppools_s$s.log 2>&1; chk $? "step pools s$s"; }
  log "s$s pools: $(tail -1 ~/VSCA/hpilot3_pools_s$s.log | cut -c1-250) | step: $(tail -1 ~/VSCA/hpilot3_steppools_s$s.log | cut -c1-90)"
  for arm in $ARMS; do
    $PY scripts/pilot.py train --arm $arm --seed $s --pools $P/pools.json --step_pools $P/step_pools.json \
      --init $S/adapter --out $P/$arm --micro_bs 2 --accum 8 --fork_mb 1 >> ~/VSCA/hpilot3_train_s$s.log 2>&1
    chk $? "train $arm s$s"
  done
  ads=""; for arm in $ARMS; do ads="$ads $P/$arm/adapter"; done
  $VPY scripts/eval_many.py --data data --splits $SPLITS --adapters $ads > ~/VSCA/hpilot3_eval_s$s.log 2>&1; chk $? "eval s$s"
  for arm in $ARMS; do for sp in $SPLITS; do
    [ -f $P/$arm/$sp.score.jsonl ] || $PY scripts/pilot.py score --data data --split $sp --samples $P/$arm/$sp.jsonl \
      --out $P/$arm/$sp.score.jsonl >> ~/VSCA/hpilot3_score.log 2>&1
  done; done
  log "s$s scored: $(for arm in $ARMS; do printf "%s=%s " $arm $(cat $P/$arm/test_id.score.jsonl $P/$arm/test_long.score.jsonl | python3 -c "import sys,json;r=[json.loads(l)['correct'] for l in sys.stdin];print(round(100*sum(r)/len(r),1))"); done)"
  push
done
mkdir -p runs/pilot3_stats
pool() { for s in 0 1 2; do d=runs/pilot3_s$s/$1; cat $d/test_id.score.jsonl $d/test_long.score.jsonl > $d/primary.score.jsonl; echo $d/primary.score.jsonl; done; }
cmp_() {
  PYTHONPATH=../vsca/src $PY -m vsca.cli stats compare --a $(pool $1) --b $(pool $2) > runs/pilot3_stats/$1_vs_$2.json
  log "$1 vs $2 (primary): $(python3 -c "import json;d=json.load(open('runs/pilot3_stats/$1_vs_$2.json'));print('%.1f vs %.1f, diff %+.2f [%+.2f, %+.2f]'%(100*d['acc_a'],100*d['acc_b'],100*d['diff'],100*d['ci95'][0],100*d['ci95'][1]))")"
}
cmp_ set rft; cmp_ neg rft; cmp_ single rft; cmp_ set single; cmp_ set_step single_step
python3 - <<'EOF' | tee -a ~/VSCA/hpilot3_progress.log
import json
d = json.load(open("runs/pilot3_stats/set_vs_rft.json"))
diff, lo = 100 * d["diff"], 100 * d["ci95"][0]
dec = "go" if diff >= 2.0 and lo > 0 else ("no_go" if diff <= 0.5 else "inconclusive")
json.dump({"set_minus_rft": diff, "ci95_low": lo, "decision": dec}, open("runs/pilot3_stats/decision.json", "w"), indent=2)
print("decision:", dec, round(diff, 2), round(lo, 2))
EOF
push
log "pilot3 done"
