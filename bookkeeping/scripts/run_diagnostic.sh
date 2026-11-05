#!/usr/bin/env bash
# DIAGNOSTIC.md on the GPU VM: FinBalance generator -> data -> SFT (seed 0) -> greedy + 8 samples per test
# problem -> diagnostic + go rule. Resumable; backed up to olukotunjosh/bookkeeping-runs.
#   screen -dmS books bash ~/VSCA/bookkeeping/scripts/run_diagnostic.sh
cd ~/VSCA/bookkeeping || exit 1
export SESSION_DEADLINE=$(cat ~/VSCA/deadline 2>/dev/null)
log() { echo "$(date +%H:%M) $*" | tee -a ~/VSCA/books_progress.log; }
R=runs/sft_s0
HUB="--runs bookkeeping/runs --repo olukotunjosh/bookkeeping-runs"
FB_COMMIT=a1062b7b
log start
(cd ~/VSCA && .venv/bin/python hub_sync.py pull $HUB >> ~/VSCA/hubsync.log 2>&1) || log "nothing to restore"
if [ ! -d external/finbalance ]; then
  git clone -q https://github.com/Devansh1105/finbalance.git external/finbalance && git -C external/finbalance checkout -q $FB_COMMIT \
    || { log "finbalance clone failed"; exit 1; }
fi
(uv pip install --python ../.venv/bin/python -q requests python-dotenv pydantic 2>/dev/null \
  || ../.venv/bin/python -m pip install -q requests python-dotenv pydantic) || { log "deps failed"; exit 1; }
[ -f data/train.jsonl ] || ../.venv/bin/python scripts/make_data.py --out data > ~/VSCA/books_data.log 2>&1 || { log "data failed"; exit 1; }
log "data ready: $(tr -d '\n ' < data/stats.json | head -c 300)"
../.venv/bin/python ../harmony/scripts/sft.py --data data --out $R --seed 0 --micro_bs 2 --max_len 4400 --grad_ckpt \
  > ~/VSCA/books_sft.log 2>&1
rc=$?
if [ $rc -eq 75 ]; then
  log "deadline during SFT; pushing the checkpoint, rerun to continue"
  (cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1)
  exit 75
fi
[ $rc -ne 0 ] && { log "SFT failed ($rc)"; exit $rc; }
log "sft done"
for s in test_id test_L3 test_ind test_year; do
  ../.venv-vllm/bin/python ../harmony/scripts/sample.py --split data/$s.jsonl --adapter $R/adapter --out $R/$s.jsonl \
    --max_tokens 1200 --max_model_len 4700 > ~/VSCA/books_sample_$s.log 2>&1 || { log "sample $s failed"; exit 1; }
  log "sampled $s"
done
../.venv/bin/python scripts/diagnose.py --data data --samples $R --out $R/diagnostic.json > ~/VSCA/books_diag.log 2>&1 \
  || { log "diagnose failed"; exit 1; }
log "diagnostic: $(tail -1 ~/VSCA/books_diag.log)"
grep -E "^test_" ~/VSCA/books_diag.log | tee -a ~/VSCA/books_progress.log
cd ~/VSCA && .venv/bin/python hub_sync.py push $HUB >> ~/VSCA/hubsync.log 2>&1; log pushed
