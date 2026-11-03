#!/usr/bin/env bash
# Exercise every method, control, evaluation and diagnostic on a tiny model.
# This checks that code paths run; the numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x ../.venv-vllm/bin/python ]; then PY=${PY:-../.venv-vllm/bin/python}; else PY=${PY:-../.venv/bin/python}; fi
C="--config configs/smoke.yaml --set out_dir=runs_smoke"
R=runs_smoke

run() { echo "=== $*"; "$PY" -m vsca.cli "$@" 2>&1 | grep -E '^\[(certify|val|rft)\]|^\{|"accuracy"|Error|error' || true; }

run train $C --set method=sft --set run_name=sft
run score $C --adapter $R/sft/adapter_best --out $R/scores_ref.json
run score $C --attention --out $R/scores_base.json
run certify $C --out $R/cert.jsonl

for m in random claim_mask claim_upweight trimsft; do run train $C --set method=$m --set run_name=$m; done
run train $C --set method=rho --set run_name=rho --set ref_scores=$R/scores_ref.json
run train $C --set method=token_cleaning --set run_name=token_cleaning --set ref_scores=$R/scores_ref.json --set base_scores=$R/scores_base.json
run train $C --set method=sstoken --set run_name=sstoken --set base_scores=$R/scores_base.json
run train $C --set method=cf_select --set run_name=cf_select --set ref_scores=$R/scores_ref.json
run train $C --set method=unlikelihood --set run_name=unlikelihood --set cert_path=$R/cert.jsonl
run train $C --set method=rft --set run_name=rft --set rft_from=$R/cert.jsonl
run train $C --set method=vsca --set run_name=vsca
for v in claim_only decision_only uncertified shuffled; do
  run train $C --set method=vsca --set variant=$v --set run_name=vsca_$v --set cert_path=$R/cert.jsonl
done
run train $C --set method=vsca --set run_name=vsca_inject --set cert_path=$R/cert.jsonl --set inject_eta=0.2
run train $C --set method=vsca --set run_name=vsca_drop --set cert_path=$R/cert.jsonl --set drop_eta=0.2

run eval $C --adapter $R/vsca/adapter_best --split gsm8k_test --out $R/eval_vsca_test.jsonl
run eval $C --adapter $R/sft/adapter_best --split gsm8k_test --out $R/eval_sft_test.jsonl
run eval $C --adapter $R/vsca/adapter_best --split svamp_all --out $R/eval_vsca_svamp.jsonl
run eval $C --split gsm8k_val --out $R/eval_base_val.jsonl

run diagnose taxonomy $C --eval_file $R/eval_base_val.jsonl --out_dir $R/diag
run diagnose mass $C --cert $R/cert.jsonl --out_dir $R/diag
run diagnose audit $C --cert $R/cert.jsonl --out_dir $R/diag
run diagnose earlier_score $C --with_model --out_dir $R/diag

"$PY" -m vsca.cli stats compare --a $R/eval_vsca_test.jsonl --b $R/eval_sft_test.jsonl --n_boot 200
"$PY" -m vsca.cli stats mcnemar --a $R/eval_vsca_test.jsonl --b $R/eval_sft_test.jsonl
echo "=== resume test"
PY="$PY" R="$R" bash scripts/resume_test.sh
echo "smoke OK"
