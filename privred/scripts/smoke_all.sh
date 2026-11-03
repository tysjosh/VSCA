#!/usr/bin/env bash
# CPU end-to-end smoke test on tiny subsets for one epoch. It checks that code paths run; the numbers are meaningless.
# Builds a mini companion tree with sg (never downloads CIFAR, never runs `sg prepare`), then runs the privred stages.
# Starts from an empty runs_smoke/ (deleted first). Full output goes to runs_smoke/smoke.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
PY=${PY:-../.venv/bin/python}
R=runs_smoke
COMP=$R/companion
CS=$COMP/cifar100
ST=$R/cifar100
SG="--config ../sg/configs/smoke.yaml --set out_dir=$COMP --set data_dir=../sg/data --set limit_test=200"
C="--config configs/base.yaml --config configs/smoke.yaml --set out_dir=$R --set companion_dir=$COMP"
rm -rf "$R"
mkdir -p "$R"
LOG=$R/smoke.log

sgrun() {  # run one sg command (companion tree), output to the log
  echo "=== sg $*"
  if ! "$PY" -m sg.cli "$@" >> "$LOG" 2>&1; then
    echo "FAILED: sg $*"
    tail -n 40 "$LOG"
    exit 1
  fi
}

run() {  # run one privred command, output to the log; on failure show the log tail and stop
  echo "=== privred $*"
  if ! "$PY" -m privred.cli "$@" >> "$LOG" 2>&1; then
    echo "FAILED: privred $*"
    tail -n 40 "$LOG"
    exit 1
  fi
}

check() {  # check <python expression over d, the command's JSON stdout> <privred command...>
  local expr=$1
  shift
  echo "=== privred $* (check: $expr)"
  if ! "$PY" -m privred.cli "$@" 2>> "$LOG" | tee -a "$LOG" | "$PY" -c "import json, sys; d = json.load(sys.stdin); assert $expr, d"; then
    echo "FAILED: privred $*"
    tail -n 40 "$LOG"
    exit 1
  fi
}

# --- mini companion tree (sg smoke config) -------------------------------------------------------
for p in 100 050; do
  b=$([ $p = 100 ] && echo 1.0 || echo 0.5)
  sgrun simclr $SG --set budget=$b --set run_name=cifar100/simclr/b$p
  sgrun partitions $SG --set budget=$b --extractor $CS/simclr/b$p --out_dir $CS/partitions/b$p
done
sgrun diagnose geometry $SG --extractor $CS/simclr/b100 --partitions_dirs $CS/partitions/b100 $CS/partitions/b050 \
  --out $CS/geometry.json
# smoke stubs for the companion's select_wd / select_rho outputs
echo '{"0.25": 0.0005, "0.5": 0.0005, "1.0": 0.0005}' > $CS/best_wd.json
echo '{"rho": 0.5, "note": "smoke stub"}' > $CS/best_rho.json
for p in 100 050; do
  b=$([ $p = 100 ] && echo 1.0 || echo 0.5)
  for s in 0 1 2 3 4; do
    sgrun train $SG --set budget=$b --set condition=ce --set seed=$s --set weight_decay=0.0005 --set save_model=false \
      --set run_name=cifar100/b$p/ce/s$s
  done
done
for p in 050 025; do
  b=$([ $p = 050 ] && echo 0.5 || echo 0.25)
  sgrun train $SG --set budget=$b --set condition=ce --set seed=0 --set weight_decay=0.0005 --set save_model=false \
    --set run_name=cifar100/b$p/ce/s0
  sgrun train $SG --set budget=$b --set condition=true_hier --set rho=0.5 --set seed=0 --set weight_decay=0.0005 \
    --set save_model=false --set run_name=cifar100/b$p/true_hier/rho0.5/s0
done

# --- privred: companion check, OOF difficulty, one Study A set ----------------------------------
check "d['ok'] and len(d['budgets']['1.0']['subset']) == 8 and d['budgets']['1.0']['reuse_true_hier']" \
  check_companion $C
check "d['n_commands'] == 2" plan $C --stage oof --budget 1.0 --out $ST/plans/oof_b100.txt
run launch --plan $ST/plans/oof_b100.txt --gpus none --jobs_per_gpu 3
check "d['n'] == 800 and d['K'] == 2 and d['check']['available'] and len(d['check']['per_seed']) == 5" \
  difficulty $C --budget 1.0
check "d['n_commands'] == 4" plan $C --stage studyA --budget 1.0 --names A_ce --out $ST/plans/studyA_b100_ce.txt
run launch --plan $ST/plans/studyA_b100_ce.txt --gpus none --jobs_per_gpu 3
check "d['M'] == 4 and d['n_pool'] == 800 and d['variance'] == 'pooled' and d['pd_available'] and d['auc'] is not None" \
  score $C --set_dir $ST/b100/sets/A_ce
# rerunning a finished member is a no-op (idempotent)
run member $C --set_dir $ST/b100/sets/A_ce --m 0
[ -s $ST/b100/difficulty.npz ] && [ -s $ST/b100/difficulty_check.json ] && [ -s $ST/b100/sets/A_ce/scores.npz ] \
  || { echo "FAILED: missing difficulty / score outputs"; exit 1; }

# --- support features and Study A at both budgets -----------------------------------------------
check "d['n'] > 800 and d['dim'] == 512" features $C
for p in 100 050; do
  b=$([ $p = 100 ] && echo 1.0 || echo 0.5)
  check "d['n'] == 800 and d['partition'] == 'true' and d['mean_N'] > 0" support $C --budget $b
done
# OOF difficulty at 50% too (Study A strata and the label-error repeat use it)
check "d['n_commands'] == 2" plan $C --stage oof --budget 0.5 --out $ST/plans/oof_b050.txt
run launch --plan $ST/plans/oof_b050.txt --gpus none --jobs_per_gpu 3
check "d['n'] == 800 and d['K'] == 2" difficulty $C --budget 0.5
for p in 100 050; do
  b=$([ $p = 100 ] && echo 1.0 || echo 0.5)
  check "d['n_commands'] == 48" plan $C --stage studyA --budget $b --out $ST/plans/studyA_b$p.txt
  run launch --plan $ST/plans/studyA_b$p.txt --gpus none --jobs_per_gpu 3
  for s in $ST/b$p/sets/A_*; do
    check "d['M'] == 4 and d['n_pool'] == 800 and d['pd_available']" score $C --set_dir $s
  done
done
# per-record variance path on A_ce (M/2 - 1 = 1 >= min_ref_per_side), then the default pooled scores again
check "d['variance'] == 'per_record'" score $C --set min_ref_per_side=1 --set_dir $ST/b100/sets/A_ce
check "d['variance'] == 'pooled'" score $C --set_dir $ST/b100/sets/A_ce
for p in 100 050; do
  b=$([ $p = 100 ] && echo 1.0 || echo 0.5)
  check "len(d['conditions']) == 12 and not d['missing'] and d['delta_priv']['available'] and d['difficulty_available']" \
    studyA $C --budget $b
  "$PY" -c "
import json
d = json.load(open('$ST/b$p/analysis/studyA_b$p.json'))
nc = [k for k in d['conditions'] if k != 'A_ce']
assert len(d['conditions']) == 12 and len(nc) == 11, list(d['conditions'])
assert all(d['patterns'][k]['n'] == 800 and d['strata'][k] and d['cs_split'][k] and d['robustness'][k] for k in nc)
assert d['delta_priv']['delta'] is not None and len(d['delta_priv']['lopo']['residuals']) == 8
pb, rk = d['delta_priv']['partition_bootstrap'], d['delta_priv']['rank_random']
assert pb['available'] and pb['n_valid'] > 0 and pb['ci95'] is not None and pb['n_partitions'] == 8, pb
assert rk['n_random'] == 4 and 1 <= rk['rank'] <= 5 and rk['descriptive'], rk
assert d['label_error_repeat']['delta_priv']['partition_bootstrap']['available']
assert d['P1']['estimate'] is not None and d['reuse_check']['A_ce']['available']
assert d['label_error_repeat']['n_kept'] + d['label_error_repeat']['n_excluded'] == 800
assert all(d['strata_pd'][k] and set(d['strata_pd'][k]) == {'0', '1', '2', '3'} for k in nc)
assert all(d['dsup_ci'][k]['ci95'] is not None for k in nc)
own = [k for k in nc if d['conditions'][k]['condition'] not in ('flat_aux',)]
assert all(d['conditions'][k]['own_frame_vs_ce']['available'] for k in own), own
assert d['conditions']['A_flat_aux']['own_frame_vs_ce'] is None
S = json.load(open('$ST/b$p/sets/A_ce/summary.json'))
assert set(S['pool_lambda_by_difficulty_quartile']) == {'0', '1', '2', '3'} and len(S['pool_lambda_by_class']) == 100
" || { echo "FAILED: studyA_b$p.json incomplete"; exit 1; }
done

# --- Study B: preliminary audit, removal sets, baseline and deletion sets ----------------------------
check "d['n_commands'] == 4" plan $C --stage prelim --budget 1.0 --out $ST/plans/prelim_b100.txt
run launch --plan $ST/plans/prelim_b100.txt --gpus none --jobs_per_gpu 3
check "d['M'] == 4 and d['n_pool'] == 800 and not d['pd_available']" score $C --set_dir $ST/b100/sets/prelim
check "d['R_size'] == 32 and d['per_class'] == 2 and len(d['cohorts']) == 2 and all(len(c['siblings']) == 16 and len(c['vis_classes']) == 16 and c['sets']['vis']['mean_dK_T'] == 0 and c['sets']['vis']['n_curated_siblings_removed'] == 0 for c in d['cohorts'])" \
  deletion_sets $C
check "d['n_commands'] == 8" plan $C --stage baseB --budget 1.0 --out $ST/plans/baseB_b100.txt
# CE: Q x 6 types, True-Hier: Q_true_hier x 4 types
check "d['n_commands'] == 64 and len(d['sets']) == 16" plan $C --stage deletion --budget 1.0 --out $ST/plans/deletion_b100.txt
cat $ST/plans/baseB_b100.txt $ST/plans/deletion_b100.txt > $ST/plans/studyB_b100.txt
run launch --plan $ST/plans/studyB_b100.txt --gpus none --jobs_per_gpu 3
for s in $ST/b100/sets/baseB_*; do
  check "d['M'] == 4 and d['n_pool'] == 800" score $C --set_dir $s
done
for s in $ST/b100/sets/del_*; do
  check "d['M'] == 4 and d['n_pool'] == 768" score $C --set_dir $s
done
check "not d['missing'] and len(d['sets']) == 16 and d['contrasts']['ce']['sem_minus_vis'] is not None and d['true_hier_interaction'] is not None and d['rows']['ce']['n_experiments'] == 12" \
  studyB $C
check "d['conditions']['ce']['cohort']['ridge'] is not None and d['P5_inputs']['available'] and not d['secondary_dataset']['available']" \
  predict $C
check "d['studyB']['ce']['all_rows'] and d['key_test']['ce']['available'] and len(d['studyA']['1.0']) == 11" linking $C
check "all(k in d['p_raw'] and k in d['p_holm'] for k in ('P1', 'P2', 'P3', 'P4', 'P5')) and all(d['p_raw'][k] is not None for k in d['p_raw'])" \
  primary $C
"$PY" -c "
import json
an = '$ST/b100/analysis/'
B = json.load(open(an + 'studyB.json'))
assert set(B['cohort_means']['ce']) == {'0', '1'} and set(B['cohort_means']['true_hier']) == {'0'}
assert all(len(B['cohort_means']['ce'][q]) == 6 for q in ('0', '1')) and len(B['cohort_means']['true_hier']['0']) == 4
assert B['label_error_repeat']['ce']['contrasts']['sem_minus_vis']['n_cohorts'] >= 0
assert all(B['contrasts']['ce'][k]['bootstrap']['available'] for k in B['contrasts']['ce'])
assert B['true_hier_interaction']['bootstrap']['available']
P = json.load(open(an + 'prediction.json'))
ce = P['conditions']['ce']
assert all(ce[k]['ridge']['available'] and ce[k]['gbt']['available'] for k in ('cohort', 'experiment', 'class', 'parent'))
assert not P['conditions']['true_hier']['cohort']['ridge']['available']  # one True-Hier cohort in the smoke
assert set(ce['record_disjoint']) == {'cohort', 'experiment'}
assert all(v['available'] is False or v['n_rows'] > 0 for s in ce['record_disjoint'].values() for v in s.values())
L = json.load(open(an + 'linking.json'))
assert L['studyB']['ce']['T_rows']['available'] and 'dK_x_PD0' in L['studyB']['ce']['all_rows']['coef']
Pr = json.load(open(an + 'primary.json'))
assert set(Pr['p_raw']) == set(Pr['p_holm']) == {'P1', 'P2', 'P3', 'P4', 'P5'}
assert all(Pr['p_holm'][k] >= Pr['p_raw'][k] for k in Pr['p_raw'])
assert Pr['n_tests'] == Pr['n_family'] == 5 and Pr['holm_warning'] is None
" || { echo "FAILED: Study B / prediction / linking / primary outputs incomplete"; exit 1; }
# removal sets are immutable once deletion sets exist: a changed R.npy makes deletion_sets refuse to overwrite it
cp $ST/b100/deletion/q0_vis/R.npy $ST/R_backup.npy
"$PY" -c "import numpy as np; p = '$ST/b100/deletion/q0_vis/R.npy'; np.save(p, np.load(p)[::-1])"
# the refusal must come before any write: record the mtimes of every removal-set file and of cohorts.json
MT="import glob, os; print(sorted((f, os.stat(f).st_mtime_ns) for f in glob.glob('$ST/b100/deletion/**', recursive=True) if os.path.isfile(f)))"
"$PY" -c "$MT" > $ST/mtimes_before.txt
echo "=== deletion_sets (expect refusal)" >> "$LOG"
if "$PY" -m privred.cli deletion_sets $C >> "$LOG" 2>&1; then echo "FAILED: deletion_sets overwrote a changed R.npy"; exit 1; fi
grep -q "would change but deletion sets already exist" "$LOG" || { echo "FAILED: no refusal message"; exit 1; }
"$PY" -c "$MT" > $ST/mtimes_after.txt
cmp -s $ST/mtimes_before.txt $ST/mtimes_after.txt || { echo "FAILED: deletion_sets wrote files before refusing"; exit 1; }
rm $ST/mtimes_before.txt $ST/mtimes_after.txt
mv $ST/R_backup.npy $ST/b100/deletion/q0_vis/R.npy

# --- iNat-100 Study A on a fake iNat2018 tree (mirrors the sg smoke iNat block) ------------------
# Companion: inat_build at 32x32 (smoke only, to keep CPU time down), simclr b100, partitions b100; no geometry.json
# (delta_priv falls back to W_train); stub best_wd.json with 0.25 and 1.0 only, as sg's iNat select_wd writes, so
# the size-matched key 0.5 is missing and wd_for uses the nearest-budget rule.
CSI=$COMP/inat100
STI=$R/inat100
SGI="--config ../sg/configs/smoke.yaml --config ../sg/configs/inat100.yaml --set out_dir=$COMP --set data_dir=$R/data_inat \
--set inat_per_class=6 --set inat_test_per_class=2 --set inat_size=32"
INAT_SETS="--set data_dir=$R/data_inat --set inat_per_class=6 --set inat_test_per_class=2"
CI="--config configs/base.yaml --config configs/inat100.yaml --config configs/smoke.yaml --set out_dir=$R \
--set companion_dir=$COMP $INAT_SETS"
echo "=== make_fake_inat"
"$PY" ../sg/scripts/make_fake_inat.py --out $R/fake_inat >> "$LOG" 2>&1
sgrun inat_build $SGI --inat_root $R/fake_inat
sgrun simclr $SGI --set budget=1.0 --set run_name=inat100/simclr/b100
sgrun partitions $SGI --set budget=1.0 --extractor $CSI/simclr/b100 --out_dir $CSI/partitions/b100
echo '{"0.25": 0.0005, "1.0": 0.0005}' > $CSI/best_wd.json
echo '{"rho": 0.5, "note": "smoke stub"}' > $CSI/best_rho.json
check "d['ok'] and len(d['budgets']['1.0']['subset']) == 8 and d['budgets']['1.0']['wd_privacy'] == 0.0005" \
  check_companion $CI
check "d['dim'] == 512" features $CI
check "d['n_commands'] == 2" plan $CI --stage oof --budget 1.0 --out $STI/plans/oof_b100.txt
run launch --plan $STI/plans/oof_b100.txt --gpus none --jobs_per_gpu 3
check "d['n'] == 400 and d['K'] == 2" difficulty $CI --budget 1.0
check "d['n'] == 400 and d['partition'] == 'true'" support $CI --budget 1.0
# CE, True-Hier and the 4 + 4 subset: 10 sets x M = 4
check "d['n_commands'] == 40" plan $CI --stage studyA --budget 1.0 --out $STI/plans/studyA_b100.txt
run launch --plan $STI/plans/studyA_b100.txt --gpus none --jobs_per_gpu 3
for s in $STI/b100/sets/A_*; do
  check "d['M'] == 4 and d['n_pool'] == 400 and d['pd_available']" score $CI --set_dir $s
done
check "len(d['conditions']) == 10 and not d['missing'] and d['delta_priv']['available'] and d['delta_priv']['W_source'] == 'W_train' and d['P1']['estimate'] is not None" \
  studyA $CI --budget 1.0

# --- remote stage script: every run_plan.sh stage against the smoke tree, training stages plan-only ----------
rp() {
  echo "=== run_plan.sh $1 (PLAN_ONLY=1)"
  if ! PLAN_ONLY=1 PY="$PY" OUT=$R COMPANION=$COMP GPUS=none JOBS_PER_GPU=3 BUDGETS="1.0 0.5" STUDYB_BUDGET=1.0 \
       SETS="--config configs/smoke.yaml" INAT_SETS="$INAT_SETS" bash scripts/run_plan.sh "$1" >> "$LOG" 2>&1; then
    echo "FAILED: run_plan.sh $1"
    tail -n 40 "$LOG"
    exit 1
  fi
}
cp $ST/b100/deletion/q0_vis/R.npy $R/R_q0_vis_before.npy
for stage in check features oof difficulty support studyA prelim deletion_sets studyB predict linking primary \
             inat_check inat_features inat_oof inat_studyA count; do
  rp $stage
done
cmp -s $R/R_q0_vis_before.npy $ST/b100/deletion/q0_vis/R.npy \
  || { echo "FAILED: deletion_sets rerun changed a removal set"; exit 1; }
for f in $ST/plans/oof_b100.txt $ST/plans/oof_b050.txt $ST/plans/studyA_b050.txt $ST/plans/prelim_b100.txt \
         $ST/plans/studyB_b100.txt $STI/plans/oof_b100.txt $STI/plans/studyA_b100.txt \
         $ST/b100/analysis/primary.json $STI/b100/analysis/studyA_b100.json; do
  [ -s "$f" ] || { echo "FAILED: run_plan.sh did not write $f"; exit 1; }
done
# compute count (Table compute): 949 runs and ~326 A100-hours at beta = 100%, iNat-100 Study A 160 runs
check "d['budgets']['1.0']['runs'] == 949 and round(d['budgets']['1.0']['gpu_h']) == 326 and round(d['budgets']['0.5']['components']['studyA']['gpu_h']) == 67 and 'assumption' in d['assumption']" \
  plan --count --config configs/base.yaml
check "d['budgets']['1.0']['components']['studyA']['runs'] == 160 and 'assumption' in d['assumption']" \
  plan --count --config configs/base.yaml --config configs/inat100.yaml

find $R -name '*.pt' -delete
[ -z "$(find $R -name '*.pt')" ] || { echo "FAILED: checkpoints left under $R"; exit 1; }
echo "smoke OK"
