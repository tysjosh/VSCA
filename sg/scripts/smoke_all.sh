#!/usr/bin/env bash
# CPU end-to-end smoke test: every condition, every analysis / stats command, every plan, the launcher and the
# iNat-100 path, on tiny subsets for one epoch. It checks that code paths run; the numbers are meaningless.
# Starts from an empty runs_smoke/ (deleted first). Full output goes to runs_smoke/smoke.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
PY=${PY:-../.venv/bin/python}
R=runs_smoke
ST=$R/cifar100
C="--config configs/smoke.yaml --set out_dir=$R"
rm -rf "$R"
mkdir -p "$R"
LOG=$R/smoke.log

run() {  # run one sg command, output to the log; on failure show the log tail and stop
  echo "=== sg $*"
  if ! "$PY" -m sg.cli "$@" >> "$LOG" 2>&1; then
    echo "FAILED: sg $*"
    tail -n 40 "$LOG"
    exit 1
  fi
}

pct() {
  case "$1" in
    0.1) echo 010 ;; 0.25) echo 025 ;; 0.5) echo 050 ;; 1.0) echo 100 ;;
    *) echo "unknown budget $1" >&2; return 1 ;;
  esac
}

wd() {  # wd <state dir> <budget>: weight decay chosen by select_wd
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[str(float(sys.argv[2]))])' "$1/best_wd.json" "$2"
}

check() {  # check <python expression over d, the command's JSON stdout> <sg command...>
  local expr=$1
  shift
  echo "=== sg $* (check: $expr)"
  if ! "$PY" -m sg.cli "$@" 2>> "$LOG" | tee -a "$LOG" | "$PY" -c "import json, sys; d = json.load(sys.stdin); assert $expr, d"; then
    echo "FAILED: sg $*"
    tail -n 40 "$LOG"
    exit 1
  fi
}

# --- data, extractors, partitions, geometry -------------------------------------------------
run prepare $C
for b in 0.1 0.25 0.5 1.0; do
  p=$(pct $b)
  run simclr $C --set budget=$b --set run_name=cifar100/simclr/b$p
  run partitions $C --set budget=$b --extractor $ST/simclr/b$p --out_dir $ST/partitions/b$p
done
run simclr $C --set simclr_scope=all --set run_name=cifar100/simclr/all
run diagnose geometry $C --extractor $ST/simclr/all \
  --partitions_dirs $ST/partitions/b010 $ST/partitions/b025 $ST/partitions/b050 $ST/partitions/b100 --out $ST/geometry.json

# --- weight decay (tune_wd plan through the launcher) and CE seeds ---------------------------
check "d['n_commands'] == 12" plan --exp tune_wd $C --out $ST/plans/tune_wd.txt --count
run launch --plan $ST/plans/tune_wd.txt --gpus none --jobs_per_gpu 2
run select_wd $C --root $ST/tune_wd --out $ST/best_wd.json
for b in 0.1 1.0; do
  for s in 0 1 2 3 4; do
    save=false
    if [ $b = 0.1 ] && [ $s = 0 ]; then save=true; fi
    run train $C --set budget=$b --set condition=ce --set seed=$s --set weight_decay=$(wd $ST $b) \
      --set save_model=$save --set run_name=cifar100/b$(pct $b)/ce/s$s
  done
done

# --- rho pilot (plan + launcher) -------------------------------------------------------------
check "d['n_commands'] == 9" plan --exp rho_pilot $C --out $ST/plans/rho_pilot.txt --count
run launch --plan $ST/plans/rho_pilot.txt --gpus none --jobs_per_gpu 2
run select_rho $C --root $ST/b100 --out $ST/best_rho.json
RHO=$("$PY" -c 'import json; print(json.load(open("'$ST'/best_rho.json"))["rho"])')
RT=$("$PY" -c "print(f'{float($RHO):g}')")

# --- every condition at 10% and 100% (names follow plans.run_name, so pilot runs are reused) ---------
for b in 0.1 1.0; do
  p=$(pct $b)
  A="--set budget=$b --set rho=$RHO --set weight_decay=$(wd $ST $b)"
  P=$ST/partitions/b$p
  for s in 0 1; do
    save=false
    if [ $b = 0.1 ] && [ $s = 0 ]; then save=true; fi
    run train $C $A --set condition=true_hier --set seed=$s --set save_model=$save --set run_name=cifar100/b$p/true_hier/rho$RT/s$s
    for i in 00 01 02 03; do
      run train $C $A --set condition=random_hier --set partition=$P/random_$i.json --set seed=$s \
        --set run_name=cifar100/b$p/random_hier_random_$i/rho$RT/s$s
    done
  done
  for i in 00 06 12 18; do  # cover subset of 4 spectrum partitions: round(linspace(0, 18, 4))
    run train $C $A --set condition=spectrum_hier --set partition=$P/spectrum_$i.json --set seed=0 \
      --set run_name=cifar100/b$p/spectrum_hier_spectrum_$i/rho$RT/s0
  done
  save=false
  if [ $b = 0.1 ]; then save=true; fi
  for c in flat_aux final_hier; do
    run train $C $A --set condition=$c --set seed=0 --set save_model=$save --set run_name=cifar100/b$p/$c/rho$RT/s0
  done
done

# --- probes (Exp C path) ---------------------------------------------------------------------
for r in ce/s0 true_hier/rho$RT/s0 flat_aux/rho$RT/s0 final_hier/rho$RT/s0; do
  run probes $C --run $ST/b010/$r
done

# --- diagnostics, outcomes, statistics -------------------------------------------------------
run diagnose baseline $C --runs_root $ST --budgets 0.1 1.0 --out $ST/diag_baseline.json
printf '%s\n' 3 17 29 42 58 77 101 150 222 299 > $R/label_errors.txt
O=$ST/outcomes
S=$ST/stats
for b in 0.1 1.0; do
  run outcomes $C --runs_root $ST --budget $b --rho $RHO --geometry $ST/geometry.json --label_errors $R/label_errors.txt \
    --out $O/b$(pct $b).jsonl
done
for m in T_acc T_C_own T_S_own T_F_true; do
  run stats rank $C --outcomes $O/b100.jsonl --metric $m --out $S/rank_$m.json
done
run stats residual $C --outcomes $O/b100.jsonl --metric T_C_own --degree 1 --out $S/residual_T_C_own.json
run stats residual $C --outcomes $O/b100.jsonl --metric T_C_own --degree 2 --out $S/residual2_T_C_own.json
run stats residual $C --outcomes $O/b010.jsonl --metric T_acc --degree 1 --out $S/residual_b010_T_acc.json
run stats h2 $C --runs_root $ST --budget 1.0 --rho $RHO --n_boot 200 --out $S/h2.json
run stats h3 $C --low $O/b010.jsonl --high $O/b100.jsonl --metric T_acc --n_boot 200 --out $S/h3.json
run stats pilot $C --runs_root $ST --budget 0.1 --rho $RHO --out $S/pilot.json
run stats primary $C --outcomes_primary $O/b100.jsonl --runs_root $ST --budget 1.0 --rho $RHO \
  --low $O/b010.jsonl --high $O/b100.jsonl --n_boot 200 --out $S/primary.json
"$PY" -c 'import json, sys; json.dump(json.load(open(sys.argv[1]))["p_raw"], open(sys.argv[2], "w"))' $S/primary.json $S/p_raw.json
run stats holm $C --pvals $S/p_raw.json --out $S/holm.json

# --- every plan, counted from the smoke state files ------------------------------------------
check "all(v['aux'] == 123 and v['ce'] == 5 for v in d['per_budget'].values()) and len(d['per_budget']) == 2" \
  plan --exp expA $C --count
check "all(v['aux'] == 57 and v['ce'] == 5 for v in d['per_budget'].values()) and len(d['per_budget']) == 4" \
  plan --exp expB $C --count
check "d['n_commands'] == 20" plan --exp ce $C --count
check "d['n_commands'] == 21" plan --exp var_pilot $C --count
check "d['n_commands'] == 256 and d['cmd'] == 'probes'" plan --exp expC $C --count
check "d['n_commands'] == 252" plan --exp rho_sens $C --count
check "d['n_commands'] == 248" plan --exp fixedstep $C --count --sec_per_epoch45k 15
for e in tune_wd ce rho_pilot var_pilot expA expB expC rho_sens fixedstep; do
  run plan --exp $e $C --state_dir $ST --out $ST/plans/$e.txt
done
head -n 1 $ST/plans/expA.txt > $R/one_line_plan.txt
run launch --plan $R/one_line_plan.txt --gpus none

# --- fixed-step training path (total_steps) --------------------------------------------------
for b in 0.1 1.0; do
  p=$(pct $b)
  F="--set budget=$b --set total_steps=3 --set weight_decay=$(wd $ST $b) --set save_model=false"
  for s in 0 1 2 3 4; do
    run train $C $F --set condition=ce --set seed=$s --set run_name=cifar100/b${p}_steps3/ce/s$s
  done
  for s in 0 1; do
    run train $C $F --set condition=true_hier --set rho=$RHO --set seed=$s --set run_name=cifar100/b${p}_steps3/true_hier/rho$RT/s$s
    for i in 00 01 02 03; do
      run train $C $F --set condition=random_hier --set rho=$RHO --set partition=$ST/partitions/b$p/random_$i.json \
        --set seed=$s --set run_name=cifar100/b${p}_steps3/random_hier_random_$i/rho$RT/s$s
    done
  done
done
run outcomes $C --runs_root $ST --budget 1.0 --rho $RHO --total_steps 3 --geometry $ST/geometry.json \
  --out $R/steps3_outcomes_b100.jsonl

# weight-decay guard: reusing a run with a different weight decay must fail
echo "=== weight-decay guard"
if "$PY" -m sg.cli train $C --set budget=1.0 --set total_steps=3 --set condition=ce --set seed=0 \
     --set weight_decay=0.123 --set run_name=cifar100/b100_steps3/ce/s0 >> "$LOG" 2>&1; then
  echo "FAILED: reused a run trained with a different weight decay"; exit 1
fi

# --- run_plan.sh analysis stages against the smoke tree --------------------------------------
rp() {  # rp <stage>: run scripts/run_plan.sh on the smoke tree
  echo "=== run_plan.sh $1"
  if ! PY=$PY OUT=$R SETS="--config configs/smoke.yaml" BUDGETS="0.1 1.0" EXPA_BUDGETS="1.0" N_BOOT=200 PLAN_ONLY=1 \
       ./scripts/run_plan.sh "$1" >> "$LOG" 2>&1; then
    echo "FAILED: run_plan.sh $1"; tail -n 40 "$LOG"; exit 1
  fi
}
rp outcomes   # also runs cover_only
rp stats      # primary, secondary (rank / residual / residual2 for every metric), h2
rp fixedstep  # plan only, then the b100_steps* parse, outcomes --total_steps, residuals, h3
for f in $ST/stats/primary.json $ST/stats/b010_rank_T_acc.json $ST/stats/b100_residual2_T_C_own.json \
         $ST/stats/h2_b100.json $ST/stats/steps3/h3.json $ST/outcomes/b100_expB.jsonl; do
  [ -s "$f" ] || { echo "FAILED: run_plan.sh did not write $f"; exit 1; }
done
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); assert d["alpha_one_sided_familywise"] == 0.025 and set(d["reject"]) == {"H1", "H2", "H3"}, d' \
  $ST/stats/primary.json

# --- iNat-100 path on a fake iNat2018 tree ------------------------------------------------------
STI=$R/inat100
CI="--config configs/smoke.yaml --config configs/inat100.yaml --set out_dir=$R --set data_dir=$R/data_inat \
--set inat_per_class=6 --set inat_test_per_class=2"
echo "=== make_fake_inat"
"$PY" scripts/make_fake_inat.py --out $R/fake_inat >> "$LOG" 2>&1
run inat_build $CI --inat_root $R/fake_inat
for b in 0.25 1.0; do
  p=$(pct $b)
  run simclr $CI --set budget=$b --set run_name=inat100/simclr/b$p
  run partitions $CI --set budget=$b --extractor $STI/simclr/b$p --out_dir $STI/partitions/b$p
done
run simclr $CI --set simclr_scope=all --set run_name=inat100/simclr/all
run diagnose geometry $CI --extractor $STI/simclr/all --partitions_dirs $STI/partitions/b025 $STI/partitions/b100 \
  --out $STI/geometry.json
check "d['n_commands'] == 6" plan --exp inat_tune_wd $CI --out $STI/plans/inat_tune_wd.txt --count
run launch --plan $STI/plans/inat_tune_wd.txt --gpus none --jobs_per_gpu 2
run select_wd $CI --root $STI/tune_wd --out $STI/best_wd.json
cp $ST/best_rho.json $STI/best_rho.json
for s in 0 1; do
  run train $CI --set budget=1.0 --set condition=ce --set seed=$s --set weight_decay=$(wd $STI 1.0) \
    --set run_name=inat100/b100/ce/s$s
  run train $CI --set budget=1.0 --set condition=true_hier --set rho=$RHO --set seed=$s \
    --set weight_decay=$(wd $STI 1.0) --set run_name=inat100/b100/true_hier/rho$RT/s$s
done
run diagnose baseline $CI --runs_root $STI --budgets 1.0 --out $STI/diag_baseline.json
run outcomes $CI --runs_root $STI --budget 1.0 --rho $RHO --geometry $STI/geometry.json --out $STI/outcomes/b100.jsonl
check "d['n_commands'] == 10" plan --exp inat_ce $CI --count
check "all(v['aux'] == 63 and v['ce'] == 5 for v in d['per_budget'].values())" plan --exp inat_A $CI --count \
  --sec_per_epoch45k 15
check "d['n_commands'] == 6" plan --exp inat_tune_wd $CI --count

echo "=== resume test"
PY="$PY" R="$R" bash scripts/resume_test.sh >> "$LOG" 2>&1 || { echo "FAILED: resume test"; tail -n 30 "$LOG"; exit 1; }
echo "smoke OK"
