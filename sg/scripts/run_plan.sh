#!/usr/bin/env bash
# Full protocol (Sections III-IV) on the remote GPU machine. Run the stages in this order:
#   CIFAR-100:  prepare simclr partitions diag_geometry tune_wd ce diag_baseline rho_pilot var_pilot
#               expA expB expC outcomes stats rho_sens fixedstep
#   iNat-100:   inat_build inat
#   ./scripts/run_plan.sh <stage>
# Environment:
#   PY              python of the shared venv (default ../.venv/bin/python)
#   OUT             output root (default runs); state files go to $OUT/cifar100 and $OUT/inat100
#   GPUS            comma-separated GPU ids used by `sg launch` (default 0)
#   JOBS_PER_GPU    concurrent runs per GPU (default 1)
#   PRIMARY_BUDGET  budget of the confirmatory H1 / H2 tests (D10; default 1.0)
#   LABEL_ERRORS    optional file of CIFAR-100 test ids flagged as mislabeled (json list or one id per line)
#   INAT_ROOT       iNaturalist 2018 root with train2018.json, categories.json, train_val2018/ (inat_build)
#   SETS            extra overrides for every command, e.g. "--set n_random=29 --set n_spectrum=29"
#   N_BOOT          bootstrap resamples (default 10000)
#   BUDGETS         budgets of the per-budget loops (default "0.1 0.25 0.5 1.0"; smoke test only)
#   EXPA_BUDGETS    Exp A budgets used by the outcomes/stats loops (default "0.25 1.0"; smoke test only)
#   PLAN_ONLY       if 1, write plan files but do not launch them (used by the smoke test)
#   SESSION_HOURS / MARGIN_MIN / SESSION_DEADLINE   stop cleanly before the machine session ends
#                   (`source scripts/session_start.sh` at login). Runs then save state and exit with code 75;
#                   rerun the same stage in the next session and finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
OUT=${OUT:-runs}
GPUS=${GPUS:-0}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
PRIMARY_BUDGET=${PRIMARY_BUDGET:-1.0}
LABEL_ERRORS=${LABEL_ERRORS:-}
INAT_ROOT=${INAT_ROOT:-}
N_BOOT=${N_BOOT:-10000}
C="--config configs/base.yaml --set out_dir=$OUT ${SETS:-}"
CI="--config configs/base.yaml --config configs/inat100.yaml --set out_dir=$OUT ${SETS:-}"
ST=$OUT/cifar100
STI=$OUT/inat100
BUDGETS=${BUDGETS:-"0.1 0.25 0.5 1.0"}
EXPA_BUDGETS=${EXPA_BUDGETS:-"0.25 1.0"}
PLAN_ONLY=${PLAN_ONLY:-0}
SESSION_HOURS=${SESSION_HOURS:-10}
MARGIN_MIN=${MARGIN_MIN:-20}
if [ -z "${SESSION_DEADLINE:-}" ] && [ "$PLAN_ONLY" != 1 ]; then
  SESSION_DEADLINE=$(( $(date +%s) + SESSION_HOURS * 3600 - MARGIN_MIN * 60 ))
fi
export SESSION_DEADLINE
trap 'rc=$?; if [ $rc -eq 75 ]; then echo "[session] deadline reached; rerun the same stage in the next session"; fi' EXIT
METRICS="T_acc T_C_true T_S_true T_F_true T_C_own T_S_own T_F_own"

sg() { "$PY" -m sg.cli "$@"; }

pct() {
  case "$1" in
    0.1) echo 010 ;; 0.25) echo 025 ;; 0.5) echo 050 ;; 1.0) echo 100 ;;
    *) echo "unknown budget $1" >&2; return 1 ;;
  esac
}

rho() {  # rho selected by the rho_pilot stage
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["rho"])' "$1/best_rho.json"
}

run_exp() {  # write a plan and run it over the GPUs: run_exp <plan> <state dir> <config args...>
  local exp=$1 st=$2
  shift 2
  sg plan --exp "$exp" --state_dir "$st" --out "$st/plans/$exp.txt" "$@"
  if [ "$PLAN_ONLY" = 1 ]; then echo "PLAN_ONLY: wrote $st/plans/$exp.txt, not launching"; return 0; fi
  sg launch --plan "$st/plans/$exp.txt" --gpus "$GPUS" --jobs_per_gpu "$JOBS_PER_GPU"
}

simclr_plan() {  # simclr_plan <dataset> <state dir> <budgets> <config args...>: budget extractors + the `all` extractor
  local ds=$1 st=$2 budgets=$3
  shift 3
  mkdir -p "$st/plans"
  : > "$st/plans/simclr.txt"
  for b in $budgets; do
    echo "$PY -m sg.cli simclr $* --set budget=$b --set run_name=$ds/simclr/b$(pct "$b")" >> "$st/plans/simclr.txt"
  done
  echo "$PY -m sg.cli simclr $* --set simclr_scope=all --set run_name=$ds/simclr/all" >> "$st/plans/simclr.txt"
  sg launch --plan "$st/plans/simclr.txt" --gpus "$GPUS" --jobs_per_gpu 1
}

le_arg() {
  if [ -n "$LABEL_ERRORS" ]; then echo "--label_errors $LABEL_ERRORS"; fi
}

cover_only() {  # keep True-Hier and the 9 + 9 cover partitions (Exp B units): cover_only <dataset> <budget> <in> <out>
  "$PY" - "$OUT" "$@" <<'EOF'
import json, sys
from sg.plans import cover_names, pct
out_dir, ds, budget, src, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
keep = {f"{ds}_b{pct(budget):03d}_{n}" for names in cover_names(out_dir, ds, budget).values() for n in names}
rows = [json.loads(line) for line in open(src) if line.strip()]
rows = [r for r in rows if r["condition"] == "true_hier" or r["partition_id"] in keep]
with open(dst, "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
print(f"{dst}: {len(rows)} rows")
EOF
}

secondary() {  # rank, linear residual and quadratic sensitivity for every metric: secondary <outcomes> <out prefix> <cfg...>
  local f=$1 pre=$2
  shift 2
  mkdir -p "$(dirname "$pre")"
  for m in $METRICS; do
    sg stats rank "$@" --outcomes "$f" --metric "$m" --out "${pre}_rank_$m.json"
    sg stats residual "$@" --outcomes "$f" --metric "$m" --degree 1 --out "${pre}_residual_$m.json"
    sg stats residual "$@" --outcomes "$f" --metric "$m" --degree 2 --out "${pre}_residual2_$m.json"
  done
}

case "${1:-}" in
prepare)  # download CIFAR-100, write data/cifar100_splits.json (Table II)
  sg prepare $C
  ;;

simclr)  # SimCLR extractors: one per budget (partitions for training) and one on all 50k images (W_analysis, D9)
  simclr_plan cifar100 "$ST" "$BUDGETS" $C
  ;;

partitions)  # true, 19 random, 19 spectrum partitions per budget (D8)
  for b in $BUDGETS; do
    p=$(pct "$b")
    sg partitions $C --set budget="$b" --extractor "$ST/simclr/b$p" --out_dir "$ST/partitions/b$p"
  done
  ;;

diag_geometry)  # Section IV-B.1
  sg diagnose geometry $C --extractor "$ST/simclr/all" \
    --partitions_dirs "$ST/partitions/b010" "$ST/partitions/b025" "$ST/partitions/b050" "$ST/partitions/b100" \
    --out "$ST/geometry.json"
  ;;

tune_wd)  # D6: CE seed 0, wd in {1e-4, 5e-4, 1e-3}, per budget
  run_exp tune_wd "$ST" $C
  sg select_wd $C --root "$ST/tune_wd" --out "$ST/best_wd.json"
  ;;

ce)  # five CE seeds per budget (strata reference models, D3)
  run_exp ce "$ST" $C
  ;;

diag_baseline)  # Section IV-B.2
  sg diagnose baseline $C --runs_root "$ST" --budgets 0.25 1.0 --out "$ST/diag_baseline.json"
  ;;

rho_pilot)  # D5: rho in {0.25, 0.5, 1} at 100%, seed 0
  run_exp rho_pilot "$ST" $C
  sg select_rho $C --root "$ST/b100" --out "$ST/best_rho.json"
  ;;

var_pilot)  # Section IV-B.3, D16
  run_exp var_pilot "$ST" $C
  sg stats pilot $C --runs_root "$ST" --budget 0.25 --rho "$(rho "$ST")" --out "$ST/var_pilot.json"
  echo "If recommend_R_at_3_seeds in $ST/var_pilot.json exceeds 19, rerun from the partitions stage with"
  echo "SETS=\"--set n_random=R --set n_spectrum=R\" before adding seeds."
  ;;

expA|expB|expC)  # Exp A (25%, 100%), Exp B (all budgets, cover subset), Exp C (probes on the Exp A runs)
  run_exp "$1" "$ST" $C
  ;;

outcomes)  # seed-paired T(G) per partition unit, strata, label-error exclusion
  RHO=$(rho "$ST")
  mkdir -p "$ST/outcomes"
  for b in $BUDGETS; do
    sg outcomes $C --runs_root "$ST" --budget "$b" --rho "$RHO" --geometry "$ST/geometry.json" $(le_arg) \
      --out "$ST/outcomes/b$(pct "$b").jsonl"
  done
  for b in $EXPA_BUDGETS; do  # Exp B units at the Exp A budgets, for the H3 contrast
    cover_only cifar100 "$b" "$ST/outcomes/b$(pct "$b").jsonl" "$ST/outcomes/b$(pct "$b")_expB.jsonl"
  done
  ;;

stats)  # H1 / H2 at PRIMARY_BUDGET, H3 (10% vs 100%, Exp B units), Holm; secondary tests for every metric
  RHO=$(rho "$ST")
  O=$ST/outcomes
  S=$ST/stats
  mkdir -p "$S"
  sg stats primary $C --outcomes_primary "$O/b$(pct "$PRIMARY_BUDGET").jsonl" --runs_root "$ST" \
    --budget "$PRIMARY_BUDGET" --rho "$RHO" --low "$O/b010.jsonl" --high "$O/b100_expB.jsonl" --n_boot "$N_BOOT" \
    --out "$S/primary.json"
  for b in $BUDGETS; do
    secondary "$O/b$(pct "$b").jsonl" "$S/b$(pct "$b")" $C
  done
  for b in $EXPA_BUDGETS; do
    sg stats h2 $C --runs_root "$ST" --budget "$b" --rho "$RHO" --n_boot "$N_BOOT" --out "$S/h2_b$(pct "$b").json"
  done
  ;;

rho_sens)  # the two non-selected rho values: True-Hier, Flat-Aux, Final-Hier, 9 + 9 cover partitions, 3 seeds
  run_exp rho_sens "$ST" $C
  BEST=$(rho "$ST")
  for r in 0.25 0.5 1.0; do
    # numeric comparison, so "1" and "1.0" are the same rho
    "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)' "$r" "$BEST" && continue
    for b in 0.25 1.0; do
      p=$(pct "$b")
      d=$ST/outcomes/rho$r
      mkdir -p "$d"
      sg outcomes $C --runs_root "$ST" --budget "$b" --rho "$r" --geometry "$ST/geometry.json" $(le_arg) --out "$d/b${p}_all.jsonl"
      cover_only cifar100 "$b" "$d/b${p}_all.jsonl" "$d/b$p.jsonl"
      secondary "$d/b$p.jsonl" "$ST/stats/rho$r/b$p" $C
    done
  done
  ;;

fixedstep)  # D17: every budget trained for the 100% step count (Exp B units); H3 under fixed steps
  run_exp fixedstep "$ST" $C
  RHO=$(rho "$ST")
  STEP_DIRS=$(ls -d "$ST"/b100_steps* 2>/dev/null || true)
  [ -n "$STEP_DIRS" ] || { echo "fixedstep: no $ST/b100_steps* runs found"; exit 1; }
  [ "$(echo "$STEP_DIRS" | wc -l)" -eq 1 ] || { echo "fixedstep: several step counts found:"; echo "$STEP_DIRS"; exit 1; }
  TS=$(basename "$STEP_DIRS")
  TS=${TS#b100_steps}
  d=$ST/outcomes/steps$TS
  mkdir -p "$d" "$ST/stats/steps$TS"
  for b in $BUDGETS; do
    sg outcomes $C --runs_root "$ST" --budget "$b" --rho "$RHO" --total_steps "$TS" --geometry "$ST/geometry.json" \
      $(le_arg) --out "$d/b$(pct "$b").jsonl"
    sg stats residual $C --outcomes "$d/b$(pct "$b").jsonl" --metric T_acc --out "$ST/stats/steps$TS/b$(pct "$b")_residual_T_acc.json"
  done
  sg stats h3 $C --low "$d/b010.jsonl" --high "$d/b100.jsonl" --metric T_acc --n_boot "$N_BOOT" --out "$ST/stats/steps$TS/h3.json"
  ;;

inat_build)  # D13: needs the iNat2018 training images and json files under INAT_ROOT
  [ -n "$INAT_ROOT" ] || { echo "set INAT_ROOT to the iNaturalist 2018 directory"; exit 1; }
  sg inat_build $CI --inat_root "$INAT_ROOT"
  ;;

inat)  # Section IV-C: Exp A at 25% and 100% with 9 random + 9 spectrum partitions; rho shared with CIFAR-100
  simclr_plan inat100 "$STI" "0.25 1.0" $CI
  for b in 0.25 1.0; do
    p=$(pct "$b")
    sg partitions $CI --set budget="$b" --extractor "$STI/simclr/b$p" --out_dir "$STI/partitions/b$p"
  done
  sg diagnose geometry $CI --extractor "$STI/simclr/all" --partitions_dirs "$STI/partitions/b025" "$STI/partitions/b100" \
    --out "$STI/geometry.json"
  run_exp inat_tune_wd "$STI" $CI
  sg select_wd $CI --root "$STI/tune_wd" --out "$STI/best_wd.json"
  cp "$ST/best_rho.json" "$STI/best_rho.json"
  run_exp inat_ce "$STI" $CI
  sg diagnose baseline $CI --runs_root "$STI" --budgets 0.25 1.0 --out "$STI/diag_baseline.json"
  run_exp inat_A "$STI" $CI
  RHO=$(rho "$STI")
  mkdir -p "$STI/outcomes" "$STI/stats"
  for b in 0.25 1.0; do
    p=$(pct "$b")
    sg outcomes $CI --runs_root "$STI" --budget "$b" --rho "$RHO" --geometry "$STI/geometry.json" --out "$STI/outcomes/b$p.jsonl"
    secondary "$STI/outcomes/b$p.jsonl" "$STI/stats/b$p" $CI
    sg stats h2 $CI --runs_root "$STI" --budget "$b" --rho "$RHO" --n_boot "$N_BOOT" --out "$STI/stats/h2_b$p.json"
  done
  ;;

*)
  echo "usage: $0 {prepare|simclr|partitions|diag_geometry|tune_wd|ce|diag_baseline|rho_pilot|var_pilot|expA|expB|expC|outcomes|stats|rho_sens|fixedstep|inat_build|inat}"
  exit 1
  ;;
esac
