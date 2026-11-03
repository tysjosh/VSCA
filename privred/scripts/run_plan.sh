#!/usr/bin/env bash
# Full privacy-redistribution protocol (Sections 3-4, Appendix inat) on the remote GPU machine. Run the stages in
# this order (each one needs the outputs of the stages before it):
#   CIFAR-100:  check features oof difficulty support studyA prelim deletion_sets studyB predict linking primary
#   iNat-100:   inat_check inat_features inat_oof inat_studyA
#   compute:    count
#   ./scripts/run_plan.sh <stage>
# The companion study (sg, ../sg/scripts/run_plan.sh) must have run first: prepare simclr partitions diag_geometry
# tune_wd ce rho_pilot expB for CIFAR-100, and inat_build inat for iNat-100 (see README.md).
# Environment:
#   PY            python of the shared venv (default ../.venv/bin/python)
#   OUT           privred output root (default runs); state goes to $OUT/cifar100 and $OUT/inat100
#   COMPANION     sg output root with the companion artifacts (default ../sg/runs)
#   GPUS          comma-separated GPU ids used by `privred launch`, or none (default 0)
#   JOBS_PER_GPU  concurrent runs per GPU (default 1)
#   SETS          extra arguments for every command, e.g. "--set n_boot=5000" (smoke: "--config configs/smoke.yaml")
#   INAT_SETS     extra arguments for the iNat-100 commands only (after SETS)
#   N_BOOT        bootstrap resamples (default: n_boot of the config)
#   BUDGETS       privacy budgets of the per-budget loops (default "1.0 0.5")
#   STUDYB_BUDGET budget of the preliminary audit and Study B (default 1.0)
#   PLAN_ONLY     if 1, write plan files but do not launch them (used by the smoke test)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
OUT=${OUT:-runs}
COMPANION=${COMPANION:-../sg/runs}
GPUS=${GPUS:-0}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
BUDGETS=${BUDGETS:-"1.0 0.5"}
STUDYB_BUDGET=${STUDYB_BUDGET:-1.0}
PLAN_ONLY=${PLAN_ONLY:-0}
NB=""
if [ -n "${N_BOOT:-}" ]; then NB="--set n_boot=$N_BOOT"; fi
C="--config configs/base.yaml --set out_dir=$OUT --set companion_dir=$COMPANION ${SETS:-} $NB"
CI="--config configs/base.yaml --config configs/inat100.yaml --set out_dir=$OUT --set companion_dir=$COMPANION \
${SETS:-} ${INAT_SETS:-} $NB"
ST=$OUT/cifar100
STI=$OUT/inat100

pr() { "$PY" -m privred.cli "$@"; }

pct() {
  case "$1" in
    0.1) echo 010 ;; 0.25) echo 025 ;; 0.5) echo 050 ;; 1.0) echo 100 ;;
    *) echo "unknown budget $1" >&2; return 1 ;;
  esac
}

launch() {  # launch <plan file>: run it over the GPUs unless PLAN_ONLY=1
  if [ "$PLAN_ONLY" = 1 ]; then echo "PLAN_ONLY: wrote $1, not launching"; return 0; fi
  pr launch --plan "$1" --gpus "$GPUS" --jobs_per_gpu "$JOBS_PER_GPU"
}

run_exp() {  # write a plan for a stage and launch it: run_exp <stage> <budget> <state dir> <config args...>
  local stage=$1 b=$2 st=$3
  shift 3
  local f=$st/plans/${stage}_b$(pct "$b").txt
  pr plan "$@" --stage "$stage" --budget "$b" --out "$f"
  launch "$f"
}

score_sets() {  # score every set matching a glob: score_sets <glob> <config args...>
  local pattern=$1
  shift
  local found=0
  for s in $pattern; do
    [ -d "$s" ] || continue
    found=1
    pr score "$@" --set_dir "$s"
  done
  [ "$found" = 1 ] || { echo "no sets match $pattern"; exit 1; }
}

case "${1:-}" in
check)  # companion artifacts present (phi-bar, partitions + index.json, best_wd, best_rho, CE / True-Hier reuse runs)
  pr check_companion $C
  ;;

features)  # phi-bar features of every pool, validation and audit row (features.npz)
  pr features $C
  ;;

oof)  # Section 3.3: K fold models per privacy budget (heldout predictions, PD on the held-out fold)
  for b in $BUDGETS; do run_exp oof "$b" "$ST" $C; done
  ;;

difficulty)  # d0, PD0, quartiles, confident-error flags, check against companion CE seeds
  for b in $BUDGETS; do pr difficulty $C --budget "$b"; done
  ;;

support)  # N, U, K, samecls, V for G_true on D_beta (Eqs. siblings, support, mass, vdens)
  for b in $BUDGETS; do pr support $C --budget "$b"; done
  ;;

studyA)  # Section 3.7: M models per condition and budget, scores, Study A analysis (P1, P2 inputs)
  for b in $BUDGETS; do
    p=$(pct "$b")
    run_exp studyA "$b" "$ST" $C
    score_sets "$ST/b$p/sets/A_*" $C
    pr studyA $C --budget "$b"
  done
  ;;

prelim)  # preliminary audit (M0 CE models) for the hv removal type
  run_exp prelim "$STUDYB_BUDGET" "$ST" $C
  score_sets "$ST/b$(pct "$STUDYB_BUDGET")/sets/prelim" $C
  ;;

deletion_sets)  # cohorts, the six removal types with Delta-V matching, deltas and composition (Section 3.9)
  pr deletion_sets $C --budget "$STUDYB_BUDGET"
  ;;

studyB)  # baseline and deletion sets (one plan), scores, Delta_del, contrasts and prediction rows
  p=$(pct "$STUDYB_BUDGET")
  mkdir -p "$ST/plans"
  pr plan $C --stage baseB --budget "$STUDYB_BUDGET" --out "$ST/plans/baseB_b$p.txt"
  pr plan $C --stage deletion --budget "$STUDYB_BUDGET" --out "$ST/plans/deletion_b$p.txt"
  cat "$ST/plans/baseB_b$p.txt" "$ST/plans/deletion_b$p.txt" > "$ST/plans/studyB_b$p.txt"
  launch "$ST/plans/studyB_b$p.txt"
  score_sets "$ST/b$p/sets/baseB_*" $C
  score_sets "$ST/b$p/sets/del_*" $C
  pr studyB $C --budget "$STUDYB_BUDGET"
  ;;

predict)  # Section 3.10: g_base vs g_full, ridge and gradient boosting, four grouped split kinds
  pr predict $C --budget "$STUDYB_BUDGET"
  ;;

linking)  # Section 3.11: difficulty-redistribution interactions and the PD0 key test
  pr linking $C
  ;;

primary)  # Section 4.3: P1-P5 with Holm at alpha_family
  pr primary $C
  ;;

inat_check)  # iNat-100 companion artifacts (sg inat_build, simclr, partitions, best_wd, best_rho)
  pr check_companion $CI
  ;;

inat_features)
  pr features $CI
  ;;

inat_oof)  # OOF difficulty and support on iNat-100 (strata and label-error repeat)
  run_exp oof 1.0 "$STI" $CI
  pr difficulty $CI --budget 1.0
  pr support $CI --budget 1.0
  ;;

inat_studyA)  # Appendix inat: CE, True-Hier and the 4 + 4 subset at beta = 100%, M = 16
  run_exp studyA 1.0 "$STI" $CI
  score_sets "$STI/b100/sets/A_*" $CI
  pr studyA $CI --budget 1.0
  ;;

count)  # runs and A100-hours (Table compute), arithmetic under t(n) = gpu_h_per_20k * n / 20000 (assumption)
  pr plan $C --count
  pr plan $CI --count
  ;;

*)
  echo "usage: $0 {check|features|oof|difficulty|support|studyA|prelim|deletion_sets|studyB|predict|linking|primary|inat_check|inat_features|inat_oof|inat_studyA|count}"
  exit 1
  ;;
esac
