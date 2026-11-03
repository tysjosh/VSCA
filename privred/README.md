# privred: Privacy Redistribution under Semantic Grouping

Code for the protocol in the privacy-redistribution paper (`../privacy_redistribution/paper/iclr2025_conference.tex`; the
paper wins over `../privacy_redistribution/source_draft.md`). It builds on the companion semantic-grouping study in
`../sg`: privred imports `sg` as a library and reads the companion's split, partitions, SimCLR extractors, weight
decay, ρ and CE runs. It implements the out-of-fold difficulty (`sec:difficulty`), the LiRA attack
with leave-one-model-out scoring (`sec:threat`, `sec:membership`), per-example vulnerability and memorization
(`sec:vulnerability`), Study A (`sec:studyA`), the support features (`sec:support`), Study B with matched removal
sets (`sec:deletion`), the prediction study (`sec:prediction`), the linking analyses (`sec:linking`), the primary
family P1–P5 with Holm (`sec:stats`), the iNat-100 Study A repetition (`app:inat`) and the run count (`app:compute`).

Training and evaluation are meant for a remote GPU machine. The only things run locally are
`scripts/sanity_check.py` and the CPU smoke test `scripts/smoke_all.sh`, which runs every code path on 8 images per
class (4 on a fake iNat tree) for 1 epoch with M = 4 models per set. Its numbers are meaningless, and this README
reports no results.

## Layout

| File | Contents |
|---|---|
| `src/privred/config.py` | `Config` (subclass of `sg.config.Config` with the privacy fields), YAML + `--set` loading, `to_sg`, `wd_for`, `rho_for`, companion paths |
| `src/privred/io.py` | pool D_β, validation V_β and audit A row indices (via `sg.data`), set / member directories, json and npz helpers, `align` |
| `src/privred/membership.py` | membership matrix u (exactly M/2 IN per record), seed registry `seed_base` |
| `src/privred/lira.py` | q from logits, LOMO IN/OUT Gaussians with the variance rule, Λ, τ_α, TPR / AUC, r_i, per-example AUC, offline attack |
| `src/privred/vulnerability.py` | Mem_i and its split-half reliability, ECE, margins, gen gap, C_G / S_G per model, `score_set` → `scores.npz`, `summary.json` |
| `src/privred/depth.py` | prediction depth on arbitrary rows with self-exclusion from the probe reference set, block-8 neighbour purity |
| `src/privred/members.py` | one member model: `sg.train.train` on `train_m{mm}.npy`, per-example outputs, PD, checkpoint deletion |
| `src/privred/difficulty.py` | K class-stratified OOF folds, d0 / loss0 / PD0 / purity, quartiles, confident-error flags, companion check |
| `src/privred/support.py` | φ̄ features, κ, N / U / K / samecls / V, deltas after a removal, brute-force reference |
| `src/privred/studyA.py` | Study A: Δsup, ΔPD, patterns, strata, C·S split, robustness, condition table, T_priv, δ_priv, reuse check, label-error repeat |
| `src/privred/deletion.py` | cohorts, curated siblings, the six removal types, ΔV swap search, deltas and composition |
| `src/privred/studyB.py` | Δdel, cohort means and contrasts, True-Hier interaction, label-error repeat, prediction rows |
| `src/privred/prediction.py` | g_base vs g_full (ridge with inner GroupKFold, gradient boosting), four grouped split kinds, MAE / R² / precision@top |
| `src/privred/linking.py` | OLS interactions with d0 and PD0 (bootstrap CIs), PD0 key test |
| `src/privred/stats.py` | bootstrap p / CI, `residual_cov`, LOPO residuals, P1 two-way bootstrap, cohort bootstrap, sign-flip, P5 bootstrap, Holm, `primary` |
| `src/privred/plans.py` | set registry, `make_sets` for each stage, plan files, `count_runs` (Table compute) |
| `src/privred/cli.py` | `python -m privred.cli <command>` (also installed as `privred`) |
| `configs/base.yaml` | paper defaults (CIFAR-100, β ∈ {100%, 50%}, M = 32, M0 = MB = 16, K = 5, η = 0.04, Q = 4) |
| `configs/smoke.yaml` | tiny CPU configuration (use after `base.yaml`) |
| `configs/inat100.yaml` | iNat-100 Study A repetition (use after `base.yaml`) |
| `scripts/sanity_check.py` | 50 identity and unit checks on synthetic data, PASS / FAIL per line |
| `scripts/smoke_all.sh` | CPU end-to-end run of every command and every `run_plan.sh` stage, ending with `smoke OK` |
| `scripts/run_plan.sh` | the full protocol on the GPU machine, by stage |

## Setup

All projects share one environment at the workspace root (`<root>/.venv`); `setup_env.sh` installs every
`*/pyproject.toml` (sg and privred) in editable mode.

```bash
cd <root> && ./setup_env.sh
cd privred
../.venv/bin/python scripts/sanity_check.py   # 50 PASS lines, exit 0
bash scripts/smoke_all.sh                     # CPU only, about 1.5 h; ends with "smoke OK"
```

The smoke test builds its own mini companion tree with `python -m sg.cli` under `runs_smoke/companion` (it reads
CIFAR-100 from `../sg/data`, never downloads it and never runs `sg prepare`), writes stub `best_wd.json` /
`best_rho.json`, and deletes every `*.pt` at the end.

## Companion prerequisites

privred reads, never rebuilds, the companion artifacts under `companion_dir` (default `../sg/runs`, sg's `out_dir`).
Run these `../sg/scripts/run_plan.sh` stages first:

| sg stage | privred needs |
|---|---|
| `prepare` | `data/cifar-100-python`, `data/cifar100_splits.json` (the shared P / V / A split) |
| `simclr` | `cifar100/simclr/b100/{backbone.pt,class_means.npy}` (φ̄, trained on exactly D_1.0 = P) |
| `partitions` | `cifar100/partitions/b100`, `b050`: `true.json`, `random_XX.json`, `spectrum_XX.json`, `index.json` |
| `diag_geometry` | `cifar100/geometry.json` (optional; W_analysis for δ_priv, else W_train is used) |
| `tune_wd` | `cifar100/best_wd.json` (keys 0.5 and 0.25 for the size-matched privacy models, 1.0 and 0.5 for OOF) |
| `ce` | `cifar100/b{100,050}/ce/s{0..4}/test.jsonl` (difficulty check) and `b{050,025}/ce` (reuse check) |
| `rho_pilot` | `cifar100/best_rho.json` |
| `expB` | `cifar100/b{050,025}/true_hier/rho{ρ}/s0/test.jsonl` (reuse check) |
| `inat_build`, `inat` | `data/inat100/`, `inat100/simclr/b100`, `inat100/partitions/b100`, `inat100/best_wd.json` (0.25, 1.0), `inat100/best_rho.json` |

`privred check_companion` (stage `check`) lists anything missing and exits 1 if so.

## Running the protocol

`scripts/run_plan.sh <stage>` on the GPU machine, in this order:

| Stage | What it does |
|---|---|
| `check` | `check_companion`: φ̄, partitions, best_wd, best_rho, the 4 + 4 subset, reuse runs |
| `features` | φ̄ features of every pool, validation and audit row → `cifar100/features.npz` |
| `oof` | per budget: plan and launch the K = 5 OOF fold models (set `oof`) |
| `difficulty` | per budget: `difficulty.npz`, `difficulty_check.json` |
| `support` | per budget: `support.npz` (G_true on D_β) |
| `studyA` | per budget: plan + launch all Study A sets (12 conditions × M = 32), score each set, `studyA` analysis |
| `prelim` | at `STUDYB_BUDGET`: preliminary audit set (M0 = 16 CE models), score |
| `deletion_sets` | cohorts and the removal sets with ΔV matching → `b100/deletion/` |
| `studyB` | plan baseB (2 × MB) and deletion sets (CE 4 × 6, True-Hier 2 × 4, × MB) as one plan, launch, score every set, `studyB` analysis |
| `predict` | `prediction.json` (g_base vs g_full, ridge and GBT, cohort / experiment / class / parent splits) |
| `linking` | `linking.json` (interactions with d0 and PD0, PD0 key test) |
| `primary` | `primary.json` (P1–P5, raw and Holm-adjusted p) |
| `inat_check`, `inat_features` | the same for iNat-100 (`configs/inat100.yaml`) |
| `inat_oof` | iNat OOF fold models, difficulty, support |
| `inat_studyA` | iNat Study A: CE, True-Hier, 4 + 4 subset at β = 100%, M = 16; score; analysis |
| `count` | `plan --count` for the CIFAR-100 and iNat-100 configs |

Environment: `PY` (default `../.venv/bin/python`), `OUT` (default `runs`), `COMPANION` (default `../sg/runs`),
`GPUS` (default `0`, or `none`), `JOBS_PER_GPU` (default 1), `SETS` (extra arguments for every command),
`INAT_SETS` (extra arguments for the iNat commands), `N_BOOT`, `BUDGETS` (default `"1.0 0.5"`),
`STUDYB_BUDGET` (default `1.0`), `PLAN_ONLY=1` (write plans, do not launch; the smoke test uses it).

Single commands (each takes `--config` and `--set key=value`, repeatable, and prints JSON):
`check_companion`, `plan --stage {oof,studyA,prelim,baseB,deletion} --budget B [--names a,b] --out F`,
`plan --count`, `member --set_dir D --m m`, `launch --plan F --gpus G --jobs_per_gpu J`, `score --set_dir D`,
`difficulty --budget B`, `features`, `support --budget B`, `studyA --budget B`, `deletion_sets`, `studyB`,
`predict`, `linking`, `primary`. A rerun member is a no-op; an existing set with a different manifest is an error
(delete the set directory to rebuild it).

### Outputs

`$OUT/<dataset>/` holds `features.npz` and per budget `b{pct}/`:
`sets/<name>/{manifest.json,pool_ids.npy,membership.npy,example_sets*.json,train_m{mm}.npy,scores.npz,summary.json}`
with members `m{mm}/` (sg run files, `examples_{pool,audit,heldout}.npz`, `pd_<set>.npz`; `model.pt` is deleted after
PD unless `keep_models`), `difficulty.npz`, `difficulty_check.json`, `support.npz`, `deletion/` (`cohorts.json`,
`q{q}_{type}/{R.npy,deltas.npz,composition.json}`), `plans/`, and `analysis/` (`studyA_b{pct}.json`,
`studyA_b{pct}_records.npz`, `studyB.json`, `studyB_contrasts.npz`, `rows_{ce,true_hier}.npz`, `prediction.json`,
`linking.json`, `primary.json`). Set names: `oof`, `A_<cond>`, `A_random_hier_random_XX`,
`A_spectrum_hier_spectrum_XX`, `prelim`, `baseB_<cond>`, `del_<cond>_q{q}_<type>`.

## Paper → code

| Paper | Code |
|---|---|
| Eq. partition (balanced 20 × 5 partition), G_true | `sg.partitions.true_partition`, companion `partitions/b{pct}/*.json` |
| Eq. train (f = Train(D, γ; s)) | `members.run_member` → `sg.train.train` with `train_indices`, seed = `seed_base + m` |
| Eq. fine, Eq. objective, λ_s and ρ (reused) | `sg.train` unchanged; ρ from `best_rho.json` (`config.rho_for`) |
| Eq. classmean (μ_c) | companion `simclr/b100/class_means.npy`, copied into `features.npz` (`class_means`) |
| Eq. w (W(G)) | companion `geometry.json` (W_analysis) or `partitions/index.json` (W_train), `studyA._w_table` |
| Eq. factor (Acc = C_G S_G, true / own frame) | `sg.metrics.outcomes_from_preds` in `vulnerability.score_set` (`condition_metrics`); own frame vs CE-B on the same G: `studyA._ce_own_frame` (`conditions.<set>.own_frame_vs_ce`) |
| Eq. pd (prediction depth, fixed probe set, self-exclusion) | `depth.run_pd`, `depth.pd_from_feats`, `depth.exclude_self`; reference = D_{probe_ref_budget} |
| Eq. residual (δ, PI, Z(G) covariate, LOPO, extrapolation) | `sg.stats.residual`, `stats.residual_cov`, `stats.lopo_residuals`, `stats.extrapolation`, `studyA._delta_priv`, `studyA._probe_z` |
| Eq. residual, descriptive rank of G_true among the random partitions | `sg.stats.rank_pvalue` in `studyA._delta_priv` (`delta_priv.rank_random`) |
| sec:stats, partition-level intervals with resampling over partitions | `stats.partition_bootstrap` in `studyA._delta_priv` (`delta_priv.partition_bootstrap`) |
| Eq. q (logit-scaled confidence, 2 views) | `sg.train._example_stats` (`q`), `lira.q_from_logits`; Q[i, m] = mean over views |
| Eq. lira (Λ), variance rule | `lira.lomo_fits` (LOMO, per-record iff M/2 − 1 ≥ `min_ref_per_side`, else pooled), `lira.lira_scores` |
| τ_α, reportable rule, TPR / AUC, offline attack | `lira.threshold`, `lira.attack_metrics` (`min_expected_fp`), `lira.offline_attack` |
| Score distributions by difficulty stratum and class (sec:threat) | `summary.json`: `pool_lambda_by_class`, `pool_lambda_by_difficulty_quartile` (pool Λ, member / nonmember pairs), `audit_offline_by_class`, `audit_offline_by_difficulty_quartile` |
| Eq. vuln (r_i), per-example AUC | `lira.vulnerability`, `lira.per_example_auc`; `scores.npz` `r_<α>`, `auc_i` |
| Eq. mem (Mem_i), reliability | `vulnerability.mem_scores`, `vulnerability.mem_reliability` |
| Membership design (u, M/2 IN, D_m) | `membership.membership_matrix`, `plans.make_sets` |
| OOF difficulty d0, PD0, strata, confident errors | `difficulty.oof_folds`, `difficulty.compute_difficulty`, `difficulty.confident_errors` |
| Eq. dsup (Δsup), Eq. dpd (ΔPD) | `studyA.analyze` (`dsup_mean`, `dpd_mean`, `studyA_b{pct}_records.npz`) |
| Four patterns, strata, C·S split, robustness of RQ2 | `studyA.pattern_codes`, `pattern_fractions`, `cs_outcome`, `cs_split`, `robustness`; d0 strata `strata`, PD0 sensitivity strata `strata_pd` |
| Condition-contrast intervals (exploratory) | Study A `dsup_ci` (`stats.p1_bootstrap` per condition vs CE); Study B `contrasts.<cond>.<contrast>.bootstrap` and `true_hier_interaction.bootstrap` (`stats.cohort_record_bootstrap`) |
| T_priv, δ_priv | `studyA.analyze` (`T_priv`, `delta_priv`) |
| Eq. siblings (N_i), Eq. support (U_i), Eq. mass (K_i), Eq. vdens (V_i), same-class support, sibling membership | `support.support`, `support.removal_stats` (`sib_frac`), `support.support_bruteforce` |
| Eq. delete (D_β^{−R}) | `plans.make_sets('deletion')`: pool = D_β minus `R.npy` |
| Eq. ddel (Δdel) | `studyB.analyze` (deletion set r_i minus baseB set r_i) |
| Cohorts, Y_sib | `deletion.cohorts`, `deletion.siblings` |
| Eq. lambda0 (Λ̄⁰, r_i⁰, Mem_i⁰) | `prelim` set `scores.npz` (`Lbar_in`, `r_0.001`, `mem`) |
| Removal sets rand / sem / vis / hv / near / far | `deletion.make_removals`, `deletion.vis_classes`, `deletion.removal_size` |
| ΔV matching | `deletion.DVIndex`, `deletion.match_dv` (`cohorts.json` `match_vis`) |
| Eq. dsupport (ΔU), ΔN, ΔK, ΔV | `support.removal_stats` → `deletion/q{q}_{type}/deltas.npz` |
| Eq. target (Y), Eq. xbase, Eq. xsem | `studyB.analyze` → `rows_<cond>.npz` (`target`, `X_base` / `BASE_NAMES`, `X_sem` / `SEM_NAMES`) |
| Eq. gbase, Eq. gfull | `prediction.fit_ridge`, `prediction.predict_split`, `prediction.evaluate` (same family for both) |
| Prediction splits and metrics | `prediction.make_splits` (`cohort`, `experiment`, `class`, `parent`), `prediction.metrics`, `precision_top`; record-disjoint variant `conditions.<cond>.record_disjoint` |
| Linking (sec:linking) | `linking.interaction_fit`, `linking.link` (`key_test` = cohort split with PD0 in both sets) |
| P1 (TPR_TrueHier − TPR_CE at 0.1%) | `stats.p1_bootstrap` (two-way bootstrap), `studyA_b100.json` `P1` |
| P2 (δ_priv of G_true) | `studyA._delta_priv`, two-sided p |
| P3 (sem − vis), P4 (sem − rand), CE | `stats.cohort_record_bootstrap`, `stats.sign_flip_p` (sensitivity) |
| P5 (MAE_base − MAE_full, leave-one-cohort-out) | `stats.p5_bootstrap` over held-out cohorts |
| Holm over P1–P5 | `stats.holm_family` → `sg.stats.holm` at `alpha_family`; `stats.primary` → `primary.json` |
| Table conditions | `config.STUDYA_CONDITIONS`, `plans._studyA_units`, `plans.select_subset` (4 lowest-index random, 4 percentile-closest spectrum) |
| Table shared | companion reads in `config.py` (`companion_root`, `partitions_dir`, `simclr_dir`, `geometry_path`, `ce_run`, `true_hier_run`, `wd_for`, `rho_for`) |
| Table compute | `plans.count_runs`, `privred plan --count` |
| Appendix inat | `configs/inat100.yaml`, `run_plan.sh` `inat_*` stages |

## Compute estimates (assumptions, not measurements)

`privred plan --count --config configs/base.yaml` is pure arithmetic over the config under the paper's assumption
t(n) = 0.35 · (n / 20,000) A100-hours for one 200-epoch run on n training images. These numbers are an
**assumption, to be replaced by the per-run `compute.json` of a pilot**; feature extraction, PD and scoring are not
included.

| Component (β = 100%) | Runs | n per run | A100-h (assumed) |
|---|---|---|---|
| OOF difficulty (K = 5) | 5 | 32,000 | 2.8 |
| Study A (12 conditions × M = 32) | 384 | 20,000 | 134.4 |
| Preliminary audit (M0 = 16) | 16 | 20,000 | 5.6 |
| Study B baselines (2 × MB = 16) | 32 | 20,000 | 11.2 |
| Study B CE deletion (4 cohorts × 6 types × 16) | 384 | 19,200 | 129.0 |
| Study B True-Hier deletion (2 × 4 × 16) | 128 | 19,200 | 43.0 |
| **Total at β = 100%** | **949** | | **326.0** |

β = 50% adds Study A (384 runs, 67.2 h) and OOF (5 runs, 1.4 h): 389 runs, 68.6 h; both budgets 1,338 runs, 394.6 h.
iNat-100 (`--config configs/base.yaml --config configs/inat100.yaml`): Study A 160 runs (10 conditions × 16) and
5 OOF runs; the arithmetic gives 33.6 h + 1.7 h with n from `inat_per_class`, but 64 × 64 images are not
costed by resolution, so this figure is a placeholder until the pilot.

## sg diff

privred needs exactly two additions to sg. Defaults keep sg's behavior; `sg/scripts/sanity_check.py` (50 PASS) and
`sg/README.md` are not edited. `diff -ru -x __pycache__ .agents/tasks/privred/sg_baseline/src sg/src` touches three
files, 54 added lines and one changed line:

1. **Explicit training indices.** `sg/src/sg/config.py`: `train_indices: Optional[str] = None` after `limit_test`.
   `sg/src/sg/data.py`: `load_index_file(path)` (`.npy` or JSON list → int64) after `limit_per_class`.
   `sg/src/sg/train.py`: the `from .data import ...` line also imports `load_index_file` (the one changed line);
   after `tr, va = split_indices(cfg, ds)`, if `train_indices` is set, `tr` is replaced by the file's rows
   (ValueError if empty, duplicated, out of range or overlapping `va`; per-class limits are not applied).
2. **Per-example outputs.** `config.py`: `example_sets: Optional[str] = None` (JSON `{name: [row ids]}`) and
   `example_logits: bool = False`. `train.py`: a pure helper `_example_stats(ids, y, logits_v0, logits_v1,
   keep_logits)` and, after the `eval` phase, a ComputeLog phase `examples` that predicts each set on the image and
   its horizontal flip and writes `examples_<name>.npz`: `ids`, `y`, and `[n, 2]` arrays `q` (stable
   `z_y − logsumexp_{y'≠y} z_{y'}`), `loss`, `p_true`, `margin`, `conf`, `pred` (int16), plus `logits` float16
   `[n, 2, C]` iff `example_logits`. Only runs with `example_sets` set do this.

## Deviations and choices

Design decisions made in the plan and implemented here:

- **Index universe.** Every index is a row of the official 50,000-image training array; D_β, V_β and A come from
  `sg.data.split_indices` / `audit_split` / `limit_per_class`, never re-implemented.
- **φ̄** is the companion's `simclr/b100` extractor (trained on exactly D_1.0 = P, 40,000 images), not sg's `all`
  extractor (45,000 images).
- **Weight decay.** Privacy models (n_β / 2 images) use `best_wd[β/2]`; OOF fold models use `best_wd[β]`. A missing
  key falls back to the available budget nearest in log-size, ties to the smaller (iNat has only 0.25 and 1.0, so
  β/2 = 0.5 maps to 0.25).
- **Study A subset**: from `sg.plans.cover_names` at budget β, the 4 random partitions with the lowest generation
  index and, for percentiles 10/37/63/90 of W_train among the 9 spectrum partitions, the closest not-yet-chosen one.
- **Membership**: exactly M/2 IN per record via argsort of iid uniforms with `default_rng([membership_seed,
  crc32(name)])` (same distribution as a per-row permutation, vectorized). Seeds: `seed_base + m` with `seed_base =
  1000 · (set position in plans.all_set_names) + 100,000 · (budget index) + 1,000,000 · (dataset index)`; subset
  placeholder slots keep their positions, so seeds do not depend on companion files. Disjointness is a sanity check.
- **LiRA variance rule**: per-record variances iff M/2 − 1 ≥ `min_ref_per_side` (= the paper's "for every target");
  pooled variances are the mean of the LOMO per-record variances over records whose side has ≥ 2 models; variance
  floor 1e-8. τ_α pooled over targets; `reportable` iff α · #nonmember pairs ≥ 100.
- **Offline attack**: z = (q − μ_out^{−m}) / σ_out^{−m}, τ calibrated on audit pairs (audit records are OUT in
  every model).
- **Mem_i** uses view-0 p_true over all models; reliability = Pearson / Spearman between even and odd models.
- **PD**: probe reference set = D at `probe_ref_budget` (10%); k + 1 neighbours, the query's own row dropped when it
  is in the reference set.
- **OOF difficulty**: K class-stratified folds (per class permutation by `default_rng([membership_seed, 7])`,
  round-robin); quartile 0 = lowest d0 = hardest; PD strata sorted by PD0 desc, loss desc, purity asc;
  confident-error flag: OOF argmax ≠ y and its OOF prob ≥ the mean OOF prob of that class over records labelled
  with it. The check against companion CE uses the CE-A rotation of `sg.metrics.crossfit_folds`. OOF is also run
  at β = 50% and on iNat-100 (strata and label-error repeat need it there).
- **Support**: κ = exp((cos − 1) / t), t = 0.1, ε = 1e-6, V over the top `n_vnn` = 50 of D \ {i}; deltas by
  recomputation on D \ R; ΔK = −Σ κ over removed siblings is checked equal.
- **Study A**: pattern tolerances `pattern_tol_r = pattern_tol_pd = 0` (exact); C·S split by the modal outcome in
  the CE OUT models; robustness within records with equal modal OUT correctness; δ_priv over the 8 non-curated
  partitions with W_analysis from `geometry.json` if present, else W_train (source recorded); P2 two-sided
  p = 2 min(p, 1 − p); Z(G) = linear-probe accuracy of the 20-way task on φ̄ features trained on D_β and tested on
  V_β (A is never used by any fitted model); reuse check against companion runs at β/2.
- **Deletion**: cohorts with `default_rng([cohort_seed, pct])`, sequential draw rejecting a class whose superclass is
  already in the cohort; |R| = round(η n_β) must be a multiple of 16; per-type rngs `[cohort_seed, q,
  crc32(type)]`; vis classes = 16 classes outside C_q ∪ Y_sib by max cosine of φ̄ class means to a cohort class
  (ties by class index); hv ties by smaller row index. ΔV matching on candidate lists of the top n_vnn + |R|
  neighbours; every proposal (including skipped ones) counts toward `dv_max_swaps` = 10^4 ("10^4 swaps" read as
  proposals); stop when the gap < 5% of |mean ΔV(sem)|; `reached` and the final gap are recorded.
- **Removal-set layout**: `deletion/q{q}_{type}/R.npy` (one directory per set, with `deltas.npz` and
  `composition.json`) instead of `deletion/q{q}_{type}.npy`. Once any `del_*` set exists, `deletion_sets` refuses
  to rewrite an `R.npy` whose contents would change (an identical rewrite is allowed). Every `(q, type)` is
  checked before any file is written, so a refusal leaves `deletion/` untouched.
- **δ_priv intervals and rank**: P2 uses the Eq. residual prediction interval. `delta_priv.partition_bootstrap` adds
  the exploratory partition-level interval: each draw resamples records (paired across sets) and target models
  within each set, recomputes T_priv for G_true and the 8 non-curated partitions, resamples those partitions with
  replacement and refits δ_priv (degree 1, seed 9, `n_boot` draws; draws with fewer than 3 distinct W are skipped,
  `n_valid` counts the rest; reference fits are not refitted). `delta_priv.rank_random` is the rank of T_priv(G_true)
  among G_true and the 4 random partitions (1 = largest, 5 positions) with `sg.stats.rank_pvalue`; descriptive only.
- **Own frame**: CE has no training partition, so for each hierarchy condition with a `partition.json` (True-Hier,
  Final-Hier, subset) CE-B's per-model test predictions are scored under that condition's G and averaged over the CE
  models; `own_frame_vs_ce` holds CE's C_G / S_G / F_G and the condition minus CE. Flat-Aux has no own frame.
- **Pool score distributions** summarize Λ over (record, target) pairs, split into member and nonmember pairs, per
  class and per pool d0 quartile (mean, sd, 10/50/90% quantiles).
- **Sibling membership** descriptor = |R ∩ N_i| / |R| (`sib_frac`).
- **Study B rows**: one row per (record, deletion set) over every record retained under all removal sets of the
  cohort (D_β minus the union of the cohort's R, T_q included and flagged `in_T`); with T_q-only rows the class and
  parent splits would equal the cohort split. Consequence: in the cohort and experiment splits most records appear
  in both training rows (other cohorts) and test rows, and their targets share the −r_i(baseB) term, which a flexible
  learner (GBT) can partly memorize. P5 keeps the paper's leave-one-cohort-out split; `prediction.json` also reports
  a record-disjoint variant (`record_disjoint`, cohort and experiment splits, both families) that drops training rows
  whose record id is in the test fold. Its cohort-split training set holds only the held-out cohort's removed
  records, so it is small. The class and parent splits are record-disjoint by construction. Class dispersion = mean 1 − cos(e_i, class mean of e) over the
  class's D_β records; class OOF accuracy = mean `oof_correct`; r0 = prelim r at α = 0.1%.
- **Prediction**: ridge on standardized features (train-median imputation, zero-sd columns kept with sd 1), α from
  `ridge_alphas` (13 values 10^{−3} … 10^{3}) by inner GroupKFold over cohorts (experiments if < 2 cohorts; the
  middle α if < 2 groups); `HistGradientBoostingRegressor(random_state=0)` secondary. Class split: 5 class folds
  (`default_rng([cohort_seed, 11])`); parent split by curated parent; every split asserts no shared experiment or
  cohort. precision@top: among the ⌈top_frac · n⌉ largest predictions, the fraction whose true value is ≥ the k-th
  largest true value.
- **Statistics**: P1 bootstrap resamples records (paired across CE and True-Hier) and target-model columns of the
  stored Λ within each set; τ and r_i are recomputed per draw but the reference Gaussians are not refitted (stated
  approximation of "resample models"). P1–P4 two-sided, P5 one-sided (improvement). The same bootstraps give
  exploratory intervals for every other Study A condition vs CE (`dsup_ci`, at the primary α) and for every Study B
  contrast and the True-Hier interaction; these are not in the Holm family. `primary.json` records `n_tests` (the
  p-values Holm adjusts over) and `n_family` = 5, and sets `holm_warning` (also printed to stderr) when a test is
  missing, because Holm over fewer tests gives smaller adjusted p-values. Two-sided p =
  min(1, 2 min((1 + #{b ≤ 0}) / (B + 1), (1 + #{b ≥ 0}) / (B + 1))). Holm via `sg.stats.holm` at 0.05. A test's
  `reportable` flag carries the α rule of the sets it uses.
- **iNat-100**: `privacy_budgets: [1.0]`, `studyB_budget: null` (no Study B), `M: 16`, conditions CE, True-Hier and
  the 4 + 4 subset. The secondary-dataset prediction split is reported as not available (the paper runs only
  Study A on iNat).
- **CPU smoke stubs**: `best_wd.json` / `best_rho.json` are written by the script; the fake iNat tree is built at
  32 × 32 (`inat_size=32`) to save CPU time, without `geometry.json`, so δ_priv on iNat uses W_train there. In the
  1-epoch smoke, r_i at α = 0.1% is almost always 0 and most OOF predictions are wrong (most records are flagged as
  label errors); these numbers are meaningless.

## Not implemented

- Splitting iNat-100 into pool / validation / audit by observation, near-duplicate or photographer group
  (`app:inat`): sg's builder metadata has no observation or photographer ids, so privred uses sg's class-stratified
  split. The dataset-difference report (resolution, imbalance, taxonomy quality) is left to the write-up.
- Before/after-deletion attacks, unlearning algorithms and pretrained backbones (out of scope in the paper).
- Rescaling the compute estimate by measured timings: replace the `count_runs` assumption with pilot `compute.json`.
- If pilot timings exceed the assumption, the paper reduces MB before Q and Q before the number of removal types;
  these are config values (`MB`, `Q`), not automated.
