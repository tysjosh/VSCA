# sg: Does Semantic Grouping Matter?

Code for the protocol in the paper on hierarchical auxiliary supervision versus matched class partitions (Overleaf project `6ad990bb16325edf2ba068d1`; local copy in `../semantic_grouping/conference_101719.tex`). It implements every condition of Table I, the matched-partition construction of Section III-D, the training recipe and gradient-norm-matched λ of Section III-E, the cross-fitted strata of Section III-F, the probes and prediction depth of Section III-G, the budgets, diagnostics and Experiments A–C of Section IV, the iNat-100 replication, and the statistics of Section IV-D.

Training and evaluation are meant for a remote GPU machine. The only things run locally are `scripts/sanity_check.py` and the CPU smoke test `scripts/smoke_all.sh`, which runs every code path on 2 images per class for 1 epoch. Its numbers are meaningless, and this README reports no results.

## Layout

| File | Contents |
|---|---|
| `src/sg/config.py` | `Config` dataclass, `CONDITIONS`, `AUX_STAGES`, YAML + `--set key=value` loading |
| `src/sg/utils.py` | seeding, device choice, `ComputeLog` (seconds and GPU-hours per phase in `compute.json`), jsonl helpers |
| `src/sg/data.py` | CIFAR-100 raw-pickle loader (with `coarse_labels`), iNat-100 npz loader, nested Table II budgets, GPU crop/flip augmentation |
| `src/sg/model.py` | CIFAR ResNet-18 (3×3 stem, no max-pool), auxiliary heads (GAP + linear) at stages 2, 3 or 4, the 8 block features |
| `src/sg/partitions.py` | W(G), curated / random / spectrum partitions, swap local search, ARI, cover subsets |
| `src/sg/simclr.py` | SimCLR extractor φ and class means μ_c |
| `src/sg/train.py` | Eq. 4 training loop, λ_s calibration, per-item test outputs, `select_wd`, `select_rho` |
| `src/sg/metrics.py` | Eqs. 1–2 factorization in both frames, cross-fitted strata (Eq. 5), probability curve, label-error exclusion |
| `src/sg/probes.py` | k-NN and linear probes at 8 blocks, neighbour purity, prediction depth (Eq. 6), manipulation-check flags |
| `src/sg/analysis.py` | run scanning, seed-paired partition-level outcomes T(G), geometry and baseline diagnostics |
| `src/sg/stats.py` | rank test (Eq. 7), similarity-adjusted residual with prediction interval (Eq. 8), H2 and H3 bootstraps, Holm, variance pilot and power |
| `src/sg/inat.py` | iNat-100 builder |
| `src/sg/plans.py` | run names, run plans for every experiment, GPU launcher, GPU-hour estimate |
| `src/sg/cli.py` | `python -m sg.cli <command>` (also installed as `sg`) |
| `configs/base.yaml` | paper defaults (Section III-E) |
| `configs/smoke.yaml` | tiny CPU configuration |
| `configs/inat100.yaml` | iNat-100 dataset and builder settings |
| `scripts/sanity_check.py` | identity and unit checks (a)–(p) on synthetic data |
| `scripts/smoke_all.sh` | CPU end-to-end run of every condition, command and plan, ending with `smoke OK` |
| `scripts/make_fake_inat.py` | tiny synthetic iNat2018-format tree, for the smoke test only |
| `scripts/run_plan.sh` | the full protocol on the GPU machine, by stage |

## Paper → code

| Paper | Code |
|---|---|
| Eq. 1, C_G and S_G; Acc = C_G·S_G | `metrics.factorize` (checked exactly with `Fraction` in sanity check (i)) |
| Eq. 2, F_G (descriptive) | `metrics.factorize` (`F` from the formula, `F_direct` by counting; check (j)) |
| True frame / own frame | `metrics.outcomes_from_preds(y, pred, assign_true, assign_own)`; own frame = the run's `partition.json` |
| Table I conditions | `config.AUX_STAGES`, `train._aux_assign`, `model.ResNet18(aux_stages, n_aux)` |
| Eq. 3, W(G) | `partitions.within_group_similarity` |
| Random-Hier, Spectrum-Hier, ARI | `partitions.random_partition`, `partitions.spectrum_partitions`, `partitions.local_search`, `partitions.ari`, `partitions.build_all` |
| SimCLR φ per budget; fixed φ on the 45k training + validation pools for analysis (audit excluded) | `simclr.train_simclr` (`simclr_scope: budget` or `all`), `analysis.geometry` (W_analysis) |
| Eq. 4 objective, λ_s from gradient norms in epoch 1 | `train.train`, `train.grad_norms`, `train.Calibrator` (check (h)) |
| ρ pilot at 100% | `plans` `rho_pilot`, `train.select_rho` |
| Weight decay per budget with CE | `plans` `tune_wd`, `train.select_wd` |
| Section III-F, five CE seeds, cross-fitting, Eq. 5 Δ_null and Δ_G | `metrics.crossfit_folds`, `metrics.stratum_deltas`, strata file of `sg outcomes` |
| Continuous curve over CE-A p_true | `metrics.probability_curve` (10 quantile bins) |
| Label-error exclusion | `metrics.load_label_errors`, `sg outcomes --label_errors` |
| Section III-G probes, purity, fixed reference set | `probes.run_probes`, `probes.knn_predict`, `probes.linear_probe`, `probes.neighbor_purity` |
| Eq. 6 prediction depth | `probes.prediction_depth` (check (k)) |
| Table II budgets, nested | `data.budget_split` (check (a)), `sg prepare` |
| IV-B.1 partition geometry | `analysis.geometry`, `sg diagnose geometry` |
| IV-B.2 baseline factors | `analysis.baseline`, `sg diagnose baseline` |
| IV-B.3 variance pilot and power | `plans` `var_pilot`, `stats.variance_pilot`, `stats.power_sim`, `sg stats pilot` |
| Exp A / Exp B / Exp C | `plans` `expA`, `expB`, `expC` |
| iNat-100 replication | `inat.build_inat100`, `plans` `inat_tune_wd`, `inat_ce`, `inat_A` |
| Fixed-step sensitivity | `plans` `fixedstep`, `train.train` with `total_steps` |
| ρ sensitivity | `plans` `rho_sens` |
| Eq. 7 rank test | `stats.rank_pvalue`, `sg stats rank` |
| Eq. 8 residual, 95% PI, quadratic sensitivity | `stats.residual`, `sg stats residual --degree 1|2` |
| H2 test | `stats.h2_bootstrap`, `sg stats h2` |
| H3 test, bootstrap over partitions and seeds | `stats.h3_bootstrap`, `sg stats h3` |
| Holm over H1–H3 | `stats.holm`, `stats.primary`, `sg stats primary` |

## Setup on the GPU machine

```bash
# All projects share one environment at the root: <root>/.venv
./setup_env.sh                                # from the root; installs every project editable
cd sg
../.venv/bin/python scripts/sanity_check.py      # all PASS, exit 0
bash scripts/smoke_all.sh                     # optional, about 15 min on CPU; ends with "smoke OK"
```

If torchvision's CIFAR-100 download fails with an SSL certificate error, fetch the archive first; `prepare` then verifies its MD5 and extracts it:

```bash
mkdir -p data && curl -C - --retry 5 https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-python.tar.gz
```

**iNaturalist 2018** (for the replication only; never downloaded by this code). On the GPU machine, download `train_val2018.tar.gz` (about 120 GB), `train2018.json.tar.gz` and `categories.json.tar.gz` from the iNat2018 competition page (github.com/visipedia/inat_comp, 2018), and extract them into one directory, for example `/data/inat2018` with `train_val2018/`, `train2018.json` and `categories.json`. Set `INAT_ROOT=/data/inat2018`. The builder reads only these files and writes `data/inat100/inat100.npz` (100 species × 350 images at 64×64, about 430 MB) and `meta.json`.

## Running the protocol

`scripts/run_plan.sh` runs one stage at a time. Training stages write a plan file (one `sg train` command per line) to `$OUT/<dataset>/plans/` and run it with `sg launch` over `GPUS`. Every run has a config-derived name, so runs shared by several experiments are trained once (`skip_if_done`), and a stage can be rerun after an interruption.

```bash
export OUT=runs GPUS=0,1,2,3 JOBS_PER_GPU=2      # PRIMARY_BUDGET=1.0 by default (D10)
./scripts/run_plan.sh prepare          # CIFAR-100 + data/cifar100_splits.json (Table II)
./scripts/run_plan.sh simclr           # SimCLR at b010 b025 b050 b100 + `all` (500 epochs each)
./scripts/run_plan.sh partitions       # true + 19 random + 19 spectrum per budget
./scripts/run_plan.sh diag_geometry    # IV-B.1 -> $OUT/cifar100/geometry.json
./scripts/run_plan.sh tune_wd          # 12 CE runs -> best_wd.json
./scripts/run_plan.sh ce               # 5 CE seeds x 4 budgets
./scripts/run_plan.sh diag_baseline    # IV-B.2 -> diag_baseline.json
./scripts/run_plan.sh rho_pilot        # 9 runs -> best_rho.json
./scripts/run_plan.sh var_pilot        # IV-B.3 -> var_pilot.json (raise partitions via SETS if power is short)
./scripts/run_plan.sh expA             # 25% and 100%
./scripts/run_plan.sh expB             # all four budgets, 9 + 9 cover partitions
./scripts/run_plan.sh expC             # probes on every Exp A run
LABEL_ERRORS=/path/cifar100_test_label_errors.txt ./scripts/run_plan.sh outcomes
./scripts/run_plan.sh stats            # primary.json (H1, H2, H3, Holm) + secondary rank/residual for every metric
./scripts/run_plan.sh rho_sens         # the two non-selected rho values
./scripts/run_plan.sh fixedstep        # every budget at the 100% step count; H3 under fixed steps
INAT_ROOT=/data/inat2018 ./scripts/run_plan.sh inat_build
./scripts/run_plan.sh inat             # iNat-100 Exp A at 25% and 100%
```

If the variance pilot recommends more than 19 partitions, rerun from `partitions` with `SETS="--set n_random=29 --set n_spectrum=29"` (the first 19 random partitions stay the same) and keep `SETS` set for every later stage.

Plans and the launcher can also be used directly:

```bash
../.venv/bin/python -m sg.cli plan --exp expA --config configs/base.yaml --set out_dir=runs --out runs/cifar100/plans/expA.txt
../.venv/bin/python -m sg.cli plan --exp expA --config configs/base.yaml --set out_dir=runs --count --sec_per_epoch45k 15
../.venv/bin/python -m sg.cli launch --plan runs/cifar100/plans/expA.txt --gpus 0,1,2,3 --jobs_per_gpu 2
```

`launch` gives each slot its own `CUDA_VISIBLE_DEVICES`, writes one log per line to `<plan>.logs/NNNN.log`, a `summary.json` with the failed lines, and exits non-zero if any line failed. Plans: `tune_wd`, `ce`, `rho_pilot`, `var_pilot`, `expA`, `expB`, `expC`, `rho_sens`, `fixedstep`, `inat_tune_wd`, `inat_ce`, `inat_A`. They read `best_wd.json` and `best_rho.json` from `--state_dir` (default `<out_dir>/<dataset>`), and `expB`, `rho_sens`, `fixedstep` and `inat_A` read the partitions' `index.json`.

Single commands, for reference:

```bash
C="--config configs/base.yaml --set out_dir=runs"
../.venv/bin/python -m sg.cli train $C --set budget=0.25 --set condition=random_hier \
  --set partition=runs/cifar100/partitions/b025/random_07.json --set rho=0.5 --set weight_decay=5e-4 --set seed=1 \
  --set run_name=cifar100/b025/random_hier_random_07/rho0.5/s1
../.venv/bin/python -m sg.cli outcomes $C --runs_root runs/cifar100 --budget 0.25 --rho 0.5 --geometry runs/cifar100/geometry.json --out o.jsonl
../.venv/bin/python -m sg.cli stats residual $C --outcomes o.jsonl --metric T_C_own --degree 1
```

### Outputs

- Training run dir (`$OUT/cifar100/b025/true_hier/rho0.5/s0`): `config.json`, `env.json`, `compute.json` (phases `calibrate`, `train`, `eval`, and `probes` after Exp C), `train_log.jsonl`, `train_summary.json` (λ_s, epoch-1 gradient norms, realized ratio per epoch in the log), `partition.json`, `model.pt`, `test.jsonl` (`id`, `y`, `pred`, `p_true`, `correct`), `test_logits.npy`, `val.jsonl`, and after Exp C `probes.json` and `pd.jsonl`.
- `outcomes/bNNN.jsonl`: one row per partition unit with `T_acc` and `T_C/S/F` in both frames (own frame null for Flat-Aux), W_train, W_analysis, ARI, and per-seed paired differences. `outcomes/bNNN.jsonl.strata.json`: per unit and metric, Δ_null and Δ_G per fold and averaged, the same with label errors excluded, and the 10-bin curve. `bNNN_expB.jsonl` keeps True-Hier and the Exp B cover partitions.
- `stats/primary.json`: H1 (residual on own-frame T_C at `PRIMARY_BUDGET`), H2 (bootstrap on own-frame S), H3 (bootstrap on δ_10% − δ_100% of T_acc over the Exp B units), raw one-sided p, Holm-adjusted p, and `reject` per hypothesis at family-wise one-sided α = 0.025 (D19).

### Label-error input

`LABEL_ERRORS` (or `--label_errors`, or `label_errors:` in a config) is a file of CIFAR-100 **test-set indices** (0-based, in the order of the official `test` pickle) flagged as mislabeled, either as a JSON list (`[12, 345, ...]`) or one integer per line (`#` lines are ignored). The flagged ids are removed from the hard strata after the strata are chosen, and the hard-stratum analyses are repeated (`excl_label_errors` in the strata file). The code does not ship such a list. One source is the CIFAR-100 test errors published with Northcutt et al. (2021) at labelerrors.com; convert them to test indices before use.

## Compute estimates (assumptions, not measurements)

`sg plan --count --sec_per_epoch45k S` prints run counts and GPU-hours = Σ epochs · (n_train / 45,000) · S / 3600 per plan. The table uses **S = 15 s per 45,000-image epoch of ResNet-18 at 32×32 on one A100-class GPU** with bf16 and GPU-side augmentation. This value is an assumption. Replace it with the `train` seconds in `compute.json` of the first pilot run divided by its epochs, scaled to 45k images. iNat-100 runs are scaled by (64/32)² = 4 for the pixel count. Not included: epoch-1 calibration overhead of auxiliary runs, evaluation, SimCLR, and probes.

> These GPU-hours were computed under the earlier 45,000-image training split. With the shared 40,000-image pool, training images per run drop by 1/9, so GPU-hours are about 11% lower; rerun `sg plan --count` on the GPU machine for current figures.

| Plan | Commands | Per budget (CE + aux) | GPU-hours at S = 15 s (assumed) |
|---|---|---|---|
| `tune_wd` | 12 | 3 CE at each of 10/25/50/100% | 4.6 |
| `ce` | 20 | 5 + 0 at each budget | 7.7 |
| `rho_pilot` | 9 | 0 + 9 at 100% | 7.5 |
| `var_pilot` | 21 | 3 + 18 at 25% | 4.4 |
| `expA` | 256 | 5 + 123 at 25% and 100% | 133.3 |
| `expB` | 248 | 5 + 57 at each budget | 95.6 |
| `expC` | 256 probe commands | | not estimated |
| `rho_sens` | 252 | 0 + 126 at 25% and 100% | 131.2 |
| `fixedstep` | 248 | 5 + 57 at each budget, 62,600 steps | 206.9 |
| `inat_tune_wd` | 6 | 3 CE at 25% and 100% | 7.5 |
| `inat_ce` | 10 | 5 + 0 at 25% and 100% | 12.5 |
| `inat_A` | 136 | 5 + 63 at 25% and 100% | 170.0 |

Commands are deduplicated by run name when executed, so the column total counts shared runs more than once. After deduplication the CIFAR-100 training plans contain about 894 distinct runs (about 509 GPU-hours at S = 15) and the iNat-100 plans 142 runs (about 178 GPU-hours); the exact number depends on which ρ the pilot selects. These figures were produced with `sg plan --count --sec_per_epoch45k 15` and are estimates under the stated assumption, not measured costs. SimCLR (500 epochs on 4,000 + 10,000 + 20,000 + 40,000 + 45,000 images for CIFAR-100, two views per image) costs roughly another 2·S per 45k images per epoch under the same assumption, about 12 GPU-hours at S = 15.

## Deviations and choices

The paper leaves the following points open. Each is decided here.

- **D1 Budgets and the shared split.** One fixed permutation per class (`split_seed`) cuts the 500 CIFAR-100 training images of each class into a training pool (400), a validation pool (`val_frac` = 0.1, 50) and an audit pool (`audit_frac` = 0.1, 50): 40,000 / 5,000 / 5,000 images. The split is shared with the privacy-redistribution study, which uses the audit pool as never-trained nonmembers; nothing in `sg` trains on it (`data.audit_split`, listed under `audit` in `data/cifar100_splits.json`). Budget f takes the first f·400 training-pool and f·50 validation-pool images of each class (fractional counts allocated so totals match), so budgets are nested and validation is 1/9 of the labeled budget: 4,000/500, 10,000/1,250, 20,000/2,500, 40,000/5,000. The `all` SimCLR extractor (W_analysis) trains on the 45,000 training + validation images and excludes the audit pool. iNat-100 uses the same fractions of its 300-image species pools (240 / 30 / 30).
- **D2 Pairing.** The data subset is fixed by `split_seed` for every run. `seed` sets initialization and data order. A model with seed s is paired with CE seed s.
- **D3 Cross-fitting.** Only hard-stratum analyses use folds: fold f takes CE seed f as CE-A and the other four as CE-B; Δ_null = mean_{b≠f} M(CE_b) − M(CE_f) and Δ_G = mean_s M(model_s) − mean_{b≠f} M(CE_b) on fold f's hard stratum, averaged over 5 folds. Full-test outcomes use the seed pairing of D2.
- **D4 λ_s.** During every step of epoch 1, ‖∂L_fine/∂h_s‖ and ‖∂L_aux,s/∂h_s‖ are computed at the stage output h_s; training uses λ_s = ρ·mean‖g_fine‖/mean‖g_aux‖ so far, and the value is frozen after epoch 1. The realized ratio λ_s‖g_aux‖/‖g_fine‖ is logged on the first batch of every later epoch.
- **D5 ρ criterion.** At 100%, seed 0, True-Hier, Random-Hier `random_00` and Flat-Aux are trained at each ρ ∈ {0.25, 0.5, 1}; the ρ with the highest mean final validation accuracy across the three is chosen, so ρ is not tuned to the curated hierarchy alone. iNat-100 reuses this ρ.
- **D6 Weight decay.** CE, seed 0, per budget, each of {1e-4, 5e-4, 1e-3}; highest final-epoch validation accuracy. No early stopping anywhere; the final model is evaluated.
- **D7 SimCLR recipe.** CIFAR ResNet-18 backbone, projection MLP 512→512→128, NT-Xent τ = 0.5, batch 512, SGD momentum 0.9, lr 0.06·bs/256, wd 5e-4, cosine, 500 epochs; RandomResizedCrop (scale 0.2–1), flip, ColorJitter(0.4, 0.4, 0.4, 0.1) with p = 0.8, grayscale p = 0.2. φ is the 512-d GAP feature of un-augmented images; μ_c is the mean of L2-normalized features. Scope `budget` trains on that budget's training images (not validation) and uses their labels for μ_c; scope `all` uses the whole 50k pool (analysis only).
- **D8 Spectrum search.** Swap moves between groups keep balance; each proposal is scored from a class-to-group similarity table. Hill-climbing (`search_iters` × `search_restarts`) finds the min-W and max-W partitions, then 17 more partitions target `linspace(W_min, W_max, 19)` interior values from random starts. Sorted by W as `spectrum_00..18`. Random partitions `random_00..18` come from `partition_seed` and are identical at every budget; spectrum partitions are built per budget from that budget's extractor.
- **D9 W in the residual fit.** W_analysis (the fixed `all` extractor). W_train (budget extractor) is recorded too and used only if W_analysis is missing (the output then says so).
- **D10 Primary budget for H1/H2.** Exp A runs at 25% and 100%, but Holm covers one test per hypothesis. H1 and H2 use the 100% budget; 25% is reported as secondary. `PRIMARY_BUDGET` changes this.
- **D11 Primary p-values.** H1: one-sided p = 1 − t_cdf(δ/se_pred, df) from the prediction interval on own-frame T_C. H2: one-sided paired bootstrap over test items and seed pairs of own-frame S (True-Hier − CE-B), p = (1 + #{boot ≥ 0})/(B + 1). H3: bootstrap over partitions and seeds of δ_10% − δ_100% on T_acc, p = (1 + #{boot ≤ 0})/(B + 1). H3 uses the Exp B units at both budgets (True-Hier and the same 9 + 9 cover partitions), so the two fits have the same design.
- **D12 Probes.** Features are the GAP of each of the 8 BasicBlock outputs, L2-normalized. k-NN: cosine, k = 20, majority vote, ties broken by the larger summed similarity. Coarse accuracy in each frame is a k-NN vote over the neighbours' group labels, plus a separate linear probe on group labels. Linear probe: standardized features, full-batch L-BFGS, `probe_linear_iters` iterations, L2 `probe_l2`, zero initialization. Reference set: the 4,000 training images of the 10% budget. Own frame is null for CE and Flat-Aux. Manipulation-check blocks: those of any stage with a head (s2 → blocks 3–4, s3 → 5–6, s4 → 7–8).
- **D13 iNat-100.** Only `train2018.json` + `categories.json` (taxonomy with `genus`) are used; iNat2018 val has about 3 images per class, too few for testing. Eligible species have ≥ 350 images; eligible genera have ≥ 5 eligible species (a genus is identified by its full taxonomy path). 20 genera and 5 species each are drawn with `inat_seed`; per species, 300 pool + 50 test images; shorter side resized to 64 (bicubic), center crop 64×64. Same ResNet-18 and crop padding 4. Mean and std come from the iNat-100 pool.
- **D14 Model saving.** `save_model: true` by default (checkpoints are about 45 MB; Exp C needs them). The smoke test saves only the runs it probes.
- **D15 Plans.** Run names come from the config: `{dataset}/b{pct}[_steps{N}]/{condition}[_{partition}]/rho{rho}/s{seed}`, CE `{dataset}/b{pct}[_steps{N}]/ce/s{seed}`, weight-decay tuning `{dataset}/tune_wd/b{pct}/wd{wd}/s0`. Runs shared by Exp A, Exp B and the pilots therefore have identical names and are trained once.
- **D16 Variance pilot and power.** One-way random effects on the per-seed paired accuracy differences of the pilot's 5 random partitions: σ²_seed = pooled within-partition variance, σ²_part = max(0, var(partition means) − σ²_seed/n). Power by Monte Carlo (2,000 simulations) for the residual test and the rank test, effect 0.01, R ∈ {19, 29, 39, 49, 99}, seeds ∈ {2, 3, 5}. To match the confirmatory H1 test, the residual fit uses 2R points (R random + R spectrum partitions), and both tests are judged at α/3 = 0.025/3, the strictest Holm step, so the reported power is a lower bound under Holm. The recommendation is the smallest R with ≥ 80% power at 3 seeds. With Holm, the rank test cannot reject at R = 19, because its minimum p is 1/20 = 0.05 > 0.0083. It needs R ≥ 119 for that, which is why the rank test is secondary and H1 uses the residual.
- **D19 Decision rule.** The family-wise α is one-sided 0.025 over H1–H3 with Holm. For a single test, p_one_sided < 0.025 is the same event as the paper's "95% prediction interval for δ above zero", so the Holm decision is never looser than the paper's rule. `stats/primary.json` records `alpha_one_sided_familywise`, the Holm-adjusted p values, `reject` per hypothesis, and `H1.pi_above_zero` for the paper's interval rule on its own.
- **D20 Weight decay and reuse.** Run names do not encode weight decay. With `skip_if_done: true`, `train` refuses to reuse an existing run whose `config.json` has a different `weight_decay`; delete the run or set `skip_if_done=false`.
- **D17 Fixed-step sensitivity.** `total_steps` overrides epochs (cosine over total steps). The `fixedstep` plan trains the Exp B units and CE at every budget for ceil(40,000/128)·200 = 62,600 steps. At 100% this equals the 200-epoch schedule, so those runs repeat Exp B under a separate name; drop budget 1.0 from the plan if that cost matters, and compare against the Exp B 100% outcomes instead.
- **D18 Dependencies.** torch, torchvision, numpy, pyyaml, tqdm, scikit-learn (ARI), scipy (t quantiles). Pillow comes with torchvision.

Other implementation choices:

- **Exp B cover subset.** The 9 random and 9 spectrum partitions of each budget are those at `round(linspace(0, 18, 9))` after sorting by W_train of that budget's partitions (`partitions.cover_subset`). Exp A at 25% and 100% contains them, so they are not trained twice.
- **Exp A, Exp B and iNat plans include their CE seeds** (5 per budget), so `sg plan --exp expA --count` reports 128 runs per budget (123 auxiliary + 5 CE).
- **Variance pilot** trains CE, True-Hier and `random_00..04` with seeds 0–2 at 25% at the selected ρ. The variance is estimated from the random partitions.
- **ρ sensitivity** trains True-Hier, Flat-Aux, Final-Hier and the 9 + 9 cover partitions with 3 seeds at 25% and 100% for each non-selected ρ (252 runs); CE is shared.
- **Batches of one.** If n_train mod batch size is 1, the last single-image batch of each epoch is dropped (BatchNorm).
- **Outcome rows** exist only for auxiliary units; CE information is in the strata (Δ_null) and in `diagnose baseline`. Full-test T values are null if any seed's metric is undefined.
- **Residual with df = 0** (for example a quadratic fit on 3 partitions) returns δ without an interval or p-value, with a note.

## Not implemented

- The exploratory extensions of Section IV-E: long-tailed CIFAR-100 (with the head/tail sibling analysis and the logit-adjustment baseline), 10- and 50-group partitions, confusion-derived hierarchies, pretrained backbones, and the semi-supervised variant with coarse labels on additional images.
- A label-error list for the CIFAR-100 test set (the input format is above).
