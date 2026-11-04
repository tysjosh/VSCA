# VSCA: Verifiable Semantic Credit Assignment

Code for the protocol in the paper *Verifiable Semantic Credit Assignment for Language Model Post-Training* (Overleaf project `6ae4fd18e6b1331c65d752c7`; local copy in `../paper/example_paper.tex`). It implements the method, all baselines and controls of Section 4.4, the pre-registered diagnostics of Section 4.2, and the statistics of Section 4.6, for GSM8K with SVAMP transfer.

Training and evaluation are meant for a remote GPU machine. The only thing run locally is the CPU smoke test (`scripts/smoke_all.sh`), which checks that every code path executes on a 135M model with a handful of examples; its numbers are meaningless.

## Layout

| File | Contents |
|---|---|
| `src/vsca/verify.py` | Claim verifier `V_c` (exact rational evaluation of `<<expr=result>>`, rounding-aware) and response verifier `V_r` (final answer matches and every annotation verifies) |
| `src/vsca/data.py` | GSM8K/SVAMP preparation, prompt format, tokenisation with claim-span alignment, counterfactual pairs for the earlier formulation, batching |
| `src/vsca/certify.py` | Algorithm 1: snapshot statistics, local certificates at claim positions, rollout certificates at decision positions; the controls (`claim_only`, `decision_only`, `uncertified`, `shuffled`) and robustness edits (`inject_eta`, `drop_eta`) |
| `src/vsca/objectives.py` | Set-valued loss (Eq. 3) and every baseline's token weighting |
| `src/vsca/train.py` | Training loop (LoRA, AdamW, cosine schedule, per-epoch certification refresh, best-on-validation checkpoint) |
| `src/vsca/scoring.py` | Per-token losses of the reference and base models; ssToken attention scores |
| `src/vsca/evaluate.py` | Greedy final-answer accuracy, trace validity, claim validity; per-item jsonl |
| `src/vsca/diagnostics.py` | Diagnostics 1–4 of Section 4.2 and the pairwise-loss saturation check |
| `src/vsca/stats.py` | Paired bootstrap over items and seeds, exact McNemar, Holm, seed-count power analysis |
| `src/vsca/cli.py` | `python -m vsca.cli {prepare,score,train,certify,eval,diagnose,stats}` |
| `configs/base.yaml` | Paper defaults |
| `configs/smoke.yaml` | Tiny CPU configuration |
| `scripts/sanity_check.py` | Verifier cases and numerical checks of Propositions 1–4 |
| `scripts/smoke_all.sh` | Runs every method, control, evaluation, diagnostic and stats command on CPU |
| `scripts/run_plan.sh` | Full protocol on the GPU machine, by stage |

## Paper → code

| Paper | Code |
|---|---|
| Sec. 3.1 verifiers `V_c`, `V_r` | `verify.claim_valid`, `verify.response_valid` |
| Reference filtering (`V_r(y_i)=1`) | `data.prepare_gsm8k` (`removed_unverified_references` in `data/prepare_stats.json`) |
| Eq. 2, acceptance set `A_t` | `certify.certify` |
| Candidates: top-k under snapshot | `certify._position_stats` (`k`) |
| Local certificates | `certify._local_ok` (`max_span_tokens`) |
| Rollout certificates | `certify.certify`, decision branch (`M`, `max_rollout_tokens`, `rollout_temperature`) |
| Which positions are searched | `certify._search_positions` (`B`, `h_min`) |
| Snapshot refresh `R` | `refresh: epoch` (re-certify before every epoch) or `once` |
| Eq. 3 objective | `objectives.set_nll`, `method: vsca` |
| Props. 1–4 | checked numerically in `scripts/sanity_check.py` |
| Sec. 4.2 diagnostics 1–4 | `vsca.cli diagnose {taxonomy,mass,audit,earlier_score}` |
| Sec. 4.4 baselines | `method:` `sft`, `random`, `rho`, `token_cleaning`, `sstoken`, `trimsft`, `claim_mask`, `claim_upweight`, `unlikelihood`, `cf_select`, `rft` |
| Sec. 4.4 VSCA controls | `variant:` `uncertified`, `shuffled`, `claim_only`, `decision_only` |
| Sec. 4.6 robustness | `inject_eta`, `drop_eta` |
| Sec. 4.5 compute | `compute.json` in every run directory (seconds and GPU-hours per phase) |
| Sec. 4.6 statistics | `vsca.cli stats {compare,mcnemar,power,holm}` |

## Setup on the GPU machine

```bash
# All projects share one environment at the root: <root>/.venv
./setup_env.sh                         # from the root; EXTRAS="./vsca[eval]" adds lm-eval (MMLU/IFEval drift)
source .venv/bin/activate
cd vsca
python scripts/sanity_check.py         # verifier and objective identities
huggingface-cli login          # needed for meta-llama/Llama-3.2-1B
```

## Running the protocol

`scripts/run_plan.sh` runs one stage at a time. Set `MODEL` to run the second backbone.

```bash
./scripts/run_plan.sh prepare
./scripts/run_plan.sh diag       # label diag/error_taxonomy.csv and diag/certificate_audit.csv by hand
./scripts/run_plan.sh pilot      # 5 SFT seeds -> power.json gives the seed count
./scripts/run_plan.sh scores     # reference-model and base-model token scores; base certification
./scripts/run_plan.sh tune       # LR grid {5e-5,1e-4,2e-4,4e-4} per method on validation
SEEDS="0 1 2 3 4" ./scripts/run_plan.sh main      # confirmatory: VSCA, SFT, claim upweighting, RFT, compute-matched SFT
SEEDS="0 1 2 3 4" ./scripts/run_plan.sh controls  # baselines and VSCA controls
SEEDS="0 1 2 3 4" ./scripts/run_plan.sh robust    # false-certificate injection / dropping
./scripts/run_plan.sh eval       # GSM8K test and SVAMP for every finished run
./scripts/run_plan.sh stats      # paired bootstrap + Holm for the confirmatory comparisons
MODEL=meta-llama/Llama-3.2-1B ./scripts/run_plan.sh prepare   # ...then repeat the stages
```

Individual commands, for reference:

```bash
python -m vsca.cli train --config configs/base.yaml --set method=vsca --set seed=0 --set run_name=vsca_s0
python -m vsca.cli eval  --config configs/base.yaml --adapter runs/vsca_s0/adapter_best --split gsm8k_test --out runs/vsca_s0/test.jsonl
python -m vsca.cli stats compare --a runs/vsca_s*/test.jsonl --b runs/sft_s*/test.jsonl
```

## Sessions and resume (machines rented in 10-hour blocks)

Every command can be stopped and rerun; finished work is skipped and unfinished work continues.

```bash
source scripts/session_start.sh        # at the start of each session: fixes the stop time (SESSION_HOURS=10, MARGIN_MIN=20)
./scripts/run_plan.sh all              # or one stage; rerun the same command next session
```

- **Clean stop.** Runs check `VSCA_DEADLINE` after every optimizer step, every certification batch and every evaluation batch. At the deadline they save their state and exit with code 75; `run_plan.sh` then stops and prints a reminder. If `session_start.sh` was not sourced, `run_plan.sh` counts the session from its own start time.
- **Training** (`runs/<name>/ckpt.pt`): LoRA weights, optimizer, scheduler, step, micro-batch position, epoch phase (certify / train / val), validation history and all RNG states. It is written every `ckpt_every` optimizer steps (default 50), at every phase change, and at the deadline, atomically. After a hard kill at most `ckpt_every` steps are repeated. `ckpt.pt` is deleted when the run finishes (`keep_ckpt=true` keeps it).
- **Certification** (`cert_epochN.jsonl`): written one example at a time; a resumed run skips certified examples, and `cert_epochN.jsonl.done` marks completion. Each example's rollouts use their own seed, so a resumed certification produces the same records as an uninterrupted one.
- **Skip and guard.** A run with `train_summary.json` is skipped. Resuming or skipping with a different config (anything except `device`, `resume`, `ckpt_every`, `keep_ckpt`, `deadline` and the debug hooks) is refused; use a new `run_name`. `score` and `eval` skip outputs that already exist. `resume=false` starts a run over.
- **Logs and compute.** `train_log.jsonl` is truncated to the resumed step. `compute.json` accumulates time across sessions, so work repeated after a hard kill is counted (at most one checkpoint interval per kill); `train_summary.json` and `compute.json` record the number of resumes.
- **Check.** `scripts/resume_test.sh` (part of the smoke test) interrupts a run during certification, mid-epoch and at a deadline, then checks the resumed run is identical to an uninterrupted one: same certification records, same losses (difference 0 on CPU) and same validation history. On GPU, nondeterministic kernels can make resumed losses differ slightly.

## vLLM backend for certification

Certification is the expensive part of VSCA. `rollout_backend` chooses how its rollouts and span completions are
generated; snapshot statistics (top-k candidates, entropy) always come from the HF model.

- `hf` (default): transformers `generate`, in length-sorted batches of `cert_bs` sequences per block of `cert_block`
  examples. Measured on the L40S (Qwen2.5-1.5B, base model): 22.7 s per example unbatched, 7.3 s with block batching.
- `vllm`: one vLLM call per block, prefix caching on, the current LoRA adapter served from disk (saved next to the
  certification file as `cert_epochN.snapshot/`, a new LoRA id per refresh). Each request has its own seed derived from
  the block seed and job index, so results do not depend on vLLM's scheduling and resume reproduces them. Use a larger
  block, e.g. `--set rollout_backend=vllm --set cert_block=256`. The engine reserves `vllm_gpu_mem` (default 0.40) of
  GPU memory for the rest of the process, next to the training model.

vLLM pins torch and transformers (vLLM 0.33: torch 2.14, transformers < 5.21), so it lives in its own environment:

```bash
bash vsca/scripts/setup_vllm_env.sh    # creates <root>/.venv-vllm; vsca scripts use it automatically when present
bash vsca/scripts/vllm_probe.sh        # GPU: same 64 examples with both backends -> runs/vllm_probe/compare.json
```

`vllm_probe.sh` reports seconds per example, the agreement of greedy local certificates (expected near 1; bf16 kernels
can differ), and certification rates. Rollout samples come from different random streams, so decision certificates are
compared as rates. Before using vLLM for the study, check that the vLLM decision rate is within sampling noise of the
HF rate (19% of searched decision positions on the 64-example probe). A run must use one backend throughout;
`rollout_backend` is part of the run config, so resuming with the other backend is refused. Locally,
`scripts/vllm_plumbing_test.py` checks the vLLM code path with a stand-in engine (CPU).

## VSCA tuning

`tune` picks VSCA's learning rate from the same 4-value grid as every other method, but on the fixed base-model
certificates written by `scores` (`cert_base.jsonl`) instead of re-certifying every epoch. A refreshed run certifies
its first epoch with the untrained model, so the two agree for the first epoch and differ only in epochs 2-3. The
12-seed main runs re-certify every epoch as the paper specifies. This must be stated in the paper's tuning protocol.

**Compute-matched SFT.** `main` trains `sft_matched_s*` with `SFT_MATCHED_EPOCHS` epochs (default 6). Set it after the first VSCA run so that total GPU-hours in `compute.json` (certification + training) match: `epochs ≈ 3 × (vsca total GPU-h) / (sft train GPU-h)`.

**Cost.** Certification is the expensive part. Per response it runs up to `B·k·M = 64` rollouts of up to 320 tokens, plus `k` short greedy completions per claim token. GSM8K train has 6,726 responses, so one refresh is roughly 430k rollouts, and `refresh: epoch` repeats it before each of 3 epochs. With HuggingFace `generate` this takes many GPU-hours per refresh. Two ways to cut it, both of which must be reported: `refresh=once`, or a vLLM-based rollout backend (not included).

## Deviations from the paper and open choices

- **ssToken.** Attention scores come from the base model, computed once (`vsca.cli score --attention`), not recomputed online from the current model. The retrospective excess loss uses the base model as the history model, as in Qin et al.'s reported setting. `sstoken_gamma=0.5` and `attn_layer=-1` (last layer) are defaults, not the paper's tuned values.
- **Token Cleaning.** Implemented as the fixed-reference variant: token score = base loss − reference loss, top-ρ per sequence.
- **Rho-style selection.** Excess loss against a fixed reference (SFT seed 0 of the pilot) with the current model's loss computed online; selection is per sequence, not per batch.
- **TrimSFT.** `trim_mu=1.5`, `trim_tau=0.8`, the main setting of Jia et al. (2026, Appendix A).
- **Normalisation.** Selection baselines divide by the number of selected tokens, as in their papers; VSCA and weighting methods divide by all target tokens (Eq. 3). Learning rate is tuned per method to absorb the difference.
- **Claim positions** are the tokens of `<<...>>` calculator annotations. A local certificate also requires that everything the substituted token emits lies inside the re-closed annotation, so the token can't smuggle text that the reference suffix would discard.
- **`V_r`** treats an undetermined annotation as a failure.
- **Validation** uses a 500-example subset of the 747 held-out training examples (`limit_val`) for per-epoch checkpoint selection.

## Not implemented yet

- **ProofWriter**, the paper's second task: it needs a parser for the formal theories, a forward-chaining prover under the open-world assumption, and claim alignment for derivations.
- **cDPO**, the preference-stage comparison.
- **MATH-500** (optional in the paper).
- **MMLU/IFEval drift.** Run with `lm-eval` on a merged adapter:

  ```bash
  lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B,peft=runs/<run>/adapter_best --tasks mmlu,ifeval
  ```
- **vLLM rollout backend** for certification.

## VSCA v2

On-policy verified credit assignment (design: `VSCA_V2_DESIGN.md`, Sections 4–5). This section is filled in as the
v2 work lands. The on-policy pipeline is `src/vsca/onpolicy.py` (`vsca.cli onpolicy --out DIR`), the V⁺ verifier is
`src/vsca/verify_plus.py`, and its checks are in `scripts/sanity_v2.py`.

### Finding: V⁺ grounding rejects some correct steps (trade-off chosen before any training)

Grounding requires every operand of a `<<expr=result>>` step to be a stated quantity. Correct GSM8K references
often use quantities that are never stated: inferred counts ("5 knots"), unit conversions (8 pints per gallon),
calendar constants and algebra lines (`x=<<97=97>>`). The table measures, on the 200 rows of
`labeling/error_taxonomy_sft.csv`, how many correct references grounding rejects ("ref. false rejection") and how
many of the 163 `needs_manual` SFT failures it still flags. Each row adds to the one above it. At every row, an
operand invented inside a restated expression (`5*9973 = <<5*9973=49865>>`) is still rejected.

| Point | Grounded operands | Ref. false rejection | Flags / 163 |
|---|---|---|---|
| FEAT-001 | problem numbers and number words, earlier annotation results, whitelist `60,24,7,12,100,1000,2,0.5` | 45.5 % (91/200) | 54 |
| C | + numbers written in earlier solution lines; numbers in the same line's prose before the annotation that are not next to an operator; percent forms (25 % ↔ 0.25, 1.25, 0.75; 0.25 → 25); unit fractions give their denominator ("a quarter" → 4); "thrice", "third"; a constant ratio `a/b` whose value is grounded | 13.0 % (26/200) | 20 |
| D | + whitelist `1, 0.01` | 9.5 % (19/200) | 17 |
| **E (chosen)** | + whitelist `30, 31, 365, 52` | **8.5 % (17/200)** | **14** |
| F | + any single-constant annotation `<<N=N>>` | 5.5 % (11/200) | 11 |
| G | + the word "fifth" | 5.0 % (10/200) | 10 |
| H | + whitelist `3, 4, 5, 6, 8, 10` | 1.5 % (3/200) | 6 |

E is the default `grounding_whitelist` (`60,24,7,12,100,1000,2,0.5,1,0.01,30,31,365,52`). At E, grounding rejects
6.8 % (458/6,726) of the GSM8K train references and 6.0 % (79/1,319) of the test references. F was rejected because
`<<9973=9973>>` would then pass, and H because it accepts invented small integers. `scripts/grounding_tradeoff.py`
reproduces every row of the table (FEAT-001 from `verify_plus.py` at commit `bc7a513`, F–H by patching the current
verifier in the script only). `scripts/sanity_v2.py` checks the 8.5 % at E, and `diagnose grounding`
(`reference.grounding_false_rejection_rate`) and `diagnose vplus_trace` (`reference_grounding_false_rejection_rate`)
report the rate.

The on-policy pipeline limits the harm of the remaining false rejections (`onpolicy.localise`; each fork records
its `rule`):
- A correct-answer sample whose only failures are grounding failures is *unverified*. It isn't trained on (it's
  excluded from the `rft_verified` positives and from SFT on verified samples) and it's **never** an error fork.
  A correct-answer sample becomes a negative (`lucky_arithmetic`) only through an arithmetic failure.
- A correct-answer sample with no `<<expr=result>>` step at all is also *unverified* and never a fork
  (`vplus_min_steps: 1`), because V⁺ has nothing to check in it. `onpolicy_stats.json` counts these samples as
  `no_step_correct`; `vplus_min_steps: 0` restores the FEAT-001 behaviour.
- In a wrong-answer sample, the first error is the earliest arithmetic failure (`arithmetic`). Without one, it's the
  earliest grounding failure only if the Monte Carlo value also drops there by at least `mc_min_drop`
  (`grounding_mc`). Otherwise it's the largest MC value drop (`mc`, kind `planning`), or the sample is skipped as
  uncertain.
### Layout (v2 files)
| File | Contents |
|---|---|
| `src/vsca/verify_plus.py` | Step verifier V⁺: arithmetic validity, grounding of every operand, final answer, lucky-guess filter; error character and kind of a failed step; step boundaries for MC localisation |
| `src/vsca/onpolicy.py` | On-policy pipeline: `sample` → `verify_samples` → `localise` → `certify_forks`, driver `prepare`, per-arm training data `build_training`, dir guard `check_dir` |
| `scripts/sanity_v2.py` | V⁺ cases, localisation and pipeline checks with a stand-in backend, fork-loss identities |
| `scripts/vllm_plumbing_v2.py` | The v2 pipeline through the vLLM code path with a stand-in engine (CPU) |
| `scripts/smoke_v2.sh` | Every v2 arm, both pipeline dirs, a dynamic run, evals, v2 diagnostics, stats and `resume_test_v2.sh` on CPU |
| `scripts/resume_test_v2.sh` | Interrupts a dynamic `vsca2` run during sampling, fork certification, mid-epoch and at a deadline; checks the resumed run is identical |
| `scripts/pilot_v2.sh` | The pre-registered v2 pilot (`PILOT_V2.md`) on the GPU machine, by stage |
| `scripts/grounding_tradeoff.py` | Reproduces the grounding trade-off table above (CPU, seconds) |
| `PILOT_V2.md` | Pre-registration of the v2 pilot |
### Design → code
| Design (`VSCA_V2_DESIGN.md`) | Code |
|---|---|
| §4.1 step 1, sample k solutions | `onpolicy.sample` (`onpolicy_k`, `rollout_temperature`, `rollout_top_p`, `gen_max_new`; `onpolicy_samples` copies an existing samples file) |
| §4.1 step 2, V⁺ and lucky-guess filter | `verify_plus.check_response`, `onpolicy.verify_samples` (`vplus_grounding`, `grounding_whitelist`, `vplus_min_steps`) |
| §4.1 step 3, first error (direct or Monte Carlo) | `onpolicy.localise` (`mc_M`, `mc_min_drop`, `max_rollout_tokens`); `verify_plus.step_boundaries` |
| §4.1 step 4, certify the fork | `onpolicy.certify_forks` (`k`, `M`, `cert_block`, `cert_bs`) |
| §4.1 step 5, fork loss | `objectives.fork_loss` (`fork_beta`), `compute_loss` branch for the v2 methods |
| §4.1 step 6, SFT on verified samples (+ references) | `onpolicy.build_training` (`v2_ref_sft`) |
| Refresh each epoch | `train` without `onpolicy_dir`: `refresh: epoch` runs the pipeline into `run_dir/onpolicy_epoch<e>` before every epoch |
| §5.1 RFT-verified | `method: rft_verified` |
| §5.1 Step-DPO-style | `method: stepdpo_fork` (unlikelihood on `e` at the first error, no positive set) |
| §5.1 VSCA v2 (full) | `method: vsca2`, `variant: full` |
| §5.1 VSCA v2 without grounding | `method: vsca2`, `vplus_grounding: false`, its own pipeline dir from the same samples |
| §5.1 VSCA v2 with uncertified sets | `method: vsca2`, `variant: uncertified` (the first \|A_t\| snapshot candidates) |
| §5.1 same starting point | `init_adapter` (the SFT adapter of the same seed) |
| §5.3 grounding coverage, 0.40 gate | `diagnose grounding` (`diagnostics.grounding_coverage`; `gate_0.40_passed` once labelled) |
| §5.3 fork recoverability | `diagnose recoverability` (`diagnostics.fork_recoverability`) |
| §5.2 valid-trace rate | `diagnose vplus_trace` (`diagnostics.vplus_trace`) |
### Commands (v2)
```bash
python scripts/sanity_v2.py                       # V+, pipeline and loss checks (CPU)
python scripts/vllm_plumbing_v2.py                # vLLM code path with a stand-in engine (CPU)
bash scripts/smoke_v2.sh                          # every v2 code path on a 135M model (CPU); ends with "smoke v2 OK"
# one shared pipeline dir per seed, then the arms train from it
python -m vsca.cli onpolicy --config configs/base.yaml --set seed=0 --set init_adapter=runs/<sft run>/adapter_best --out runs/v2_onpolicy_s0
python -m vsca.cli onpolicy --config configs/base.yaml --set seed=0 --set init_adapter=runs/<sft run>/adapter_best \
  --set vplus_grounding=false --set onpolicy_samples=runs/v2_onpolicy_s0/samples.jsonl --out runs/v2_onpolicy_nogr_s0
python -m vsca.cli train --config configs/base.yaml --set method=vsca2 --set seed=0 --set lr=5e-5 \
  --set init_adapter=runs/<sft run>/adapter_best --set onpolicy_dir=runs/v2_onpolicy_s0 --set run_name=v2_vsca2_s0
python -m vsca.cli diagnose grounding --config configs/base.yaml --out_dir runs/diag_v2
python -m vsca.cli diagnose recoverability --config configs/base.yaml --onpolicy_dir runs/v2_onpolicy_s0 --out_dir runs/diag_v2
python -m vsca.cli diagnose vplus_trace --config configs/base.yaml --eval_file runs/v2_vsca2_s0/test.jsonl --split gsm8k_test --out_dir runs/v2_vsca2_s0
bash scripts/pilot_v2.sh [diag|onpolicy|train|passk|stats|all]   # GPU machine; resumable; FORCE=1 overrides the gate
python scripts/grounding_tradeoff.py              # the grounding trade-off table (CPU)
```
`pilot_v2.sh` needs `best_lr.json` and the go/no-go pilot's SFT and claim-span masking runs (seeds 0–2); it reuses
them and aborts with a message if an adapter is missing. Its `diag` stage fails closed: it stops if
`labeling/error_taxonomy_sft.csv` has no `manual_label`s or if the 0.40 gate fails, unless `FORCE=1`. The outcome
(`passed`, `failed_stopped`, `unlabelled_stopped`, `failed_force`, `ungated_force`) is written to
`v2pilot/diag/gate.json` and copied into `decision.json`. A pipeline dir is refused by `onpolicy` (on resume) and
`train` if any of `seed`, `model_name`, `init_adapter`, `limit_train`, `vplus_grounding`, `grounding_whitelist`,
`vplus_min_steps`, `onpolicy_k`, `mc_M`, `mc_min_drop`, `k` or `M` differ from the run's config.
### Deviations from the design and open choices (v2)
- **Loss sign.** Design §4.1 writes `+β·log(1−π(e|h))`, which as a term to minimise would *raise* π(e). The code
  minimises `β·(−log(1−π(e|h)))` (unlikelihood), the intended term. An empty certified set leaves only that term.
- **SFT initialisation.** v2 arms and their pipelines start from the SFT adapter of the same seed (`init_adapter`).
  The base model writes no `<<…>>` annotations (0/200 in the base-model taxonomy sample), so V⁺ would have nothing
  to check on its samples; the design's failure taxonomy is of SFT failures.
- **Matched step budget.** v2 arms train `ceil(n_train_problems / eff_bs)` optimizer steps per epoch, as SFT does,
  drawing micro-batches from a seeded shuffle of the epoch's examples (cycled if shorter). The design says "same
  budget"; its 25-minute training estimate only holds at SFT's step count.
- **Steps and fork tokens.** A step is a `<<expr=result>>` annotation; its span starts at the start of its line, and
  the text after the last step line (the `####` line) is one more step for MC localisation. Fork token positions come
  from re-tokenising the sample text exactly as the loss sees it. A fork example is the sample cut after the fork
  token and supervised only there, so the wrong continuation after the error is never trained on.
- **Error character.** An arithmetic slip points at the wrong result where it is first written on the line, an
  ungrounded operand at its first mention; if a step fails both checks, the earlier character wins.
- **Grounding numbers.** Problem numbers include number words (zero–twenty, tens, hundred, thousand, million,
  dozen, half, quarter, twice/double, triple, thrice, third) and the loosened forms of point E above (earlier
  solution lines, percent forms, unit fractions, constant ratios), plus the whitelist. Operands are the numeric
  constants of the annotation's expression, so `10-3` gives 10 and 3.
### Cost estimate (assumption, not a measurement)
These figures are copied from Design §5.2 (L40S) and have not been measured with this code:
- sampling 8 × 6,726 problems with vLLM: about 1 GPU-hour per refresh;
- MC rollouts at the forks of failed samples (about 40 % of 54k): about 2–3 GPU-hours;
- verification: CPU;
- each training run: about 25 minutes;
- total about 12 GPU-hours for 4 new arms × 3 seeds with one refresh.
`pilot_v2.sh` trains 5 arms × 3 seeds and builds two pipeline dirs per seed (the no-grounding dir reuses the
samples but repeats localisation and certification). Under the same assumptions that is roughly
3 × (1 + 2 × (2–3)) + 15 × 0.42 ≈ 21–27 GPU-hours, before pass@8 and evaluation. Use the `compute.json` files of
the first seed to replace these numbers.
