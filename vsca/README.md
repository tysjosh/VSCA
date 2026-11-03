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
