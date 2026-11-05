# Probe after the v2 pilot (pre-registered 2026-11-05, before any result of these runs)
Two cheap probes on Qwen2.5-1.5B, seeds 0–1, greedy / sampled GSM8K test (1,319 problems). They decide whether
symbolic verification is worth more work in **training** (experiment 1) or at **inference** (experiment 2).
Runner: `scripts/run_v3_probe.sh`; checks: `scripts/sanity_v3.py`. Outputs under `runs/Qwen2.5-1.5B/` and
`runs/Qwen2.5-1.5B/v3probe/`.

## Background (v2 pilot, `PILOT_V2.md`, stopped after 2 of 3 seeds)
Greedy test, mean of seeds 0–1: rft_verified 61.2, stepdpo_fork 61.0, vsca2 60.3, vsca2_uncertified 59.8,
claim-span masking 60.8, SFT 58.7. **Confound:** every v2 arm took the same 211 optimizer steps per epoch from a
seeded cut of its own example list. vsca2's list was 60 % verified samples and 40 % forks (24,318 + 16,061 for
seed 0), so it saw about 40 % fewer verified solutions than rft_verified. The loss is averaged over supervised
tokens (`objectives.compute_loss`), so a fork token is not over-weighted; the confound is in the data mix.

## Experiment 1: the fork loss added on top of rft_verified's batches
`fork_mode=add`: the arm trains on **exactly** rft_verified's SFT batches (same verified samples, same seeded
order; checked by `main_order_sha256` in `onpolicy_stats_epoch*.json`), and every micro-batch adds
`fork_lambda × mean fork loss` over a separate micro-batch of fork examples from the same pipeline dir.
Fork loss as in v2 (certified set + β = 1.0 unlikelihood on the erroneous token). lr 5e-5, init from the SFT
adapter of the same seed, pipeline dirs `v2_onpolicy_s<seed>`, refresh once, 3 epochs, selection on validation.

| Run | Config | Role |
|---|---|---|
| `v3_vsca2add_l0.3_s<seed>` | `method=vsca2 fork_mode=add fork_lambda=0.3` | **primary** |
| `v3_vsca2add_l0.1_s<seed>` | `fork_lambda=0.1` | secondary |
| `v3_stepdpoadd_l0.3_s<seed>` | `method=stepdpo_fork fork_mode=add fork_lambda=0.3` | secondary: negative term only |
| `v2_rft_verified_s<seed>` | reused from the v2 pilot | comparison |

**Rule** (primary run vs rft_verified, mean over seeds 0–1 of the greedy-test difference, with the paired
bootstrap reported):
- ≥ +1.0 point: the credit-assignment idea is alive; pre-register a full study (≥ 3 seeds, all arms).
- ≤ +0.3 point: drop fork-level credit assignment with this verifier.
- otherwise: inconclusive at 2 seeds; discuss before more GPU time.
λ = 0.1 and the stepdpo run are reported but do not change the rule (no picking the best λ after the fact).

## Experiment 2: V⁺ as a gold-free filter for voting at inference
For each adapter, 8 samples per test problem at temperature 0.7 (`vsca.cli passk`, vLLM, seed = run seed), saved
with their texts (`test_pass8v.jsonl`). **vmaj@8**: majority vote among the samples whose every step passes V⁺
**without the gold answer** (arithmetic + grounding, at least one step; `evaluate.steps_valid`), falling back to
plain majority when no sample passes. **maj@8**: plain majority over the same 8 samples.
Adapters: SFT, claim-span masking, rft_verified, vsca2, seeds 0–1 (the existing runs).

**Rule** (vmaj@8 − maj@8 on the same samples, rft_verified adapter, pooled over seeds 0–1, paired bootstrap):
- ≥ +1.0 point: V⁺ adds accuracy at inference; next, verifier-guided search or V⁺ as an RL reward.
- ≤ +0.3 point: the verifier adds nothing beyond self-consistency.
- otherwise: discuss. The other adapters are reported, not part of the rule.

## Not changed
Verifier V⁺ (point E, `vplus_min_steps=1`), data split, evaluation code paths, and the v2 runs reused here.
The probe is ungated with respect to the 0.40 grounding gate (unlabelled CSV), as the v2 pilot was.
