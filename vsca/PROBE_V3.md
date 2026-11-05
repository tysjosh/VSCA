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

## Experiment 3 (added 2026-11-05 09:05, after experiments 1–2 and before any experiment-3 result): answer check by program
Experiments 1–2 were negative: V⁺ checks steps locally and passes 84 % of samples, while pass@8 (86 %) exceeds
maj@8 (70 %) by 16 points. This probe asks whether an **answer-level** symbolic check closes part of that gap.
For every test problem the base model (Qwen2.5-1.5B, no adapter, 4-shot PAL-style prompt) writes 8 Python programs
`def solution(): ... return answer` at temperature 0.7; each is executed in a restricted subprocess (only `math`,
`fractions`, `datetime`, `itertools` imports; 3 s timeout; 512 MB). **p\*** = majority answer of the programs that
run. **pmaj@8**: among the 8 existing chain-of-thought samples (`test_pass8v.jsonl`), choose p\* if some sample's
answer equals it, else plain majority. No gold answer is used. Script: `scripts/pal_check.py`.
**Rule** (pmaj@8 − maj@8, rft_verified adapter, seeds 0–1 pooled, paired bootstrap; same thresholds): ≥ +1.0 point:
an answer-level symbolic check is the next verifier to build (training filter / reward); ≤ +0.3: it isn't; else
discuss. Reported, not in the rule: program-only accuracy (p\* vs gold), the other adapters, coverage (share of
problems where p\* matches one of the 8 samples).

## Outcome (recorded 2026-11-05)
Paired bootstrap over test items and seeds 0–1; comparison files in `runs/Qwen2.5-1.5B/v3probe/`.

**Experiment 1: no-go.** Greedy test vs `rft_verified` (61.2 mean):
| Run | Seed 0 | Seed 1 | Δ vs rft_verified (95 % CI) |
|---|---|---|---|
| `v3_vsca2add_l0.3` (primary) | 47.5 | 53.1 | −10.9 (−15.5, −6.1) |
| `v3_vsca2add_l0.1` | 55.5 | 57.8 | −4.6 (−7.6, −1.4) |
| `v3_stepdpoadd_l0.3` | 48.1 | 51.6 | −11.3 (−14.9, −7.7) |
Accuracy falls as λ grows; with or without certified sets it is the same (+0.45, CI −4.6, +5.4), so the
unlikelihood term on the erroneous token does the damage. Caveat: the fork term is a mean over about 8 forks per
micro-batch, so at these λ a fork token weighs roughly 15–45× an SFT token; smaller λ was not tested, but the trend
gives no sign of a gain above λ = 0 (= rft_verified). Fork-level credit assignment with V⁺ is dropped.

**Experiment 2: no-go.** vmaj@8 − maj@8: rft_verified −1.0 (−2.7, +0.6); SFT −1.1; claim mask −1.0; vsca2 −1.1.
84 % of samples pass the gold-free step check; 60 % of those are correct vs 22 % of the rest (rft_verified s0), so
the filter removes few wrong answers and some right ones.

**Experiment 3: go.** Programs (base model, 8 per problem): 97.6 % run; program-majority accuracy alone 63.7 %.
p\* matches one of the 8 chain-of-thought answers on 73–75 % of problems.
| Adapter | maj@8 (s0, s1) | pmaj@8 (s0, s1) | Δ pooled (95 % CI) |
|---|---|---|---|
| rft_verified (rule) | 70.6, 70.2 | 72.4, 72.1 | **+1.9 (+0.1, +3.6)** |
| SFT | 68.3, 67.4 | 69.6, 70.3 | +2.1 (+0.1, +4.0) |
| claim-span masking | 68.8, 67.7 | 70.8, 71.4 | +2.8 (+0.9, +4.9) |
| vsca2 | 68.6, 69.1 | 71.6, 71.0 | +2.5 (+0.6, +4.3) |
An answer-level symbolic check (independent re-solution as a program) adds accuracy where the step-level check V⁺
did not. Per the rule, it is the next verifier to build. Not shown here: whether it helps as a training signal, and
how it compares with simply voting over more chain-of-thought samples at equal compute.
