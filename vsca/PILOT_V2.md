# VSCA v2 pilot (pre-registered 2026-11-04, before any v2 training or evaluation result)
**Question.** Does VSCA v2 (on-policy verified credit assignment, `VSCA_V2_DESIGN.md` §4) beat both RFT on
V⁺-verified samples and claim-span masking on held-out GSM8K test accuracy by enough to justify a full study?
The only measurements taken before this pre-registration are the verifier trade-off and grounding coverage in the
finding below (no model was trained or evaluated with v2 code).
**Runs** (Qwen2.5-1.5B, LoRA, `configs/base.yaml`, seeds 0–2; script `scripts/pilot_v2.sh`, outputs under
`runs/Qwen2.5-1.5B/`):
| Arm (Design §5.1) | Run name | Config | What it isolates |
|---|---|---|---|
| SFT | seed 0 `tune_sft_lr<best>`, seeds 1–2 `gonogo_sft_s1/s2` | reused, not retrained | baseline |
| Claim-span masking (ρ = 0.5) | seed 0 `tune_claim_mask_lr5e-5`, seeds 1–2 `gonogo_claim_mask_s1/s2` | reused, not retrained | baseline |
| RFT-verified | `v2_rft_verified_s<seed>` | `method=rft_verified` | symbolic filtering alone |
| Step-DPO-style | `v2_stepdpo_fork_s<seed>` | `method=stepdpo_fork` | first-error negative alone |
| VSCA v2 (full) | `v2_vsca2_s<seed>` | `method=vsca2 variant=full` | certified set + negative at the fork |
| VSCA v2 without grounding | `v2_vsca2_nogrounding_s<seed>` | `method=vsca2 vplus_grounding=false` | the planning-error verifier |
| VSCA v2, uncertified sets | `v2_vsca2_uncertified_s<seed>` | `method=vsca2 variant=uncertified` | verification itself |
The reused runs are the go/no-go pilot's runs at their tuned learning rates (`best_lr.json`); claim-span masking's
is 5e-5. The SFT run names follow the same rule. All five v2 arms use **lr 5e-5**, not tuned per arm.
**Pre-registered choices** (fixed now, not changed after seeing results):
- **Initialisation.** Every v2 arm starts from the SFT adapter of the same seed (`init_adapter=<sft run>/adapter_best`).
  The base model writes no `<<…>>` annotations, so V⁺ would have nothing to check on its samples.
- **Shared samples, refreshed once.** Per seed, one pipeline dir `v2_onpolicy_s<seed>` (`vsca.cli onpolicy`:
  sample → verify → localise → certify) is built from the SFT adapter and shared by `rft_verified`, `stepdpo_fork`,
  `vsca2` and `vsca2_uncertified` (`onpolicy_dir=…`, `refresh=once`). `vsca2_nogrounding` uses
  `v2_onpolicy_nogr_s<seed>`, built from a copy of the same samples file (`onpolicy_samples=`) with
  `vplus_grounding=false`. So every arm of a seed sees the same sampled solutions.
- **Matched step budget.** v2 arms train `ceil(n_train_problems / eff_bs)` optimizer steps per epoch for 3 epochs,
  the same as SFT and claim-span masking, cycling a seeded shuffle of their example list.
- **Sampling and localisation.** k = 8 samples per problem (`onpolicy_k=8`) at the rollout temperature. A failing
  arithmetic step gives the first error directly; otherwise Monte Carlo values with M = 4 rollouts per step boundary
  (`mc_M=4`), first error at the largest value drop, sample skipped as uncertain if that drop is below 0.5
  (`mc_min_drop=0.5`). A correct-answer sample whose only failures are grounding failures is unverified and never a
  fork (see the finding below).
- **Fork certification.** Top k = 4 snapshot candidates at the fork, excluding the erroneous token `e`, M = 4
  verified rollouts each; a candidate is certified if any rollout passes V⁺.
- **Loss.** At a fork, `−log Σ_{a∈A_t} π(a|h) + β·(−log(1 − π(e|h)))` with β = 1.0 (`fork_beta=1.0`); with an empty
  `A_t` only the unlikelihood term. SFT on V⁺-verified samples elsewhere. No reference SFT (`v2_ref_sft=false`).
- **Verifier.** V⁺ = arithmetic validity + grounding + final answer, lucky guesses not verified; grounding whitelist
  `60,24,7,12,100,1000,2,0.5,1,0.01,30,31,365,52` (point E below).
- Checkpoint selection on the validation subset, as everywhere else.
**Diagnostics first** (`pilot_v2.sh diag`, Design §5.3):
1. Grounding coverage: `diagnose grounding` on `labeling/error_taxonomy_sft.csv`. If the share of
   `wrong_quantity_or_operation` failures that V⁺ flags is **below 0.40**, the script stops before training (a better
   verifier is built first; `FORCE=1` overrides and must be reported). The gate needs manual labels; with an
   unlabelled CSV the script reports per `auto_hint` and continues with a warning.
2. Fork recoverability: `diagnose recoverability` on 500 training problems from the seed-0 SFT adapter (share of
   forks with at least one certified alternative). Reported, not gated: the design gives no threshold. If it is low,
   `vsca2` is expected to behave like `stepdpo_fork`.
**Evaluation** on the full GSM8K test split (1,319 problems) for all seven arms × 3 seeds:
- primary: greedy final-answer accuracy (`vsca.cli eval`, `test.jsonl`);
- secondary: SVAMP transfer (`svamp.jsonl`), pass@8 from 8 samples at temperature 0.7 (`vsca.cli passk`, vLLM,
  `test_pass8.jsonl`), and the V⁺ valid-trace rate of the greedy test generations (`diagnose vplus_trace`,
  `vplus_verified_share`).
**Statistic.** Paired bootstrap over test items and seeds (`vsca.cli stats compare`), `vsca2` against every other arm,
written to `runs/Qwen2.5-1.5B/v2pilot/`; the rule below is applied in `v2pilot/decision.json`.
**Decision rule** (greedy GSM8K test accuracy, observed paired-bootstrap difference):
- `vsca2` ahead of **both** `rft_verified` and claim-span masking by **≥ 1.0 point**: go (full study).
- `vsca2` within 0.5 points of **either** of them, or behind it: no-go for this version.
- Otherwise: discuss before spending more GPU time.
- If `rft_verified` matches `vsca2` (within 0.5 points), the contribution is the V⁺ filter, not the certified sets,
  and is reported that way.
The secondary metrics are reported but do not change the rule; a secondary gain with no greedy gain is a reason
for a new pre-registration, not a change applied after the fact.
## Finding before pre-registration: grounding coverage is far below the 0.40 gate
V⁺ grounding false-rejects correct reference steps. The trade-off between false rejection and coverage was measured
on `labeling/error_taxonomy_sft.csv` (200 SFT failures and their references) before any training. Point E was chosen
and is the default `grounding_whitelist`:
| Point | Ref. false rejection | Flags / 163 `needs_manual` failures |
|---|---|---|
| FEAT-001 (strict) | 45.5 % | 54 |
| C | 13.0 % | 20 |
| D | 9.5 % | 17 |
| **E (chosen)** | **8.5 %** | **14** |
| F | 5.5 % | 11 |
| G | 5.0 % | 10 |
| H | 1.5 % | 6 |
The definition of each point is in README, section "VSCA v2". F, G and H were rejected because they let invented
operands pass.
**Grounding coverage at E is 14/163 = 8.6 % of the auto-hinted `needs_manual` failures, before manual labels. That
is far below the 0.40 gate of Design §5.3.** The gate stays. It is evaluated on `manual_label` once
`error_taxonomy_sft.csv` is labelled; the auto-hint figure is not a substitute. Until then, grounding is expected to
contribute little. Planning errors are carried by Monte Carlo localisation: a grounding failure alone never makes
a fork, and in wrong-answer samples it is used only when the MC value also drops at that step. The
`vsca2_nogrounding` arm tests this expectation.
