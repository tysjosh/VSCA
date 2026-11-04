# Go/no-go pilot (pre-registered 2026-11-03, before any result)

**Question.** Does VSCA, in the version tuned so far, beat SFT and claim-span masking on held-out GSM8K test accuracy
by enough to justify the full 12-seed study?

**Runs** (Qwen2.5-1.5B, LoRA, `configs/base.yaml`, learning rate from `tune` / `best_lr.json`):

| Method | Certificates | Seeds |
|---|---|---|
| VSCA | fixed base-model certificates (`cert_base.jsonl`), as in `tune` | 0–4 |
| SFT | — | 0–4 |
| Claim-span masking (ρ = 0.5) | — | 0–4 |

Seed 0 of each method is its `tune` run at the chosen learning rate (identical configuration); seeds 1–4 are new.
Checkpoint selection on the validation subset, as everywhere else.

**Evaluation** on the full GSM8K test split (1,319 problems):
- primary: greedy final-answer accuracy (`vsca.cli eval`);
- secondary: pass@1, pass@8 and maj@8 from 8 samples per problem at temperature 0.7 (`vsca.cli passk`, vLLM).

**Statistic.** Paired bootstrap over test items and seeds (`vsca.cli stats compare`), for VSCA − SFT and
VSCA − claim-span masking. Expected standard error of each difference ≈ 0.8 points (from the SFT pilot's seed spread).

**Decision rule** (on greedy accuracy, the paper's primary metric):
- VSCA ahead of **both** baselines by ≥ 1.0 point: run the full study as planned (12 seeds, re-certification every
  epoch).
- VSCA within 0.5 points of **either** baseline (or behind it): do not run the full study with this version. Either
  report a smaller null-result study, or change the method (e.g. search more decision positions than B = 4) and pilot
  the changed version.
- Otherwise: discuss before spending more GPU time.

pass@8 / maj@8 are reported but do not change the rule; a large sampled-answer gain with no greedy gain would be
discussed as a reason to change the primary metric in a new pre-registration, not applied after the fact.

## Outcome (recorded 2026-11-04): no-go, stopped after 3 of 5 seeds
**Deviation from the plan.** The pilot was stopped after seeds 0–2 (9 of the 15 runs evaluated) instead of 5 seeds,
and pass@8 / maj@8 were not computed. The decision was made after seeing the 3-seed greedy results below. Two more
seeds would have had to move VSCA − claim-span masking from +0.2 to ≥ 1.0 point, which needs a mean gain of about
+2.2 points on the remaining seeds, against per-seed differences of +0.5, −0.1 and +0.2 so far. This early stop is a
departure from the pre-registration and is reported as such.

Greedy GSM8K test accuracy (%):
| Seed | SFT | VSCA | Claim-span masking |
|---|---|---|---|
| 0 | 60.3 | 62.6 | 62.1 |
| 1 | 57.0 | 59.3 | 59.4 |
| 2 | 60.7 | 60.9 | 60.7 |
| Mean | 59.3 | 60.9 | 60.7 |

VSCA − claim-span masking = +0.2 points (mean of seeds 0–2), inside the 0.5-point band of the rule, so the full
12-seed study is **not** run with this version. VSCA − SFT = +1.6 points, but claim-span masking alone gives +1.4,
so the gain is attributed to masking the claim spans, not to the certified sets. The changed method is VSCA v2
(`VSCA_V2_DESIGN.md`, pre-registered in `PILOT_V2.md`). The differences above are means of per-seed accuracies;
the paired bootstrap (`vsca.cli stats compare`) was not run on the 3-seed results.
