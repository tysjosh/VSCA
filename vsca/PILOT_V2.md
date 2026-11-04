# VSCA v2 pilot

The pre-registration (arms, seeds, evaluation, decision rule) is written before any v2 result. This file records
the findings fixed before it.

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
