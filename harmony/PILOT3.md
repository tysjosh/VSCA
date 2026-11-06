# Harmony pilot 3: replication on unseen exercises (pre-registered 2026-11-05 19:15, before any pilot-3 run)

**Why.** Pilots 1 and 2 drew on-policy samples from the first 4,000 **SFT training** exercises. The samples were not
memorised (31 % fully valid), so their forks are real errors, but the design favours the fork arms over `rft`: `rft`'s
verified samples are near-copies of SFT data. Pilot 3 repeats the comparison on exercises **never seen in SFT**.

**On-policy exercises.** A new split `data/onpolicy.jsonl`, 4,000 exercises from the training distribution (major,
training keys, 6–8 chords), generator seed 5, deduplicated against every other split. Earlier splits are regenerated
byte-identically (checked with `cmp` before running), so the SFT adapters `runs/sft_s<seed>` of pilot 1 are reused.

**Pools, arms, settings.** As pilots 1–2 (`pilot.py pools` / `pools_step` with `--split onpolicy`). Arms: `rft`,
`neg`, `single`, `set` (token level) and `single_step`, `set_step` (step level). All arms: lr 5e-5, 2 epochs,
16 main-stream sequences and 8 forks per optimizer step (micro-batch 2 × 8 accumulation, 1 fork per micro-batch),
the same seeded main stream for every arm of a seed. Outputs in `runs/pilot3_s<seed>/`.

**Evaluation, seeds, statistic.** As pilot 1: greedy; test_id + test_long pooled primary; seeds 0–2; paired bootstrap.

**Decision rule** (points): primary claim, exact first-error training beats verified filtering:
**go** if `set` − `rft` ≥ +2.0 with the 95 % CI above 0; **no-go** if `set` − `rft` ≤ +0.5; otherwise inconclusive.
Reported, not in the rule: `neg` − `rft`, `single` − `rft`, `set` − `single`, `set_step` − `single_step`, per split.
