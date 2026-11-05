# Harmony pilot 2: is the certified *set* better than one certified target? (pre-registered 2026-11-05, before any run)

**Why.** Pilot 1 (PILOT.md) ended in *discuss*: training at the exact first error beat verified filtering
(`set` − `rft` = +2.6, 95 % CI +0.1 to +5.0), but `set` − `single` was +0.7 (CI −1.5 to +2.9). That comparison was
weak by construction: certification was at the next **token**, and only 23 % of forks had ≥ 2 certified tokens, so
`set` and `single` were the same loss on 77 % of forks. Different certified voicings usually share the first token
where they leave the sample. Pilot 2 certifies the whole **step** (one chord line) instead, where the diagnostic
found ≥ 2 certified voicings for 58–60 % of errors.

**Step forks** (`pilot.py pools_step`, from pilot 1's on-policy samples `runs/pilot_s<seed>/train_samples.jsonl`):
for every sample whose exact first error is `rule` or `dead_end`, the verified prefix (the response before the
erroneous chord line) and the candidate lines = every certified voicing of that chord (legal after the prefix and
completable). At most 8 candidates: the star plus a seeded sample of the rest. **Star** = the candidate with the least
total voice movement from the previous chord (closest spacing for a first chord; ties: enumeration order).

**Arms** (from the SFT adapter of the seed, `pilot.py train`, same settings and the **same main stream** as pilot 1:
the seed's verified pool, same seeded order and step count, 4 step forks per micro-batch):
| Arm | Fork loss at the erroneous step |
|---|---|
| `single_step` | −log π(star line \| prefix): SFT on one certified line |
| `set_step` | −log Σ_{v ∈ certified} π(v line \| prefix) |
Each step fork counts as the tokens of its star line in the loss's token mean, so both arms give forks the same
weight. `rft` (pilot 1) is the shared baseline: identical main stream, no forks.

**Evaluation, seeds, statistic:** as pilot 1 (greedy; test_id + test_long pooled primary; seeds 0–2; paired bootstrap).

**Decision rule** (points): **go** for the set-valued claim if `set_step` − `single_step` ≥ +1.0 with the 95 % CI
above 0; **no-go** if `set_step` − `single_step` ≤ +0.3; otherwise inconclusive. Also reported: `set_step` − `rft`,
`single_step` − `rft`, per-split results, and the share of step forks with ≥ 2 candidates.
