# Harmony VSCA pilot (pre-registered 2026-11-05, before any pilot run; diagnostic result in DIAGNOSTIC.md)

**Question.** With exact first errors and exact certified sets (the conditions GSM8K lacked), does training on the
**certified set** at the first error improve over training on verified samples alone, and over a single
certified target?

**Setup.** Qwen2.5-1.5B. Per seed s ∈ {0, 1, 2}: SFT adapter `runs/sft_s<s>` (DIAGNOSTIC.md settings, seed s).
On-policy data: the SFT adapter samples 4 solutions at temperature 0.7 for the first 4,000 training exercises
(`data/train_pilot.jsonl`). `pilot.py pools` finds each sample's exact first error: fully valid samples are the
**verified pool**; samples with a `rule` or `dead_end` first error give one **fork** each: token index k, the
erroneous token e, the certified next tokens A (tokens that continue some certified voicing; forks with empty A are
dropped for every arm), and a\* = the token of A shared by the most certified voicings (ties: smallest id).

**Arms** (all start from the seed's SFT adapter; `pilot.py train`, lr 5e-5 cosine, 2 epochs, 16 sequences of the
main stream per step plus 8 fork examples for fork arms):
| Arm | Main stream | Fork loss at position k |
|---|---|---|
| `sft_cont` | reference solutions of the 4,000 exercises | none |
| `rft` | verified pool | none |
| `neg` | verified pool | −log(1 − π(e)) |
| `single` | verified pool | −log π(a\*) |
| `set` | verified pool | −log Σ_{a∈A} π(a) |
| `full` | verified pool | set + neg |
Every arm of a seed takes the same number of steps, ⌈|verified| / 16⌉ × 2, and the fork arms see the **same**
main stream as `rft` (same seeded order) plus the same fork stream. Each fork position counts as one token in the
token mean of the loss, so forks neither displace verified samples nor outweigh ordinary tokens (the two confounds
of the GSM8K pilots).

**Evaluation.** Greedy decoding, `pilot.py score`: a solution is correct if it has no first error (every chord
legal, no dead end, well formed). Splits: test_id and test_long (**primary**, pooled: 600 exercises), test_minor and
test_keys (secondary). Also scored: the SFT adapter itself.

**Statistic.** Paired bootstrap over exercises and seeds (`vsca.cli stats compare`), pooled primary splits.

**Decision rule** (differences in points of the correct rate, observed paired-bootstrap difference):
- **go**: `set` − `rft` ≥ +2.0 **and** `set` − `single` ≥ +1.0 (the set beats both filtering and one target);
- **no-go**: `set` − `rft` ≤ +0.5;
- otherwise discuss.
`full`, `neg`, `sft_cont`, SFT and the secondary splits are reported and do not change the rule.
