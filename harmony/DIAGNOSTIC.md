# Harmony fit diagnostic (pre-registered 2026-11-05, before any model was trained or sampled on this task)

**Question.** Does four-part harmonisation give verified credit assignment (VSCA) the conditions it lacked on
GSM8K: failures the checker catches, exact first errors, and certified sets with more than one correct option?
No method is trained here; this decides whether a VSCA pilot on harmony is worth running.

**Task.** Given a key, a Roman-numeral progression and the soprano line, write alto, tenor and bass, one chord per
line (`harmony/src/harmony/textfmt.py`). Rules (`rules.py`): chord tones spelled in the key, correct inversion,
complete chords, SATB ranges, no crossing, spacing ≤ an octave in the upper voices, no doubled leading tone or
seventh, leaps ≤ an octave (S, B) or a fifth (A, T), no parallel fifths or octaves, no direct fifths/octaves in the
outer voices with a soprano leap, no overlap, the soprano leading tone resolves to the tonic in a dominant → tonic
move, the chordal seventh resolves down by step.

**Exact first error.** The first step whose voicing breaks a rule (`rule`) or breaks none but has no rule-free
completion to the end (`dead_end`, found by dynamic programming), or the first malformed line (`format`).
**Certified set** at a `rule` / `dead_end` error: every voicing that follows the verified prefix legally and can
still be completed. No sampling is involved anywhere.

**Data** (`scripts/make_data.py`, seed 0, every exercise generated solvable, no duplicates across splits):
train 12,000 major-key exercises of 6–8 chords (9 keys); `test_id` (same distribution), `test_long` (12–16 chords),
`test_minor` (minor keys, never seen), `test_keys` (F#, Db, B major, never seen); 300 each.

**Model.** Qwen2.5-1.5B, LoRA r = 16 on all projections, 2 epochs, lr 1e-4, batch 16, loss on the response,
seed 0 (`scripts/sft.py`). **Samples:** greedy plus 8 at temperature 0.7 per test problem (`scripts/sample.py`).

**Measured on the 8 samples per problem, per test split** (`scripts/diagnose.py`):
1. failure rate (any first error);
2. parse rate (share without a `format` first error);
3. located share: `rule` + `dead_end` first errors among failures;
4. certified ≥ 2 share: located errors with at least two certified voicings (the set-valued case VSCA needs);
also reported: dead-end share, mean certified voicings, token-level fork with ≥ 1 / ≥ 2 certified next tokens,
rules broken, and the greedy failure rate.

**Go rule for a harmony VSCA pilot.** At least one held-out split (`test_long`, `test_minor`, `test_keys`) with
failure rate ≥ 0.30 **and**, on that split, parse rate ≥ 0.90, located share ≥ 0.50 and certified ≥ 2 share ≥ 0.50.
Otherwise no-go: either the task is too easy after SFT (no room), or the failures are not of the kind VSCA can use.
`test_id` is reported but not in the rule.

**If go,** the pilot mirrors VSCA v2 with exact localisation and certification: SFT, rft (checker-verified
samples), first-error negative only, and VSCA (certified set at the fork), 3 seeds, evaluated on the held-out
splits; pre-registered separately before it runs.

## Outcome (2026-11-05, SFT seed 0; `runs/sft_s0/diagnostic.json`, backed up to `olukotunjosh/harmony-runs`)
SFT: 1,500 steps, 9 minutes on an L40S. Samples: greedy + 8 at T = 0.7, 300 problems per split.

| Split | Greedy fail | Sampled fail | Parse | Located | Dead-end share | Certified ≥ 2 | Fork token with ≥ 1 certified alternative | Mean certified voicings |
|---|---|---|---|---|---|---|---|---|
| test_id | 0.540 | 0.722 | 0.997 | 0.996 | 0.317 | 0.597 | 0.948 | 2.3 |
| test_long | 0.957 | 0.990 | **0.888** | 0.887 | 0.290 | 0.578 | 0.914 | 2.2 |
| test_minor | 0.993 | 0.992 | 0.992 | 0.992 | 0.069 | 0.796 | 0.708 | 3.3 |
| test_keys | 1.000 | 1.000 | 0.999 | 0.999 | 0.085 | 0.813 | 0.933 | 3.6 |

**Rule result: go** (test_minor and test_keys pass every condition; test_long misses the 0.90 parse threshold).

**Caveat on what passed.** On test_minor and test_keys, 80 % and 92 % of rule errors are `non_chord_tone`: the model
spells chords as in the keys it trained on (A natural instead of A♭ in F minor). That is a missing-knowledge
failure about keys, which SFT on those keys would fix directly. On test_id and test_long the errors are the
voice-leading kind VSCA targets: parallel octaves 23–25 %, parallel fifths 21–24 %, overlap 17 %, tenor leaps
13–15 %, and **29–32 % of located first errors are exact dead ends** (legal chords with no legal completion,
the planning-error type GSM8K could not localise). test_id is not in the rule but meets every condition; test_long
misses only on parse (268 of 2,400 samples have a malformed or missing line, mostly in 12–16-chord exercises).

**Implication for the pilot pre-registration** (to be written before it runs): primary evaluation on test_id and
test_long (voice-leading and dead-end errors), test_minor and test_keys secondary.
