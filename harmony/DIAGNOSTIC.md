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
