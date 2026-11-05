# harmony: four-part harmonisation as an exactly checkable domain for verified credit assignment

On GSM8K, VSCA failed because the step verifier missed most errors and first errors had to be estimated by Monte
Carlo (`vsca/PILOT_V2.md`, `vsca/PROBE_V3.md`). Here every step (one chord voiced as S, A, T, B) is checked exactly
by voice-leading rules, the first error is exact (including *dead ends*: a legal chord after which no legal
completion exists, found by dynamic programming), and the certified set at an error is every legal, completable
voicing, usually more than one.

| File | Contents |
|---|---|
| `src/harmony/theory.py` | spelled notes, keys, Roman-numeral chords |
| `src/harmony/rules.py` | the rule checker, voicing enumeration, exact completability (`Certifier`) |
| `src/harmony/generate.py` | solvable exercises with a reference realisation |
| `src/harmony/textfmt.py` | prompt / response format and strict parser |
| `src/harmony/diagnose.py` | exact first error, certified set, fork token |
| `scripts/make_data.py` | train / test splits (`data/`, deterministic from seed 0) |
| `scripts/sft.py` | LoRA SFT, resumable (checkpoints, deadline exit 75) |
| `scripts/sample.py` | greedy + n samples with vLLM |
| `scripts/diagnose.py` | the fit diagnostic and its go rule |
| `scripts/run_diagnostic.sh` | the whole diagnostic on the GPU VM, resumable, backed up to `olukotunjosh/harmony-runs` |
| `DIAGNOSTIC.md` | pre-registration |

```bash
# on the VM, after vm_start.sh
screen -dmS harmony bash ~/VSCA/harmony/scripts/run_diagnostic.sh     # progress: ~/VSCA/harmony_progress.log
# on CPU
python scripts/make_data.py --out /tmp/hdata --n_train 200 --n_test 30
python scripts/sft.py --data /tmp/hdata --out /tmp/hrun --model HuggingFaceTB/SmolLM2-135M --epochs 1 --limit 32
```
