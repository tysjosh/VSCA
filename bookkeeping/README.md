# bookkeeping: FinBalance as a domain for verified credit assignment

Document-grounded bookkeeping from [FinBalance](https://github.com/Devansh1105/finbalance) (EMNLP 2026 Findings):
source documents → journal entries → closing balance sheet, with FinBalance's deterministic ledger as the checker.
FinBalance is a benchmark only; this project tests whether the task suits step-level verified training.

| File | Contents |
|---|---|
| `src/bookkeeping/task.py` | prompt, canonical response, parser, ledger replay, gold-free entry checks |
| `src/bookkeeping/check.py` | exact first error (order-free entries), certified sets, fork token, BS_exact / BS_recon audit |
| `scripts/make_data.py` | splits from FinBalance's generator |
| `scripts/diagnose.py` | fit diagnostic and go rule |
| `scripts/run_diagnostic.sh` | the diagnostic on the GPU VM (clones FinBalance at `a1062b7b`), backed up to `olukotunjosh/bookkeeping-runs` |
| `DIAGNOSTIC.md` | pre-registration |

SFT and sampling reuse `harmony/scripts/sft.py` and `harmony/scripts/sample.py`. FinBalance is cloned into
`external/finbalance` (not committed): `git clone https://github.com/Devansh1105/finbalance external/finbalance`.

```bash
screen -dmS books bash ~/VSCA/bookkeeping/scripts/run_diagnostic.sh   # progress: ~/VSCA/books_progress.log
```
