# Bookkeeping fit diagnostic (pre-registered 2026-11-05, before any model was trained or sampled on this task)

**Question.** Does document-grounded bookkeeping (FinBalance, [arXiv 2606.15949](https://arxiv.org/abs/2606.15949))
give verified credit assignment the conditions it needs: failures a checker locates exactly, and certified sets
with more than one correct next step? Same question and thresholds as `harmony/DIAGNOSTIC.md`.

**Task.** From a bundle of source documents (OCR text), the opening balance and the allowed accounts, write the
journal entries (`E<k> | <cited docs> | <debit> | <credit> | <amount>`), then the closing balance sheet in a
canonical account order (`bookkeeping/src/bookkeeping/task.py`). Clean records only (no forced inconsistencies).
The ledger is FinBalance's own (pinned commit `a1062b7b`).

**Exact first error** (`check.py`). Entries are order-free: an entry is verified if it matches an expected entry
not yet used (accounts, amount, cited documents: FinBalance's strict match). Kinds: `invalid` (gold-free ledger
check: unknown account, debit = credit, amount ≤ 0, unknown document), `no_match` (sub-kind amount / docs /
account), `premature_end`, `extra_entry`; balance-sheet lines (reached only when every entry is verified, so the
ledger replay of the model's own entries is the truth): `bs_value` (aggregation error), `bs_account`, `bs_extra`;
`format`. **Certified set:** every remaining expected entry (any order), or the single correct balance-sheet line.
Note: unlike harmony, entry certification uses the expected entries (available at training time, not at
inference); only `invalid` and the balance-sheet checks are gold-free. Reported as `gold_free_share_of_located`.

**Data** (`scripts/make_data.py`, FinBalance generator; prompt ≤ 3,300 and response ≤ 1,100 tokens):
train 6,000 records (6 industries, month and quarter periods, difficulty 1–2); `test_id` (same distribution),
`test_L3` (difficulty 3), `test_ind` (healthcare clinic and property management, never seen), `test_year` (year
periods, never seen); 300 each. Length-skipped counts in `data/stats.json`. The length limit selects the shorter
records of `test_L3` and `test_year` (in a CPU dry run 117 and 228 records were skipped to keep 300), so those splits
under-represent their longest bundles.

**Model.** Qwen2.5-1.5B, LoRA r = 16, 2 epochs, lr 1e-4, batch 16, loss on the response, gradient checkpointing,
seed 0 (`harmony/scripts/sft.py`). **Samples:** greedy + 8 at temperature 0.7 (`harmony/scripts/sample.py`).

**Measured** (8 samples per problem, per split): failure rate, parse rate, located share, certified ≥ 2 share
(as in harmony), plus first-error kinds, gold-free share, token-level fork with certified alternatives, and
FinBalance's BS_exact, BS_recon (replay of the model's own entries) and self-consistency.

**Go rule** (identical to harmony): at least one held-out split (`test_L3`, `test_ind`, `test_year`) with failure
rate ≥ 0.30 and, on that split, parse rate ≥ 0.90, located share ≥ 0.50 and certified ≥ 2 share ≥ 0.50.
