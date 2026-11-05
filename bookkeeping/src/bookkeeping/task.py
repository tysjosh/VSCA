"""FinBalance records as a text task: prompt, canonical response, parser and ledger replay.

Response format (one step per line):
    ENTRIES
    E1 | D002 | Accounts Receivable | Service Revenue | 4530.84
    ...
    BALANCE SHEET
    assets | Cash | 20384.98
    ...
    equity | Retained Earnings | 2064.59
    END
Balance-sheet lines are in a canonical order: assets, liabilities, equity; accounts in the record's allowed-account
order, then any other account alphabetically. The ledger is FinBalance's own (``finbalance.ledger``), so balances
match the benchmark's ground truth.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

FB = Path(__file__).resolve().parents[2] / "external" / "finbalance"
if str(FB) not in sys.path:
    sys.path.insert(0, str(FB))

from finbalance.accounts import ACCOUNT_TYPES  # noqa: E402
from finbalance.ledger import Ledger  # noqa: E402
from finbalance.schemas import OpeningBalance, Posting  # noqa: E402

SECTIONS = ("assets", "liabilities", "equity")
ENTRY_RE = re.compile(r"^E(\d+) \| (D\d+(?:,D\d+)*) \| ([^|]+?) \| ([^|]+?) \| (\d+\.\d\d)$")
BS_RE = re.compile(r"^(assets|liabilities|equity) \| ([^|]+?) \| (-?\d+\.\d\d)$")


def entry_key(docs, debit, credit, amount) -> tuple:
    return (",".join(sorted(docs)), debit, credit, f"{float(amount):.2f}")


def entry_line(k: int, key: tuple) -> str:
    return f"E{k} | {key[0]} | {key[1]} | {key[2]} | {key[3]}\n"


def bs_lines(rec: dict, bs: dict) -> list[str]:
    order = {a: i for i, a in enumerate(rec["allowed_accounts"])}
    out = []
    for sec in SECTIONS:
        accts = sorted(bs[sec], key=lambda a: (order.get(a, len(order)), a))
        out += [f"{sec} | {a} | {float(bs[sec][a]):.2f}\n" for a in accts]
    return out


def gold_keys(rec: dict) -> list[tuple]:
    return [entry_key(e["doc_refs"], e["debit_account"], e["credit_account"], e["amount"]) for e in rec["expected_entries"]]


def prompt(rec: dict, entries_only: bool = False) -> str:
    ob = rec["opening_balance"]
    opening = "".join(f"{sec} | {a} | {float(v):.2f}\n" for sec in SECTIONS for a, v in ob[sec].items())
    docs = "".join(f"[{d['doc_id']}] {d['doc_type']} - {d['title']} ({d['date']})\n{d['ocr_text'].strip()}\n\n"
                   for d in rec["documents"])
    task = ("Record the journal entries for the period from the documents.\n" if entries_only else
            "Record the journal entries for the period from the documents, then the closing balance sheet.\n")
    return (task +
            f"Period: {rec['period_start']} to {rec['period_end']}\n"
            f"Opening balance:\n{opening}"
            f"Allowed accounts: {'; '.join(rec['allowed_accounts'])}\n"
            f"Documents:\n{docs}"
            f"Answer:\n")


def response(rec: dict, entries_only: bool = False) -> str:
    """Entries-only (pilot): the entries then END; the ledger computes the balance sheet from them."""
    ents = "".join(entry_line(k + 1, key) for k, key in enumerate(gold_keys(rec)))
    if entries_only:
        return "ENTRIES\n" + ents + "END"
    return "ENTRIES\n" + ents + "BALANCE SHEET\n" + "".join(bs_lines(rec, rec["expected_balance_sheet"])) + "END"


def replay(rec: dict, keys: list[tuple]):
    """Balance sheet (dict with assets / liabilities / equity) from the opening balance and the given entries, or
    None if an entry uses an account the ledger does not know."""
    led = Ledger(OpeningBalance(**rec["opening_balance"]))
    try:
        for docs, debit, credit, amount in keys:
            led.apply_posting(Posting(doc_refs=docs.split(","), debit_account=debit, credit_account=credit,
                                      amount=float(amount), posting_date=rec["period_end"], label="model"))
    except (KeyError, ValueError):
        return None
    bs = led.build_balance_sheet(rec["period_end"])
    return {"assets": bs.assets, "liabilities": bs.liabilities, "equity": bs.equity}


def valid_entry(rec: dict, debit: str, credit: str, amount: str, docs: list[str]) -> list[str]:
    """Gold-free checks of one entry: known and allowed accounts, debit != credit, positive amount, cited documents
    exist in the bundle."""
    errs = []
    allowed = set(rec["allowed_accounts"])
    for acc in (debit, credit):
        if acc not in allowed or acc not in ACCOUNT_TYPES:
            errs.append("unknown_account")
    if debit == credit:
        errs.append("same_account")
    if float(amount) <= 0:
        errs.append("nonpositive_amount")
    ids = {d["doc_id"] for d in rec["documents"]}
    if any(d not in ids for d in docs):
        errs.append("unknown_doc")
    return sorted(set(errs))


__all__ = ["ENTRY_RE", "BS_RE", "entry_key", "entry_line", "bs_lines", "gold_keys", "prompt", "response", "replay",
           "valid_entry", "Counter"]
