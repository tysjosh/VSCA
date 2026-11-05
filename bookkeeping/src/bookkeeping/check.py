"""Exact first error, certified set and fork token for a bookkeeping response.

Walk the response line by line:
  * entry lines: an entry is *verified* if it matches one of the expected entries not yet used (any order:
    accounts, amount and cited documents, as FinBalance's strict match). First-error kinds:
      ``invalid``        breaks a gold-free ledger check (unknown account, debit = credit, amount <= 0, unknown doc);
      ``no_match``       valid but matches no remaining expected entry (sub-kind: which fields differ from the
                         closest remaining entry: amount / docs / account);
      ``premature_end``  the balance-sheet header while expected entries remain;
      ``extra_entry``    an entry after every expected entry was used;
    certified set: every remaining expected entry (in any order), or the balance-sheet header if none remains.
  * balance-sheet lines (reached only after all entries verified, so the replay of the model's own entries equals
    the ground truth): ``bs_value`` (right account, wrong amount: an aggregation error), ``bs_account`` (wrong or
    missing account line), ``bs_extra`` (a line after the last one); certified set: the single correct line.
  * ``format``: a line that does not parse.
Separately, ``audit`` replays the model's own parsed entries (right or wrong) to compare its reported balance
sheet with the replay (FinBalance's BS_exact / BS_recon / self-consistency).
"""

from __future__ import annotations

from collections import Counter

from .task import BS_RE, ENTRY_RE, bs_lines, entry_key, entry_line, gold_keys, replay, valid_entry


def _closest(key, remaining: Counter) -> str:
    best = None
    for r in remaining:
        diff = [f for f, a, b in zip(("docs", "debit", "credit", "amount"), key, r) if a != b]
        if best is None or len(diff) < len(best):
            best = diff
    if not best:
        return "none"
    if best == ["amount"]:
        return "amount"
    if best == ["docs"]:
        return "docs"
    return "account" if any(f in ("debit", "credit") for f in best) else "+".join(best)


def first_error(rec: dict, text: str, entries_only: bool = False) -> dict:
    """``entries_only``: the response ends with END after the entries (the ledger computes the balance sheet)."""
    lines = text.split("\n")
    stop = "END" if entries_only else "BALANCE SHEET"
    remaining = Counter(gold_keys(rec))
    pos = 0

    def err(kind, i, certified, **kw):
        return {"ok": False, "kind": kind, "line": i, "line_start": starts[i] if i < len(starts) else pos,
                "certified": certified, **kw}

    starts = []
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    pos = len(text)
    if not lines or lines[0].strip() != "ENTRIES":
        return err("format", 0, ["ENTRIES\n"], phase="header")
    i, k = 1, 1
    while True:
        cert_entries = [entry_line(k, r) for r in remaining] or [stop + "\n"]
        if i >= len(lines):
            return err("format", i, cert_entries, phase="entries")
        ln = lines[i].strip()
        if ln == stop:
            if remaining:
                return err("premature_end", i, cert_entries, phase="entries", n_remaining=sum(remaining.values()))
            if entries_only:
                return {"ok": True, "kind": None, "line": None, "line_start": None, "certified": []}
            break
        m = ENTRY_RE.match(ln)
        if not m or int(m.group(1)) != k:
            return err("format", i, cert_entries, phase="entries")
        docs = m.group(2).split(",")
        key = entry_key(docs, m.group(3), m.group(4), m.group(5))
        bad = valid_entry(rec, m.group(3), m.group(4), m.group(5), docs)
        if not remaining:
            return err("extra_entry", i, [stop + "\n"], phase="entries", rules=bad)
        if bad:
            return err("invalid", i, cert_entries, phase="entries", rules=bad)
        if remaining[key] == 0:
            return err("no_match", i, cert_entries, phase="entries", sub=_closest(key, remaining))
        remaining[key] -= 1
        if remaining[key] == 0:
            del remaining[key]
        i, k = i + 1, k + 1
    gold = bs_lines(rec, rec["expected_balance_sheet"])
    i += 1
    for j, g in enumerate(gold):
        if i >= len(lines):
            return err("format", i, [g], phase="bs")
        ln = lines[i].strip()
        m = BS_RE.match(ln)
        if not m:
            return err("format", i, [g], phase="bs")
        if ln + "\n" != g:
            same_acct = ln.rsplit(" | ", 1)[0] == g.rsplit(" | ", 1)[0]
            return err("bs_value" if same_acct else "bs_account", i, [g], phase="bs")
        i += 1
    if i >= len(lines) or lines[i].strip() not in ("END", ""):
        return err("bs_extra", i, ["END"], phase="bs")
    return {"ok": True, "kind": None, "line": None, "line_start": None, "certified": []}


def audit(rec: dict, text: str) -> dict:
    """BS_exact (reported = gold), BS_recon (replay of the model's own entries = gold), self-consistency."""
    keys, reported = [], {s: {} for s in ("assets", "liabilities", "equity")}
    phase = None
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln in ("ENTRIES", "BALANCE SHEET", "END"):
            phase = ln
            continue
        if phase == "ENTRIES" and (m := ENTRY_RE.match(ln)):
            keys.append(entry_key(m.group(2).split(","), m.group(3), m.group(4), m.group(5)))
        elif phase == "BALANCE SHEET" and (m := BS_RE.match(ln)):
            reported[m.group(1)][m.group(2)] = float(m.group(3))
    gold = bs_lines(rec, rec["expected_balance_sheet"])
    rep_lines = bs_lines(rec, reported)
    rp = replay(rec, keys)
    rp_lines = bs_lines(rec, rp) if rp is not None else None
    return {"bs_exact": rep_lines == gold, "bs_recon": rp_lines == gold, "self_consistent": rp_lines == rep_lines,
            "entries_exact": Counter(keys) == Counter(gold_keys(rec))}


def fork_token(tok, text: str, err: dict):
    """(token index, erroneous token id, certified next-token ids) at a located first error, as in harmony."""
    start = err["line_start"]
    line = text[start:].split("\n", 1)[0] + "\n"
    cert = err["certified"]
    lcp = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    char = start + max(lcp(line, c) for c in cert)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    k = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > char)
    head = text[: enc["offset_mapping"][k][0]]
    alts = set()
    for c in cert:
        full = text[:start] + c
        if not full.startswith(head):
            continue
        e2 = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        j = next((i for i, (s, e) in enumerate(e2["offset_mapping"]) if s >= len(head)), None)
        if j is not None and e2["input_ids"][:j] == enc["input_ids"][:k]:
            alts.add(e2["input_ids"][j])
    return k, enc["input_ids"][k], sorted(alts - {enc["input_ids"][k]})
