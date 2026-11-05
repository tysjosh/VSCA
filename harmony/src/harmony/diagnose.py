"""The pre-registered fit diagnostic (harmony/DIAGNOSTIC.md) for model samples.

For each sample: parse it, walk the steps, and find the **first error**, exactly:
  * ``format``   - a malformed or missing line, a wrong chord label, or a changed soprano note;
  * ``rule``     - the voicing breaks a voice-leading rule (vertical, or in the move from the previous chord);
  * ``dead_end`` - the voicing breaks no rule but no rule-free completion of the exercise exists from it
                   (a planning error, located exactly by dynamic programming).
At a ``rule`` / ``dead_end`` error, the **certified set** is every voicing that follows the verified prefix without
a rule violation and can still be completed. It is never empty there, because the prefix was completable. The
diagnostic also maps the error to the **fork token**: the first token of the sample's line that no certified line
shares, with the set of certified next tokens at that point.
"""

from __future__ import annotations

from collections import Counter

from .rules import Certifier, Exercise, check_step
from .textfmt import parse, step_line


def first_error(ex: Exercise, text: str, cert: Certifier | None = None) -> dict:
    cert = cert or Certifier(ex)
    vs, starts, bad = parse(ex, text)
    prev = None
    for t, v in enumerate(vs):
        errs = check_step(prev, v, ex.chords[t - 1] if t else None, ex.chords[t], ex.key)
        if errs or not cert.completable(t, v):
            return {"ok": False, "step": t, "kind": "rule" if errs else "dead_end", "rules": errs,
                    "certified": cert.certified(t, prev), "line_start": starts[t], "prev": prev}
        prev = v
    if bad is not None:
        return {"ok": False, "step": bad, "kind": "format", "rules": [], "certified": cert.certified(bad, prev),
                "line_start": starts[bad], "prev": prev}
    return {"ok": True, "step": None, "kind": None, "rules": [], "certified": [], "line_start": None, "prev": prev}


def fork_token(tok, ex: Exercise, prompt: str, text: str, err: dict):
    """(token index in the response, erroneous token id, certified next-token ids) for a rule / dead-end error.
    The response is tokenized as the model produced it, so the index lines up with the sample's token ids."""
    t, start = err["step"], err["line_start"]
    line = text[start:].split("\n", 1)[0] + "\n"
    cert_lines = [step_line(ex, t, v) for v in err["certified"]]
    lcp = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    char = start + max(lcp(line, c) for c in cert_lines)  # first char of the sample no certified line shares
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    k = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > char)
    head = text[: enc["offset_mapping"][k][0]]
    # certified next tokens: the token that follows ``head`` when the line is completed with a certified voicing
    alts = set()
    for c in cert_lines:
        full = text[:start] + c
        if not full.startswith(head):
            continue
        e2 = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        j = next((i for i, (s, e) in enumerate(e2["offset_mapping"]) if s >= len(head)), None)
        if j is not None and e2["input_ids"][:j] == enc["input_ids"][:k]:
            alts.add(e2["input_ids"][j])
    return k, enc["input_ids"][k], sorted(alts - {enc["input_ids"][k]})


def summarise(rows: list[dict]) -> dict:
    """rows: one per sample with ``ok``, ``kind``, ``n_certified``, ``n_cert_tokens`` (None if not computed)."""
    n = len(rows)
    fail = [r for r in rows if not r["ok"]]
    kinds = Counter(r["kind"] for r in fail)
    located = [r for r in fail if r["kind"] in ("rule", "dead_end")]
    share = lambda a, b: a / b if b else 0.0
    tok_rows = [r for r in located if r.get("n_cert_tokens") is not None]
    return {
        "samples": n,
        "failure_rate": share(len(fail), n),
        "parse_rate": share(sum(r["kind"] != "format" for r in rows), n),
        "first_error_kinds": dict(kinds),
        "located_share_of_failures": share(len(located), len(fail)),
        "dead_end_share_of_located": share(kinds.get("dead_end", 0), len(located)),
        "certified_alt_share": share(sum(r["n_certified"] >= 1 for r in located), len(located)),
        "certified_ge2_share": share(sum(r["n_certified"] >= 2 for r in located), len(located)),
        "mean_certified_voicings": share(sum(r["n_certified"] for r in located), len(located)),
        "token_fork_with_cert_alt_share": share(sum(r["n_cert_tokens"] >= 1 for r in tok_rows), len(tok_rows)),
        "token_fork_ge2_share": share(sum(r["n_cert_tokens"] >= 2 for r in tok_rows), len(tok_rows)),
        "rules_broken": dict(Counter(x for r in fail for x in r.get("rules", []))),
    }
