"""Greedy final-answer accuracy and trace validity (Section 4.5)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

from .data import format_prompt
from .modeling import generate_from_ids
from .utils import write_jsonl
from .verify import answer_correct, parse_claims, response_valid, truncate_response


def evaluate(model, tok, records: list[dict], bs: int, max_new: int, out_path: Path | None = None, desc: str = "eval",
             should_stop=None) -> dict:
    """``should_stop`` (optional) is called before each batch and may raise to stop cleanly."""
    was_training = model.training
    model.eval()
    rows = []
    for i in tqdm(range(0, len(records), bs), desc=desc):
        if should_stop is not None:
            should_stop()
        group = records[i : i + bs]
        prefixes = [tok(format_prompt(r["question"]), add_special_tokens=True).input_ids for r in group]
        gens = generate_from_ids(model, tok, prefixes, max_new, do_sample=False)
        for r, g in zip(group, gens):
            text = truncate_response(tok.decode(g, skip_special_tokens=True))
            gold = Fraction(r["gold"])
            claims = parse_claims(text)
            rows.append(
                {
                    "id": r["id"],
                    "correct": answer_correct(text, gold),
                    "trace_valid": response_valid(text, gold),
                    "n_claims": len(claims),
                    "n_valid_claims": sum(c.valid is True for c in claims),
                    "n_invalid_claims": sum(c.valid is False for c in claims),
                    "generation": text,
                }
            )
    if was_training:
        model.train()
    n = max(len(rows), 1)
    n_claims = sum(r["n_claims"] for r in rows)
    summary = {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "trace_valid": sum(r["trace_valid"] for r in rows) / n,
        "claim_validity": (sum(r["n_valid_claims"] for r in rows) / n_claims) if n_claims else None,
    }
    if out_path is not None:
        write_jsonl(out_path, rows)
    return summary
