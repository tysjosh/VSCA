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


def steps_valid(question: str, text: str, whitelist, grounding: bool, min_steps: int) -> bool:
    """Gold-free V+ check usable at inference: at least ``min_steps`` annotation steps and every step's arithmetic
    (and, with ``grounding``, every operand) verifies. The final answer is not compared with anything."""
    from .verify_plus import check_response

    res = check_response(question, text, None, whitelist, grounding, min_steps)
    return len(res.steps) >= min_steps and res.first_failed is None


def verified_vote(answers: list, step_ok: list[bool]):
    """Majority answer among the samples with ``step_ok``; plain majority if none has. Ties: first occurrence."""
    from collections import Counter

    def vote(idx):
        c = Counter(answers[i] for i in idx if answers[i] is not None)
        if not c:
            return None
        first = {a: min(i for i in idx if answers[i] == a) for a in c}
        return max(c, key=lambda a: (c[a], -first[a]))

    ok = [i for i, s in enumerate(step_ok) if s]
    a = vote(ok) if ok else None  # explicit None test: an answer of 0 is falsy
    return a if a is not None else vote(list(range(len(answers))))


def evaluate_passk(cfg, tok, records: list[dict], adapter, n: int, temperature: float, out_path: Path | None = None,
                   seed: int = 0, backend=None) -> dict:
    """Sampled evaluation with vLLM: n answers per question at ``temperature``.

    Reports pass@1 (mean fraction correct), pass@n (any correct) and maj@n (majority vote over extracted answers,
    ties broken by first occurrence). Each question samples with its own seed, so results are reproducible and
    independent of batching. Per-item rows carry ``correct`` = pass@n, so `vsca.cli stats compare` works on them.
    """
    from collections import Counter

    from .rollout import VLLMBackend
    from .verify import final_answer
    from .verify_plus import vplus_from_cfg

    wl, grounding, min_steps = vplus_from_cfg(cfg)
    be = backend or VLLMBackend(cfg, tok)
    be.set_snapshot(Path(adapter) if adapter else None)
    prefixes = [tok(format_prompt(r["question"]), add_special_tokens=True).input_ids for r in records]
    seeds = [(seed * 1_000_003 + i) % (2**31) for i in range(len(records))]
    outs = be.generate(prefixes, cfg.gen_max_new, do_sample=True, temperature=temperature, top_p=1.0, n=n,
                       seeds=seeds)
    rows = []
    for i, r in enumerate(records):
        gold = Fraction(r["gold"])
        texts = [truncate_response(tok.decode(g, skip_special_tokens=True)) for g in outs[i * n : (i + 1) * n]]
        ok = [answer_correct(t, gold) for t in texts]
        answers = [final_answer(t, strict=False) for t in texts]
        votes = Counter(a for a in answers if a is not None)
        maj = max(votes, key=lambda a: (votes[a], -answers.index(a))) if votes else None
        step_ok = [steps_valid(r["question"], t, wl, grounding, min_steps) for t in texts]
        vmaj = verified_vote(answers, step_ok)
        rows.append({
            "id": r["id"], "n": n, "n_correct": int(sum(ok)), "correct": bool(any(ok)),
            "maj_correct": maj is not None and maj == gold,
            "vmaj_correct": vmaj is not None and vmaj == gold,
            "answers": [None if a is None else str(a) for a in answers],
            "step_ok": step_ok,
            "texts": texts,
        })
    m = max(len(rows), 1)
    summary = {
        "n_items": len(rows), "n_samples": n, "temperature": temperature,
        "pass@1": sum(x["n_correct"] for x in rows) / (m * n),
        f"pass@{n}": sum(x["correct"] for x in rows) / m,
        f"maj@{n}": sum(x["maj_correct"] for x in rows) / m,
        # majority vote among the samples whose every step passes V+ without the gold answer (falls back to maj@n
        # when no sample passes): PROBE_V3.md, experiment 2
        f"vmaj@{n}": sum(x["vmaj_correct"] for x in rows) / m,
        "step_ok_share": sum(sum(x["step_ok"]) for x in rows) / (m * n),
        "items_without_step_ok": sum(not any(x["step_ok"]) for x in rows),
    }
    if out_path is not None:
        write_jsonl(out_path, rows)
    return summary
