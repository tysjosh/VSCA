"""GSM8K / SVAMP loading, prompt format, tokenisation with claim alignment, batching."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import torch

from .utils import read_jsonl, write_jsonl
from .verify import Claim, claim_valid, final_answer, parse_claims, parse_number

PROMPT = "Question: {q}\nAnswer:\n"


def format_prompt(question: str) -> str:
    return PROMPT.format(q=question.strip())


# ---------------------------------------------------------------- raw data


def prepare_gsm8k(data_dir: str, val_frac: float, split_seed: int) -> dict:
    """Download GSM8K, hold out ``val_frac`` of the official train split, save jsonl."""
    from datasets import load_dataset

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    train_all = [
        {"id": f"gsm8k-train-{i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(ds["train"])
    ]
    test = [{"id": f"gsm8k-test-{i}", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(ds["test"])]
    rng = random.Random(split_seed)
    idx = list(range(len(train_all)))
    rng.shuffle(idx)
    n_val = int(round(val_frac * len(idx)))
    val_set = set(idx[:n_val])
    train = [train_all[i] for i in range(len(train_all)) if i not in val_set]
    val = [train_all[i] for i in range(len(train_all)) if i in val_set]

    stats = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        kept = []
        n_bad = 0
        for r in rows:
            gold = final_answer(r["answer"])
            claims = parse_claims(r["answer"])
            r["gold"] = str(gold) if gold is not None else None
            r["n_claims"] = len(claims)
            r["n_valid_claims"] = sum(c.valid is True for c in claims)
            r["n_invalid_claims"] = sum(c.valid is False for c in claims)
            r["n_undetermined_claims"] = sum(c.valid is None for c in claims)
            # V_r(x_i, y_i, E_i) = 1 is assumed for training references; remove failures (Sec. 3.1).
            ref_ok = gold is not None and r["n_invalid_claims"] == 0 and r["n_undetermined_claims"] == 0
            r["reference_verified"] = ref_ok
            if name == "train" and not ref_ok:
                n_bad += 1
                continue
            kept.append(r)
        write_jsonl(out / f"gsm8k_{name}.jsonl", kept)
        stats[name] = {"n": len(kept), "removed_unverified_references": n_bad}
    return stats


def prepare_svamp(data_dir: str) -> dict:
    from datasets import load_dataset

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "test"):
        for r in load_dataset("ChilleD/SVAMP", split=split):
            gold = parse_number(str(r["Answer"]))
            rows.append(
                {
                    "id": f"svamp-{r['ID']}",
                    "question": r["question_concat"].strip(),
                    "answer": None,
                    "gold": str(gold) if gold is not None else None,
                }
            )
    write_jsonl(out / "svamp_all.jsonl", rows)
    return {"svamp": {"n": len(rows)}}


def load_split(data_dir: str, name: str, limit: Optional[int] = None) -> list[dict]:
    rows = read_jsonl(Path(data_dir) / f"{name}.jsonl")
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- tokenisation


@dataclass
class Example:
    id: str
    prompt_ids: list[int]
    resp_ids: list[int]
    resp_offsets: list[tuple[int, int]]
    answer: str
    gold: Fraction
    claims: list[Claim]
    claim_of_token: list[int]  # index of a *valid* claim each response token overlaps, else -1
    cf_pairs: list[dict] = field(default_factory=list)  # for CF-Select (Sec. 3.4)
    supervise_from: int = 0  # labels only from response token supervise_from on (v2 fork examples)

    @property
    def claim_mask(self) -> list[bool]:
        return [c >= 0 for c in self.claim_of_token]


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and a1 > b0


def counterfactual_result(result: str) -> Optional[str]:
    """Hand-built counterfactual used by the earlier formulation: result + 1, same format."""
    v = parse_number(result)
    if v is None:
        return None
    s = result.strip().replace(",", "")
    decimals = len(s.split(".", 1)[1]) if "." in s else 0
    nv = v + 1
    if decimals == 0 and nv.denominator == 1:
        return str(nv.numerator)
    return f"{float(nv):.{max(decimals, 1)}f}"


def tokenize_example(tok, rec: dict, max_len: int, with_cf: bool = False) -> Optional[Example]:
    prompt_ids = tok(format_prompt(rec["question"]), add_special_tokens=True).input_ids
    answer = rec["answer"]
    enc = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids = list(enc.input_ids) + [tok.eos_token_id]
    offsets = [tuple(o) for o in enc.offset_mapping] + [(len(answer), len(answer))]
    budget = max_len - len(prompt_ids)
    if budget <= 8:
        return None
    resp_ids, offsets = resp_ids[:budget], offsets[:budget]
    claims = parse_claims(answer)
    claim_of_token = []
    for s, e in offsets:
        ci = -1
        for j, c in enumerate(claims):
            if c.valid is True and _overlap(s, e, c.start, c.end):
                ci = j
                break
        claim_of_token.append(ci)
    ex = Example(
        id=rec["id"],
        prompt_ids=prompt_ids,
        resp_ids=resp_ids,
        resp_offsets=offsets,
        answer=answer,
        gold=Fraction(rec["gold"]),
        claims=claims,
        claim_of_token=claim_of_token,
    )
    if with_cf:
        ex.cf_pairs = build_cf_pairs(tok, ex)
    return ex


def build_cf_pairs(tok, ex: Example) -> list[dict]:
    """Token spans of (gold result, counterfactual result) sharing a prefix, per valid claim."""
    pairs = []
    for j, c in enumerate(ex.claims):
        if c.valid is not True or c.result_end <= c.result_start:
            continue
        cf = counterfactual_result(ex.answer[c.result_start : c.result_end])
        if cf is None:
            continue
        z = [t for t, (s, e) in enumerate(ex.resp_offsets) if _overlap(s, e, c.result_start, c.result_end)]
        if not z or z[-1] >= len(ex.resp_ids) - 1:
            continue
        z0, z1 = z[0], z[-1] + 1
        if ex.resp_offsets[z0][0] != c.result_start:
            continue  # gold result does not start on a token boundary; skip pair
        cf_answer = ex.answer[: c.result_start] + cf + ex.answer[c.result_end :]
        enc = tok(cf_answer, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc.input_ids), [tuple(o) for o in enc.offset_mapping]
        if ids[:z0] != ex.resp_ids[:z0]:
            continue  # shared prefix h must be identical
        cf_ids = [ids[t] for t, (s, e) in enumerate(offs) if _overlap(s, e, c.result_start, c.result_start + len(cf))]
        if not cf_ids:
            continue
        cf_claim = claim_valid(c.expr, cf)
        cf_final = final_answer(cf_answer)
        pairs.append(
            {
                "claim": j,
                "z0": z0,
                "z1": z1,
                "cf_ids": cf_ids,
                "cf_claim_valid": cf_claim,  # V_c of the counterfactual claim
                "cf_final_correct": cf_final is not None and cf_final == ex.gold,  # final answer unchanged?
            }
        )
    return pairs


# ---------------------------------------------------------------- batching


def collate(examples: list[Example], pad_id: int, per_token: Optional[dict] = None, extras: Optional[dict] = None,
            bad: Optional[dict] = None):
    """Right-padded batch. Per-token arrays are aligned to full-sequence positions.

    ``per_token`` maps a name to {example_id: list[float] over response tokens}.
    ``extras`` maps example_id to {resp_t: list[token_id]} (certified alternatives) and
    is materialised as ``extra_ids`` [B, L, K] padded with -1.
    ``bad`` maps example_id to {resp_t: token_id} (v2: the erroneous token at a fork) and is
    materialised as ``bad_ids`` [B, L], -1 elsewhere (always present).
    Labels start at response token ``Example.supervise_from`` (0 unless set).
    """
    L = max(len(e.prompt_ids) + len(e.resp_ids) for e in examples)
    B = len(examples)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    claim = torch.zeros((B, L), dtype=torch.bool)
    claim_id = torch.full((B, L), -1, dtype=torch.long)
    resp_pos = torch.full((B, L), -1, dtype=torch.long)
    named = {k: torch.zeros((B, L), dtype=torch.float32) for k in (per_token or {})}
    kmax = 0
    if extras:
        for e in examples:
            for alts in extras.get(e.id, {}).values():
                kmax = max(kmax, len(alts))
    extra_ids = torch.full((B, L, max(kmax, 1)), -1, dtype=torch.long)
    bad_ids = torch.full((B, L), -1, dtype=torch.long)
    for b, e in enumerate(examples):
        P = len(e.prompt_ids)
        seq = e.prompt_ids + e.resp_ids
        input_ids[b, : len(seq)] = torch.tensor(seq)
        attn[b, : len(seq)] = 1
        s0 = e.supervise_from
        labels[b, P + s0 : len(seq)] = torch.tensor(e.resp_ids[s0:], dtype=torch.long)
        claim[b, P : len(seq)] = torch.tensor(e.claim_mask)
        claim_id[b, P : len(seq)] = torch.tensor(e.claim_of_token)
        resp_pos[b, P : len(seq)] = torch.arange(len(e.resp_ids))
        for k, table in (per_token or {}).items():
            vals = table.get(e.id)
            if vals is not None:
                n = min(len(vals), len(e.resp_ids))
                named[k][b, P : P + n] = torch.tensor(vals[:n], dtype=torch.float32)
        if extras:
            for t, alts in extras.get(e.id, {}).items():
                t = int(t)
                if t < len(e.resp_ids) and alts:
                    extra_ids[b, P + t, : len(alts)] = torch.tensor(alts)
        if bad:
            for t, a in bad.get(e.id, {}).items():
                t = int(t)
                if t < len(e.resp_ids):
                    bad_ids[b, P + t] = int(a)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "claim": claim,
        "claim_id": claim_id,
        "resp_pos": resp_pos,
        "extra_ids": extra_ids,
        "bad_ids": bad_ids,
        "examples": examples,
    }
    batch.update({f"tok_{k}": v for k, v in named.items()})
    return batch
