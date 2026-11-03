"""Certified acceptance sets A_t (Section 3.2, Algorithm 1) and their control variants.

For each training response we take a snapshot of the model and
  * score every response position (top-k alternatives, entropy, gold probability);
  * search claim positions (tokens inside annotations whose claim verifies) with
    *local* certificates: complete the annotation greedily from ``y_<t ⊕ a``,
    require the new claim to verify and to have the reference value, append the
    reference suffix, and check V_r;
  * search up to B decision positions (outside claims, entropy >= h_min) with
    *rollout* certificates: sample M continuations from ``y_<t ⊕ a`` and accept
    ``a`` if any continuation passes V_r.
Certificates are positive only; an unaccepted candidate is treated exactly as SFT
treats it (Proposition 4).
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .data import Example, collate
from .rollout import make_backend, save_snapshot
from .utils import check_deadline, stop_for_resume
from .verify import parse_claims, response_valid, safe_eval, truncate_response, values_match


@torch.no_grad()
def _position_stats(model, tok, examples: list[Example], k: int) -> list[dict]:
    """Snapshot statistics for every response position of each example."""
    device = next(model.parameters()).device
    batch = collate(examples, tok.pad_token_id)
    logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
    out = []
    for b, ex in enumerate(examples):
        P, T = len(ex.prompt_ids), len(ex.resp_ids)
        lg = logits[b, P - 1 : P + T - 1].float()  # row t predicts response token t
        logp = torch.log_softmax(lg, -1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        gold = torch.tensor(ex.resp_ids, device=lg.device)
        gold_p = p.gather(-1, gold[:, None]).squeeze(-1)
        top_p, top_i = p.topk(k + 1, dim=-1)
        rows = []
        for t in range(T):
            ids, ps = [], []
            for i, q in zip(top_i[t].tolist(), top_p[t].tolist()):
                if i != ex.resp_ids[t] and len(ids) < k:
                    ids.append(i)
                    ps.append(q)
            rows.append({"t": t, "entropy": float(ent[t]), "gold_prob": float(gold_p[t]), "cands": ids, "cand_probs": ps})
        out.append(rows)
    return out


def _search_positions(ex: Example, rows: list[dict], cfg: Config) -> list[dict]:
    T = len(ex.resp_ids)
    claim_rows = [dict(r, type="claim") for r in rows if ex.claim_of_token[r["t"]] >= 0]
    dec = [r for r in rows if ex.claim_of_token[r["t"]] < 0 and r["t"] < T - 1 and r["entropy"] >= cfg.h_min]
    dec = sorted(dec, key=lambda r: -r["entropy"])[: cfg.B]
    return claim_rows + [dict(r, type="decision") for r in sorted(dec, key=lambda r: r["t"])]


def _local_ok(tok, ex: Example, t: int, a: int, gen: list[int]) -> tuple[bool, str]:
    c = ex.claims[ex.claim_of_token[t]]
    head = ex.answer[: ex.resp_offsets[t][0]]
    a_end = len(head) + len(tok.decode([a], skip_special_tokens=True))
    new_text = head + tok.decode([a] + gen, skip_special_tokens=True)
    if new_text[c.start : c.start + 2] != "<<":
        return False, ""
    close = new_text.find(">>", c.start + 2)
    # Everything token `a` emits must lie inside the re-closed annotation; otherwise
    # text it produces would be discarded when the reference suffix is appended.
    if close < 0 or a_end > close + 2:
        return False, ""
    new_claims = parse_claims(new_text[c.start : close + 2])
    if len(new_claims) != 1 or new_claims[0].valid is not True:
        return False, ""
    v_new = safe_eval(new_claims[0].expr)
    if v_new is None or values_match(v_new, c.result) is not True:
        return False, ""
    full = new_text[: close + 2] + ex.answer[c.end :]
    return response_valid(full, ex.gold), full


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _load_partial(path: Path) -> dict:
    """Records already written to ``path``; a truncated last line (from a kill) is dropped and the file rewritten."""
    if not path.exists():
        return {}
    good, recs = [], {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs[rec["id"]] = rec
        good.append(line)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(g + "\n" for g in good))
    os.replace(tmp, path)
    return recs


def _block_seed(seed: int, epoch: int, first_idx: int) -> int:
    # Each block of examples samples with its own seed, so a resumed certification produces exactly what an
    # uninterrupted one would have (blocks are fixed over the full example list).
    return (seed * 1_000_003 + epoch * 10_007 + first_idx) % (2**62)


def _length_batches(jobs: list, prefix_len, size: int) -> Iterable[list]:
    """Batches of ``jobs`` sorted by prefix length (stable), so each generate call pads little."""
    order = sorted(range(len(jobs)), key=lambda j: prefix_len(jobs[j]))
    for i in range(0, len(order), size):
        yield [jobs[j] for j in order[i : i + size]]


def certify(model, tok, examples: list[Example], cfg: Config, out_path: Path, desc: str = "certify",
            epoch: int = 0, backend=None) -> list[dict]:
    """Run Algorithm 1 on ``examples`` with the current model as snapshot; write jsonl.

    Resumable: examples already in ``out_path`` are skipped, and ``<out_path>.done`` marks completion.
    Returns records in ``examples`` order.
    """
    out_path = Path(out_path)
    done_marker = out_path.with_name(out_path.name + ".done")
    if cfg.resume and done_marker.exists():
        by_id = {r["id"]: r for r in load_cert(out_path)}
        return [by_id[ex.id] for ex in examples if ex.id in by_id]
    by_id = _load_partial(out_path) if cfg.resume else {}
    if not cfg.resume and out_path.exists():
        out_path.unlink()
    if by_id:
        print(f"[resume] {out_path.name}: {len(by_id)}/{len(examples)} examples already certified")
    was_training = model.training
    model.eval()
    backend = backend or make_backend(cfg, model, tok)
    if backend.name == "vllm":
        # vLLM serves the snapshot from disk: base model + the current adapter (none for the base model).
        backend.set_snapshot(save_snapshot(model, out_path.with_name(out_path.stem + ".snapshot")))
    stat_bs = max(1, cfg.micro_bs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    # Examples are processed in fixed blocks of the full list. All generation jobs of a block (every example,
    # position and candidate) share generate calls of up to ``cert_bs`` sequences, so the GPU is kept full.
    # Statistics are computed in groups of ``stat_bs`` aligned to the full list, and each block has its own
    # seed, so batch shapes and samples are the same whether or not a run was resumed.
    block = max(stat_bs, (max(cfg.cert_block, 1) // stat_bs) * stat_bs)
    blocks = [list(range(i, min(i + block, len(examples)))) for i in range(0, len(examples), block)]
    blocks = [b for b in blocks if any(examples[i].id not in by_id for i in b)]
    with open(out_path, "a") as fh:
        for bidx in tqdm(blocks, desc=desc):
            check_deadline(cfg, desc)
            rows_of = {}
            for g in range(0, len(bidx), stat_bs):
                gidx = bidx[g : g + stat_bs]
                rows_of.update(zip(gidx, _position_stats(model, tok, [examples[i] for i in gidx], cfg.k)))
            todo = [i for i in bidx if examples[i].id not in by_id]
            torch.manual_seed(_block_seed(cfg.seed, epoch, bidx[0]))
            positions_of = {i: _search_positions(examples[i], rows_of[i], cfg) for i in todo}
            for i in todo:
                for pos in positions_of[i]:
                    pos["certified"] = []
                    pos["cert_text"] = {}
            rollouts_of: dict[int, list[str]] = {i: [] for i in todo}

            def prefix(job):
                i, pos, a = job
                ex = examples[i]
                return ex.prompt_ids + ex.resp_ids[: pos["t"]] + [a]

            def run_jobs(jobs, n, **kw):
                """(job, [n outputs]) pairs. HF: length-sorted batches of ``cert_bs`` sequences under the block seed.
                vLLM: one call for the whole block, with a seed per job from the block seed and the job's index."""
                if backend.name == "vllm":
                    base = _block_seed(cfg.seed, epoch, bidx[0])
                    seeds = [(base * 31 + 7 * k + (1 if kw.get("do_sample") else 0)) % (2**31) for k in range(len(jobs))]
                    outs = backend.generate([prefix(j) for j in jobs], n=n, seeds=seeds, **kw)
                    return [(j, outs[k * n : (k + 1) * n]) for k, j in enumerate(jobs)]
                res = []
                for chunk in _length_batches(jobs, lambda j: len(prefix(j)), max(1, cfg.cert_bs // max(n, 1))):
                    outs = backend.generate([prefix(j) for j in chunk], n=n, batch=len(chunk), **kw)
                    res += [(j, outs[k * n : (k + 1) * n]) for k, j in enumerate(chunk)]
                return res

            # local certificates (greedy span completion) for every claim (position, candidate) in the block
            jobs = [(i, pos, a) for i in todo for pos in positions_of[i] if pos["type"] == "claim" for a in pos["cands"]]
            if jobs:
                for (i, pos, a), (gen,) in run_jobs(jobs, 1, max_new_tokens=cfg.max_span_tokens, do_sample=False):
                    ok, text = _local_ok(tok, examples[i], pos["t"], a, gen)
                    if ok:
                        pos["certified"].append(a)
                        pos["cert_text"][str(a)] = text

            # rollout certificates: M samples for every decision (position, candidate) in the block
            jobs = [(i, pos, a) for i in todo for pos in positions_of[i] if pos["type"] == "decision" for a in pos["cands"]]
            if jobs:
                for (i, pos, a), gens in run_jobs(jobs, cfg.M, max_new_tokens=cfg.max_rollout_tokens, do_sample=True,
                                                   temperature=cfg.rollout_temperature, top_p=cfg.rollout_top_p):
                    ex = examples[i]
                    head = ex.answer[: ex.resp_offsets[pos["t"]][0]]
                    for gen in gens:
                        text = truncate_response(head + tok.decode([a] + gen, skip_special_tokens=True))
                        if response_valid(text, ex.gold):
                            rollouts_of[i].append(text)
                            if a not in pos["certified"]:
                                pos["certified"].append(a)
                                pos["cert_text"][str(a)] = text

            for i in todo:
                for pos in positions_of[i]:  # candidate order, independent of batch order
                    pos["certified"] = [a for a in pos["cands"] if a in pos["certified"]]
                rec = {"id": examples[i].id, "positions": positions_of[i], "rollouts": sorted(set(rollouts_of[i]))}
                fh.write(json.dumps(rec) + "\n")
                by_id[examples[i].id] = rec
            fh.flush()
            n_new += len(todo)
            if cfg.debug_stop_after_cert_examples and n_new >= cfg.debug_stop_after_cert_examples:
                stop_for_resume(f"debug stop after {n_new} certified examples")
    done_marker.write_text("")
    if was_training:
        model.train()
    return [by_id[ex.id] for ex in examples]


def load_cert(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- variants and controls


def build_extras(records: list[dict], cfg: Config, seed: int) -> tuple[dict, dict]:
    """Map certification output to accepted alternatives per (example, position).

    Returns ``extras[id][t] = [token ids]`` (gold excluded; it is always accepted) and
    summary statistics. Variants follow Sections 4.4 and 4.6 of the paper.
    """
    rng = random.Random(seed)
    positions = [(rec["id"], pos) for rec in records for pos in rec["positions"]]
    if cfg.variant == "claim_only":
        positions = [(i, p) for i, p in positions if p["type"] == "claim"]
    elif cfg.variant == "decision_only":
        positions = [(i, p) for i, p in positions if p["type"] == "decision"]

    accepted: dict[tuple, list[int]] = {}
    for i, p in positions:
        accepted[(i, p["t"])] = list(p["certified"])

    if cfg.variant == "uncertified":
        # same number of alternatives, chosen by snapshot probability, no verification
        for i, p in positions:
            n = len(p["certified"])
            accepted[(i, p["t"])] = list(p["cands"][:n])
    elif cfg.variant == "shuffled":
        # permute certified-set sizes between positions in the same entropy bin
        ents = np.array([p["entropy"] for _, p in positions]) if positions else np.zeros(0)
        if len(ents):
            edges = np.quantile(ents, np.linspace(0, 1, cfg.shuffle_bins + 1)[1:-1])
            bins = np.digitize(ents, edges)
            by_bin = defaultdict(list)
            for idx, bn in enumerate(bins):
                by_bin[int(bn)].append(idx)
            for idxs in by_bin.values():
                sizes = [len(positions[j][1]["certified"]) for j in idxs]
                rng.shuffle(sizes)
                for j, n in zip(idxs, sizes):
                    i, p = positions[j]
                    accepted[(i, p["t"])] = list(p["cands"][:n])

    n_injected = n_dropped = 0
    if cfg.inject_eta > 0 or cfg.drop_eta > 0:
        for i, p in positions:
            acc = accepted[(i, p["t"])]
            if cfg.drop_eta > 0:
                keep = [a for a in acc if rng.random() >= cfg.drop_eta]
                n_dropped += len(acc) - len(keep)
                acc = keep
            if cfg.inject_eta > 0 and rng.random() < cfg.inject_eta:
                wrong = [a for a in p["cands"] if a not in p["certified"] and a not in acc]
                if wrong:
                    acc = acc + [wrong[0]]
                    n_injected += 1
            accepted[(i, p["t"])] = acc

    extras: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for (i, t), acc in accepted.items():
        if acc:
            extras[i][t] = acc
    n_search = len(positions)
    n_nontrivial = sum(1 for acc in accepted.values() if acc)
    stats = {
        "variant": cfg.variant,
        "searched_positions": n_search,
        "positions_with_alternatives": n_nontrivial,
        "frac_with_alternatives": n_nontrivial / n_search if n_search else 0.0,
        "injected": n_injected,
        "dropped": n_dropped,
    }
    return dict(extras), stats


def uncertified_candidates(records: list[dict]) -> dict:
    """For the unlikelihood baseline: top-k candidates that were searched but not certified."""
    out: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for rec in records:
        for p in rec["positions"]:
            bad = [a for a in p["cands"] if a not in p["certified"]]
            if bad:
                out[rec["id"]][p["t"]] = bad
    return dict(out)
