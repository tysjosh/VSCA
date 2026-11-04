"""Pre-registered diagnostics (Section 4.2)."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .certify import load_cert
from .data import Example
from .verify import FINAL_RE, parse_claims, values_match


def error_taxonomy(eval_rows: list[dict], records: list[dict], out_csv: Path, n: int = 200, seed: int = 0) -> dict:
    """Diagnostic 1: sample base-model failures for manual labelling, with automatic hints.

    Manual labels: arithmetic_slip / wrong_quantity_or_operation / misread_problem / format.
    """
    by_id = {r["id"]: r for r in records}
    fails = [r for r in eval_rows if not r["correct"]]
    random.Random(seed).shuffle(fails)
    fails = fails[:n]
    hints = Counter()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "auto_hint", "question", "gold", "reference", "generation", "manual_label"])
        for r in fails:
            text = r["generation"]
            claims = parse_claims(text)
            if not FINAL_RE.search(text):
                hint = "format(no ####)"
            elif any(c.valid is False for c in claims):
                hint = "arithmetic_slip(invalid annotation)"
            else:
                hint = "needs_manual(arithmetic valid)"
            hints[hint] += 1
            src = by_id[r["id"]]
            w.writerow([r["id"], hint, src["question"], src["gold"], src["answer"], text, ""])
    return {"n_failures_sampled": len(fails), "auto_hints": dict(hints)}


def certified_mass(cert_path) -> dict:
    """Diagnostic 2: how often |A_t| > 1, and pi(A_t) - pi(g_t) under the snapshot."""
    recs = load_cert(cert_path)
    by_type = defaultdict(lambda: {"n": 0, "with_alt": 0, "gaps": []})
    n_rollouts = 0
    for rec in recs:
        n_rollouts += len(rec.get("rollouts", []))
        for p in rec["positions"]:
            d = by_type[p["type"]]
            d["n"] += 1
            d["with_alt"] += bool(p["certified"])
            prob = dict(zip(p["cands"], p["cand_probs"]))
            d["gaps"].append(sum(prob.get(a, 0.0) for a in p["certified"]))
    out = {"examples": len(recs), "verified_rollouts": n_rollouts}
    for t, d in by_type.items():
        g = np.array(d["gaps"]) if d["gaps"] else np.zeros(1)
        out[t] = {
            "searched_positions": d["n"],
            "frac_with_alternatives": d["with_alt"] / d["n"] if d["n"] else 0.0,
            "gap_mean": float(g.mean()),
            "gap_quantiles": {q: float(np.quantile(g, q)) for q in (0.5, 0.9, 0.99)},
            "frac_gap_above_0.01": float((g > 0.01).mean()),
        }
    return out


def audit_sample(cert_path, examples: list[Example], tok, out_csv: Path, n: int = 200, seed: int = 0) -> dict:
    """Diagnostic 3: stratified sample of certified alternatives for manual precision audit."""
    by_id = {e.id: e for e in examples}
    items = defaultdict(list)
    for rec in load_cert(cert_path):
        for p in rec["positions"]:
            for a in p["certified"]:
                items[p["type"]].append((rec["id"], p, a))
    rng = random.Random(seed)
    per = max(1, n // max(len(items), 1))
    rows = []
    for typ, xs in items.items():
        rng.shuffle(xs)
        rows += [(typ, *x) for x in xs[:per]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "id", "t", "context", "gold_token", "alt_token", "certificate", "is_valid_manual"])
        for typ, eid, p, a in rows:
            ex = by_id.get(eid)
            if ex is None:
                continue
            t = p["t"]
            ctx = ex.answer[: ex.resp_offsets[t][0]][-300:]
            w.writerow([typ, eid, t, ctx, tok.decode([ex.resp_ids[t]]), tok.decode([a]), p["cert_text"].get(str(a), ""), ""])
    return {"available": {k: len(v) for k, v in items.items()}, "sampled": len(rows)}


def earlier_score(examples: list[Example]) -> dict:
    """Diagnostic 4: distribution of the earlier binary score s under two verifier regimes."""
    s_claim, s_final, n_valid = [], [], 0
    for ex in examples:
        n_valid += sum(c.valid is True for c in ex.claims)
        for p in ex.cf_pairs:
            s_claim.append(1.0 if p["cf_claim_valid"] is False else 0.0)
            s_final.append(0.0 if p["cf_final_correct"] else 1.0)
    f = lambda xs: float(np.mean(xs)) if xs else None
    return {
        "valid_claims": n_valid,
        "pairs_built": len(s_claim),
        "mean_s_claim_verifier": f(s_claim),
        "mean_s_final_answer_verifier_no_propagation": f(s_final),
    }


def _vplus_flags(res) -> tuple[bool, bool, bool]:
    """(flagged, any arithmetic failure, any ungrounded operand) for a VPlusResult."""
    arith = any(s.arithmetic is not True for s in res.steps)
    ground = any(not s.grounded for s in res.steps)
    return arith or ground, arith, ground


def grounding_coverage(csv_path, cfg) -> dict:
    """V2 diagnostic (Design 5.3 item 1): which failures V+ flags, per manual label (or auto_hint if unlabelled)."""
    from .verify_plus import check_response, vplus_from_cfg

    wl, grounding = vplus_from_cfg(cfg)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    labelled = any((r.get("manual_label") or "").strip() for r in rows)
    key = "manual_label" if labelled else "auto_hint"
    groups = defaultdict(lambda: {"n": 0, "flagged": 0, "arithmetic": 0, "grounding": 0, "grounding_only": 0})
    ref_n, ref_rejected, ref_ops = 0, 0, Counter()
    for r in rows:
        gold = Fraction(r["gold"])
        res = check_response(r["question"], r["generation"], gold, wl, grounding)
        flagged, arith, ground = _vplus_flags(res)
        g = groups[(r.get(key) or "").strip() or "unlabelled"]
        g["n"] += 1
        g["flagged"] += flagged
        g["arithmetic"] += arith
        g["grounding"] += ground
        g["grounding_only"] += ground and not arith
        if r.get("reference"):
            ref = check_response(r["question"], r["reference"], gold, wl, grounding)
            ref_n += 1
            ref_rejected += not ref.verified
            for s in ref.steps:
                ref_ops.update(s.ungrounded)
    share = lambda a, n: a / n if n else 0.0
    out = {
        "csv": str(csv_path),
        "grouped_by": key,
        "n_rows": len(rows),
        "vplus_grounding": grounding,
        "grounding_whitelist": cfg.grounding_whitelist,
        "flagged_share": share(sum(g["flagged"] for g in groups.values()), len(rows)),
        "groups": {
            name: {
                "n": g["n"],
                "flagged_share": share(g["flagged"], g["n"]),
                "arithmetic_share": share(g["arithmetic"], g["n"]),
                "grounding_share": share(g["grounding"], g["n"]),
                "grounding_only_share": share(g["grounding_only"], g["n"]),
            }
            for name, g in sorted(groups.items())
        },
        "reference": {
            "n": ref_n,
            "false_rejection_rate": share(ref_rejected, ref_n),
            "top_ungrounded_operands": ref_ops.most_common(20),
        },
    }
    if labelled:
        wq = out["groups"].get("wrong_quantity_or_operation")
        out["wrong_quantity_flag_share"] = wq["flagged_share"] if wq else None
        out["gate_0.40_passed"] = bool(wq and wq["flagged_share"] >= 0.40)
    return out


def vplus_trace(eval_rows: list[dict], records: list[dict], cfg) -> dict:
    """V2 diagnostic: V+ valid-trace rate of eval generations, next to V_r trace validity and lucky guesses."""
    from .verify_plus import check_response, vplus_from_cfg

    wl, grounding = vplus_from_cfg(cfg)
    by_id = {r["id"]: r for r in records}
    n = correct = trace_valid = vplus = lucky = missing = 0
    kinds = Counter()
    for row in eval_rows:
        src = by_id.get(row["id"])
        if src is None:
            missing += 1
            continue
        res = check_response(src["question"], row["generation"], Fraction(src["gold"]), wl, grounding)
        n += 1
        correct += bool(row.get("correct"))
        trace_valid += bool(row.get("trace_valid"))
        vplus += res.verified
        lucky += bool(row.get("correct")) and not res.verified
        if res.first_failed is not None:
            kinds[res.steps[res.first_failed].kind] += 1
    share = lambda a: a / n if n else 0.0
    return {
        "n": n,
        "missing_ids": missing,
        "vplus_grounding": grounding,
        "accuracy": share(correct),
        "vr_trace_valid_share": share(trace_valid),
        "vplus_verified_share": share(vplus),
        "lucky_guess_share": share(lucky),
        "lucky_guess_share_of_correct": lucky / correct if correct else 0.0,
        "first_failure_kind": dict(kinds),
    }


@torch.no_grad()
def pairwise_saturation(model, tok, examples: list[Example], tau: float, max_pairs: int = 2000) -> dict:
    """Fraction of counterfactual pairs whose pairwise-loss gradient scale sigma(-margin/tau) exceeds 0.1."""
    device = next(model.parameters()).device
    model.eval()
    scales = []
    for ex in examples:
        if len(scales) >= max_pairs or not ex.cf_pairs:
            continue
        P = len(ex.prompt_ids)
        seq = torch.tensor([ex.prompt_ids + ex.resp_ids], device=device)
        lp = torch.log_softmax(model(input_ids=seq).logits[0].float(), -1)
        for p in ex.cf_pairs:
            z = ex.resp_ids[p["z0"] : p["z1"]]
            sg = torch.stack([lp[P + p["z0"] - 1 + i, z[i]] for i in range(len(z))]).mean()
            cf_seq = torch.tensor([ex.prompt_ids + ex.resp_ids[: p["z0"]] + p["cf_ids"]], device=device)
            lpc = torch.log_softmax(model(input_ids=cf_seq).logits[0].float(), -1)
            start = cf_seq.size(1) - len(p["cf_ids"])
            sc = torch.stack([lpc[start - 1 + i, p["cf_ids"][i]] for i in range(len(p["cf_ids"]))]).mean()
            scales.append(float(torch.sigmoid(-(sg - sc) / tau)))
    s = np.array(scales) if scales else np.zeros(1)
    return {"pairs": len(scales), "frac_scale_above_0.1": float((s > 0.1).mean()), "median_scale": float(np.median(s))}
