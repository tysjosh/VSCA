"""Training loop shared by VSCA and all baselines."""

from __future__ import annotations

import json
import math
import os
import random
import shutil

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from .certify import build_extras, certify, load_cert, uncertified_candidates
from .config import Config
from .data import Example, collate, load_split, tokenize_example
from .evaluate import evaluate
from .modeling import load_model, load_tokenizer
from .objectives import compute_loss, pairwise_cf_loss
from .scoring import load_scores
from .utils import ComputeLog, check_deadline, env_info, past_deadline, set_seed, stop_for_resume


def _rft_records(train_recs: list[dict], cert_path: str) -> list[dict]:
    by_id = {r["id"]: r for r in train_recs}
    extra = []
    for rec in load_cert(cert_path):
        base = by_id.get(rec["id"])
        if base is None:
            continue
        for j, text in enumerate(rec.get("rollouts", [])):
            if text.strip() != base["answer"].strip():
                extra.append({"id": f"{rec['id']}-rft{j}", "question": base["question"], "answer": text, "gold": base["gold"]})
    return extra


def _cf_s_table(examples: list[Example], regime: str) -> dict:
    table = {}
    for ex in examples:
        s_claim = {}
        for p in ex.cf_pairs:
            changed = (p["cf_claim_valid"] is False) if regime == "claim" else (not p["cf_final_correct"])
            s_claim[p["claim"]] = 1.0 if changed else 0.0
        table[ex.id] = [s_claim.get(c, 0.0) if c >= 0 else 0.0 for c in ex.claim_of_token]
    return table


def _cf_term(model, tok, batch, logits, cfg: Config, rng: random.Random, device):
    """Pairwise loss of the earlier formulation on (gold result, result+1) spans."""
    jobs = []
    for b, ex in enumerate(batch["examples"]):
        for p in ex.cf_pairs:
            jobs.append((b, ex, p))
    if not jobs:
        return None
    jobs = rng.sample(jobs, min(len(jobs), cfg.cf_max_pairs))
    s_gold = []
    for b, ex, p in jobs:
        P = len(ex.prompt_ids)
        pos = torch.arange(P + p["z0"] - 1, P + p["z1"] - 1, device=device)
        lg = logits[b, pos].float()
        gold = torch.tensor(ex.resp_ids[p["z0"] : p["z1"]], device=device)
        s_gold.append((torch.log_softmax(lg, -1).gather(-1, gold[:, None])).mean())
    seqs = [ex.prompt_ids + ex.resp_ids[: p["z0"]] + p["cf_ids"] for _, ex, p in jobs]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    # Run the decoder (LoRA layers included) and apply the LM head only at the counterfactual positions:
    # full-vocabulary logits for every position of every sequence do not fit next to the main batch.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    s_cf = []
    for i, (_, ex, p) in enumerate(jobs):
        n = len(p["cf_ids"])
        start = len(seqs[i]) - n
        lg = base.lm_head(hidden[i, start - 1 : start - 1 + n]).float()
        tgt = torch.tensor(p["cf_ids"], device=device)
        s_cf.append(torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).mean())
    return pairwise_cf_loss(torch.stack(s_gold), torch.stack(s_cf), cfg.cf_tau)


# ---------------------------------------------------------------- resume support

# Config keys that may differ between sessions of the same run without changing what it computes.
_SESSION_KEYS = {
    "device", "resume", "ckpt_every", "keep_ckpt", "deadline", "debug_stop_after_steps",
    "debug_stop_after_cert_examples", "extra",
}


def _same_config(run_dir, cfg: Config) -> None:
    p = run_dir / "config.json"
    if not p.exists():
        return
    old = json.loads(p.read_text())
    new = json.loads(json.dumps(cfg.to_dict()))
    keys = (set(old) | set(new)) - _SESSION_KEYS
    defaults = Config().to_dict()
    # A setting added after the run was created is absent from its config.json; it is a mismatch only if the
    # requested value differs from the default (the value the run implicitly used).
    diff = sorted(k for k in keys
                  if not (k not in old and new.get(k) == defaults.get(k)) and old.get(k) != new.get(k))
    if diff:
        raise SystemExit(
            f"{run_dir}: an existing run used a different config ({', '.join(diff)}). "
            "Use a new run_name, delete the run directory, or set resume=false to start over."
        )


def _rng_state(gen: torch.Generator, rng: random.Random) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "gen": gen.get_state(),
        "rng": rng.getstate(),
    }


def _set_rng_state(s: dict, gen: torch.Generator, rng: random.Random) -> None:
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    gen.set_state(s["gen"])
    rng.setstate(s["rng"])


def _save_ckpt(run_dir, model, opt, sched, state: dict, gen, rng) -> None:
    payload = {
        "model": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "state": state,
        "rng": _rng_state(gen, rng),
    }
    tmp = run_dir / "ckpt.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, run_dir / "ckpt.pt")  # atomic: a kill leaves either the old or the new checkpoint


def _load_ckpt(run_dir, model, opt, sched, gen, rng) -> dict:
    payload = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    res = model.load_state_dict(payload["model"], strict=False)
    if res.unexpected_keys:
        raise SystemExit(f"{run_dir}/ckpt.pt does not match the model: unexpected keys {res.unexpected_keys[:5]}")
    opt.load_state_dict(payload["opt"])
    sched.load_state_dict(payload["sched"])
    _set_rng_state(payload["rng"], gen, rng)
    return payload["state"]


def _truncate_log(path, max_step: int) -> None:
    if not path.exists():
        return
    keep = []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line)["step"] <= max_step:
                keep.append(line)
        except (json.JSONDecodeError, KeyError):
            continue
    path.write_text("".join(k + "\n" for k in keep))


def _fresh_start(run_dir) -> None:
    """Remove outputs of an earlier attempt that left no checkpoint (or resume=false)."""
    for pattern in ("train_log.jsonl", "compute.json", "cert_epoch*", "extras_stats_*", "val_epoch*.jsonl", "ckpt.pt*"):
        for f in run_dir.glob(pattern):
            f.unlink()
    for d in ("adapter_last", "adapter_best"):
        if (run_dir / d).exists():
            shutil.rmtree(run_dir / d)


def train(cfg: Config) -> dict:
    """Train one run. Resumable: rerunning the same command continues from ``ckpt.pt``.

    State machine per epoch: certify -> train (micro-batch position saved every ``ckpt_every`` optimizer steps)
    -> val. A finished run (train_summary.json) is returned as is.
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "train_summary.json").exists() and cfg.resume:
        _same_config(run_dir, cfg)
        print(f"[resume] {run_dir} already finished; skipping")
        return json.loads((run_dir / "train_summary.json").read_text())
    resuming = cfg.resume and (run_dir / "ckpt.pt").exists()
    if resuming:
        _same_config(run_dir, cfg)
    else:
        _fresh_start(run_dir)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    set_seed(cfg.seed)
    clog = ComputeLog(run_dir / "compute.json")

    tok = load_tokenizer(cfg.model_name)
    train_recs = load_split(cfg.data_dir, "gsm8k_train", cfg.limit_train)
    val_recs = load_split(cfg.data_dir, "gsm8k_val", cfg.limit_val)
    if cfg.method == "rft":
        added = _rft_records(train_recs, cfg.rft_from)
        train_recs = train_recs + added
        print(f"[rft] added {len(added)} verified rollouts")
    examples = [tokenize_example(tok, r, cfg.max_len, with_cf=cfg.method == "cf_select") for r in train_recs]
    examples = [e for e in examples if e is not None]

    per_token: dict = {}
    if cfg.ref_scores:
        per_token["ref_loss"] = load_scores(cfg.ref_scores)["loss"]
    if cfg.base_scores:
        bs = load_scores(cfg.base_scores)
        per_token["base_loss"] = bs["loss"]
        if "attn" in bs:
            per_token["attn"] = bs["attn"]
    if cfg.method == "cf_select":
        per_token["cf_s"] = _cf_s_table(examples, cfg.cf_regime)

    model, device = load_model(cfg, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum = max(1, cfg.eff_bs // cfg.micro_bs)
    steps_per_epoch = math.ceil(len(examples) / cfg.eff_bs)
    total = steps_per_epoch * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    fixed_extras = None
    if cfg.method == "unlikelihood":
        fixed_extras = uncertified_candidates(load_cert(cfg.cert_path))
    elif cfg.method == "vsca" and cfg.cert_path:
        fixed_extras, st_fixed = build_extras(load_cert(cfg.cert_path), cfg, cfg.seed)
        (run_dir / "extras_stats_fixed.json").write_text(json.dumps(st_fixed, indent=2))
    dynamic = cfg.method == "vsca" and not cfg.cert_path

    def cert_epoch_of(epoch: int) -> int:
        return epoch if cfg.refresh == "epoch" else 0

    def extras_for(epoch: int):
        if not dynamic:
            return fixed_extras
        e = cert_epoch_of(epoch)
        recs = load_cert(run_dir / f"cert_epoch{e}.jsonl")
        return build_extras(recs, cfg, cfg.seed + e)[0]

    if resuming:
        state = _load_ckpt(run_dir, model, opt, sched, gen, rng)
        state["resumes"] = state.get("resumes", 0) + 1
        _truncate_log(run_dir / "train_log.jsonl", state["step"])
        print(f"[resume] {run_dir}: epoch {state['epoch']} phase {state['phase']} step {state['step']} "
              f"(resume #{state['resumes']})")
        _save_ckpt(run_dir, model, opt, sched, state, gen, rng)  # count this session even if it stops at once
    else:
        state = {"epoch": 0, "phase": "certify", "mb_next": 0, "step": 0, "best_acc": -1.0, "history": [], "resumes": 0}
        _save_ckpt(run_dir, model, opt, sched, state, gen, rng)  # LoRA initialisation is part of the run
    clog.data["session"] = {"resumes": state["resumes"]}
    clog.save()

    def save():
        _save_ckpt(run_dir, model, opt, sched, state, gen, rng)

    micro_starts = list(range(0, len(examples), cfg.micro_bs))
    extras = None
    log_fh = open(run_dir / "train_log.jsonl", "a")
    try:
        while state["epoch"] < cfg.epochs:
            epoch = state["epoch"]
            check_deadline(cfg, f"epoch {epoch} ({state['phase']})")

            if state["phase"] == "certify":
                if dynamic and (epoch == 0 or cfg.refresh == "epoch"):
                    with clog.phase("certify"):
                        recs = certify(model, tok, examples, cfg, run_dir / f"cert_epoch{epoch}.jsonl",
                                       desc=f"certify e{epoch}", epoch=epoch)
                    extras, st = build_extras(recs, cfg, cfg.seed + epoch)
                    st["epoch"] = epoch
                    (run_dir / f"extras_stats_epoch{epoch}.json").write_text(json.dumps(st, indent=2))
                    print(f"[certify] epoch {epoch}: {st}")
                # Training randomness (dropout) is seeded per epoch, independent of certification sampling.
                torch.manual_seed((cfg.seed * 7_919 + epoch * 104_729) % (2**62))
                state.update(phase="train", mb_next=0)
                save()

            if state["phase"] == "train":
                if extras is None:
                    extras = extras_for(epoch)
                model.train()
                order = list(range(len(examples)))
                random.Random(cfg.seed + epoch).shuffle(order)
                with clog.phase("train"):
                    agg: dict = {}
                    for k_mb in range(state["mb_next"], len(micro_starts)):
                        mb_i = micro_starts[k_mb]
                        group = [examples[j] for j in order[mb_i : mb_i + cfg.micro_bs]]
                        batch = collate(group, tok.pad_token_id, per_token=per_token, extras=extras)
                        ids = batch["input_ids"].to(device)
                        out = model(input_ids=ids, attention_mask=batch["attention_mask"].to(device))
                        labels = batch["labels"][:, 1:].to(device)
                        sel = labels != -100
                        logits_full = out.logits
                        lg = logits_full[:, :-1][sel].float()
                        lab = labels[sel]
                        seq = torch.arange(labels.size(0), device=device)[:, None].expand_as(labels)[sel]
                        aux = {
                            "claim": batch["claim"][:, 1:].to(device)[sel],
                            "claim_id": batch["claim_id"][:, 1:].to(device)[sel],
                            "extra": batch["extra_ids"][:, 1:].to(device)[sel],
                        }
                        for k in per_token:
                            aux[k] = batch[f"tok_{k}"][:, 1:].to(device)[sel]
                        loss, st = compute_loss(cfg, lg, lab, seq, aux, gen)
                        if cfg.method == "cf_select" and cfg.cf_lambda > 0:
                            cf = _cf_term(model, tok, batch, logits_full, cfg, rng, device)
                            if cf is not None:
                                loss = loss + cfg.cf_lambda * cf
                                st["cf_loss"] = float(cf.detach())
                        (loss / accum).backward()
                        for k, v in st.items():
                            agg.setdefault(k, []).append(v)
                        if (k_mb + 1) % accum == 0 or k_mb == len(micro_starts) - 1:
                            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                            opt.step()
                            sched.step()
                            opt.zero_grad(set_to_none=True)
                            state["step"] += 1
                            state["mb_next"] = k_mb + 1
                            row = {"step": state["step"], "epoch": epoch, "lr": sched.get_last_lr()[0],
                                   "loss": float(loss.detach())}
                            row.update({k: sum(v) / len(v) for k, v in agg.items()})
                            log_fh.write(json.dumps(row) + "\n")
                            log_fh.flush()
                            agg = {}
                            if state["step"] % 10 == 0:
                                print(f"[train] epoch {epoch} step {state['step']}/{total} loss {row['loss']:.4f}")
                            debug_stop = cfg.debug_stop_after_steps is not None and state["step"] >= cfg.debug_stop_after_steps
                            out_of_time = past_deadline(cfg)
                            if debug_stop or out_of_time or state["step"] % cfg.ckpt_every == 0:
                                save()
                            if out_of_time:
                                stop_for_resume(f"session deadline reached at step {state['step']}")
                            if debug_stop:
                                stop_for_resume(f"debug stop after step {state['step']}")
                state.update(phase="val", mb_next=0)
                save()

            if state["phase"] == "val":
                with clog.phase("val_eval"):
                    summ = evaluate(model, tok, val_recs, cfg.eval_bs, cfg.gen_max_new,
                                    run_dir / f"val_epoch{epoch}.jsonl", desc=f"val e{epoch}",
                                    should_stop=lambda: check_deadline(cfg, f"validation of epoch {epoch}"))
                summ["epoch"] = epoch
                state["history"].append(summ)
                print(f"[val] epoch {epoch}: {summ}")
                model.save_pretrained(run_dir / "adapter_last")
                if not cfg.select_best_on_val or summ["accuracy"] > state["best_acc"]:
                    state["best_acc"] = summ["accuracy"]
                    model.save_pretrained(run_dir / "adapter_best")
                state.update(epoch=epoch + 1, phase="certify", mb_next=0)
                if dynamic and cfg.refresh == "epoch":
                    extras = None  # the next epoch has its own certification
                save()
    finally:
        log_fh.close()

    result = {
        "best_val_accuracy": state["best_acc"],
        "history": state["history"],
        "n_train_examples": len(examples),
        "resumes": state["resumes"],
    }
    (run_dir / "train_summary.json").write_text(json.dumps(result, indent=2))
    if not cfg.keep_ckpt:
        (run_dir / "ckpt.pt").unlink(missing_ok=True)
    return result
