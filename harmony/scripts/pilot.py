"""Harmony VSCA pilot (harmony/PILOT.md).

  python scripts/pilot.py pools --data data --samples runs/p_s0/train_samples.jsonl --out runs/p_s0/pools.json
      exact first errors of on-policy samples -> verified samples + forks (token index, erroneous token e,
      certified next tokens A, single target a*)
  python scripts/pilot.py train --arm set --seed 0 --pools runs/p_s0/pools.json --init runs/sft_s0/adapter \
      --out runs/p_s0/set
  python scripts/pilot.py score --data data --samples runs/p_s0/set/test_id.jsonl --out runs/p_s0/set/test_id.score.jsonl

Arms (all from the SFT adapter of the same seed, identical verified-sample stream and step count):
  sft_cont  reference solutions of the sampled problems (more SFT, no on-policy data)
  rft       verified samples only
  neg       rft + fork loss -log(1 - pi(e))
  single    rft + fork loss -log pi(a*)            (one certified token)
  set       rft + fork loss -log sum_{a in A} pi(a) (the certified set)
  full      rft + set + neg
Fork positions enter the same token mean as the SFT tokens (each counts as one token).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("sft_cont", "rft", "neg", "single", "set", "full", "single_step", "set_step")
FORK_ARMS = ("neg", "single", "set", "full")
# PILOT2.md: certification at the level of the whole step (one chord line) instead of the next token
STEP_ARMS = ("single_step", "set_step")
MAX_CANDS = 8


def with_end(text: str) -> str:
    """vLLM drops the stop string ``END``; restore the training format (last chord line, newline, END)."""
    text = text.rstrip()
    return text if text.endswith("END") else text + "\nEND"


def _exercise(r):
    from harmony.rules import Exercise
    from harmony.theory import MAJOR_KEYS, MINOR_KEYS, parse_note

    keys = {str(k): k for k in MAJOR_KEYS + MINOR_KEYS}
    return Exercise(keys[r["key"]], tuple(r["labels"]), tuple(parse_note(s) for s in r["soprano"]))


# ---------------------------------------------------------------- pools


def cmd_pools(a):
    from transformers import AutoTokenizer

    from harmony.diagnose import first_error, fork_token
    from harmony.rules import Certifier
    from harmony.textfmt import prompt, step_line

    tok = AutoTokenizer.from_pretrained(a.model)
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.data) / f"{a.split}.jsonl")}
    verified, forks, refs, kinds = [], [], [], Counter()
    for line in open(a.samples):
        s = json.loads(line)
        r = probs[s["id"]]
        ex = _exercise(r)
        cert = Certifier(ex)
        refs.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"]})
        for j, text in enumerate(s["samples"]):
            text = with_end(text)
            e = first_error(ex, text, cert)
            kinds[e["kind"] or "ok"] += 1
            if e["ok"]:
                verified.append({"id": f"{r['id']}-s{j}", "prompt": r["prompt"], "response": text})
            elif e["kind"] in ("rule", "dead_end"):
                try:
                    k, bad, alts = fork_token(tok, ex, prompt(ex), text, e)
                except StopIteration:
                    continue
                if not alts:
                    kinds["fork_without_certified_token"] += 1
                    continue
                # a*: the certified token shared by the most certified voicings (ties: smallest id)
                votes = Counter()
                ids = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                head = text[: ids["offset_mapping"][k][0]]
                for v in e["certified"]:
                    full = text[: e["line_start"]] + step_line(ex, e["step"], v)
                    if full.startswith(head):
                        e2 = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                        jj = next((i for i, (st, en) in enumerate(e2["offset_mapping"]) if st >= len(head)), None)
                        if jj is not None and e2["input_ids"][jj] in alts:
                            votes[e2["input_ids"][jj]] += 1
                a_star = min(votes, key=lambda t: (-votes[t], t)) if votes else alts[0]
                forks.append({"id": f"{r['id']}-s{j}-f", "prompt": r["prompt"], "response": text, "k": k,
                              "e": bad, "A": alts, "a_star": a_star, "kind": e["kind"]})
    out = {"verified": verified, "forks": forks, "refs": refs,
           "stats": {"samples": sum(kinds.values()) - kinds["fork_without_certified_token"], "kinds": dict(kinds),
                     "verified": len(verified), "forks": len(forks),
                     "forks_A_ge2": sum(len(f["A"]) >= 2 for f in forks),
                     "dead_end_forks": sum(f["kind"] == "dead_end" for f in forks)}}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out["stats"]))


def cmd_pools_step(a):
    """Step-level forks (PILOT2.md): for every sample with a `rule` / `dead_end` first error, the verified prefix
    (text before the erroneous chord line) and the certified voicings of that chord as candidate lines; ``star`` is
    the candidate with the least total voice movement from the previous chord (closest spacing for the first chord;
    ties: enumeration order). At most MAX_CANDS candidates: the star plus a seeded sample of the others."""
    from harmony.diagnose import first_error
    from harmony.rules import Certifier
    from harmony.textfmt import step_line

    probs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.data) / f"{a.split}.jsonl")}
    out, sizes = [], Counter()
    for line in open(a.samples):
        s = json.loads(line)
        r = probs[s["id"]]
        ex = _exercise(r)
        cert = Certifier(ex)
        for j, text in enumerate(s["samples"]):
            text = with_end(text)
            e = first_error(ex, text, cert)
            if e["kind"] not in ("rule", "dead_end") or not e["certified"]:
                continue
            vs, prev = e["certified"], e["prev"]
            cost = (lambda v: sum(abs(x.midi - y.midi) for x, y in zip(v, prev))) if prev else \
                (lambda v: v[0].midi - v[3].midi)
            star = min(range(len(vs)), key=lambda i: (cost(vs[i]), i))
            keep = list(range(len(vs)))
            if len(vs) > MAX_CANDS:
                others = [i for i in keep if i != star]
                random.Random(f"{r['id']}-{j}").shuffle(others)
                keep = sorted([star] + others[: MAX_CANDS - 1])
            cands = [step_line(ex, e["step"], vs[i]) for i in keep]
            sizes[min(len(cands), MAX_CANDS)] += 1
            out.append({"id": f"{r['id']}-s{j}-F", "prompt": r["prompt"], "prefix": text[: e["line_start"]],
                        "cands": cands, "star": keep.index(star), "kind": e["kind"]})
    n = len(out)
    stats = {"step_forks": n, "cands_ge2_share": sum(v for k, v in sizes.items() if k >= 2) / n if n else 0.0,
             "mean_cands": sum(k * v for k, v in sizes.items()) / n if n else 0.0, "sizes": dict(sorted(sizes.items()))}
    Path(a.out).write_text(json.dumps({"step_forks": out, "stats": stats}))
    print(json.dumps(stats))


# ---------------------------------------------------------------- train


def fork_loss(arm: str, logits, e, A_pad):
    import torch

    lse = torch.logsumexp(logits, -1)
    p_e = (logits.gather(-1, e[:, None]).squeeze(-1) - lse).exp()
    neg = -torch.log((1 - p_e).clamp(min=1e-6))
    valid = A_pad >= 0
    picked = logits.gather(-1, A_pad.clamp(min=0)).masked_fill(~valid, float("-inf"))
    st = lse - torch.logsumexp(picked, -1)
    if arm == "neg":
        return neg
    if arm == "set":
        return st
    if arm == "full":
        return st + neg
    raise ValueError(arm)


def cmd_train(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    out = Path(a.out)
    if (out / "adapter" / "adapter_config.json").exists():
        print(f"[pilot] {out}/adapter exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(a).copy()
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        diff = [k for k in ("arm", "seed", "pools", "init", "lr", "epochs", "micro_bs", "accum", "fork_mb")
                if old.get(k) != cfg.get(k)]
        if diff:
            raise SystemExit(f"{out}: existing run has a different {diff}")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    pools = json.loads(Path(a.pools).read_text())
    ver = pools["verified"]
    main = pools["refs"] if a.arm == "sft_cont" else ver
    steps_per_epoch = math.ceil(len(ver) / (a.micro_bs * a.accum))  # identical for every arm of a seed
    total = steps_per_epoch * a.epochs
    n_mb = total * a.accum
    order = []
    for ep in range(math.ceil(n_mb * a.micro_bs / len(main)) + 1):
        idx = list(range(len(main)))
        random.Random(a.seed * 1000 + ep).shuffle(idx)
        order += idx
    forks = pools["forks"] if a.arm in FORK_ARMS else []
    if a.arm in STEP_ARMS:
        if not a.step_pools:
            raise SystemExit(f"arm {a.arm} needs --step_pools (pilot.py pools_step)")
        forks = json.loads(Path(a.step_pools).read_text())["step_forks"]
    forder = []
    if forks:
        for ep in range(math.ceil(n_mb * a.fork_mb / len(forks)) + 1):
            idx = list(range(len(forks)))
            random.Random(a.seed * 1000 + 7 + ep).shuffle(idx)
            forder += idx

    def enc_main(r):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        y = tok(r["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return p + y, [-100] * len(p) + y

    def enc_cand(f, c):
        """prompt + verified prefix + candidate line; labels only on the candidate line's tokens."""
        p = tok(f["prompt"], add_special_tokens=False)["input_ids"]
        text = f["prefix"] + c
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        y, offs = enc["input_ids"], enc["offset_mapping"]
        lab = [t if s >= len(f["prefix"]) else -100 for t, (s, _) in zip(y, offs)]
        return p + y, [-100] * len(p) + lab

    def enc_fork(f):
        p = tok(f["prompt"], add_special_tokens=False)["input_ids"]
        y = tok(f["response"], add_special_tokens=False)["input_ids"]
        assert y[f["k"]] == f["e"], f["id"]
        return p + y[: f["k"] + 1], len(p) + f["k"]  # position (in input ids) of the fork token

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if cuda else torch.float32)
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True).to(dev)
    if a.grad_ckpt:  # long inputs (bookkeeping)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
    step = 0
    ck = out / "ckpt"
    if (ck / "state.json").exists():
        st = json.loads((ck / "state.json").read_text())
        model.load_state_dict(torch.load(ck / "lora.pt", map_location=dev), strict=False)
        opt.load_state_dict(torch.load(ck / "opt.pt", map_location=dev))
        sched.load_state_dict(torch.load(ck / "sched.pt"))
        step = st["step"]
        print(f"[pilot] resumed at step {step}/{total}")
    deadline = float(os.environ["SESSION_DEADLINE"]) if os.environ.get("SESSION_DEADLINE") else None

    def save():
        ck.mkdir(exist_ok=True)
        torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, ck / "lora.pt")
        torch.save(opt.state_dict(), ck / "opt.pt")
        torch.save(sched.state_dict(), ck / "sched.pt")
        (ck / "state.json").write_text(json.dumps({"step": step}))

    log = open(out / "train_log.jsonl", "a")
    model.train()
    while step < total:
        opt.zero_grad(set_to_none=True)
        agg = Counter()
        for m in range(a.accum):
            mb = step * a.accum + m
            seqs, labels, fpos = [], [], []
            for j in order[mb * a.micro_bs : (mb + 1) * a.micro_bs]:
                x, y = enc_main(main[j])
                seqs.append(x)
                labels.append(y)
            fk = [forks[j] for j in forder[mb * a.fork_mb : (mb + 1) * a.fork_mb]] if forks else []
            cand_rows, cand_labels = [], []  # step arms: (fork index, is_star) per candidate row
            for fi, f in enumerate(fk):
                if a.arm in STEP_ARMS:
                    for ci, c in enumerate(f["cands"]):
                        x, cl = enc_cand(f, c)
                        seqs.append(x)
                        labels.append([-100] * len(x))
                        cand_labels.append((len(seqs) - 1, cl))
                        cand_rows.append((fi, ci == f["star"]))
                else:
                    x, pos = enc_fork(f)
                    seqs.append(x)
                    labels.append([-100] * len(x))
                    fpos.append((len(seqs) - 1, pos))
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), tok.pad_token_id)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            lab = torch.full((len(seqs), L), -100)
            clab = torch.full((len(seqs), L), -100)
            for i, (s, y) in enumerate(zip(seqs, labels)):
                ids[i, : len(s)], att[i, : len(s)], lab[i, : len(s)] = torch.tensor(s), 1, torch.tensor(y)
            for r, cl in cand_labels:
                clab[r, : len(cl)] = torch.tensor(cl)
            ids, att, lab, clab = ids.to(dev), att.to(dev), lab.to(dev), clab.to(dev)
            # logits stay in the model dtype; only the positions a loss uses are upcast (selecting before or after
            # the upcast gives identical values, but upcasting everything costs B x L x vocab x 4 bytes)
            logits = model(input_ids=ids, attention_mask=att).logits
            tgt = lab[:, 1:]
            lg = logits[:, :-1]
            sel = tgt != -100
            nll = torch.nn.functional.cross_entropy(lg[sel].float(), tgt[sel], reduction="sum")
            n_tok = int(sel.sum())
            floss = torch.zeros((), device=dev)
            if cand_rows:
                rows = torch.tensor([r for r, _ in cand_labels], device=dev)
                ct = clab[rows, 1:]
                cm = ct != -100
                r_idx, t_idx = cm.nonzero(as_tuple=True)  # labelled positions of the candidate lines
                tok_lp = -torch.nn.functional.cross_entropy(lg[rows[r_idx], t_idx].float(), ct[r_idx, t_idx],
                                                            reduction="none")
                seq_lp = torch.zeros(len(cand_rows), device=dev).index_add(0, r_idx, tok_lp)  # log pi(line | prefix)
                fl = []
                for fi in range(len(fk)):
                    idx = [k for k, (f_, _) in enumerate(cand_rows) if f_ == fi]
                    if a.arm == "set_step":
                        fl.append(-torch.logsumexp(seq_lp[idx], 0))
                    else:
                        star = [k for k in idx if cand_rows[k][1]]
                        fl.append(-seq_lp[star[0]])
                fl = torch.stack(fl)
                floss = fl.sum()
                agg["fork_loss"] += float(fl.detach().mean())
                # each step fork counts as the tokens of its star line in the token mean, so single_step is
                # exactly SFT on one certified line after the verified prefix and set_step has the same weight
                n_line = int(cm[[k for k, (_, st_) in enumerate(cand_rows) if st_]].sum())
                fpos = [None] * n_line
            elif fpos:
                rows = torch.tensor([r for r, _ in fpos], device=dev)
                cols = torch.tensor([p - 1 for _, p in fpos], device=dev)  # logits at p-1 predict token p
                flg = lg[rows, cols].float()
                e = torch.tensor([f["e"] for f in fk], device=dev)
                if a.arm == "single":
                    astar = torch.tensor([f["a_star"] for f in fk], device=dev)
                    fl = torch.logsumexp(flg, -1) - flg.gather(-1, astar[:, None]).squeeze(-1)
                else:
                    K = max(len(f["A"]) for f in fk)
                    A = torch.full((len(fk), K), -1, device=dev)
                    for i, f in enumerate(fk):
                        A[i, : len(f["A"])] = torch.tensor(f["A"], device=dev)
                    fl = fork_loss(a.arm, flg, e, A)
                floss = fl.sum()
                agg["fork_loss"] += float(fl.detach().mean())
            loss = (nll + floss) / max(1, n_tok + len(fpos))
            (loss / a.accum).backward()
            agg["loss"] += float(loss.detach()) / a.accum
            agg["nll_tok"] += float(nll.detach()) / max(1, n_tok) / a.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        log.write(json.dumps({"step": step, **agg}) + "\n")
        log.flush()
        if step % 20 == 0:
            print(f"[pilot] {a.arm} s{a.seed} step {step}/{total} loss {agg['loss']:.4f}", flush=True)
        if step % 100 == 0:
            save()
        if deadline and time.time() > deadline and step < total:
            save()
            print("[pilot] session deadline reached; rerun to continue")
            sys.exit(75)
    model.save_pretrained(out / "adapter")
    (out / "train_summary.json").write_text(json.dumps({"steps": total, "verified": len(ver),
                                                        "forks_in_pool": len(pools["forks"]),
                                                        "fork_draws": total * a.accum * a.fork_mb if forks else 0}))
    print(f"[pilot] wrote {out}/adapter")


# ---------------------------------------------------------------- score


def cmd_score(a):
    from harmony.diagnose import first_error

    probs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.data) / f"{a.split}.jsonl")}
    rows = []
    for line in open(a.samples):
        s = json.loads(line)
        text = with_end(s["greedy"])
        e = first_error(_exercise(probs[s["id"]]), text)
        rows.append({"id": s["id"], "correct": bool(e["ok"]), "kind": e["kind"]})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.out}: {sum(r['correct'] for r in rows) / len(rows):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pools")
    p.add_argument("--data", default="data")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", help="the exercises the samples were drawn for")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    t = sp.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--pools", required=True)
    t.add_argument("--init", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--micro_bs", type=int, default=8)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--fork_mb", type=int, default=4, help="fork examples added to each micro-batch")
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--step_pools", default=None, help="step-level forks (pilot.py pools_step), for the *_step arms")
    q = sp.add_parser("pools_step")
    q.add_argument("--data", default="data")
    q.add_argument("--samples", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--split", default="train")
    s = sp.add_parser("score")
    s.add_argument("--data", default="data")
    s.add_argument("--split", required=True)
    s.add_argument("--samples", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pools": cmd_pools, "pools_step": cmd_pools_step, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
