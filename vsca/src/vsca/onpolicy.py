"""On-policy pipeline of VSCA v2 (Design Section 4.1, steps 1-4).

Per training problem, with the current policy as snapshot:
  1. ``sample``        ``onpolicy_k`` solutions                                            -> samples.jsonl
  2. ``verify_samples`` every sample with V+ (verify_plus.check_response)                  -> verify.jsonl
  3. ``localise``      the first error of each failed sample: the earliest step failing V+,
                       else (planning error) the step whose Monte Carlo value drops most
                       (``mc_M`` V+-verified rollouts per step boundary); mapped to the fork
                       token index t and the erroneous token e                              -> localise.jsonl
  4. ``certify_forks`` the top-k next tokens at t (excluding e); a candidate is certified if
                       one of ``M`` rollouts from it passes V+ (positive certificates only)  -> forks.jsonl
``prepare`` runs the stages into one directory; ``build_training`` turns that directory into the training
examples of one arm, so every arm trained from it sees the same samples.

Every model-driven stage runs in fixed blocks of ``cert_block`` problems with a seed per block (salts +101,
+202, +303 on ``cfg.seed``); the vLLM path passes one seed per request. A block is always recomputed in full,
so a resumed stage writes exactly what an uninterrupted one would have. ``<file>.done`` marks completion.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager, nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .certify import _block_seed, _length_batches, _load_partial, _position_stats, load_cert
from .config import V2_METHODS, Config
from .data import Example, counterfactual_result, format_prompt, tokenize_example
from .rollout import make_backend, save_snapshot
from .utils import check_deadline, stop_for_resume
from .verify import NUM_RE, parse_claims, truncate_response
from .verify_plus import check_response, step_boundaries, vplus_from_cfg

SALT_SAMPLE, SALT_LOCALISE, SALT_CERTIFY = 101, 202, 303
INVENTED = "9973"  # operand of the debug_seed_samples "invented number" sample
CONFIG_FILE, STATS_FILE = "onpolicy_config.json", "onpolicy_stats.json"
CHECKED_KEYS = ("seed", "model_name", "limit_train", "vplus_grounding", "grounding_whitelist")


# ---------------------------------------------------------------- helpers


def _done(path: Path) -> Path:
    return path.with_name(path.name + ".done")


def _gold(rec: dict) -> Fraction:
    return Fraction(rec["gold"])


def _by_id(rows) -> dict:
    return {r["id"]: r for r in rows}


def _sample_example(tok, rec: dict, text: str, cfg: Config, ex_id: Optional[str] = None) -> Optional[Example]:
    """The sample tokenised exactly as the loss sees it (Decision 5)."""
    return tokenize_example(tok, {"id": ex_id or rec["id"], "question": rec["question"], "answer": text,
                                  "gold": rec["gold"]}, cfg.max_len)


def _token_at(ex: Example, c: int) -> Optional[int]:
    """First response token that ends after char ``c`` or starts at or after it (None if cut by max_len)."""
    for t, (s, e) in enumerate(ex.resp_offsets):
        if e > c or s >= c:
            return t
    return None


def _prefix_example(ex: Example, t: int) -> Example:
    """``ex`` cut after response token t (the fork token)."""
    return Example(id=ex.id, prompt_ids=ex.prompt_ids, resp_ids=ex.resp_ids[: t + 1],
                   resp_offsets=ex.resp_offsets[: t + 1], answer=ex.answer, gold=ex.gold, claims=ex.claims,
                   claim_of_token=ex.claim_of_token[: t + 1])


def samples_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_samples(rec: dict) -> list[str]:
    """debug_seed_samples: the reference, the reference with its first annotation's result +1 inside ``<<..>>``,
    and the reference with that annotation's first operand replaced by an invented number in line and annotation."""
    ref = truncate_response(rec["answer"])
    out = [ref]
    claims = parse_claims(ref)
    if not claims:
        return out
    c = claims[0]
    cf = counterfactual_result(c.result)
    if cf is not None:
        out.append(ref[: c.result_start] + cf + ref[c.result_end :])
    m = NUM_RE.search(c.expr)
    if m:
        lo = c.start + 2
        text = ref[: lo + m.start()] + INVENTED + ref[lo + m.end() :]
        line_start = ref.rfind("\n", 0, c.start) + 1
        for n in NUM_RE.finditer(ref, line_start, c.start):
            if n.group() == m.group():  # first mention in the line text (before the annotation)
                text = text[: n.start()] + INVENTED + text[n.end() :]
                break
        out.append(text)
    return out


@contextmanager
def _inference(model):
    was = model is not None and model.training
    if model is not None:
        model.eval()
    try:
        yield
    finally:
        if was:
            model.train()


def _set_snapshot(model, backend, out_dir: Path) -> None:
    # vLLM serves the snapshot from disk (base model + current adapter), one shared dir per pipeline dir.
    if backend is not None and backend.name == "vllm":
        backend.set_snapshot(save_snapshot(model, out_dir / "snapshot") if model is not None else None)


def _generate(backend, cfg: Config, prefixes: list, n: int, seed: int, **kw) -> list[list[list[int]]]:
    """``n`` outputs per prefix, in prefix order. HF: length-sorted batches of ``cert_bs`` sequences under
    ``seed``. vLLM: one call, one seed per request from ``seed`` and the request index (as certify.run_jobs)."""
    if not prefixes or n <= 0:
        return [[] for _ in prefixes]
    if backend.name == "vllm":
        ds = 1 if kw.get("do_sample") else 0
        seeds = [(seed * 31 + 7 * k + ds) % (2**31) for k in range(len(prefixes))]
        outs = backend.generate(prefixes, n=n, seeds=seeds, **kw)
        return [outs[k * n : (k + 1) * n] for k in range(len(prefixes))]
    torch.manual_seed(seed)
    res: list = [None] * len(prefixes)
    for chunk in _length_batches(list(range(len(prefixes))), lambda i: len(prefixes[i]), max(1, cfg.cert_bs // n)):
        outs = backend.generate([prefixes[i] for i in chunk], n=n, batch=len(chunk), **kw)
        for k, i in enumerate(chunk):
            res[i] = outs[k * n : (k + 1) * n]
    return res


def _run_stage(desc: str, records: list[dict], cfg: Config, out_path: Path, epoch: int, salt: int, work,
               model=None, backend=None, stop_after: Optional[int] = None) -> list[dict]:
    """Resumable block loop shared by the stages. ``work(block_indices, seed) -> {index: record}``."""
    out_path = Path(out_path)
    if cfg.resume and _done(out_path).exists():
        by_id = _by_id(load_cert(out_path))
        return [by_id[r["id"]] for r in records]
    if not cfg.resume:
        for p in (out_path, _done(out_path)):
            if p.exists():
                p.unlink()
    by_id = _load_partial(out_path) if cfg.resume else {}
    if by_id:
        print(f"[resume] {out_path.name}: {len(by_id)}/{len(records)} problems already done")
    size = max(1, cfg.cert_block)
    blocks = [list(range(i, min(i + size, len(records)))) for i in range(0, len(records), size)]
    blocks = [b for b in blocks if any(records[i]["id"] not in by_id for i in b)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if blocks:
        _set_snapshot(model, backend, out_path.parent)
    n_new = 0
    with open(out_path, "a") as fh:
        for bidx in tqdm(blocks, desc=desc):
            check_deadline(cfg, desc)
            recs = work(bidx, _block_seed(cfg.seed + salt, epoch, bidx[0]))
            for i in bidx:
                if records[i]["id"] not in by_id:
                    fh.write(json.dumps(recs[i]) + "\n")
                    by_id[records[i]["id"]] = recs[i]
                    n_new += 1
            fh.flush()
            if stop_after and n_new >= stop_after:
                stop_for_resume(f"debug stop after {n_new} problems in {desc}")
    _done(out_path).write_text("")
    return [by_id[r["id"]] for r in records]


# ---------------------------------------------------------------- stage 1: sample


def _copy_samples(cfg: Config, records: list[dict], out_path: Path) -> list[dict]:
    src = Path(cfg.onpolicy_samples)
    if not _done(src).exists():
        raise SystemExit(f"onpolicy_samples={src}: not a finished samples file (no {src.name}.done)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out_path.resolve():
        shutil.copyfile(src, out_path)
        _done(out_path).write_text("")
    by_id = _by_id(load_cert(out_path))
    missing = [r["id"] for r in records if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"onpolicy_samples={src}: no samples for {len(missing)} training problems (e.g. {missing[0]})")
    return [by_id[r["id"]] for r in records]


def sample(model, tok, records: list[dict], cfg: Config, out_path, epoch: int = 0, backend=None) -> list[dict]:
    """Stage 1: ``onpolicy_k`` samples per problem -> ``{"id", "samples": [text, ...]}`` per problem."""
    out_path = Path(out_path)
    if cfg.onpolicy_samples:
        return _copy_samples(cfg, records, out_path)
    if backend is None and cfg.onpolicy_k > 0:
        backend = make_backend(cfg, model, tok)

    def work(bidx, seed):
        prefixes = [tok(format_prompt(records[i]["question"]), add_special_tokens=True).input_ids for i in bidx]
        gens = _generate(backend, cfg, prefixes, cfg.onpolicy_k, seed, max_new_tokens=cfg.gen_max_new,
                         do_sample=True, temperature=cfg.rollout_temperature, top_p=cfg.rollout_top_p)
        out = {}
        for i, gs in zip(bidx, gens):
            texts = [truncate_response(tok.decode(g, skip_special_tokens=True)) for g in gs]
            if cfg.debug_seed_samples:
                texts += seed_samples(records[i])
            out[i] = {"id": records[i]["id"], "samples": texts}
        return out

    with _inference(model):
        return _run_stage("onpolicy sample", records, cfg, out_path, epoch, SALT_SAMPLE, work, model, backend,
                          cfg.debug_stop_after_sample_examples)


# ---------------------------------------------------------------- stage 2: verify


def _vplus_record(res) -> dict:
    first = res.steps[res.first_failed] if res.first_failed is not None else None
    return {
        "verified": res.verified,
        "answer_ok": res.answer_ok,
        "answer": None if res.answer is None else str(res.answer),
        "first_failed": res.first_failed,
        "first_arith": next((s.index for s in res.steps if s.arithmetic is not True), None),
        "first_ground": next((s.index for s in res.steps if not s.grounded), None),
        "kind": first.kind if first else None,
        "error_char": first.error_char if first else None,
        "steps": [{"index": s.index, "line_start": s.line_start, "expr": s.claim.expr, "result": s.claim.result,
                   "arithmetic": s.arithmetic, "grounded": s.grounded, "ungrounded": s.ungrounded, "ok": s.ok,
                   "kind": s.kind, "error_char": s.error_char} for s in res.steps],
    }


def verify_samples(records: list[dict], samples: list[dict], cfg: Config, out_path) -> list[dict]:
    """Stage 2: V+ on every sample -> ``{"id", "results": [per sample]}`` per problem (CPU)."""
    wl, grounding = vplus_from_cfg(cfg)
    s_by = _by_id(samples)

    def work(bidx, seed):
        return {i: {"id": records[i]["id"],
                    "results": [_vplus_record(check_response(records[i]["question"], text, _gold(records[i]), wl,
                                                             grounding))
                                for text in s_by[records[i]["id"]]["samples"]]}
                for i in bidx}

    return _run_stage("onpolicy verify", records, cfg, Path(out_path), 0, 0, work)


# ---------------------------------------------------------------- stage 3: localise


def localise(model, tok, records: list[dict], samples: list[dict], verif: list[dict], cfg: Config, out_path,
             epoch: int = 0, backend=None) -> list[dict]:
    """Stage 3: first error of every failed sample -> ``{"id", "forks": [...], "n_failed", "n_uncertain", ...}``.

    Grounding false-rejects some correct steps (README, grounding trade-off), so a grounding failure alone never
    makes a fork. Each fork records the ``rule`` that picked it:
      * correct answer: ``lucky_arithmetic`` = the earliest step failing arithmetic; with no arithmetic failure
        (grounding-only) the sample is unverified (not trained on) but never a fork (``n_grounding_only_correct``);
      * wrong answer, some arithmetic failure: ``arithmetic`` = the earliest step failing arithmetic;
      * wrong answer otherwise: Monte Carlo values. V_j of each step boundary j = 0..S (S = the part after the last
        step line) is the share of ``mc_M`` rollouts from it that pass V+, V_{S+1} = 0, drop_j = V_j - V_{j+1}.
        ``grounding_mc`` = the earliest grounding failure g if drop_g >= ``mc_min_drop``; else ``mc`` = the earliest
        j with the largest drop (kind planning), or the sample is skipped as uncertain if that drop is below
        ``mc_min_drop`` (Decision 10).
    For the direct rules the error character and kind are the chosen step's (Decision 6).
    """
    wl, grounding = vplus_from_cfg(cfg)
    s_by, v_by = _by_id(samples), _by_id(verif)
    if backend is None:
        backend = make_backend(cfg, model, tok)

    def work(bidx, seed):
        plans, jobs = {}, []  # jobs: (problem index, prefix ids, head text)
        for i in bidx:
            rec = records[i]
            entries, n_gc = [], 0
            for j, (text, v) in enumerate(zip(s_by[rec["id"]]["samples"], v_by[rec["id"]]["results"])):
                if v["verified"]:
                    continue
                fa = v["first_arith"]
                if v["answer_ok"] and fa is None:
                    n_gc += 1  # correct answer, grounding-only failure: unverified, never a fork
                    continue
                ex = _sample_example(tok, rec, text, cfg)
                en = {"sample": j, "ex": ex, "v": v, "skip": ex is None}
                if fa is not None:
                    en["rule"], en["step"] = ("lucky_arithmetic" if v["answer_ok"] else "arithmetic"), fa
                elif ex is not None:
                    bounds = step_boundaries(check_response(rec["question"], text, _gold(rec), wl, grounding))
                    ts = [_token_at(ex, c) for c in bounds]
                    if any(t is None for t in ts):
                        en["skip"] = True
                    else:
                        en["bounds"], en["jobs"] = bounds, []
                        for t in ts:
                            en["jobs"].append(len(jobs))
                            jobs.append((i, ex.prompt_ids + ex.resp_ids[:t], text[: ex.resp_offsets[t][0]]))
                entries.append(en)
            plans[i] = (entries, n_gc)
        gens = _generate(backend, cfg, [j[1] for j in jobs], cfg.mc_M, seed, max_new_tokens=cfg.max_rollout_tokens,
                         do_sample=True, temperature=cfg.rollout_temperature, top_p=cfg.rollout_top_p)
        passed = []
        for (i, _, head), gs in zip(jobs, gens):
            rec = records[i]
            passed.append(sum(check_response(rec["question"], truncate_response(head + tok.decode(g, skip_special_tokens=True)),
                                             _gold(rec), wl, grounding).verified for g in gs))
        out = {}
        for i in bidx:
            forks, n_unc, n_skip = [], 0, 0
            entries, n_gc = plans[i]
            for en in entries:
                if en["skip"]:
                    n_skip += 1
                    continue
                v, ex = en["v"], en["ex"]
                if "rule" in en:
                    st = v["steps"][en["step"]]
                    rule, kind, step, c, mc = en["rule"], st["kind"], en["step"], st["error_char"], None
                else:
                    counts = [passed[k] for k in en["jobs"]] + [0]
                    drops = [counts[b] - counts[b + 1] for b in range(len(counts) - 1)]
                    mc = [x / cfg.mc_M for x in counts[:-1]]
                    g = v["first_ground"]
                    if g is not None and drops[g] / cfg.mc_M >= cfg.mc_min_drop:
                        rule, kind, step, c = "grounding_mc", "grounding", g, v["steps"][g]["error_char"]
                    else:
                        best = max(range(len(drops)), key=lambda b: (drops[b], -b))
                        if drops[best] / cfg.mc_M < cfg.mc_min_drop:
                            n_unc += 1
                            continue
                        rule, kind, step, c = "mc", "planning", best, en["bounds"][best]
                t = _token_at(ex, c)
                if t is None:
                    n_skip += 1
                    continue
                forks.append({"sample": en["sample"], "rule": rule, "kind": kind, "step": step, "t": t,
                              "e": ex.resp_ids[t], "error_char": c, "mc_values": mc})
            out[i] = {"id": records[i]["id"], "forks": forks, "n_failed": len(entries), "n_uncertain": n_unc,
                      "n_grounding_only_correct": n_gc,
                      "n_skipped": n_skip}
        return out

    with _inference(model):
        return _run_stage("onpolicy localise", records, cfg, Path(out_path), epoch, SALT_LOCALISE, work, model, backend)


# ---------------------------------------------------------------- stage 4: certify the forks


def certify_forks(model, tok, records: list[dict], samples: list[dict], loc: list[dict], cfg: Config, out_path,
                  epoch: int = 0, backend=None) -> list[dict]:
    """Stage 4: certified alternatives at each fork -> ``{"id", "forks": [fork + cands/certified/...]}``.

    Candidates are the snapshot's top-k next tokens at t excluding e (certify._position_stats); a candidate is
    certified if one of ``M`` rollouts from ``prompt + y_<t + [a]`` passes V+. Certificates are positive only.
    """
    wl, grounding = vplus_from_cfg(cfg)
    s_by, l_by = _by_id(samples), _by_id(loc)
    if backend is None:
        backend = make_backend(cfg, model, tok)
    stat_bs = max(1, cfg.micro_bs)

    def work(bidx, seed):
        forks = []  # (problem index, fork record, full sample Example)
        for i in bidx:
            rec = records[i]
            for f in l_by[rec["id"]]["forks"]:
                ex = _sample_example(tok, rec, s_by[rec["id"]]["samples"][f["sample"]], cfg)
                if ex.resp_ids[f["t"]] != f["e"]:
                    raise RuntimeError(f"{rec['id']} sample {f['sample']}: fork token differs from localise output")
                forks.append((i, dict(f), ex))
        for g in range(0, len(forks), stat_bs):
            grp = forks[g : g + stat_bs]
            stats = _position_stats(model, tok, [_prefix_example(ex, f["t"]) for _, f, ex in grp], cfg.k)
            for (_, f, _), rows in zip(grp, stats):
                row = rows[f["t"]]
                f.update(cands=row["cands"], cand_probs=row["cand_probs"], e_prob=row["gold_prob"],
                         entropy=row["entropy"], certified=[], cert_text={})
        jobs = [(i, f, ex, a) for i, f, ex in forks for a in f["cands"]]
        gens = _generate(backend, cfg, [ex.prompt_ids + ex.resp_ids[: f["t"]] + [a] for _, f, ex, a in jobs], cfg.M,
                         seed, max_new_tokens=cfg.max_rollout_tokens, do_sample=True,
                         temperature=cfg.rollout_temperature, top_p=cfg.rollout_top_p)
        for (i, f, ex, a), gs in zip(jobs, gens):
            head = ex.answer[: ex.resp_offsets[f["t"]][0]]
            for g in gs:
                text = truncate_response(head + tok.decode([a] + g, skip_special_tokens=True))
                if check_response(records[i]["question"], text, ex.gold, wl, grounding).verified:
                    if str(a) not in f["cert_text"]:
                        f["certified"].append(a)
                        f["cert_text"][str(a)] = text
        out = {}
        for i in bidx:
            out[i] = {"id": records[i]["id"], "forks": []}
        for i, f, _ in forks:
            f["certified"] = [a for a in f["cands"] if a in f["certified"]]  # candidate order
            out[i]["forks"].append(f)
        return out

    with _inference(model):
        return _run_stage("onpolicy certify", records, cfg, Path(out_path), epoch, SALT_CERTIFY, work, model, backend,
                          cfg.debug_stop_after_cert_examples)


# ---------------------------------------------------------------- driver


def _config_record(cfg: Config, sha: Optional[str]) -> dict:
    keys = ("seed", "model_name", "init_adapter", "limit_train", "onpolicy_k", "mc_M", "mc_min_drop", "k", "M",
            "vplus_grounding", "grounding_whitelist", "rollout_backend", "rollout_temperature", "rollout_top_p",
            "max_rollout_tokens", "gen_max_new", "debug_seed_samples", "onpolicy_samples")
    d = {k: getattr(cfg, k) for k in keys}
    d["samples_sha256"] = sha
    return d


def pipeline_stats(verif: list[dict], loc: list[dict], forks: Optional[list[dict]], sha: Optional[str]) -> dict:
    results = [x for r in verif for x in r["results"]]
    n = len(results)
    n_ver = sum(x["verified"] for x in results)
    fl = [f for r in loc for f in r["forks"]]
    by_kind: dict = {}
    by_rule: dict = {}
    for f in fl:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    st = {
        "problems": len(verif),
        "samples": n,
        "verified": n_ver,
        "verified_share": n_ver / n if n else 0.0,
        "lucky_guesses": sum(x["answer_ok"] and not x["verified"] for x in results),
        "failed": sum(r["n_failed"] for r in loc),
        "grounding_only_correct": sum(r["n_grounding_only_correct"] for r in loc),
        "failed_by_kind": dict(sorted(by_kind.items())),
        "forks_by_rule": dict(sorted(by_rule.items())),
        "forks": len(fl),
        "uncertain": sum(r["n_uncertain"] for r in loc),
        "skipped": sum(r.get("n_skipped", 0) for r in loc),
        "samples_sha256": sha,
    }
    if forks is not None:
        cf = [f for r in forks for f in r["forks"]]
        st["forks_with_certified"] = sum(1 for f in cf if f["certified"])
        st["mean_certified"] = sum(len(f["certified"]) for f in cf) / len(cf) if cf else 0.0
    return st


def prepare(model, tok, records: list[dict], cfg: Config, out_dir, epoch: int = 0, backend=None, clog=None,
            need_certify: bool = True) -> dict:
    """Run sample -> verify -> localise (-> certify) into ``out_dir``; resumable by rerunning. Returns stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume and (out_dir / CONFIG_FILE).exists():
        check_dir(out_dir, cfg, complete=False)
    phase = clog.phase if clog is not None else (lambda name: nullcontext())
    (out_dir / CONFIG_FILE).write_text(json.dumps(_config_record(cfg, None), indent=2))
    if backend is None:
        backend = make_backend(cfg, model, tok)
    with phase("onpolicy_sample"):
        samples = sample(model, tok, records, cfg, out_dir / "samples.jsonl", epoch=epoch, backend=backend)
    sha = samples_sha256(out_dir / "samples.jsonl")
    (out_dir / CONFIG_FILE).write_text(json.dumps(_config_record(cfg, sha), indent=2))
    with phase("onpolicy_verify"):
        verif = verify_samples(records, samples, cfg, out_dir / "verify.jsonl")
    with phase("onpolicy_localise"):
        loc = localise(model, tok, records, samples, verif, cfg, out_dir / "localise.jsonl", epoch=epoch,
                       backend=backend)
    forks = None
    if need_certify:
        with phase("onpolicy_certify"):
            forks = certify_forks(model, tok, records, samples, loc, cfg, out_dir / "forks.jsonl", epoch=epoch,
                                  backend=backend)
    stats = pipeline_stats(verif, loc, forks, sha)
    (out_dir / STATS_FILE).write_text(json.dumps(stats, indent=2))
    return stats


def check_dir(out_dir, cfg: Config, complete: bool = True) -> None:
    """Refuse (SystemExit) a pipeline dir built with a different seed/model/data/verifier, or (``complete``) an
    unfinished one."""
    out_dir = Path(out_dir)
    p = out_dir / CONFIG_FILE
    if not p.exists():
        raise SystemExit(f"{out_dir}: no {CONFIG_FILE}; build it with `vsca onpolicy --out {out_dir}`")
    old = json.loads(p.read_text())
    diff = [k for k in CHECKED_KEYS if old.get(k) != getattr(cfg, k)]
    if diff:
        raise SystemExit(f"{out_dir}: on-policy dir was built with a different " + ", ".join(
            f"{k} ({old.get(k)!r} != {getattr(cfg, k)!r})" for k in diff))
    if complete:
        need = ["samples.jsonl", "verify.jsonl", "localise.jsonl"] + (["forks.jsonl"] if cfg.method == "vsca2" else [])
        missing = [n for n in need if not _done(out_dir / n).exists()]
        if missing or not (out_dir / STATS_FILE).exists():
            raise SystemExit(f"{out_dir}: on-policy pipeline unfinished ({', '.join(missing) or STATS_FILE}); "
                             f"rerun `vsca onpolicy --out {out_dir}`")


# ---------------------------------------------------------------- training data per arm


def build_training(tok, records: list[dict], out_dir, cfg: Config) -> tuple[list[Example], dict, dict, dict]:
    """Training examples of one v2 arm from a pipeline dir (Decision 8) -> (examples, extras, bad, stats).

    * every arm: each V+-verified sample as a full SFT example (id ``{id}-s{j}``), plus the reference when
      ``v2_ref_sft``;
    * ``vsca2`` / ``stepdpo_fork``: one fork example per localised first error (id ``{id}-s{j}-f``): the sample cut
      after token t with ``supervise_from=t``, ``bad[id][t] = e``, and ``extras[id][t]`` = the certified set
      (``variant=full``), the first |certified| snapshot candidates (``uncertified``), none (``stepdpo_fork``);
    * ``rft_verified``: no forks.
    """
    if cfg.method not in V2_METHODS:
        raise ValueError(f"build_training is for methods {V2_METHODS}, not {cfg.method!r}")
    out_dir = Path(out_dir)
    s_by = _by_id(load_cert(out_dir / "samples.jsonl"))
    v_by = _by_id(load_cert(out_dir / "verify.jsonl"))
    forks_by = {}
    if cfg.method == "vsca2":
        if not (out_dir / "forks.jsonl").exists():
            raise SystemExit(f"{out_dir}: no forks.jsonl (fork certification); method vsca2 needs it")
        forks_by = _by_id(load_cert(out_dir / "forks.jsonl"))
    elif cfg.method == "stepdpo_fork":
        forks_by = _by_id(load_cert(out_dir / "localise.jsonl"))
    examples: list[Example] = []
    extras: dict = {}
    bad: dict = {}
    n_ver = n_ref = n_fork = n_set = n_mismatch = 0
    for rec in records:
        if rec["id"] not in s_by:
            raise SystemExit(f"{out_dir}: no samples for training problem {rec['id']}")
        texts, res = s_by[rec["id"]]["samples"], v_by[rec["id"]]["results"]
        if cfg.v2_ref_sft:
            ex = tokenize_example(tok, rec, cfg.max_len)
            if ex is not None:
                examples.append(ex)
                n_ref += 1
        for j, (text, v) in enumerate(zip(texts, res)):
            if v["verified"]:
                ex = _sample_example(tok, rec, text, cfg, f"{rec['id']}-s{j}")
                if ex is not None:
                    examples.append(ex)
                    n_ver += 1
        for f in forks_by.get(rec["id"], {}).get("forks", []):
            fid, t, e = f"{rec['id']}-s{f['sample']}-f", f["t"], f["e"]
            ex = _sample_example(tok, rec, texts[f["sample"]], cfg, fid)
            if ex is None or t >= len(ex.resp_ids) or ex.resp_ids[t] != e:
                n_mismatch += 1
                continue
            ex = _prefix_example(ex, t)
            ex.supervise_from = t
            examples.append(ex)
            bad[fid] = {t: e}
            n_fork += 1
            if cfg.method == "vsca2":
                acc = f["certified"] if cfg.variant == "full" else f["cands"][: len(f["certified"])]
                if acc:
                    extras[fid] = {t: list(acc)}
                    n_set += 1
    stats = {
        "method": cfg.method,
        "variant": cfg.variant,
        "onpolicy_dir": str(out_dir),
        "samples_sha256": samples_sha256(out_dir / "samples.jsonl"),
        "examples": len(examples),
        "verified_examples": n_ver,
        "reference_examples": n_ref,
        "fork_examples": n_fork,
        "forks_with_set": n_set,
        "frac_forks_with_set": n_set / n_fork if n_fork else 0.0,
        "forks_skipped_token_mismatch": n_mismatch,
    }
    return examples, extras, bad, stats
