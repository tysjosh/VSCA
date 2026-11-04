"""CPU test of the VSCA v2 on-policy pipeline through the vLLM path, with a stand-in engine (vLLM needs CUDA).

FakeLLM and the ``_params``/``set_snapshot`` patches are copied from scripts/vllm_plumbing_test.py: the stand-in
has vLLM's interface and runs the HF model, sampling each request under its own seed. Checks, on 4 smoke
problems with debug_seed_samples=true:
  1. ``onpolicy.prepare`` via the vLLM path is deterministic and independent of request order;
  2. a run stopped by debug_stop_after_sample_examples resumes to identical files;
  3. every stage ran (MC localisation and fork certification had work), every sampling request had a seed.
Run: ../.venv/bin/python scripts/vllm_plumbing_v2.py
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from vsca import rollout
from vsca.config import load_config
from vsca.data import load_split
from vsca.modeling import generate_from_ids, load_model, load_tokenizer
from vsca.onpolicy import load_cert, prepare

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    ok &= bool(cond)


class FakeLLM:
    def __init__(self, model, tok, shuffle=False):
        self.model, self.tok, self.shuffle, self.calls, self.loras, self.seeds = model, tok, shuffle, 0, [], []

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls += 1
        self.loras.append(lora_request)
        order = list(range(len(prompts)))
        if self.shuffle:  # vLLM schedules requests in its own order; results must not depend on it
            random.Random(self.calls).shuffle(order)
        res = [None] * len(prompts)
        for k in order:
            p, sp = prompts[k]["prompt_token_ids"], params[k]
            if sp["sample"]:
                self.seeds.append(sp["seed"])
                torch.manual_seed(sp["seed"])
            outs = generate_from_ids(self.model, self.tok, [p], sp["max_tokens"], sp["sample"], sp["temperature"],
                                     sp["top_p"], num_return_sequences=sp["n"])
            res[k] = SimpleNamespace(outputs=[SimpleNamespace(index=i, token_ids=o + [self.tok.eos_token_id])
                                              for i, o in reversed(list(enumerate(outs)))])
        return res


def fake_params(self, max_new_tokens, do_sample, temperature, top_p, n, seed):
    return {"sample": do_sample, "temperature": temperature, "top_p": top_p, "n": n, "seed": seed,
            "max_tokens": max_new_tokens}


rollout.VLLMBackend._params = fake_params
rollout.VLLMBackend.set_snapshot = lambda self, d: setattr(self, "_lora", None if d is None else ("lora", self._next_id, str(d))) or setattr(self, "_next_id", self._next_id + 1)

cfg = load_config(["configs/smoke.yaml"], ["limit_train=4", "cert_block=2", "debug_seed_samples=true",
                                           "onpolicy_k=2", "mc_M=2", "gen_max_new=32", "max_rollout_tokens=24",
                                           "rollout_backend=vllm"])
tok = load_tokenizer(cfg.model_name)
recs = load_split(cfg.data_dir, "gsm8k_train", cfg.limit_train)
model, _ = load_model(cfg)
tmp = Path(tempfile.mkdtemp(prefix="vllm_v2_"))
FILES = ("samples.jsonl", "verify.jsonl", "localise.jsonl", "forks.jsonl", "onpolicy_stats.json")
mk = lambda shuffle: rollout.VLLMBackend(cfg, tok, llm=FakeLLM(model, tok, shuffle))
same = lambda a, b: all((a / f).read_bytes() == (b / f).read_bytes() for f in FILES)

b1 = mk(False)
s1 = prepare(model, tok, recs, cfg, tmp / "a", backend=b1)
s2 = prepare(model, tok, recs, cfg, tmp / "b", backend=mk(True))
check("1. vLLM-path prepare deterministic and independent of request order", same(tmp / "a", tmp / "b"))

cfg.debug_stop_after_sample_examples = 1
try:
    prepare(model, tok, recs, cfg, tmp / "c", backend=mk(True))
    check("2a. debug stop triggered", False)
except SystemExit as e:
    check("2a. debug_stop_after_sample_examples exits with 75 after one block",
          e.code == 75 and len(load_cert(tmp / "c" / "samples.jsonl")) == 2
          and not (tmp / "c" / "samples.jsonl.done").exists())
cfg.debug_stop_after_sample_examples = None
prepare(model, tok, recs, cfg, tmp / "c", backend=mk(True))
check("2b. resumed prepare == uninterrupted (all files)", same(tmp / "a", tmp / "c"))

loc = load_cert(tmp / "a" / "localise.jsonl")
n_mc = sum(1 for r in loc for f in r["forks"] if f["mc_values"] is not None) + sum(r["n_uncertain"] for r in loc)
check(f"3a. stages ran: {s1['samples']} samples, {s1['forks']} forks, {n_mc} MC-localised or uncertain",
      s1["samples"] == 4 * (2 + 3) and s1["forks"] > 0 and n_mc > 0)
check("3b. every sampling request carried a seed", b1._llm.seeds and all(s is not None for s in b1._llm.seeds))

shutil.rmtree(tmp)
sys.exit(0 if ok else 1)
