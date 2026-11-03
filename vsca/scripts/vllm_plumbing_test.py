"""CPU test of the vLLM certification path with a stand-in engine (vLLM itself needs CUDA).

The stand-in has vLLM's interface (``generate(prompts, params, lora_request=..)`` returning outputs with
``.outputs[k].index/.token_ids``) and runs the same HF model, sampling each request under its own seed. Checks:
  1. greedy local certificates from the vLLM path equal those of the HF path;
  2. the vLLM path is deterministic, independent of request order, and resumes to identical records;
  3. a LoRA model gets a snapshot adapter written and a new LoRA request id per certification.
The real-engine comparison runs on the GPU machine: scripts/vllm_probe.sh.
Run: ../.venv/bin/python scripts/vllm_plumbing_test.py
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
from vsca.certify import certify, load_cert
from vsca.config import load_config
from vsca.data import load_split, tokenize_example
from vsca.modeling import generate_from_ids, load_model, load_tokenizer

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    ok &= bool(cond)


class FakeLLM:
    def __init__(self, model, tok, shuffle=False):
        self.model, self.tok, self.shuffle, self.calls, self.loras = model, tok, shuffle, 0, []

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

cfg = load_config(["configs/smoke.yaml"], ["limit_train=6", "cert_block=4", "h_min=0.0"])
tok = load_tokenizer(cfg.model_name)
exs = [e for e in (tokenize_example(tok, r, cfg.max_len) for r in load_split(cfg.data_dir, "gsm8k_train", 6)) if e]
model, _ = load_model(cfg)
tmp = Path(tempfile.mkdtemp())

hf = certify(model, tok, exs, cfg, tmp / "hf.jsonl")
cfg.rollout_backend = "vllm"
mk = lambda shuffle: rollout.VLLMBackend(cfg, tok, llm=FakeLLM(model, tok, shuffle))
v1 = certify(model, tok, exs, cfg, tmp / "v1.jsonl", backend=mk(False))
v2 = certify(model, tok, exs, cfg, tmp / "v2.jsonl", backend=mk(True))

claims = lambda recs: [[(p["t"], p["certified"]) for p in r["positions"] if p["type"] == "claim"] for r in recs]
check("1. greedy local certificates: vLLM path == HF path", claims(v1) == claims(hf))
check("2a. vLLM path deterministic and independent of request order", v1 == v2)
n_dec = sum(1 for r in v1 for p in r["positions"] if p["type"] == "decision")
check(f"2b. decision positions searched ({n_dec})", n_dec > 0)

cfg.debug_stop_after_cert_examples = 4
try:
    certify(model, tok, exs, cfg, tmp / "v3.jsonl", backend=mk(True))
    check("2c. debug stop triggered", False)
except SystemExit as e:
    check("2c. debug stop exits with 75", e.code == 75)
cfg.debug_stop_after_cert_examples = None
v3 = certify(model, tok, exs, cfg, tmp / "v3.jsonl", backend=mk(True))
check("2d. resumed vLLM certification == uninterrupted", v3 == v1 and load_cert(tmp / "v3.jsonl") == v1)

cfg.limit_train = 2
lmodel, _ = load_model(cfg, trainable=True)
b = mk(False)
certify(lmodel, tok, exs[:2], cfg, tmp / "lora_e0.jsonl", backend=b, epoch=0)
certify(lmodel, tok, exs[:2], cfg, tmp / "lora_e1.jsonl", backend=b, epoch=1)
ids = [l[1] for l in b._llm.loras if l is not None]
check("3a. snapshot adapter written for a LoRA model",
      (tmp / "lora_e0.snapshot" / "adapter_config.json").exists())
check("3b. every certification of a LoRA model uses a LoRA request, new id per snapshot",
      all(l is not None for l in b._llm.loras) and sorted(set(ids)) == [1, 2])

shutil.rmtree(tmp)
sys.exit(0 if ok else 1)
