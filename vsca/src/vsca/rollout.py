"""Generation backends for certification (Section 3.2): Hugging Face ``generate`` or vLLM.

Both take token-id prefixes and return the new token ids of each output sequence, ``n`` outputs per prefix in
prefix order, cut at EOS. Certification code calls ``backend.generate`` and does not know which one it uses.

vLLM specifics
  * The snapshot model is the base model plus the current LoRA adapter, which certification saves to disk and
    passes as a ``LoRARequest`` (a new id per snapshot, so vLLM never serves a stale adapter).
  * One engine per process, created on first use and reused for every later certification (each epoch refresh).
    It keeps ``vllm_gpu_mem`` of GPU memory for the rest of the process, next to the training model.
  * Each request gets its own sampling seed (``seeds``), derived by the caller from the block seed and the job
    index, so results do not depend on how vLLM schedules requests and a resumed certification samples what an
    uninterrupted one would have.
  * Prefix caching is on: all jobs of one example share the prompt and most of the reference prefix.
Sampling streams differ between backends, so their rollouts are not identical, only identically distributed.
Greedy local certificates agree up to bf16 kernel differences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

STOP = ["\nQuestion:"]


def _trim(seq: list[int], eos: int, pad: Optional[int]) -> list[int]:
    if eos in seq:
        seq = seq[: seq.index(eos)]
    while seq and pad is not None and seq[-1] == pad:
        seq = seq[:-1]
    return seq


class HFBackend:
    name = "hf"

    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    def set_snapshot(self, adapter_dir: Optional[Path]) -> None:
        pass  # generates with the in-memory model, which is the snapshot

    def generate(self, prefixes, max_new_tokens, do_sample, temperature=1.0, top_p=1.0, n=1, seeds=None, batch=16):
        from .modeling import generate_from_ids

        out = []
        for i in range(0, len(prefixes), batch):
            out += generate_from_ids(self.model, self.tok, prefixes[i : i + batch], max_new_tokens, do_sample,
                                     temperature, top_p, num_return_sequences=n)
        return out


class VLLMBackend:
    name = "vllm"
    _engines: dict = {}

    def __init__(self, cfg, tok, llm=None):
        self.cfg, self.tok = cfg, tok
        self._llm = llm  # tests pass a stand-in engine
        self._lora = None
        self._next_id = 1

    @property
    def llm(self):
        if self._llm is None:
            key = self.cfg.model_name
            if key not in VLLMBackend._engines:
                import os

                # FlashInfer's sampling kernels are compiled at first use and need nvcc, which the VCL VMs lack.
                # vLLM's PyTorch sampler needs no compilation. Set VLLM_USE_FLASHINFER_SAMPLER=1 to override.
                os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
                try:
                    from vllm import LLM
                except ImportError as e:
                    raise SystemExit("rollout_backend=vllm needs vLLM: `uv pip install --python .venv/bin/python vllm` "
                                     "(see README, 'vLLM backend')") from e
                VLLMBackend._engines[key] = LLM(
                    model=self.cfg.model_name,
                    dtype="bfloat16",
                    gpu_memory_utilization=self.cfg.vllm_gpu_mem,
                    max_model_len=self.cfg.max_len + max(self.cfg.max_rollout_tokens, self.cfg.max_span_tokens) + 8,
                    max_num_seqs=self.cfg.vllm_max_num_seqs,
                    enable_prefix_caching=True,
                    enable_lora=True,
                    max_lora_rank=max(self.cfg.lora_r, 8),
                    max_loras=1,
                    seed=self.cfg.seed,
                    enforce_eager=self.cfg.vllm_enforce_eager,
                )
            self._llm = VLLMBackend._engines[key]
        return self._llm

    def set_snapshot(self, adapter_dir: Optional[Path]) -> None:
        if adapter_dir is None:
            self._lora = None
            return
        from vllm.lora.request import LoRARequest

        self._lora = LoRARequest(f"snapshot{self._next_id}", self._next_id, str(adapter_dir))
        self._next_id += 1

    def _params(self, max_new_tokens, do_sample, temperature, top_p, n, seed):
        from vllm import SamplingParams

        if do_sample:
            return SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
                                  stop=STOP, seed=seed)
        return SamplingParams(n=n, temperature=0.0, max_tokens=max_new_tokens, stop=STOP)

    def generate(self, prefixes, max_new_tokens, do_sample, temperature=1.0, top_p=1.0, n=1, seeds=None, batch=None):
        if not prefixes:
            return []
        seeds = seeds if seeds is not None else [None] * len(prefixes)
        prompts = [{"prompt_token_ids": list(p)} for p in prefixes]
        params = [self._params(max_new_tokens, do_sample, temperature, top_p, n, s) for s in seeds]
        outs = self.llm.generate(prompts, params, lora_request=self._lora, use_tqdm=False)
        res = []
        for o in outs:  # vLLM returns results in prompt order
            seqs = sorted(o.outputs, key=lambda c: c.index)
            if len(seqs) != n:
                raise RuntimeError(f"vLLM returned {len(seqs)} sequences, expected {n}")
            res += [_trim(list(c.token_ids), self.tok.eos_token_id, self.tok.pad_token_id) for c in seqs]
        return res


def make_backend(cfg, model, tok):
    if cfg.rollout_backend == "hf":
        return HFBackend(model, tok)
    if cfg.rollout_backend == "vllm":
        return VLLMBackend(cfg, tok)
    raise ValueError(f"unknown rollout_backend {cfg.rollout_backend!r}")


def save_snapshot(model, out_dir: Path) -> Optional[Path]:
    """Write the current LoRA adapter for vLLM; None if the model is the plain base model."""
    if not hasattr(model, "peft_config"):
        return None
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    return out_dir
