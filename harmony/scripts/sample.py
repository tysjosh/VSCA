"""Sample responses for a test split with vLLM (optionally with a LoRA adapter): greedy plus n samples at T.

  python scripts/sample.py --split data/test_id.jsonl --adapter runs/sft_s0/adapter --out runs/sft_s0/test_id.jsonl
Writes one row per problem: id, greedy, samples (texts). Skips if the output exists.
"""

import argparse
import json
import os
from pathlib import Path


def gpu_share(cap: float = 0.85) -> float:
    """vLLM memory share: 0.85, or less when another job already uses the GPU (free share minus a 5 % margin)."""
    import torch

    free, total = torch.cuda.mem_get_info()
    share = min(cap, free / total - 0.05)
    if share < 0.15:
        raise SystemExit(f"only {free / 2**30:.1f} GiB free on the GPU; not starting vLLM")
    return round(share, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--max_model_len", type=int, default=1024)
    a = ap.parse_args()
    if Path(a.out).exists():
        print(f"[sample] {a.out} exists; skipping")
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in open(a.split)]
    llm = LLM(model=a.model, dtype="bfloat16", enable_lora=bool(a.adapter), max_lora_rank=16, seed=a.seed,
              gpu_memory_utilization=gpu_share(), max_model_len=a.max_model_len)
    lora = LoRARequest("a", 1, a.adapter) if a.adapter else None
    prompts = [r["prompt"] for r in rows]
    stop = ["END"]
    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=stop),
                          lora_request=lora)
    samp = [None] * len(rows)
    if a.n > 0:  # --n 0: greedy only
        samp = llm.generate(prompts, [SamplingParams(n=a.n, temperature=a.temperature, max_tokens=a.max_tokens,
                                                     stop=stop, seed=a.seed * 1_000_003 + i) for i in range(len(rows))],
                            lora_request=lora)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r, g, s in zip(rows, greedy, samp):
            outs = [] if s is None else [o.text for o in sorted(s.outputs, key=lambda o: o.index)]
            f.write(json.dumps({"id": r["id"], "greedy": g.outputs[0].text, "samples": outs}) + "\n")
    print(f"[sample] wrote {a.out}")


if __name__ == "__main__":
    main()
