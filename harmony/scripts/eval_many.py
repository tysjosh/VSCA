"""Greedy decoding of several LoRA adapters on several splits with one vLLM engine (skips finished outputs).

  python scripts/eval_many.py --data data --splits test_id test_long --adapters runs/p_s0/set/adapter ...
Writes <adapter dir>/../<split>.jsonl (id, greedy, samples=[]) next to each adapter.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--max_model_len", type=int, default=1024)
    a = ap.parse_args()
    jobs = [(Path(ad), s) for ad in a.adapters for s in a.splits if not (Path(ad).parent / f"{s}.jsonl").exists()]
    if not jobs:
        print("[eval_many] nothing to do")
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=a.model, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1, seed=0,
              gpu_memory_utilization=0.85, max_model_len=a.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=["END"])
    lora_id = {}
    for ad, split in jobs:
        rows = [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl")]
        lid = lora_id.setdefault(str(ad), len(lora_id) + 1)
        outs = llm.generate([r["prompt"] for r in rows], sp, lora_request=LoRARequest(f"a{lid}", lid, str(ad)),
                            use_tqdm=False)
        out = ad.parent / f"{split}.jsonl"
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "greedy": o.outputs[0].text, "samples": []}) + "\n")
        tmp.rename(out)
        print(f"[eval_many] {out}")


if __name__ == "__main__":
    main()
