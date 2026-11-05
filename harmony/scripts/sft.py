"""LoRA SFT on harmony/data/train.jsonl (loss on the response only). Resumable: checkpoints every 200 steps and at
the session deadline (exit 75); rerun the same command to continue.

  python scripts/sft.py --data data --out runs/sft_s0 [--model Qwen/Qwen2.5-1.5B --epochs 2 --lr 1e-4 --seed 0]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)


class Deadline(TrainerCallback):
    def __init__(self, at):
        self.at = at

    def on_step_end(self, args, state, control, **kw):
        if self.at and time.time() > self.at:
            control.should_save = True
            control.should_training_stop = True
            state.stopped_for_deadline = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None, help="drop examples longer than this many tokens")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (long examples)")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "adapter" / "adapter_config.json").exists():
        print(f"[sft] {out}/adapter exists; skipping")
        return
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    rows = [json.loads(l) for l in open(Path(a.data) / "train.jsonl")][: a.limit]

    def enc(r):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        y = tok(r["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return {"input_ids": p + y, "labels": [-100] * len(p) + y}

    data = [enc(r) for r in rows]
    if a.max_len:
        n0 = len(data)
        data = [d for d in data if len(d["input_ids"]) <= a.max_len]
        print(f"[sft] kept {len(data)}/{n0} examples of <= {a.max_len} tokens")

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        lab = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if cuda else torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * a.epochs * -(-len(data) // a.bs))), per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=a.bs // a.micro_bs,
        logging_steps=20, save_steps=200, save_total_limit=2, bf16=cuda, seed=a.seed, report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0, use_cpu=not cuda,
        gradient_checkpointing=a.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.grad_ckpt else None)
    deadline = float(os.environ["SESSION_DEADLINE"]) if os.environ.get("SESSION_DEADLINE") else None
    cb = Deadline(deadline)
    tr = Trainer(model=model, args=args, train_dataset=data, data_collator=collate, callbacks=[cb])
    last = None
    if (out / "ckpt").exists():
        cks = sorted((out / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        last = str(cks[-1]) if cks else None
    tr.train(resume_from_checkpoint=last)
    if getattr(tr.state, "stopped_for_deadline", False):
        print("[sft] session deadline reached; rerun to continue")
        sys.exit(75)
    model.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps(tr.state.log_history, indent=1))
    print(f"[sft] wrote {out}/adapter")


if __name__ == "__main__":
    main()
