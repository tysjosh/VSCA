"""Model / tokenizer loading, LoRA wrapping, and batched generation."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import pick_device, pick_dtype


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("a fast tokenizer is required for offset mappings")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: Config, adapter: Optional[str] = None, trainable: bool = False, attn_impl: Optional[str] = None):
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype, device)
    kwargs = {"torch_dtype": dtype}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
    if trainable and cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, device


@torch.no_grad()
def generate_from_ids(
    model,
    tok,
    prefixes: list[list[int]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[list[int]]:
    """Left-padded batched generation; returns only the new tokens per output sequence."""
    device = next(model.parameters()).device
    L = max(len(p) for p in prefixes)
    pad = tok.pad_token_id
    ids = torch.full((len(prefixes), L), pad, dtype=torch.long)
    mask = torch.zeros((len(prefixes), L), dtype=torch.long)
    for i, p in enumerate(prefixes):
        ids[i, L - len(p) :] = torch.tensor(p)
        mask[i, L - len(p) :] = 1
    kwargs = dict(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=pad,
        eos_token_id=tok.eos_token_id,
        stop_strings=["\nQuestion:"],
        tokenizer=tok,
    )
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    else:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    out = model.generate(**kwargs)
    new = out[:, L:].tolist()
    res = []
    for seq in new:
        if tok.eos_token_id in seq:
            seq = seq[: seq.index(tok.eos_token_id)]
        while seq and seq[-1] == pad:
            seq.pop()
        res.append(seq)
    return res
