"""Per-token scores for the selection baselines.

* reference losses (Rho-style, Token Cleaning, CF-Select): a LoRA model fine-tuned
  with SFT on the same training split; its cost is part of those baselines.
* base losses and attention scores (ssToken, Token Cleaning): the base model.
  The ssToken attention score of response token i is the head-averaged attention
  mass from query i to the prompt at a deep layer (Qin et al., 2025, Eq. 9).
  Here it is computed once with the base model rather than online; see README.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .data import Example, collate


def _decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if not hasattr(m, "layers"):
        raise RuntimeError("could not find decoder layers on model")
    return m.layers


@torch.no_grad()
def score_tokens(model, tok, examples: list[Example], bs: int, out_path: Path, with_attention: bool, attn_layer: int = -1) -> dict:
    device = next(model.parameters()).device
    model.eval()
    captured = {}
    handle = None
    if with_attention:
        layer = _decoder_layers(model)[attn_layer]

        def hook(_module, _inp, out):
            captured["w"] = out[1] if isinstance(out, tuple) and len(out) > 1 else None

        handle = layer.self_attn.register_forward_hook(hook)
    losses, attn = {}, {}
    try:
        for i in tqdm(range(0, len(examples), bs), desc="score"):
            group = examples[i : i + bs]
            batch = collate(group, tok.pad_token_id)
            ids = batch["input_ids"].to(device)
            logits = model(input_ids=ids, attention_mask=batch["attention_mask"].to(device)).logits.float()
            w = captured.get("w")
            if with_attention and w is None:
                raise RuntimeError("attention weights not returned; load the model with attn_implementation='eager'")
            for b, ex in enumerate(group):
                P, T = len(ex.prompt_ids), len(ex.resp_ids)
                lg = logits[b, P - 1 : P + T - 1]
                gold = torch.tensor(ex.resp_ids, device=device)
                nll = torch.logsumexp(lg, -1) - lg.gather(-1, gold[:, None]).squeeze(-1)
                losses[ex.id] = [round(float(v), 5) for v in nll.tolist()]
                if with_attention:
                    a = w[b].float().mean(0)  # [L, L], averaged over heads
                    attn[ex.id] = [round(float(v), 5) for v in a[P : P + T, :P].sum(-1).tolist()]
    finally:
        if handle is not None:
            handle.remove()
    out = {"loss": losses}
    if with_attention:
        out["attn"] = attn
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out))
    return out


def load_scores(path) -> dict:
    return json.loads(Path(path).read_text())
