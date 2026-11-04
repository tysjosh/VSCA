"""Run configuration. YAML files set defaults; ``--set key=value`` overrides them."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

METHODS = (
    "sft",            # full SFT (Eq. 1)
    "vsca",           # certified set-valued likelihood (Eq. 3)
    "random",         # random token mask at ratio rho
    "rho",            # Rho-style excess loss vs. fine-tuned reference, online
    "token_cleaning", # fixed-reference influence: base loss - reference loss
    "sstoken",        # retrospective excess loss vs. base + attention score
    "trimsft",        # Gaussian weight on gold-vs-competitor logit gap
    "claim_mask",     # claim spans first, filled to rho with random tokens
    "claim_upweight", # w = 1 + beta * 1[claim]
    "unlikelihood",   # SFT + unlikelihood on uncertified top-k candidates
    "cf_select",      # earlier formulation: binary counterfactual score + pairwise loss
    "rft",            # SFT on reference data + verified rollouts from certification
)

VSCA_VARIANTS = ("full", "claim_only", "decision_only", "uncertified", "shuffled")


@dataclass
class Config:
    # run
    run_name: str = "debug"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"  # auto -> bf16 on CUDA, fp32 elsewhere

    # model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: str = "all-linear"
    gradient_checkpointing: bool = False

    # data
    data_dir: str = "data"
    val_frac: float = 0.1
    split_seed: int = 1234
    max_len: int = 1024
    limit_train: Optional[int] = None
    limit_val: Optional[int] = 500
    limit_test: Optional[int] = None

    # optimisation
    method: str = "sft"
    lr: float = 2e-4
    epochs: int = 3
    micro_bs: int = 4
    eff_bs: int = 32
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    norm: str = "auto"  # "all" (divide by all target tokens), "selected", or "auto"
    select_best_on_val: bool = True

    # selection baselines
    rho: float = 0.5
    ref_scores: Optional[str] = None   # per-token losses of the fine-tuned reference model
    base_scores: Optional[str] = None  # per-token losses (and attention) of the base model
    sstoken_gamma: float = 0.5
    attn_layer: int = -1
    trim_mu: float = 1.5   # TrimSFT margin m (Jia et al., 2026, main setting)
    trim_tau: float = 0.8  # TrimSFT bandwidth tau
    upweight_beta: float = 1.0

    # VSCA certification (Section 3.2)
    k: int = 4
    M: int = 4
    B: int = 4
    h_min: float = 0.5  # nats
    refresh: str = "epoch"  # "epoch" (R = one epoch) or "once"
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    max_rollout_tokens: int = 320
    max_span_tokens: int = 48
    rollout_backend: str = "hf"  # "hf" (transformers generate) or "vllm" (CUDA only; see rollout.py)
    vllm_gpu_mem: float = 0.40   # fraction of GPU memory the vLLM engine reserves (training shares the GPU)
    vllm_max_num_seqs: int = 256
    vllm_enforce_eager: bool = False
    cert_bs: int = 16    # HF backend: sequences per generate call during certification
    cert_block: int = 64  # examples whose generation jobs are batched together (also the resume granularity)
    cert_path: Optional[str] = None  # reuse an existing certification file
    variant: str = "full"
    inject_eta: float = 0.0
    drop_eta: float = 0.0
    shuffle_bins: int = 10
    vplus_grounding: bool = True  # V+ (v2): also require every annotation operand to be grounded
    grounding_whitelist: str = "60,24,7,12,100,1000,2,0.5"  # V+ constants always accepted as operands

    # baselines that use certification output
    ul_lambda: float = 1.0
    rft_from: Optional[str] = None

    # earlier formulation (CF-Select)
    cf_alpha: float = 0.5
    cf_gamma: float = 0.0
    cf_lambda: float = 0.1
    cf_tau: float = 1.0
    cf_max_pairs: int = 4  # counterfactual pairs sampled per micro-batch for the pairwise term
    cf_regime: str = "claim"  # "claim" or "final" (final-answer verifier, no propagation)

    # evaluation
    eval_bs: int = 16
    gen_max_new: int = 400

    # resume (sessions of limited length)
    resume: bool = True          # continue from runs/<name>/ckpt.pt if present; skip runs with train_summary.json
    ckpt_every: int = 50         # optimizer steps between training checkpoints
    keep_ckpt: bool = False      # keep ckpt.pt (adapter + optimizer state) after a run finishes
    deadline: Optional[float] = None  # unix time to stop cleanly; default from $VSCA_DEADLINE
    debug_stop_after_steps: Optional[int] = None         # test hook: checkpoint and exit after N optimizer steps
    debug_stop_after_cert_examples: Optional[int] = None  # test hook: exit after N newly certified examples

    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def resolved_norm(self) -> str:
        if self.norm != "auto":
            return self.norm
        # Published selection baselines normalise by the selected count; VSCA and
        # weighting methods normalise by all target tokens (Eq. 3).
        return "selected" if self.method in ("random", "rho", "token_cleaning", "sstoken", "claim_mask", "cf_select") else "all"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.variant not in VSCA_VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; choose from {VSCA_VARIANTS}")
        if self.rollout_backend not in ("hf", "vllm"):
            raise ValueError("rollout_backend must be 'hf' or 'vllm'")
        if self.refresh not in ("epoch", "once"):
            raise ValueError("refresh must be 'epoch' or 'once'")
        if self.method in ("rho", "token_cleaning", "cf_select") and not self.ref_scores:
            raise ValueError(f"method {self.method} needs ref_scores (run `vsca score` with the reference model)")
        if self.method in ("token_cleaning", "sstoken") and not self.base_scores:
            raise ValueError(f"method {self.method} needs base_scores (run `vsca score` with the base model)")
        if self.method == "rft" and not self.rft_from:
            raise ValueError("method rft needs rft_from (a certification file with rollouts)")
        if self.method == "unlikelihood" and not self.cert_path:
            raise ValueError("method unlikelihood needs cert_path (a certification file)")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(value: str, current: Any) -> Any:
    if value.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return yaml.safe_load(value) if current is None else value


def load_config(paths: list[str], overrides: list[str]) -> Config:
    cfg = Config()
    names = {f.name for f in dataclasses.fields(Config)}
    for p in paths:
        with open(p) as fh:
            data = yaml.safe_load(fh) or {}
        for key, val in data.items():
            if key not in names:
                raise KeyError(f"{p}: unknown config key {key!r}")
            setattr(cfg, key, val)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        if key not in names:
            raise KeyError(f"unknown config key {key!r}")
        setattr(cfg, key, _coerce(val, getattr(cfg, key)))
    cfg.validate()
    return cfg
