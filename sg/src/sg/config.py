"""Run configuration. YAML files set defaults; ``--set key=value`` overrides them."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

CONDITIONS = ("ce", "true_hier", "random_hier", "spectrum_hier", "flat_aux", "final_hier")

# Stages that carry an auxiliary head (Table I).
AUX_STAGES = {
    "ce": (),
    "true_hier": ("s2", "s3"),
    "random_hier": ("s2", "s3"),
    "spectrum_hier": ("s2", "s3"),
    "flat_aux": ("s2", "s3"),
    "final_hier": ("s4",),
}

DATASETS = ("cifar100", "inat100")
SIMCLR_SCOPES = ("budget", "all")


@dataclass
class Config:
    # run
    run_name: str = "debug"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    amp: bool = True  # bf16 autocast, CUDA only
    num_workers: int = 4
    skip_if_done: bool = True
    save_model: bool = True
    # resume (sessions of limited length)
    ckpt_every_epochs: int = 5          # training / SimCLR checkpoint interval (also saved at the deadline)
    keep_ckpt: bool = False             # keep ckpt.pt after a run finishes
    deadline: Optional[float] = None    # unix time to stop cleanly; default $SESSION_DEADLINE
    debug_stop_after_epochs: Optional[int] = None  # test hook: checkpoint and exit after N epochs

    # data
    dataset: str = "cifar100"
    data_dir: str = "data"
    split_seed: int = 1234
    budget: float = 1.0
    val_frac: float = 0.1    # validation pool, fraction of each class (CIFAR-100: 50 of 500)
    audit_frac: float = 0.1  # nonmember audit pool shared with the privacy study (CIFAR-100: 50 of 500)
    limit_train_per_class: Optional[int] = None
    limit_val_per_class: Optional[int] = None
    limit_test: Optional[int] = None
    # privacy study (privred): explicit training rows and per-example outputs; defaults change nothing
    train_indices: Optional[str] = None  # .npy or JSON list of training-pool rows; replaces the budget's train split
    example_sets: Optional[str] = None   # JSON {name: [rows]}; writes examples_<name>.npz after evaluation
    example_logits: bool = False         # also store float16 logits in examples_<name>.npz

    # condition
    condition: str = "ce"
    partition: Optional[str] = None
    rho: float = 0.5

    # optimisation (Section III-E)
    epochs: int = 200
    total_steps: Optional[int] = None
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    crop_pad: int = 4
    eval_batch_size: int = 512

    # SimCLR extractor (D7)
    simclr_scope: str = "budget"
    simclr_epochs: int = 500
    simclr_batch_size: int = 512
    simclr_lr_per256: float = 0.06
    simclr_wd: float = 5e-4
    simclr_temperature: float = 0.5
    simclr_proj_dim: int = 128

    # partitions (D8)
    partition_seed: int = 0
    n_groups: int = 20
    n_random: int = 19
    n_spectrum: int = 19
    search_iters: int = 20000
    search_restarts: int = 20

    # probes / strata
    knn_k: int = 20
    probe_linear_iters: int = 500
    probe_l2: float = 1e-4
    probe_ref_budget: float = 0.1
    hard_frac: float = 0.25
    label_errors: Optional[str] = None

    # iNat-100 (D13)
    inat_root: Optional[str] = None
    inat_seed: int = 0
    inat_per_class: int = 300
    inat_test_per_class: int = 50
    inat_size: int = 64

    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; choose from {CONDITIONS}")
        if self.condition in ("random_hier", "spectrum_hier") and not self.partition:
            raise ValueError(f"condition {self.condition} needs partition (path to a partition json)")
        if not self.rho > 0:
            raise ValueError("rho must be > 0")
        if self.simclr_scope not in SIMCLR_SCOPES:
            raise ValueError(f"simclr_scope must be one of {SIMCLR_SCOPES}")
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}; choose from {DATASETS}")

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
