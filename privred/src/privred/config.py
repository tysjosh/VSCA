"""privred configuration: sg's Config plus the privacy-study fields. YAML files set defaults; ``--set key=value``
overrides them (list / dict values are parsed with yaml.safe_load). ``to_sg`` gives the pure sg Config used for
every ``sg.train.train`` call, so member config.json files stay sg configs."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from sg.config import Config as SgConfig
from sg.config import _coerce
from sg.plans import partitions_dir as _sg_partitions_dir
from sg.plans import pct

STUDYA_CONDITIONS = ("ce", "true_hier", "flat_aux", "final_hier", "subset")


@dataclass
class Config(SgConfig):
    # companion study (sg out_dir) and data
    companion_dir: str = "../sg/runs"
    # budgets and model-set sizes (Sections 3.4, 3.6, 4.1)
    privacy_budgets: list = field(default_factory=lambda: [1.0, 0.5])
    studyB_budget: Optional[float] = 1.0  # None: no Study B (iNat-100)
    studyA_conditions: list = field(default_factory=lambda: list(STUDYA_CONDITIONS))
    M: int = 32
    M0: int = 16
    MB: int = 16
    K_folds: int = 5
    # attack (Section 3.4)
    alpha: float = 0.001
    alphas: list = field(default_factory=lambda: [0.001, 0.01, 0.1])
    min_expected_fp: int = 100
    min_ref_per_side: int = 15
    # support features (Section 3.8)
    kernel_t: float = 0.1
    support_eps: float = 1.0e-6
    n_vnn: int = 50
    # deletion (Section 3.9)
    eta: float = 0.04
    Q: int = 4
    Q_true_hier: int = 2
    cohort_seed: int = 0
    membership_seed: int = 0
    dv_tol: float = 0.05
    dv_max_swaps: int = 10000
    # Study A subset (Section 3.7)
    n_subset_random: int = 4
    n_subset_spectrum: int = 4
    spectrum_percentiles: list = field(default_factory=lambda: [10, 37, 63, 90])
    pattern_tol_r: float = 0.0
    pattern_tol_pd: float = 0.0
    # prediction / statistics (Sections 3.10, 4.3)
    top_frac: float = 0.05
    ridge_alphas: list = field(default_factory=lambda: [float(10 ** (e / 2)) for e in range(-6, 7)])
    inner_folds: int = 3
    n_boot: int = 2000
    alpha_family: float = 0.05
    # misc
    limit_audit_per_class: Optional[int] = None
    ece_bins: int = 15
    keep_models: bool = False
    gpu_h_per_20k: float = 0.35  # Table compute assumption: A100-hours of one 200-epoch run on 20,000 images

    def validate(self) -> None:
        super().validate()
        for c in self.studyA_conditions:
            if c not in STUDYA_CONDITIONS:
                raise ValueError(f"unknown Study A condition {c!r}; choose from {STUDYA_CONDITIONS}")
        for name in ("M", "M0", "MB"):
            v = getattr(self, name)
            if v < 4 or v % 2:
                raise ValueError(f"{name} must be even and >= 4 (got {v})")
        if self.K_folds < 2:
            raise ValueError("K_folds must be >= 2")


SG_FIELDS = tuple(f.name for f in dataclasses.fields(SgConfig))


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
        cur = getattr(cfg, key)
        setattr(cfg, key, yaml.safe_load(val) if isinstance(cur, (list, dict)) else _coerce(val, cur))
    cfg.validate()
    return cfg


def to_sg(cfg, **overrides) -> SgConfig:
    """Plain sg Config with the sg fields of ``cfg`` and ``overrides`` applied."""
    vals = {k: getattr(cfg, k) for k in SG_FIELDS}
    for k in overrides:
        if k not in vals:
            raise KeyError(f"to_sg: {k!r} is not an sg config field")
    vals.update(overrides)
    out = SgConfig(**vals)
    out.validate()
    return out


# --- companion artifacts (read, never rebuilt) ------------------------------------------------

def companion_root(cfg) -> Path:
    return Path(cfg.companion_dir) / cfg.dataset


def partitions_dir(cfg, budget) -> Path:
    return _sg_partitions_dir(cfg.companion_dir, cfg.dataset, budget)


def simclr_dir(cfg, budget=1.0) -> Path:
    """phi-bar = companion simclr/b100 (trained on exactly D_1.0 = P); phi_beta = simclr/b{pct}."""
    return companion_root(cfg) / "simclr" / f"b{pct(budget):03d}"


def geometry_path(cfg) -> Path:
    return companion_root(cfg) / "geometry.json"


def ce_run(cfg, budget, seed) -> Path:
    return companion_root(cfg) / f"b{pct(budget):03d}" / "ce" / f"s{seed}"


def true_hier_run(cfg, budget, rho, seed=0) -> Path:
    return companion_root(cfg) / f"b{pct(budget):03d}" / "true_hier" / f"rho{float(rho):g}" / f"s{seed}"


def _state(cfg, name):
    p = companion_root(cfg) / name
    if not p.exists():
        raise SystemExit(f"privred: companion file {p} missing (run the sg stage that writes it, or set companion_dir)")
    return json.loads(p.read_text())


def wd_for(cfg, budget, size_matched: bool = True) -> float:
    """Weight decay from the companion's best_wd.json. Privacy models (n_beta / 2 images) use the size-matched key
    beta / 2; OOF fold models use key beta (size_matched=False). A missing key falls back to the available budget
    nearest in log-size, ties to the smaller budget."""
    table = _state(cfg, "best_wd.json")
    target = float(budget) / 2 if size_matched else float(budget)
    key = str(float(target))
    if key in table:
        return float(table[key])
    avail = sorted(float(k) for k in table)
    if not avail:
        raise SystemExit(f"privred: {companion_root(cfg)}/best_wd.json is empty")
    best = min(avail, key=lambda b: (abs(math.log(b) - math.log(target)), b))
    return float(table[str(best)])


def rho_for(cfg) -> float:
    return float(_state(cfg, "best_rho.json")["rho"])
