"""Seeding, device selection, compute logging, and small I/O helpers."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class ComputeLog:
    """Wall-clock and GPU-hours per phase (Section 4.5: matched end-to-end compute)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    @contextmanager
    def phase(self, name: str):
        n_gpu = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            entry = self.data.setdefault(name, {"seconds": 0.0, "gpu_hours": 0.0, "calls": 0})
            entry["seconds"] += dt
            entry["gpu_hours"] += dt * n_gpu / 3600.0
            entry["calls"] += 1
            if torch.cuda.is_available():
                entry["peak_mem_gb"] = max(entry.get("peak_mem_gb", 0.0), torch.cuda.max_memory_allocated() / 1e9)
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def env_info() -> dict:
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "host": os.uname().nodename,
    }


# ---------------------------------------------------------------- session deadline and resume

DEADLINE_EXIT = 75  # EX_TEMPFAIL: "stopped cleanly, rerun the same command to continue"


def deadline_ts(cfg=None):
    """Unix time after which work must stop: cfg.deadline, else $VSCA_DEADLINE, else None."""
    v = getattr(cfg, "deadline", None) if cfg is not None else None
    if v is None:
        v = os.environ.get("VSCA_DEADLINE") or os.environ.get("SESSION_DEADLINE")
    return float(v) if v not in (None, "") else None


def past_deadline(cfg=None) -> bool:
    d = deadline_ts(cfg)
    return d is not None and time.time() >= d


def stop_for_resume(reason: str):
    print(f"[resume] {reason}; state is saved. Exiting with code {DEADLINE_EXIT}; rerun the same command to continue.",
          flush=True)
    raise SystemExit(DEADLINE_EXIT)


def check_deadline(cfg, where: str) -> None:
    if past_deadline(cfg):
        stop_for_resume(f"session deadline reached during {where}")
