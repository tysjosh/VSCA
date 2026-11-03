"""Run plans for every experiment (D15), a subprocess GPU launcher, and a GPU-hour estimate.

Run names come from the run's configuration, not from the experiment, so runs shared by Exp A, Exp B and the
pilots get identical names and ``skip_if_done`` runs each of them once. ``sg plan`` writes one shell command
per line; ``sg launch`` runs a plan file over GPUs.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .partitions import cover_subset
from .utils import DEADLINE_EXIT, past_deadline

WDS = (1e-4, 5e-4, 1e-3)               # Section III-E
RHOS = (0.25, 0.5, 1.0)                # Section III-E
CIFAR_BUDGETS = (0.1, 0.25, 0.5, 1.0)  # Table II
EXPA_BUDGETS = (0.25, 1.0)             # Exp A, rho sensitivity, iNat-100
N_CE_SEEDS = 5
N_SEEDS = 3
N_COVER = 9
EXPS = ("tune_wd", "ce", "rho_pilot", "var_pilot", "expA", "expB", "expC", "rho_sens", "fixedstep",
        "inat_tune_wd", "inat_ce", "inat_A")


# --- names -----------------------------------------------------------------------------------

def pct(budget) -> int:
    return int(round(float(budget) * 100))


def _get(c, key, default=None):
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def run_name(c, tune_wd: bool = False) -> str:
    """D15 run name from a config-like mapping/object (dataset, budget, condition, seed, rho, partition or
    partition_name, total_steps, weight_decay)."""
    ds, b = _get(c, "dataset"), pct(_get(c, "budget"))
    if tune_wd:
        return f"{ds}/tune_wd/b{b:03d}/wd{float(_get(c, 'weight_decay')):g}/s0"
    ts = _get(c, "total_steps")
    head = f"{ds}/b{b:03d}" + (f"_steps{int(ts)}" if ts else "")
    cond, seed = _get(c, "condition"), int(_get(c, "seed"))
    if cond == "ce":
        return f"{head}/ce/s{seed}"
    pname = _get(c, "partition_name") or (Path(_get(c, "partition")).stem if _get(c, "partition") else None)
    unit = f"{cond}_{pname}" if cond in ("random_hier", "spectrum_hier") else cond
    return f"{head}/{unit}/rho{float(_get(c, 'rho')):g}/s{seed}"


def simclr_name(dataset, budget=None) -> str:
    return f"{dataset}/simclr/all" if budget is None else f"{dataset}/simclr/b{pct(budget):03d}"


def partitions_dir(out_dir, dataset, budget) -> Path:
    return Path(out_dir) / dataset / "partitions" / f"b{pct(budget):03d}"


def cover_names(out_dir, dataset, budget, k: int = N_COVER) -> dict:
    """The k random and k spectrum partitions that cover the budget's W_train range (Exp B)."""
    idx = partitions_dir(out_dir, dataset, budget) / "index.json"
    if not idx.exists():
        raise SystemExit(f"plan: {idx} missing; build the partitions for budget {budget} first")
    parts = json.loads(idx.read_text())["partitions"]
    return {kind: [p["name"] for p in cover_subset([p for p in parts if p["kind"] == kind], k)]
            for kind in ("random", "spectrum")}


# --- state written by earlier stages --------------------------------------------------------

class State:
    def __init__(self, state_dir):
        self.dir = Path(state_dir)

    def _load(self, name, stage):
        p = self.dir / name
        if not p.exists():
            raise SystemExit(f"plan: {p} missing; run the `{stage}` stage first (or pass --state_dir)")
        return json.loads(p.read_text())

    def wd(self, budget) -> float:
        t = self._load("best_wd.json", "tune_wd")
        key = str(float(budget))
        if key not in t:
            raise SystemExit(f"plan: {self.dir}/best_wd.json has no weight decay for budget {key}")
        return float(t[key])

    def rho(self) -> float:
        return float(self._load("best_rho.json", "rho_pilot")["rho"])


# --- jobs ------------------------------------------------------------------------------------

def n_train(cfg, budget) -> float:
    """Training images at a budget (Table II for CIFAR-100; 100 species x inat_per_class pool for iNat-100)."""
    pool = 500 if cfg.dataset == "cifar100" else cfg.inat_per_class
    train_pool = pool - round(cfg.val_frac * pool) - round(cfg.audit_frac * pool)  # CIFAR-100: 400 per class
    per_class = float(budget) * train_pool
    if cfg.limit_train_per_class is not None:
        per_class = min(per_class, cfg.limit_train_per_class)
    return 100 * per_class


def fixed_steps(cfg) -> int:
    """D17: steps of the 100% budget, ceil(n_train(100%) / batch_size) * epochs (62,600 for CIFAR-100)."""
    pool = 500 if cfg.dataset == "cifar100" else cfg.inat_per_class
    full = 100 * (pool - round(cfg.val_frac * pool) - round(cfg.audit_frac * pool))
    return int(math.ceil(full / cfg.batch_size) * cfg.epochs)


def _train(cfg, budget, condition, seed, wd, rho=None, pname=None, total_steps=None, tune_wd=False) -> dict:
    sets = {"dataset": cfg.dataset, "budget": float(budget), "condition": condition, "seed": int(seed),
            "weight_decay": float(wd)}
    if condition != "ce":
        sets["rho"] = float(rho)
    if pname:
        sets["partition"] = str(partitions_dir(cfg.out_dir, cfg.dataset, budget) / f"{pname}.json")
    if total_steps:
        sets["total_steps"] = int(total_steps)
    sets["run_name"] = run_name({**sets, "partition_name": pname}, tune_wd=tune_wd)
    return {"cmd": "train", "sets": sets}


def _ce(cfg, st, budgets, seeds=range(N_CE_SEEDS), total_steps=None):
    return [_train(cfg, b, "ce", s, st.wd(b), total_steps=total_steps) for b in budgets for s in seeds]


def _units(cfg, budget, kinds):
    """(condition, partition name) units. kinds: 'true', 'flat', 'final', 'all' (19 random + 19 spectrum),
    'cover' (9 + 9 covering W)."""
    out = []
    for k in kinds:
        if k == "true":
            out.append(("true_hier", None))
        elif k == "flat":
            out.append(("flat_aux", None))
        elif k == "final":
            out.append(("final_hier", None))
        elif k == "all":
            out += [("random_hier", f"random_{i:02d}") for i in range(cfg.n_random)]
            out += [("spectrum_hier", f"spectrum_{i:02d}") for i in range(cfg.n_spectrum)]
        elif k == "cover":
            cov = cover_names(cfg.out_dir, cfg.dataset, budget)
            out += [("random_hier", n) for n in cov["random"]] + [("spectrum_hier", n) for n in cov["spectrum"]]
    return out


def _aux(cfg, st, budgets, kinds, rho, seeds=range(N_SEEDS), total_steps=None):
    return [_train(cfg, b, c, s, st.wd(b), rho, pname, total_steps)
            for b in budgets for c, pname in _units(cfg, b, kinds) for s in seeds]


def _need(cfg, dataset, exp):
    if cfg.dataset != dataset:
        raise SystemExit(f"plan {exp} needs dataset={dataset} (got {cfg.dataset}); "
                         + ("add --config configs/inat100.yaml" if dataset == "inat100" else "drop the iNat config"))


def generate(exp, cfg, state_dir=None) -> list:
    if exp not in EXPS:
        raise SystemExit(f"unknown plan {exp!r}; choose from {EXPS}")
    _need(cfg, "inat100" if exp.startswith("inat_") else "cifar100", exp)
    st = State(state_dir or Path(cfg.out_dir) / cfg.dataset)
    if exp in ("tune_wd", "inat_tune_wd"):
        budgets = CIFAR_BUDGETS if exp == "tune_wd" else EXPA_BUDGETS
        return [_train(cfg, b, "ce", 0, wd, tune_wd=True) for b in budgets for wd in WDS]
    if exp == "ce":
        return _ce(cfg, st, CIFAR_BUDGETS)
    if exp == "inat_ce":
        return _ce(cfg, st, EXPA_BUDGETS)
    if exp == "rho_pilot":  # D5
        return [_train(cfg, 1.0, c, 0, st.wd(1.0), r, pname)
                for r in RHOS for c, pname in (("true_hier", None), ("random_hier", "random_00"), ("flat_aux", None))]
    if exp == "var_pilot":  # Section IV-B.3: 3 seeds of CE, True-Hier and 5 random partitions at 25%
        rho = st.rho()
        seeds = range(3)
        return _ce(cfg, st, (0.25,), seeds) + [_train(cfg, 0.25, c, s, st.wd(0.25), rho, pname)
                                              for c, pname in [("true_hier", None)] + [("random_hier", f"random_{i:02d}") for i in range(5)]
                                              for s in seeds]
    if exp in ("expA", "expC"):
        jobs = _ce(cfg, st, EXPA_BUDGETS) + _aux(cfg, st, EXPA_BUDGETS, ("true", "all", "flat", "final"), st.rho())
        if exp == "expA":
            return jobs
        return [{"cmd": "probes", "run": str(Path(cfg.out_dir) / j["sets"]["run_name"]), "sets": j["sets"]} for j in jobs]
    if exp == "expB":
        return _ce(cfg, st, CIFAR_BUDGETS) + _aux(cfg, st, CIFAR_BUDGETS, ("true", "cover"), st.rho())
    if exp == "rho_sens":
        best = st.rho()
        return [j for r in RHOS if abs(r - best) > 1e-9
                for j in _aux(cfg, st, EXPA_BUDGETS, ("true", "flat", "final", "cover"), r)]
    if exp == "fixedstep":  # D17
        ts = fixed_steps(cfg)
        return _ce(cfg, st, CIFAR_BUDGETS, total_steps=ts) + _aux(cfg, st, CIFAR_BUDGETS, ("true", "cover"), st.rho(),
                                                                 total_steps=ts)
    # inat_A
    return _ce(cfg, st, EXPA_BUDGETS) + _aux(cfg, st, EXPA_BUDGETS, ("true", "cover", "flat", "final"), st.rho())


def to_lines(jobs, configs, user_sets) -> list:
    base = [sys.executable, "-m", "sg.cli"]
    cfg_args = [a for c in (configs or []) for a in ("--config", c)]
    user = [a for s in (user_sets or []) for a in ("--set", s)]
    lines = []
    for j in jobs:
        if j["cmd"] == "train":
            sets = [a for k, v in j["sets"].items() for a in ("--set", f"{k}={v}")]
            args = base + ["train"] + cfg_args + user + sets
        else:
            args = base + ["probes"] + cfg_args + user + ["--run", j["run"]]
        lines.append(" ".join(shlex.quote(a) for a in args))
    return lines


def count(exp, jobs, cfg, sec_per_epoch45k=None) -> dict:
    per_budget: dict = {}
    images = 0.0
    for j in jobs:
        s = j["sets"]
        b = per_budget.setdefault(str(s["budget"]), {"total": 0, "ce": 0, "aux": 0, "by_condition": {}})
        b["total"] += 1
        b["ce" if s["condition"] == "ce" else "aux"] += 1
        b["by_condition"][s["condition"]] = b["by_condition"].get(s["condition"], 0) + 1
        if j["cmd"] == "train":
            nt = n_train(cfg, s["budget"])
            ts = s.get("total_steps")
            images += ts * cfg.batch_size if ts else cfg.epochs * nt
    out = {"exp": exp, "cmd": jobs[0]["cmd"] if jobs else None, "n_commands": len(jobs), "per_budget": per_budget}
    if sec_per_epoch45k is not None and jobs and jobs[0]["cmd"] == "train":
        px = (cfg.inat_size / 32) ** 2 if cfg.dataset == "inat100" else 1.0
        out["gpu_hours_assumed"] = images / 45000 * sec_per_epoch45k * px / 3600
        out["assumption"] = (f"{sec_per_epoch45k} s per 45,000-image 32x32 epoch on one GPU (replace with compute.json "
                             f"of a pilot run); cost scaled by pixels x{px:g}; epoch-1 calibration overhead, eval, "
                             "SimCLR and probes not included")
    return out


# --- launcher --------------------------------------------------------------------------------

def launch(plan, gpus: str = "none", jobs_per_gpu: int = 1) -> list:
    """Run every line of ``plan`` in a subprocess pool. Each slot has its own CUDA_VISIBLE_DEVICES (``none``:
    the environment is left as is). Logs go to ``<plan>.logs/NNNN.log``. Returns the failed lines."""
    lines = [ln.strip() for ln in Path(plan).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    log_dir = Path(f"{plan}.logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    if gpus == "none":
        slots = [None] * max(1, jobs_per_gpu)
    else:
        slots = [g.strip() for g in gpus.split(",") if g.strip() for _ in range(jobs_per_gpu)]
    free: queue.Queue = queue.Queue()
    for s in slots:
        free.put(s)

    def run(i, line):
        slot = free.get()
        try:
            if past_deadline():  # do not start new runs after the session deadline
                return {"line": i, "rc": DEADLINE_EXIT, "gpu": slot, "log": None, "cmd": line, "skipped": True}
            env = os.environ.copy()
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = slot
            log = log_dir / f"{i:04d}.log"
            with open(log, "w") as fh:
                fh.write(f"# {line}\n")
                fh.flush()
                rc = subprocess.run(line, shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            print(f"[{i + 1}/{len(lines)}] rc={rc} gpu={slot} log={log}", flush=True)
            return {"line": i, "rc": rc, "gpu": slot, "log": str(log), "cmd": line}
        finally:
            free.put(slot)

    with ThreadPoolExecutor(len(slots)) as ex:
        results = list(ex.map(lambda a: run(*a), enumerate(lines)))
    failures = [r for r in results if r["rc"] != 0]
    (log_dir / "summary.json").write_text(json.dumps({"plan": str(plan), "n": len(lines), "failures": failures}, indent=2))
    return failures
