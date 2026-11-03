"""Model sets, run plans and the compute count.

A model set is <out_dir>/<dataset>/b{pct}/sets/<name>/ with manifest.json, pool_ids.npy (sorted pool D),
membership.npy (bool [|D|, M]), example_sets.json (pool, audit) or example_sets_m{mm}.json (OOF: heldout, audit),
and train_m{mm}.npy. ``privred plan`` writes one ``python -m privred.cli member ...`` command per model;
``privred launch`` runs a plan file with sg.plans.launch.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
from sg.plans import cover_names, pct

from .config import partitions_dir, rho_for, wd_for
from .io import audit_indices, budget_dir, pool_indices, set_dir, write_json
from .membership import membership_matrix, seed_base

STAGES = ("oof", "studyA", "prelim", "baseB", "deletion")
DEL_TYPES = {"ce": ("rand", "sem", "vis", "hv", "near", "far"), "true_hier": ("rand", "sem", "vis", "hv")}
STUDYB_CONDITIONS = ("ce", "true_hier")


# --- Study A subset ---------------------------------------------------------------------------

def select_subset(random_names, spectrum_parts, n_random: int, n_spectrum: int, percentiles) -> list:
    """Random: the n_random names with the lowest generation index. Spectrum: for each percentile p of W_train among
    the given partitions, the closest not-yet-chosen one (ties: lower W). Returns [(condition, name)]."""
    rnd = sorted(random_names, key=lambda s: int(s.rsplit("_", 1)[1]))[:n_random]
    parts = sorted(spectrum_parts, key=lambda p: (p["W_train"], p["name"]))
    W = np.array([p["W_train"] for p in parts], np.float64)
    chosen = []
    for q in list(percentiles)[:n_spectrum]:
        target = float(np.percentile(W, q))
        order = sorted(range(len(parts)), key=lambda j: (abs(W[j] - target), W[j]))
        chosen.append(next(j for j in order if j not in chosen))
    return [("random_hier", n) for n in rnd] + [("spectrum_hier", parts[j]["name"]) for j in chosen]


def subset_partitions(cfg, budget, required: bool = True) -> list:
    """The 4 + 4 Study A partitions from the 9 + 9 the companion trains at budget beta (sg.plans.cover_names)."""
    idx = partitions_dir(cfg, budget) / "index.json"
    if not idx.exists():
        if required:
            raise SystemExit(f"plan: {idx} missing; the companion partitions for budget {budget} are needed")
        return [("random_hier", f"slot{k}") for k in range(cfg.n_subset_random)] + \
               [("spectrum_hier", f"slot{k}") for k in range(cfg.n_subset_spectrum)]
    cov = cover_names(cfg.companion_dir, cfg.dataset, budget)
    parts = {p["name"]: p for p in json.loads(idx.read_text())["partitions"]}
    return select_subset(cov["random"], [parts[n] for n in cov["spectrum"]], cfg.n_subset_random,
                         cfg.n_subset_spectrum, cfg.spectrum_percentiles)


def _studyA_units(cfg, budget, required: bool = True) -> list:
    """[(set name, condition, partition name or None)] in registry order."""
    out = []
    for c in cfg.studyA_conditions:
        if c == "subset":
            out += [(f"A_{cond}_{p}", cond, p) for cond, p in subset_partitions(cfg, budget, required)]
        else:
            out.append((f"A_{c}", c, None))
    return out


def _deletion_units(cfg) -> list:
    out = []
    for cond, nq in (("ce", cfg.Q), ("true_hier", cfg.Q_true_hier)):
        out += [(f"del_{cond}_q{q}_{t}", cond, q, t) for q in range(nq) for t in DEL_TYPES[cond]]
    return out


def all_set_names(cfg, budget) -> list:
    """Every model set name that can exist at a budget, in a fixed order (the seed registry). Subset slots keep
    their position whether or not the companion partitions are available."""
    names = ["oof"] + [u[0] for u in _studyA_units(cfg, budget, required=False)]
    names += ["prelim"] + [f"baseB_{c}" for c in STUDYB_CONDITIONS] + [u[0] for u in _deletion_units(cfg)]
    return names


# --- writing sets -----------------------------------------------------------------------------

def removal_path(cfg, budget, q, typ) -> Path:
    return budget_dir(cfg, budget) / "deletion" / f"q{q}_{typ}" / "R.npy"


def _specs(stage, cfg, budget) -> list:
    b = float(budget)
    base = {"budget": b, "dataset": cfg.dataset, "removal": None, "example_logits": False, "pd_sets": []}
    if stage == "oof":
        return [{**base, "name": "oof", "kind": "oof", "condition": "ce", "partition": None, "M": cfg.K_folds,
                 "wd": wd_for(cfg, b, size_matched=False), "rho": None, "example_logits": True,
                 "pd_sets": ["heldout"]}]
    wd = wd_for(cfg, b)
    if stage == "studyA":
        out = []
        for name, cond, pname in _studyA_units(cfg, b):
            part = str((partitions_dir(cfg, b) / f"{pname}.json").resolve()) if pname else None
            out.append({**base, "name": name, "kind": "studyA", "condition": cond, "partition": part, "M": cfg.M,
                        "wd": wd, "rho": None if cond == "ce" else rho_for(cfg), "pd_sets": ["pool"]})
        return out
    if cfg.studyB_budget is None or abs(b - float(cfg.studyB_budget)) > 1e-9:
        raise SystemExit(f"plan {stage}: Study B runs only at studyB_budget={cfg.studyB_budget}")
    if stage == "prelim":
        return [{**base, "name": "prelim", "kind": "prelim", "condition": "ce", "partition": None, "M": cfg.M0,
                 "wd": wd, "rho": None}]
    if stage == "baseB":
        return [{**base, "name": f"baseB_{c}", "kind": "baseB", "condition": c, "partition": None, "M": cfg.MB,
                 "wd": wd, "rho": None if c == "ce" else rho_for(cfg)} for c in STUDYB_CONDITIONS]
    if stage == "deletion":
        out = []
        for name, cond, q, t in _deletion_units(cfg):
            rp = removal_path(cfg, b, q, t)
            if not rp.exists():
                raise SystemExit(f"plan deletion: {rp} missing; run `privred deletion_sets` first")
            out.append({**base, "name": name, "kind": "deletion", "condition": cond, "partition": None, "M": cfg.MB,
                        "wd": wd, "rho": None if cond == "ce" else rho_for(cfg), "removal": str(rp), "cohort": q,
                        "type": t})
        return out
    raise SystemExit(f"unknown stage {stage!r}; choose from {STAGES}")


SPEC_KEYS = ("name", "kind", "condition", "partition", "budget", "dataset", "M", "seed_base", "wd", "rho", "pd_sets",
             "removal", "example_logits")


def make_sets(stage, cfg, budget, names=None, ds=None) -> list:
    """Write the model sets of a stage (only those in ``names`` if given). Existing sets with the same manifest are
    kept; a changed manifest is an error (delete the set directory to rebuild it)."""
    from sg.data import load_dataset

    from .difficulty import oof_folds

    specs = [s for s in _specs(stage, cfg, budget) if names is None or s["name"] in names]
    if names is not None and len(specs) != len(set(names)):
        raise SystemExit(f"plan: unknown set names {sorted(set(names) - {s['name'] for s in specs})} for stage {stage}")
    ds = ds or load_dataset(cfg)
    pool_all = pool_indices(cfg, ds, budget)
    aud = audit_indices(cfg, ds)
    dirs = []
    for s in specs:
        s["seed_base"] = seed_base(cfg, budget, s["name"])
        d = set_dir(cfg, budget, s["name"])
        dirs.append(d)
        man = {k: s.get(k) for k in SPEC_KEYS} | {k: s[k] for k in ("cohort", "type") if k in s}
        mp = d / "manifest.json"
        if mp.exists():
            old = json.loads(mp.read_text())
            if old != json.loads(json.dumps(man)):
                raise SystemExit(f"plan: {mp} exists with a different manifest; delete {d} to rebuild it")
            continue
        pool = pool_all
        if s["removal"]:
            R = np.load(s["removal"])
            pool = np.setdiff1d(pool_all, R)
        M = s["M"]
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "pool_ids.npy", pool)
        if s["kind"] == "oof":
            fold = oof_folds(ds.y_train[pool], M, cfg.membership_seed)
            np.save(d / "folds.npy", fold)
            U = fold[:, None] != np.arange(M)[None, :]
            for m in range(M):
                write_json(d / f"example_sets_m{m:02d}.json",
                           {"heldout": pool[fold == m].tolist(), "audit": aud.tolist()})
        else:
            U = membership_matrix(len(pool), M, s["name"], cfg.membership_seed)
            write_json(d / "example_sets.json", {"pool": pool.tolist(), "audit": aud.tolist()})
        np.save(d / "membership.npy", U)
        for m in range(M):
            np.save(d / f"train_m{m:02d}.npy", np.sort(pool[U[:, m]]))
        write_json(mp, man)
    return dirs


def plan_lines(set_dirs, configs, user_sets) -> list:
    base = [sys.executable, "-m", "privred.cli", "member"]
    cfg_args = [a for c in (configs or []) for a in ("--config", c)]
    user = [a for s in (user_sets or []) for a in ("--set", s)]
    lines = []
    for d in set_dirs:
        M = json.loads((Path(d) / "manifest.json").read_text())["M"]
        for m in range(M):
            args = base + cfg_args + user + ["--set_dir", str(d), "--m", str(m)]
            lines.append(" ".join(shlex.quote(a) for a in args))
    return lines


# --- compute count (Table compute) ------------------------------------------------------------

def n_pool(cfg, budget) -> float:
    """n_beta = 100 classes x beta x training pool per class (CIFAR-100: 400; iNat-100: inat_per_class - val - audit)."""
    per = 500 if cfg.dataset == "cifar100" else cfg.inat_per_class
    train_pool = per - round(cfg.val_frac * per) - round(cfg.audit_frac * per)
    return 100 * float(budget) * train_pool


def count_runs(cfg) -> dict:
    """Runs and A100-hours per budget and component, t(n) = gpu_h_per_20k * (n / 20000) for one run on n images.
    Pure arithmetic over the config (no files)."""
    t = lambda n: cfg.gpu_h_per_20k * n / 20000.0  # noqa: E731
    n_cond = sum(cfg.n_subset_random + cfg.n_subset_spectrum if c == "subset" else 1 for c in cfg.studyA_conditions)
    out = {"budgets": {}}
    for b in cfg.privacy_budgets:
        nb = n_pool(cfg, b)
        comp = {"oof": (cfg.K_folds, nb * (cfg.K_folds - 1) / cfg.K_folds), "studyA": (n_cond * cfg.M, nb / 2)}
        if cfg.studyB_budget is not None and abs(float(b) - float(cfg.studyB_budget)) < 1e-9:
            R = round(cfg.eta * nb)
            comp["prelim"] = (cfg.M0, nb / 2)
            comp["baseB"] = (len(STUDYB_CONDITIONS) * cfg.MB, nb / 2)
            comp["del_ce"] = (cfg.Q * len(DEL_TYPES["ce"]) * cfg.MB, (nb - R) / 2)
            comp["del_true_hier"] = (cfg.Q_true_hier * len(DEL_TYPES["true_hier"]) * cfg.MB, (nb - R) / 2)
        rows = {k: {"runs": int(r), "n_train": float(n), "gpu_h": r * t(n)} for k, (r, n) in comp.items()}
        out["budgets"][str(float(b))] = {"components": rows, "runs": sum(v["runs"] for v in rows.values()),
                                          "gpu_h": sum(v["gpu_h"] for v in rows.values())}
    out["total_runs"] = sum(v["runs"] for v in out["budgets"].values())
    out["total_gpu_h"] = sum(v["gpu_h"] for v in out["budgets"].values())
    out["assumption"] = (f"t(n) = {cfg.gpu_h_per_20k} * (n / 20000) A100-hours per {cfg.epochs}-epoch run on n images; "
                         "assumption; replace with pilot compute.json. Feature extraction, PD and scoring not included"
                         + ("; iNat-100 images are 64x64, cost not scaled by resolution" if cfg.dataset == "inat100" else ""))
    return out
