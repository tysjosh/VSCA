"""``privred member --set_dir D --m m``: train model m of a set with sg.train.train (companion recipe), compute PD on
the manifest's pd_sets, then delete model.pt unless keep_models. Idempotent: finished work is skipped."""

from __future__ import annotations

import json
from pathlib import Path

from .config import to_sg
from .io import member_dir


def member_sg_config(cfg, set_dir, m):
    sdir = Path(set_dir).resolve()
    man = json.loads((sdir / "manifest.json").read_text())
    M = int(man["M"])
    if not 0 <= m < M:
        raise SystemExit(f"member: m={m} outside 0..{M - 1} for set {man['name']}")
    ex = sdir / f"example_sets_m{m:02d}.json"
    if not ex.exists():
        ex = sdir / "example_sets.json"
    over = {"out_dir": str(sdir), "run_name": f"m{m:02d}", "train_indices": str(sdir / f"train_m{m:02d}.npy"),
            "example_sets": str(ex), "example_logits": bool(man.get("example_logits", False)),
            "dataset": man["dataset"], "budget": float(man["budget"]), "condition": man["condition"],
            "partition": man.get("partition"), "weight_decay": float(man["wd"]), "seed": int(man["seed_base"]) + m,
            "save_model": bool(man["pd_sets"]) or bool(cfg.keep_models), "skip_if_done": True}
    if man.get("rho") is not None:
        over["rho"] = float(man["rho"])
    return to_sg(cfg, **over), man, ex


def run_member(cfg, set_dir, m) -> dict:
    from sg.train import train

    from .depth import run_pd

    scfg, man, ex = member_sg_config(cfg, set_dir, m)
    run_dir = member_dir(Path(set_dir).resolve(), m)
    pd_files = {s: run_dir / f"pd_{s}.npz" for s in man["pd_sets"]}
    summary = train(scfg)
    todo = [s for s, f in pd_files.items() if not f.exists()]
    if todo:
        if not (run_dir / "model.pt").exists():
            raise SystemExit(f"member: {run_dir} is trained but model.pt is gone and {todo} PD files are missing; "
                             "delete the run directory and rerun")
        sets = json.loads(Path(ex).read_text())
        run_pd(scfg, run_dir, {s: sets[s] for s in todo})
    if (run_dir / "model.pt").exists() and not cfg.keep_models:
        (run_dir / "model.pt").unlink()
    return {"set": man["name"], "m": m, "run_dir": str(run_dir), "test_acc": summary.get("test_acc"),
            "n_train": summary.get("n_train"), "pd_sets": man["pd_sets"]}
