"""Back up run outputs to a private Hugging Face dataset repo, and restore them on a fresh VM.

Runs directly on the VM (no relay through another machine). Needs HF_TOKEN (write scope) in the environment
or a prior `hf auth login`.

  python hub_sync.py push  [--runs vsca/runs] [--repo olukotunjosh/vsca-runs]          # one upload
  python hub_sync.py loop  [--every 600]                                               # upload every N s
  python hub_sync.py pull  [--full]                                                    # restore onto this VM

push uploads only new or changed files (content-addressed, so unchanged files are not re-sent) and never
deletes anything in the repo. Each push is a commit, so earlier states can be recovered.
pull skips adapters of tuning runs and every adapter_last/ by default; --full restores everything.
Training checkpoints (ckpt.pt) exist only for unfinished runs and are always kept, so resume works.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORE = ["**/*.tmp", "**/*.snapshot/**", "**/__pycache__/**"]
LIGHT_SKIP = ["**/adapter_last/**", "tune_*/adapter_best/**"]


def api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo(a, repo: str) -> None:
    a.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)


def push(runs: Path, repo: str) -> None:
    a = api()
    ensure_repo(a, repo)
    t0 = time.time()
    info = a.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(runs), path_in_repo="runs",
                           ignore_patterns=IGNORE,
                           commit_message=f"sync {datetime.now():%Y-%m-%d %H:%M} from {os.uname().nodename}")
    logs = [p for p in (ROOT.glob("*.log")) if p.is_file()]
    for p in logs:
        a.upload_file(path_or_fileobj=str(p), path_in_repo=f"logs/{p.name}", repo_id=repo, repo_type="dataset",
                      commit_message=f"log {p.name}")
    print(f"[hub] {datetime.now():%H:%M:%S} pushed {runs} -> {repo} in {time.time() - t0:.0f}s "
          f"({getattr(info, 'oid', '')[:8]})", flush=True)


def pull(runs: Path, repo: str, full: bool) -> None:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    tmp = snapshot_download(repo_id=repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                            local_dir=str(runs.parent / ".hub_pull"), allow_patterns=["runs/**"],
                            ignore_patterns=IGNORE + ([] if full else [f"runs/{p}" for p in LIGHT_SKIP]))
    src = Path(tmp) / "runs"
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            dst = runs / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != f.stat().st_size:
                os.replace(f, dst)
                n += 1
    print(f"[hub] restored {n} files from {repo} into {runs} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "loop", "pull"])
    p.add_argument("--runs", default="vsca/runs")
    p.add_argument("--repo", default=os.environ.get("HUB_REPO", "olukotunjosh/vsca-runs"))
    p.add_argument("--every", type=int, default=600)
    p.add_argument("--full", action="store_true")
    a = p.parse_args()
    runs = (ROOT / a.runs).resolve()
    if a.cmd == "pull":
        runs.mkdir(parents=True, exist_ok=True)
        pull(runs, a.repo, a.full)
        return
    if not runs.exists():
        sys.exit(f"{runs} does not exist")
    while True:
        try:
            push(runs, a.repo)
        except Exception as e:  # keep looping through transient network errors
            print(f"[hub] {datetime.now():%H:%M:%S} push failed: {type(e).__name__}: {e}", flush=True)
        if a.cmd == "push":
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
