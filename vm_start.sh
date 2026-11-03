#!/usr/bin/env bash
# One-command setup on a fresh VCL VM (run ON the VM). Clones the code from GitHub, builds the environments,
# restores runs from the Hugging Face dataset repo, and starts the 10-minute backup loop.
#   export HF_TOKEN=hf_...          # write-scoped token (or `hf auth login` beforehand)
#   export GH_TOKEN=github_pat_...  # read access to the private repo (or set up a deploy key)
#   END="2026-11-09 09:15" bash vm_start.sh
# Then start work, e.g.:  cd ~/VSCA/vsca && screen -dmS work bash -c "./scripts/run_plan.sh tune > ~/VSCA/work.log 2>&1"
set -euo pipefail
REPO=${REPO:-tysjosh/VSCA}
MARGIN_MIN=${MARGIN_MIN:-25}
[ -n "${HF_TOKEN:-}" ] || [ -f ~/.cache/huggingface/token ] || { echo "set HF_TOKEN first"; exit 1; }

echo "== code"
if [ -d ~/VSCA/.git ]; then git -C ~/VSCA pull -q; else
  git clone -q "https://${GH_TOKEN:+x-access-token:$GH_TOKEN@}github.com/$REPO.git" ~/VSCA; fi
cd ~/VSCA && chmod +x *.sh */scripts/*.sh

echo "== environments"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
./setup_env.sh > /tmp/setup_env.log 2>&1 && tail -1 /tmp/setup_env.log
bash vsca/scripts/setup_vllm_env.sh > /tmp/setup_vllm.log 2>&1 && tail -2 /tmp/setup_vllm.log

echo "== restore runs"
.venv/bin/python hub_sync.py pull

echo "== data and deadline"
(cd vsca && ../.venv/bin/python -m vsca.cli prepare --config configs/base.yaml > /tmp/prepare.log 2>&1)
if [ -n "${END:-}" ]; then END_TS=$(date -d "$END" +%s); else END_TS=$(( $(date +%s) + 36000 )); fi
echo $(( END_TS - MARGIN_MIN * 60 )) > ~/VSCA/deadline
echo "runs stop cleanly at $(date -d @"$(cat ~/VSCA/deadline)")"

echo "== backup loop"
screen -S hubsync -X quit 2>/dev/null || true
screen -dmS hubsync bash -c "cd ~/VSCA && .venv/bin/python hub_sync.py loop > ~/VSCA/hubsync.log 2>&1"
sleep 3; screen -ls | grep -q hubsync && echo "backup loop running (~/VSCA/hubsync.log)"
