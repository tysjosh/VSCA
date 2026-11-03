#!/usr/bin/env bash
# Run on this Mac at the start of each VCL reservation (the VM is fresh every time):
#   VM=ojolukot@<ip> ./start_reservation.sh            # code, environments, restore from Hazel, start backup
#   END="2026-11-09 09:15" VM=... ./start_reservation.sh   # reservation end time (default: now + 10 h)
# It then prints the command to start work on the VM. Needs the Mac's open Hazel connection (`ssh hazel`).
set -euo pipefail
cd "$(dirname "$0")"
VM=${VM:?set VM=ojolukot@<ip>}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
MARGIN_MIN=${MARGIN_MIN:-25}

echo "== 1. check access"
$SSH "$VM" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h ~ | tail -1'
$SSH hazel true || { echo "Hazel connection is closed: run 'ssh hazel' once (Duo), then rerun"; exit 1; }

echo "== 2. copy code"
$SSH "$VM" 'mkdir -p ~/VSCA'
rsync -az --exclude .venv --exclude ".venv-*" --exclude runs --exclude runs_smoke --exclude data \
  --exclude __pycache__ --exclude "*.egg-info" -e "$SSH" setup_env.sh vsca "$VM:VSCA/"
$SSH "$VM" 'chmod +x ~/VSCA/setup_env.sh ~/VSCA/vsca/scripts/*.sh'

echo "== 3. environments (shared .venv; .venv-vllm for vLLM)"
$SSH "$VM" 'set -e; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=$HOME/.local/bin:$PATH; cd ~/VSCA && ./setup_env.sh >/tmp/setup_env.log 2>&1 && tail -1 /tmp/setup_env.log
  bash vsca/scripts/setup_vllm_env.sh >/tmp/setup_vllm.log 2>&1 && tail -2 /tmp/setup_vllm.log || { echo "vLLM setup failed: /tmp/setup_vllm.log"; tail -5 /tmp/setup_vllm.log; }'

echo "== 4. restore runs from Hazel"
RESTORE=1 VM="$VM" bash sync_remote.sh

echo "== 5. data and session deadline"
if [ -n "${END:-}" ]; then END_TS=$(date -j -f "%Y-%m-%d %H:%M" "$END" +%s); else END_TS=$(( $(date +%s) + 36000 )); fi
DL=$(( END_TS - MARGIN_MIN * 60 ))
$SSH "$VM" "echo $DL > ~/VSCA/deadline; cd ~/VSCA/vsca && ../.venv/bin/python -m vsca.cli prepare --config configs/base.yaml >/tmp/prepare.log 2>&1 && tail -1 /tmp/prepare.log"
echo "runs stop cleanly at $(date -r "$DL")"

echo "== 6. start the backup loop (every 10 min)"
pkill -f "bash sync_remote.sh" 2>/dev/null || true
VM="$VM" nohup bash sync_remote.sh > sync_remote.log 2>&1 &
sleep 5
pgrep -f "bash sync_remote.sh" >/dev/null && echo "backup running (log: $(pwd)/sync_remote.log)" || { echo "backup did NOT start"; exit 1; }

cat <<EOF
== ready. Start work on the VM inside screen, e.g.:
  $SSH $VM 'cd ~/VSCA/vsca && SESSION_DEADLINE=\$(cat ~/VSCA/deadline) VSCA_DEADLINE=\$(cat ~/VSCA/deadline) screen -dmS work bash -c "./scripts/run_plan.sh <stage> > ~/VSCA/work.log 2>&1"'
EOF
