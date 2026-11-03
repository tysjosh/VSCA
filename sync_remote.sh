#!/usr/bin/env bash
# Back up remote run outputs to Hazel scratch, because the VCL VM is reset between reservations.
# Hazel cannot reach the VM and needs Duo for new logins, so this Mac relays: it compares file lists on both
# sides and streams only new or changed files VM -> Mac -> Hazel (nothing is stored on the Mac).
# It uses the Mac's existing authenticated Hazel connection (ControlMaster "hazel" in ~/.ssh/config).
#   ./sync_remote.sh                 # every INTERVAL seconds
#   ONCE=1 ./sync_remote.sh          # one pass
#   RESTORE=1 VM=ojolukot@<new-ip> ./sync_remote.sh   # new reservation: copy Hazel's backup onto the new VM
# Files are never deleted on Hazel (a fresh, empty VM must not wipe the backup).
set -uo pipefail
VM=${VM:-ojolukot@152.7.177.74}
HZ=${HZ:-hazel}
INTERVAL=${INTERVAL:-600}
VM_DIR=${VM_DIR:-VSCA/vsca/runs}
HZ_DIR=${HZ_DIR:-/share/nesylearning/ojolukot/vsca_runs}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"

listing() {  # listing <host> <dir>: "path<TAB>size<TAB>mtime" for every finished file
  $SSH "$1" "mkdir -p '$2' && cd '$2' && find . -type f ! -name '*.tmp' -printf '%P\t%s\t%T@\n' | awk -F'\t' '{printf \"%s\t%s\t%d\n\", \$1, \$2, \$3}' | sort"
}

copy_changed() {  # copy_changed <src host> <src dir> <dst host> <dst dir>
  local src=$1 sdir=$2 dst=$3 ddir=$4 a b todo n
  a=$(mktemp) b=$(mktemp) todo=$(mktemp)
  listing "$src" "$sdir" > "$a" || { echo "[sync] cannot list $src:$sdir"; rm -f "$a" "$b" "$todo"; return 1; }
  listing "$dst" "$ddir" > "$b" || { echo "[sync] cannot list $dst:$ddir (is the Hazel connection open? run: ssh hazel)"; rm -f "$a" "$b" "$todo"; return 1; }
  comm -23 "$a" "$b" | cut -f1 > "$todo"
  n=$(wc -l < "$todo" | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    # tar keeps mtimes, so the next comparison sees these files as unchanged.
    $SSH "$src" "cd '$sdir' && tar cf - -T -" < "$todo" | $SSH "$dst" "cd '$ddir' && tar xpf -"
    local rc=$?
    [ $rc -ne 0 ] && { echo "[sync] transfer failed (exit $rc)"; rm -f "$a" "$b" "$todo"; return 1; }
  fi
  echo "[sync] $(date '+%H:%M:%S') $src -> $dst: $n files copied, $(wc -l < "$a" | tr -d ' ') files on source"
  rm -f "$a" "$b" "$todo"
}

if [ -n "${RESTORE:-}" ]; then
  copy_changed "$HZ" "$HZ_DIR" "$VM" "$VM_DIR"
  exit $?
fi
while true; do
  copy_changed "$VM" "$VM_DIR" "$HZ" "$HZ_DIR"
  $SSH "$VM" "cd VSCA && tar cf - *.log 2>/dev/null" | $SSH "$HZ" "mkdir -p '$HZ_DIR/../vsca_logs' && cd '$HZ_DIR/../vsca_logs' && tar xpf -" 2>/dev/null
  [ -n "${ONCE:-}" ] && break
  sleep "$INTERVAL"
done
