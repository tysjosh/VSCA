# Source this once when a machine session starts:   source scripts/session_start.sh
# It fixes the stop time for every command in this shell (sg, privred and vsca all read SESSION_DEADLINE).
SESSION_HOURS=${SESSION_HOURS:-10}
MARGIN_MIN=${MARGIN_MIN:-20}
export SESSION_DEADLINE=$(( $(date +%s) + SESSION_HOURS * 3600 - MARGIN_MIN * 60 ))
export VSCA_DEADLINE=$SESSION_DEADLINE
echo "[session] runs stop cleanly at $(date -d @"$SESSION_DEADLINE" 2>/dev/null || date -r "$SESSION_DEADLINE")"
