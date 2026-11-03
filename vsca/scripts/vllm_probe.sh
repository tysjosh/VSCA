#!/usr/bin/env bash
# GPU check of the vLLM backend against the HF backend on the same N training examples (base model, epoch 0).
# Reports seconds per example, agreement of greedy local certificates, and certification rates. vLLM samples
# with different random streams, so decision certificates are compared as rates, not record by record.
#   bash scripts/vllm_probe.sh            # N=64
set -uo pipefail
cd "$(dirname "$0")/.."
N=${N:-64}
HF_PY=${HF_PY:-../.venv/bin/python}
VLLM_PY=${VLLM_PY:-../.venv-vllm/bin/python}
O=runs/vllm_probe
mkdir -p "$O"
C="--config configs/base.yaml --set limit_train=$N --set resume=false"

t0=$(date +%s)
[ -s "$O/hf.jsonl" ] || "$HF_PY" -m vsca.cli certify $C --set rollout_backend=hf --out "$O/hf.jsonl" > "$O/hf.log" 2>&1
t1=$(date +%s)
"$VLLM_PY" -m vsca.cli certify $C --set rollout_backend=vllm --set cert_block=256 --out "$O/vllm.jsonl" > "$O/vllm.log" 2>&1 \
  || { echo "vLLM certification failed; see $O/vllm.log"; tail -n 30 "$O/vllm.log"; exit 1; }
t2=$(date +%s)

"$HF_PY" - "$O" "$N" "$((t1 - t0))" "$((t2 - t1))" <<'EOF'
import json, sys
o, n, t_hf, t_vl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
load = lambda f: {r["id"]: r for r in map(json.loads, open(f"{o}/{f}"))}
hf, vl = load("hf.jsonl"), load("vllm.jsonl")
assert set(hf) == set(vl), "different examples"
def rates(recs, typ):
    ps = [p for r in recs.values() for p in r["positions"] if p["type"] == typ]
    return len(ps), sum(bool(p["certified"]) for p in ps) / max(len(ps), 1)
agree = tot = 0
for i in hf:
    a = {p["t"]: p["certified"] for p in hf[i]["positions"] if p["type"] == "claim"}
    b = {p["t"]: p["certified"] for p in vl[i]["positions"] if p["type"] == "claim"}
    for t in a:
        tot += 1
        agree += a[t] == b.get(t)
same_dec = sum([p["t"] for p in hf[i]["positions"] if p["type"] == "decision"] ==
               [p["t"] for p in vl[i]["positions"] if p["type"] == "decision"] for i in hf)
res = {
    "examples": n,
    "hf_seconds": t_hf if t_hf > 1 else None, "vllm_seconds_incl_engine_start": t_vl,
    "hf_s_per_example": round(t_hf / n, 2) if t_hf > 1 else None, "vllm_s_per_example": round(t_vl / n, 2),
    "claim_positions": tot, "claim_certificate_agreement": round(agree / max(tot, 1), 4),
    "examples_with_same_decision_positions": f"{same_dec}/{n}",
    "hf": {"claim": rates(hf, "claim"), "decision": rates(hf, "decision"),
           "verified_rollouts": sum(len(r["rollouts"]) for r in hf.values())},
    "vllm": {"claim": rates(vl, "claim"), "decision": rates(vl, "decision"),
             "verified_rollouts": sum(len(r["rollouts"]) for r in vl.values())},
}
json.dump(res, open(f"{o}/compare.json", "w"), indent=2)
print(json.dumps(res, indent=2))
EOF
