"""Checks of the verifier and of the objective identities stated in the paper.

Run: python scripts/sanity_check.py
"""

from fractions import Fraction

import torch

from vsca.certify import build_extras
from vsca.config import Config
from vsca.objectives import set_nll, token_nll
from vsca.verify import claim_valid, final_answer, parse_claims, response_valid

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# --- verifier -------------------------------------------------------------
ref = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nTotal <<48+24=72>>72.\n#### 72"
check("reference verifies", response_valid(ref, Fraction(72)))
check("wrong final answer fails", not response_valid(ref.replace("#### 72", "#### 73"), Fraction(72)))
check("invalid annotation fails", not response_valid(ref.replace("48+24=72", "48+24=73"), Fraction(72)))
check("rounded result accepted", claim_valid("10/3", "3.33") is True)
check("rounding too coarse rejected", claim_valid("10/3", "3.4") is False)
check("commas and $ parsed", final_answer("#### $1,250") == Fraction(1250))
check("undetermined expression -> None", claim_valid("x+y", "3") is None)
cl = parse_claims(ref)
check("claim spans", ref[cl[0].start : cl[0].end] == "<<48/2=24>>" and ref[cl[0].result_start : cl[0].result_end] == "24")

# --- objective identities -------------------------------------------------
torch.manual_seed(0)
N, V = 6, 11
z = torch.randn(N, V, dtype=torch.float64, requires_grad=True)
y = torch.randint(0, V, (N,))
none = torch.full((N, 2), -1)
check("Prop 3a: no extras == cross-entropy", torch.allclose(set_nll(z, y, none), token_nll(z, y)))

extra = none.clone()
extra[0, 0] = (y[0] + 1) % V
extra[0, 1] = (y[0] + 2) % V
loss = set_nll(z, y, extra)
loss[0].backward()
g = z.grad[0]
pi = torch.softmax(z[0].detach(), -1)
A = torch.zeros(V, dtype=torch.bool)
A[torch.stack([y[0], extra[0, 0], extra[0, 1]])] = True
expected = pi - A.double() * pi / pi[A].sum()
check("Prop 1: gradient formula", torch.allclose(g, expected, atol=1e-10))
check("Prop 1: L1 norm = 2(1 - pi(A))", torch.isclose(g.abs().sum(), 2 * (1 - pi[A].sum())))
check("Prop 2: loss = -log pi(A)", torch.isclose(loss[0], -torch.log(pi[A].sum())))
check("Prop 4: CE >= set loss", bool((set_nll(z, y, extra) <= token_nll(z, y) + 1e-12).all()))

# --- variants -------------------------------------------------------------
recs = [
    {"id": "a", "positions": [
        {"t": 1, "type": "claim", "entropy": 0.1, "cands": [5, 6], "cand_probs": [0.2, 0.1], "certified": [6], "cert_text": {}},
        {"t": 3, "type": "decision", "entropy": 2.0, "cands": [7, 8], "cand_probs": [0.3, 0.2], "certified": [], "cert_text": {}},
    ]},
]
cfg = Config()
ex, _ = build_extras(recs, cfg, 0)
check("full variant keeps certified set", ex == {"a": {1: [6]}})
cfg.variant = "uncertified"
ex, _ = build_extras(recs, cfg, 0)
check("uncertified takes same count by probability", ex == {"a": {1: [5]}})
cfg.variant = "decision_only"
ex, _ = build_extras(recs, cfg, 0)
check("decision_only drops claim positions", ex == {})
cfg.variant, cfg.inject_eta = "full", 1.0
ex, st = build_extras(recs, cfg, 0)
check("injection adds an uncertified candidate", ex["a"][3] == [7] and st["injected"] == 2)

raise SystemExit(0 if ok else 1)
