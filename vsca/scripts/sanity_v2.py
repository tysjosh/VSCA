"""Checks of the V+ step verifier (VSCA v2, Design Section 4).

Run: python scripts/sanity_v2.py
"""

from fractions import Fraction

from vsca.config import Config
from vsca.verify_plus import check_response, expr_operands, parse_whitelist, step_boundaries, vplus_from_cfg

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


WL, _ = vplus_from_cfg(Config())

# --- operands -------------------------------------------------------------
check("10-3 operands are {10, 3}", set(expr_operands("10-3")) == {Fraction(10), Fraction(3)})
check("x and commas normalised", expr_operands("1,000x2") == [Fraction(1000), Fraction(2)])
check("unparsable expression -> None", expr_operands("a+b") is None)

# --- grounding ------------------------------------------------------------
q = "Tom has 15 apples and buys three more. How many apples does he have?"
r = check_response(q, "He has <<15+3=18>>18 apples.\n#### 18", Fraction(18), WL)
check("problem numeral and number word accepted", r.verified and r.steps[0].grounded)

q2 = "A box holds 5 pens. Ann buys 4 boxes and gives away 3 pens. How many pens are left?"
ref2 = "She buys <<5*4=20>>20 pens.\nShe keeps <<20-3=17>>17 pens.\n#### 17"
r = check_response(q2, ref2, Fraction(17), WL)
check("earlier-step result accepted", r.verified and all(s.grounded for s in r.steps))

q3 = "Sam works 3 hours a day. How many minutes does he work in a day?"
r = check_response(q3, "He works <<3*60=180>>180 minutes.\n#### 180", Fraction(180), WL)
check("whitelist constant accepted", r.verified)

inv = "She buys 5*9973 = <<5*9973=49865>>49865 pens.\n#### 49865"
r = check_response(q2, inv, Fraction(49865), WL)
s = r.steps[0]
check(
    "invented operand rejected with error_char at its first mention",
    not r.verified and not s.grounded and s.kind == "grounding" and s.ungrounded == ["9973"]
    and s.error_char == inv.index("9973"),
)
inv2 = "She buys <<5*9973=49865>>49865 pens.\n#### 49865"
s = check_response(q2, inv2, Fraction(49865), WL).steps[0]
check("invented operand only in annotation -> error_char inside expr", s.error_char == inv2.index("9973"))

check("custom whitelist changes the outcome",
      not check_response(q3, "He works <<3*60=180>>180 minutes.\n#### 180", Fraction(180), parse_whitelist("24,7")).verified)

# --- arithmetic and lucky guesses -------------------------------------------
bad = "She buys <<5*4=21>>20 pens.\nShe keeps <<20-3=17>>17 pens.\n#### 17"
r = check_response(q2, bad, Fraction(17), WL)
check(
    "correct answer + invalid annotation not verified",
    r.answer_ok and not r.verified and r.first_failed == 0 and r.steps[0].kind == "arithmetic"
    and r.steps[0].error_char == r.steps[0].claim.result_start,
)
lucky = "She buys <<5*4=20>>20 pens.\nShe keeps <<20-11=9>>9 pens, plus 8 makes <<9+8=17>>17.\n#### 17"
r = check_response(q2, lucky, Fraction(17), WL)
check("correct answer + ungrounded operand not verified", r.answer_ok and not r.verified and r.first_failed == 1)
check("... but verified with grounding=False", check_response(q2, lucky, Fraction(17), WL, grounding=False).verified)
check("wrong final answer not verified", not check_response(q2, ref2.replace("#### 17", "#### 18"), Fraction(17), WL).verified)

# --- GSM8K reference (sanity_check.py) ---------------------------------------
natalia_q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
             "How many clips did Natalia sell altogether in April and May?")
natalia = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nTotal <<48+24=72>>72.\n#### 72"
r = check_response(natalia_q, natalia, Fraction(72), WL)
check("Natalia reference verified", r.verified)
check("step boundaries = line starts + final part",
      step_boundaries(r) == [0, natalia.index("Total"), natalia.index("####")])

# --- on-policy pipeline with a scripted stand-in backend (tokenizer only, no model weights) ---------------
import json
import shutil
import tempfile
from pathlib import Path

from vsca import onpolicy as OP
from vsca.config import load_config
from vsca.data import collate
from vsca.modeling import load_tokenizer

CFG = load_config(["configs/smoke.yaml"], ["mc_M=4", "onpolicy_k=0"])
tok = load_tokenizer(CFG.model_name)
TMP = Path(tempfile.mkdtemp(prefix="sanity_v2_"))


class Scripted:
    """Backend stand-in: decodes each prefix and returns the tokens of ``fn(question, response_so_far, r)``."""

    name = "hf"

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def set_snapshot(self, d):
        pass

    def generate(self, prefixes, max_new_tokens, do_sample, temperature=1.0, top_p=1.0, n=1, seeds=None, batch=16):
        self.calls += 1
        out = []
        for p in prefixes:
            q, _, resp = tok.decode(p, skip_special_tokens=True).partition("\nAnswer:\n")
            for r in range(n):
                out.append(tok(self.fn(q[len("Question: "):], resp, r), add_special_tokens=False).input_ids)
        return out


def no_calls(q, resp, r):
    raise AssertionError("no rollouts expected")


def run_localise(name, recs, samples, fn, **over):
    cfg = load_config(["configs/smoke.yaml"], ["mc_M=4", "onpolicy_k=0"] + [f"{k}={v}" for k, v in over.items()])
    d = TMP / name
    smp = [{"id": r["id"], "samples": s} for r, s in zip(recs, samples)]
    d.mkdir(parents=True)
    (d / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in smp))
    ver = OP.verify_samples(recs, smp, cfg, d / "verify.jsonl")
    return OP.localise(None, tok, recs, smp, ver, cfg, d / "localise.jsonl", backend=Scripted(fn))


def scripted(good, marker, gold):
    """Rollouts complete ``good`` (verified) unless the prefix contains ``marker``, then answer wrongly."""
    def fn(q, resp, r):
        sep = "" if not resp or resp.endswith("\n") else "\n"
        if marker in resp:
            return sep + "#### 1"
        return good[len(resp):] if good.startswith(resp) else sep + f"#### {gold}"
    return fn


def fork_ok(rec, text, f, char, first_char):
    ex = OP._sample_example(tok, rec, text, CFG)
    return (f["error_char"] == char and f["e"] == ex.resp_ids[f["t"]] and ex.resp_offsets[f["t"]][0] == char
            and tok.decode([f["e"]]) == first_char)


rec2 = {"id": "p2", "question": q2, "answer": ref2, "gold": "17"}
slip = "She buys 5*4 = <<5*4=21>>21 pens.\nShe keeps <<21-3=18>>18 pens.\n#### 18"
ungr = "She buys 5*9973 = <<5*9973=49865>>49865 pens.\n#### 49865"
g_then_a = "She buys <<5*4+0=20>>20 pens.\nShe keeps <<20-3=18>>18 pens.\n#### 18"  # 0: invented, harmless
g_only_ok = "She buys <<5*4+0=20>>20 pens.\nShe keeps <<20-3=17>>17 pens.\n#### 17"
loc = run_localise("direct", [rec2], [[ref2, slip, g_then_a, g_only_ok, bad]], no_calls)[0]  # no MC rollouts
fs = {f["sample"]: f for f in loc["forks"]}
check("localise: wrong-answer arithmetic slip -> rule arithmetic, step 0, t/e at the wrong result",
      fs[1]["rule"] == "arithmetic" and fs[1]["kind"] == "arithmetic" and fs[1]["step"] == 0
      and fs[1]["mc_values"] is None and fork_ok(rec2, slip, fs[1], slip.index("21>>"), "2"))
check("localise: earliest ARITHMETIC failure wins over an earlier grounding failure",
      fs[2]["rule"] == "arithmetic" and fs[2]["step"] == 1 and fork_ok(rec2, g_then_a, fs[2], g_then_a.index("18>>"), "1"))
check("correct answer + grounding-only failure: unverified, never a fork",
      3 not in fs and loc["n_grounding_only_correct"] == 1 and loc["n_failed"] == 3)
check("correct answer + arithmetic failure: rule lucky_arithmetic", fs[4]["rule"] == "lucky_arithmetic"
      and fs[4]["step"] == 0)
rft = OP.build_training(tok, [rec2], TMP / "direct", load_config(["configs/smoke.yaml"], ["method=rft_verified"]))
check("grounding-only correct sample excluded from rft_verified positives", [e.id for e in rft[0]] == ["p2-s0"])

loc = run_localise("ground_mc", [rec2], [[ungr]], scripted(ref2, "9973", 17))[0]
f = loc["forks"][0] if loc["forks"] else {}
check("wrong answer, grounding failure confirmed by MC drop -> rule grounding_mc at its first mention",
      f.get("rule") == "grounding_mc" and f.get("kind") == "grounding" and f.get("step") == 0
      and f.get("mc_values") == [1.0, 0.0] and fork_ok(rec2, ungr, f, ungr.index("9973"), "9"))
nc = "She buys <<5+4=9>>9 pens.\nShe keeps <<9-3+0=6>>6 pens.\n#### 6"  # planning error at 0, harmless 0 at 1
loc = run_localise("ground_nomc", [rec2], [[nc]], scripted(ref2, "5+4", 17))[0]
f = loc["forks"][0] if loc["forks"] else {}
check("wrong answer, grounding failure without MC drop -> falls back to MC location (rule mc, step 0)",
      f.get("rule") == "mc" and f.get("kind") == "planning" and f.get("step") == 0
      and f.get("mc_values") == [1.0, 0.0, 0.0] and f.get("error_char") == 0)

qp = "Ann has 3 apples. She buys 4 more apples, then doubles her apples. How many apples does she have?"
recp = {"id": "pp", "question": qp, "answer": "", "gold": "14"}
plan = "3+4=<<3+4=7>>7\n7*3=<<7*3=21>>21\n#### 21"
good = "3+4=<<3+4=7>>7\n7*2=<<7*2=14>>14\n#### 14"


def rest(resp):  # a verified completion of ``resp`` when one exists
    if good.startswith(resp):
        return good[len(resp):]
    return ("" if resp.endswith("\n") else "\n") + "#### 14"


def planning_fn(q, resp, r):
    return "\n#### 21" if "7*3" in resp else rest(resp)


loc = run_localise("planning", [recp], [[plan]], planning_fn)[0]
f = loc["forks"][0] if loc["forks"] else {}
check("localise: MC planning error -> rule mc, step 1 (values 1, 1, 0)",
      f.get("rule") == "mc" and f.get("kind") == "planning" and f.get("step") == 1 and f.get("mc_values") == [1.0, 1.0, 0.0]
      and fork_ok(recp, plan, f, plan.index("7*3"), "7"))
loc = run_localise("uncertain", [recp], [[plan]], lambda q, resp, r: rest(resp) if r == 0 else "\n#### 3")[0]
check("localise: largest value drop 0.25 < mc_min_drop -> uncertain, no fork",
      loc["forks"] == [] and loc["n_uncertain"] == 1 and loc["n_failed"] == 1)

# hand-made pipeline dir: synthetic samples only (onpolicy_k=0, debug_seed_samples), hand-written certificates
natalia_rec = {"id": "p0", "question": natalia_q, "answer": natalia, "gold": "72"}
RECS = [natalia_rec, rec2]
cfg0 = load_config(["configs/smoke.yaml"], ["onpolicy_k=0", "debug_seed_samples=true"])
D = TMP / "op"
st = OP.prepare(None, tok, RECS, cfg0, D, backend=Scripted(no_calls), need_certify=False)
smp = OP.load_cert(D / "samples.jsonl")
check("debug_seed_samples: reference, result+1 slip, invented operand in line and annotation",
      smp[0]["samples"][0] == natalia and "<<48/2=25>>24" in smp[0]["samples"][1]
      and smp[0]["samples"][2].startswith("Natalia sold 9973/2 = <<9973/2=24>>24"))
loc = OP.load_cert(D / "localise.jsonl")
check("seeded samples localised as arithmetic and grounding (both lucky_arithmetic: invented operand breaks the sum)",
      [[f["kind"] for f in r["forks"]] for r in loc] == [["arithmetic", "grounding"]] * 2 and st["verified"] == 2
      and st["forks_by_rule"] == {"lucky_arithmetic": 4})
with open(D / "forks.jsonl", "w") as fh:
    for r in loc:
        fks = []
        for n, fk in enumerate(r["forks"]):
            cands = [fk["e"] + 1, fk["e"] + 2]
            fks.append(dict(fk, cands=cands, cand_probs=[0.2, 0.1], certified=[cands[1]] if n == 0 else [],
                            cert_text={}))
        fh.write(json.dumps({"id": r["id"], "forks": fks}) + "\n")
(D / "forks.jsonl.done").write_text("")

arms = {name: load_config(["configs/smoke.yaml"], ["onpolicy_k=0", "debug_seed_samples=true"] + s) for name, s in {
    "vsca2": ["method=vsca2"], "vsca2_uncertified": ["method=vsca2", "variant=uncertified"],
    "stepdpo_fork": ["method=stepdpo_fork"], "rft_verified": ["method=rft_verified"]}.items()}
built = {name: OP.build_training(tok, RECS, D, c) for name, c in arms.items()}
check("all arms: same samples_sha256", len({b[3]["samples_sha256"] for b in built.values()}) == 1)
ver_set = lambda b: sorted((e.id, tuple(e.resp_ids)) for e in b[0] if not e.id.endswith("-f"))
check("rft_verified's verified set == vsca2's (2 references)",
      ver_set(built["rft_verified"]) == ver_set(built["vsca2"]) and len(ver_set(built["vsca2"])) == 2)
check("rft_verified: no forks", not built["rft_verified"][2] and not built["rft_verified"][1])
fk = {(r["id"], fk["sample"]): fk for r in OP.load_cert(D / "forks.jsonl") for fk in r["forks"]}
exp_full = {f"{i}-s{j}-f": {fk_["t"]: fk_["certified"]} for (i, j), fk_ in fk.items() if fk_["certified"]}
exp_unc = {f"{i}-s{j}-f": {fk_["t"]: fk_["cands"][:len(fk_["certified"])]} for (i, j), fk_ in fk.items()
           if fk_["certified"]}
check("vsca2 full: extras = certified sets", built["vsca2"][1] == exp_full and len(exp_full) == 2)
check("vsca2 uncertified: extras = cands[:|certified|]", built["vsca2_uncertified"][1] == exp_unc
      and exp_unc != exp_full)
check("stepdpo_fork: no extras, same bad tokens as vsca2",
      built["stepdpo_fork"][1] == {} and built["stepdpo_fork"][2] == built["vsca2"][2] and len(built["vsca2"][2]) == 4)
forks_ex = [e for e in built["vsca2"][0] if e.id.endswith("-f")]
b = collate(forks_ex, tok.pad_token_id, extras=built["vsca2"][1], bad=built["vsca2"][2])
lab_ok = True
for i, e in enumerate(forks_ex):
    P, t = len(e.prompt_ids), e.supervise_from
    lab_ok &= (b["labels"][i] != -100).nonzero().flatten().tolist() == [P + t] and len(e.resp_ids) == t + 1
    lab_ok &= int(b["bad_ids"][i, P + t]) == built["vsca2"][2][e.id][t] and int((b["bad_ids"][i] >= 0).sum()) == 1
check("fork examples: one supervised position t, bad_ids[t] = e", lab_ok)
cfg_cp = load_config(["configs/smoke.yaml"], ["onpolicy_k=0", "debug_seed_samples=true", "vplus_grounding=false",
                                              f"onpolicy_samples={D / 'samples.jsonl'}"])
st_cp = OP.prepare(None, tok, RECS, cfg_cp, TMP / "op_nogr", backend=Scripted(no_calls), need_certify=False)
check("onpolicy_samples copy: identical samples file",
      (TMP / "op_nogr" / "samples.jsonl").read_bytes() == (D / "samples.jsonl").read_bytes()
      and st_cp["samples_sha256"] == st["samples_sha256"])
try:
    OP.check_dir(D, load_config(["configs/smoke.yaml"], ["seed=1", "method=stepdpo_fork"]))
    check("check_dir refuses a different seed", False)
except SystemExit:
    check("check_dir refuses a different seed", True)

shutil.rmtree(TMP)

# --- grounding false rejections on correct GSM8K references (decision: trade-off point E, README) ----------
from vsca.diagnostics import grounding_coverage

ref = grounding_coverage("../labeling/error_taxonomy_sft.csv", Config())["reference"]
check(f"grounding false rejection on the 200 SFT-failure references <= 8.5% "
      f"({ref['grounding_false_rejection_rate']:.3f})", ref["n"] == 200 and ref["grounding_false_rejection_rate"] <= 0.085)

raise SystemExit(0 if ok else 1)
