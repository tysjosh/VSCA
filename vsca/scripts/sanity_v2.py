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

raise SystemExit(0 if ok else 1)
