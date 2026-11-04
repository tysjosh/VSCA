"""V+ step verifier for VSCA v2 (Design Section 4).

A step is a calculator annotation ``<<expr=result>>``. V+ accepts a step if its
arithmetic verifies (``Claim.valid is True``, as in V_c) and, with grounding on,
every numeric operand of ``expr`` is a number from the problem text, the result
of an earlier step, or a whitelisted constant. A response is V+-verified if its
``####`` answer equals the gold answer and every step is accepted, so a correct
answer reached through an invalid or invented step (a lucky guess) is rejected.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Optional, Union

from .config import Config
from .verify import NUM_RE, Claim, final_answer, parse_claims, parse_number, truncate_response

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_TENS = {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_OTHER = {
    "hundred": Fraction(100), "thousand": Fraction(1000), "million": Fraction(10**6), "dozen": Fraction(12),
    "half": Fraction(1, 2), "quarter": Fraction(1, 4), "twice": Fraction(2), "double": Fraction(2),
    "triple": Fraction(3), "thrice": Fraction(3), "third": Fraction(1, 3),
}
_WORDS = {**{k: Fraction(v) for k, v in {**_UNITS, **_TENS}.items()}, **_OTHER}
_WORD_RE = re.compile(r"\b(" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")s?\b", re.IGNORECASE)
_COMPOUND_RE = re.compile(
    r"\b(twenty|" + "|".join(_TENS) + r")[- ](one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE
)


def parse_whitelist(s: Union[str, Iterable, None]) -> set[Fraction]:
    """``"60,24,0.5"`` (or a list) -> set of Fractions."""
    if s is None:
        return set()
    items = s.split(",") if isinstance(s, str) else list(s)
    out = set()
    for x in items:
        x = str(x).strip()
        if x:
            out.add(Fraction(x))
    return out


_PERCENT_RE = re.compile(r"\s?(%|percent\b)", re.IGNORECASE)
_OPS = set("+-*/×÷^=")


def _touches_operator(text: str, start: int, end: int) -> bool:
    """True if the numeral text[start:end] is next to an arithmetic operator (spaces skipped), as in ``5*9973 =``."""
    i = start - 1
    while i >= 0 and text[i] == " ":
        i -= 1
    j = end
    while j < len(text) and text[j] == " ":
        j += 1
    standalone_x = lambda k: text[k] == "x" and not (k > 0 and text[k - 1].isalpha()) and not (
        k + 1 < len(text) and text[k + 1].isalpha())
    return (i >= 0 and (text[i] in _OPS or standalone_x(i))) or (j < len(text) and (text[j] in _OPS or standalone_x(j)))


def text_numbers(text: str, lo: int = 0, hi: Optional[int] = None, prose_only: bool = False) -> set[Fraction]:
    """Numbers stated in text[lo:hi]: numerals (with absolute values) and number words, plus their equivalent forms:
    ``25%`` also gives 0.25 and the factors 1.25 / 0.75; a value strictly between 0 and 1 also gives its percent
    value (0.25 -> 25) and, for a unit fraction, its denominator ("a quarter" -> 4). With ``prose_only``, numerals
    next to an arithmetic operator are skipped: they restate a step's expression rather than state a quantity."""
    out = set()
    hi = len(text) if hi is None else hi
    for m in NUM_RE.finditer(text, lo, hi):
        v = parse_number(m.group())
        if v is None or (prose_only and _touches_operator(text, m.start(), m.end())):
            continue
        out.update((v, abs(v)))
        if _PERCENT_RE.match(text, m.end()):
            p = abs(v) / 100
            out.update((p, 1 + p, 1 - p))
    for m in _WORD_RE.finditer(text, lo, hi):
        out.add(_WORDS[m.group(1).lower()])
    for m in _COMPOUND_RE.finditer(text, lo, hi):
        out.add(Fraction(_UNITS.get(m.group(1).lower(), _TENS.get(m.group(1).lower(), 0)) + _UNITS[m.group(2).lower()]))
    for v in list(out):
        if 0 < v < 1:
            out.add(v * 100)
            if v.numerator == 1:
                out.add(Fraction(v.denominator))
    return out


def problem_numbers(question: str) -> set[Fraction]:
    """Numbers in the problem text: numerals and number words, with the equivalent forms of ``text_numbers``."""
    return text_numbers(question)


def _normalise(expr: str) -> str:
    # same normalisation as verify.safe_eval
    e = re.sub(r"(?<=\d),(?=\d{3})", "", expr.strip())
    return e.replace("x", "*").replace("×", "*").replace("÷", "/").replace("^", "**")


def _collect(node: ast.AST, out: list) -> None:
    if isinstance(node, ast.Expression):
        _collect(node.body, out)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        out.append(Fraction(str(node.value)))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _collect(node.operand, out)
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        _collect(node.left, out)
        _collect(node.right, out)
    else:
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _const_divisions(expr: str) -> list[tuple[Fraction, Fraction]]:
    """(a, b) for every ``a/b`` of two numeric constants in the expression (``1/4``), for ratio grounding."""
    try:
        tree = ast.parse(_normalise(expr), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return []
    const = lambda n: isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
    return [(Fraction(str(n.left.value)), Fraction(str(n.right.value))) for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and const(n.left) and const(n.right)]


def expr_operands(expr: str) -> Optional[list[Fraction]]:
    """Numeric constants of an annotation expression (``10-3`` -> [10, 3]), or None if unparsable."""
    e = _normalise(expr)
    if not e or len(e) > 200:
        return None
    out: list[Fraction] = []
    try:
        _collect(ast.parse(e, mode="eval"), out)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    return out


@dataclass
class StepResult:
    index: int
    claim: Claim
    line_start: int
    arithmetic: Optional[bool]
    grounded: bool
    ungrounded: list[str] = field(default_factory=list)
    ok: bool = False
    error_char: Optional[int] = None
    kind: Optional[str] = None  # "arithmetic" | "grounding"


@dataclass
class VPlusResult:
    text: str
    steps: list[StepResult]
    answer: Optional[Fraction]
    answer_ok: bool
    verified: bool
    first_failed: Optional[int]


def fmt_number(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else str(float(v))


def _num_value(s: str) -> Optional[Fraction]:
    return parse_number(s.lstrip("-"))


def _find_value(text: str, lo: int, hi: int, value: Fraction) -> Optional[int]:
    """Char offset of the first number in text[lo:hi] whose (unsigned) value equals ``value``."""
    for m in NUM_RE.finditer(text, lo, hi):
        if _num_value(m.group()) == abs(value):
            return m.start() + (1 if m.group().startswith("-") else 0)
    return None


def _arith_char(text: str, c: Claim, line_start: int) -> int:
    r = parse_number(c.result)
    if r is not None:
        eq = text.rfind("=", line_start, c.start)
        pos = _find_value(text, eq + 1 if eq >= 0 else line_start, c.start, r)
        if pos is not None:
            return pos
    return c.result_start


def _ground_char(text: str, c: Claim, line_start: int, bad: list[Fraction]) -> int:
    best = None
    expr_lo = c.start + 2
    for v in bad:
        pos = _find_value(text, line_start, c.start, v)
        if pos is None:
            pos = _find_value(text, expr_lo, expr_lo + len(c.expr), v)
        if pos is None:
            pos = expr_lo
        best = pos if best is None else min(best, pos)
    return expr_lo if best is None else best


def check_response(
    question: str,
    text: str,
    gold: Union[Fraction, str, int, None],
    whitelist: Optional[set[Fraction]] = None,
    grounding: bool = True,
) -> VPlusResult:
    """Run V+ on one response (truncated after its first ``####`` line)."""
    if whitelist is None:
        whitelist = parse_whitelist(Config.grounding_whitelist)
    if gold is not None and not isinstance(gold, Fraction):
        gold = parse_number(str(gold))
    text = truncate_response(text)
    allowed = problem_numbers(question) | set(whitelist)
    steps = []
    seen_upto = 0  # numbers written in the solution text before this step's line are grounded too
    for j, c in enumerate(parse_claims(text)):
        line_start = text.rfind("\n", 0, c.start) + 1
        arith = c.valid
        bad: list[Fraction] = []
        if grounding:
            if line_start > seen_upto:
                allowed |= text_numbers(text, seen_upto, line_start)
                seen_upto = line_start
            # quantities stated in this line's prose before the annotation ("with 31 days, ..."), but not the
            # numerals of an expression restated there ("5*9973 = <<5*9973=..>>")
            step_allowed = allowed | text_numbers(text, line_start, c.start, prose_only=True)
            ops = expr_operands(c.expr)
            # a constant ratio a/b is grounded if its value is (``1/4`` for a problem's "25%" or "quarter")
            ratio_ok = {x for a, b in _const_divisions(c.expr) if b != 0 and a / b in step_allowed for x in (a, b)}
            bad = [v for v in (ops or []) if v not in step_allowed and v not in ratio_ok]
        grounded = not bad
        chars = []
        if arith is not True:
            chars.append((_arith_char(text, c, line_start), 0, "arithmetic"))
        if not grounded:
            chars.append((_ground_char(text, c, line_start, bad), 1, "grounding"))
        err = min(chars) if chars else None
        seen = []
        for v in bad:
            if fmt_number(v) not in seen:
                seen.append(fmt_number(v))
        steps.append(
            StepResult(
                index=j, claim=c, line_start=line_start, arithmetic=arith, grounded=grounded, ungrounded=seen,
                ok=err is None, error_char=err[0] if err else None, kind=err[2] if err else None,
            )
        )
        r = parse_number(c.result)
        if r is not None:
            allowed.add(r)
            allowed.add(abs(r))
    answer = final_answer(text, strict=True)
    answer_ok = answer is not None and gold is not None and answer == gold
    failed = [s.index for s in steps if not s.ok]
    return VPlusResult(
        text=text, steps=steps, answer=answer, answer_ok=answer_ok,
        verified=answer_ok and not failed, first_failed=failed[0] if failed else None,
    )


def vplus_from_cfg(cfg) -> tuple[set[Fraction], bool]:
    """(whitelist, grounding) from a Config."""
    return parse_whitelist(cfg.grounding_whitelist), bool(cfg.vplus_grounding)


def step_boundaries(result: VPlusResult) -> list[int]:
    """Char offsets where step j's span starts (its line start), plus the start of the final part S.

    The final entry is the offset just past the newline that ends the last step line (len(text) if
    none), so text[boundaries[S]:] is the ``####`` part. Steps sharing a line share a boundary. With
    no steps the only boundary is 0.
    """
    if not result.steps:
        return [0]
    text = result.text
    out = [s.line_start for s in result.steps]
    nl = text.find("\n", result.steps[-1].claim.end)
    out.append(len(text) if nl < 0 else nl + 1)
    return out
