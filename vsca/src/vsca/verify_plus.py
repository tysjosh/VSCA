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
    "triple": Fraction(3),
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


def problem_numbers(question: str) -> set[Fraction]:
    """Numbers in the problem text: numerals (with their absolute values) and cardinal number words."""
    out = set()
    for m in NUM_RE.finditer(question):
        v = parse_number(m.group())
        if v is not None:
            out.add(v)
            out.add(abs(v))
    for m in _WORD_RE.finditer(question):
        out.add(_WORDS[m.group(1).lower()])
    for m in _COMPOUND_RE.finditer(question):
        out.add(Fraction(_UNITS.get(m.group(1).lower(), _TENS.get(m.group(1).lower(), 0)) + _UNITS[m.group(2).lower()]))
    return out


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
    for j, c in enumerate(parse_claims(text)):
        line_start = text.rfind("\n", 0, c.start) + 1
        arith = c.valid
        bad: list[Fraction] = []
        if grounding:
            ops = expr_operands(c.expr)
            bad = [v for v in (ops or []) if v not in allowed]
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
