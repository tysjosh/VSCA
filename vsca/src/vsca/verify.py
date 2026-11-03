"""Claim verifier V_c and response verifier V_r for GSM8K-style solutions.

Claims are the calculator annotations ``<<expr=result>>`` that GSM8K reference
solutions contain. ``V_c`` evaluates ``expr`` exactly with rational arithmetic
and compares it with ``result``, allowing the rounding implied by the number of
decimals written in ``result``. ``V_r`` accepts a response only if its
``#### <answer>`` line matches the gold answer and every annotation in it has
``V_c = 1`` (Section 3.1 of the paper).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

ANNOT_RE = re.compile(r"<<([^<>]*)>>")
FINAL_RE = re.compile(r"####\s*([^\n]*)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")

_MAX_POW = 16


@dataclass
class Claim:
    start: int  # char offset of "<<"
    end: int  # char offset just after ">>"
    expr: str
    result: str
    result_start: int  # char offset of the result substring
    result_end: int
    valid: Optional[bool]  # True / False / None (= undetermined, the paper's ⊥)


def parse_number(s: str) -> Optional[Fraction]:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s or s.endswith("%"):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_node(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b  # ZeroDivisionError handled by caller
        if isinstance(node.op, ast.FloorDiv):
            return Fraction(a // b)
        if isinstance(node.op, ast.Mod):
            return a % b
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b.numerator) > _MAX_POW:
                raise ValueError("unsupported exponent")
            return a ** b.numerator
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def safe_eval(expr: str) -> Optional[Fraction]:
    """Exact rational evaluation of an arithmetic expression, or None."""
    e = re.sub(r"(?<=\d),(?=\d{3})", "", expr.strip())
    e = e.replace("x", "*").replace("×", "*").replace("÷", "/").replace("^", "**")
    if not e or len(e) > 200:
        return None
    try:
        return _eval_node(ast.parse(e, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError, RecursionError):
        return None


def values_match(value: Fraction, result_str: str) -> Optional[bool]:
    """True if ``result_str`` equals ``value`` up to the rounding its decimals imply."""
    r = parse_number(result_str)
    if r is None:
        return None
    s = result_str.strip().replace(",", "")
    decimals = len(s.split(".", 1)[1]) if "." in s else 0
    if decimals == 0:
        return value == r
    tol = Fraction(1, 2 * 10**decimals)
    return abs(value - r) <= tol


def claim_valid(expr: str, result: str) -> Optional[bool]:
    v = safe_eval(expr)
    if v is None:
        return None
    return values_match(v, result)


def parse_claims(text: str) -> list[Claim]:
    claims = []
    for m in ANNOT_RE.finditer(text):
        inner = m.group(1)
        inner_start = m.start() + 2
        if "=" not in inner:
            claims.append(Claim(m.start(), m.end(), inner, "", m.end() - 2, m.end() - 2, None))
            continue
        expr, res = inner.rsplit("=", 1)
        rs = inner_start + len(expr) + 1
        lead = len(res) - len(res.lstrip())
        trail = len(res) - len(res.rstrip())
        claims.append(
            Claim(
                start=m.start(),
                end=m.end(),
                expr=expr,
                result=res.strip(),
                result_start=rs + lead,
                result_end=rs + len(res) - trail,
                valid=claim_valid(expr, res),
            )
        )
    return claims


def truncate_response(text: str) -> str:
    """Cut a generated response after its first ``####`` line."""
    m = FINAL_RE.search(text)
    if m:
        return text[: m.end()]
    cut = text.find("\nQuestion:")
    return text if cut < 0 else text[:cut]


def final_answer(text: str, strict: bool = True) -> Optional[Fraction]:
    """Numeric value on the ``####`` line; if not strict, fall back to the last number."""
    m = FINAL_RE.search(text)
    if m:
        nums = NUM_RE.findall(m.group(1))
        return parse_number(nums[0]) if nums else None
    if strict:
        return None
    nums = NUM_RE.findall(text)
    return parse_number(nums[-1]) if nums else None


def response_valid(text: str, gold: Fraction) -> bool:
    """V_r: final answer matches and every annotation verifies (undetermined counts as failure)."""
    text = truncate_response(text)
    ans = final_answer(text, strict=True)
    if ans is None or ans != gold:
        return False
    return all(c.valid is True for c in parse_claims(text))


def answer_correct(text: str, gold: Fraction) -> bool:
    """Primary metric: final-answer accuracy (lenient extraction)."""
    ans = final_answer(truncate_response(text), strict=False)
    return ans is not None and ans == gold
