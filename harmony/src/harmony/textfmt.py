"""Prompt / response text for the model, and a strict parser.

Prompt:   Harmonise in four parts (SATB).\\nKey: D major\\nChords: I IV V7 I\\nSoprano: F#4 G4 A4 D5\\nSATB:\\n
Response: one line per chord, ``<k> <label>: S=<n> A=<n> T=<n> B=<n>``, then ``END``.
"""

from __future__ import annotations

import re

from .rules import Exercise
from .theory import parse_note

LINE_RE = re.compile(r"^(\d+) (\S+): S=(\S+) A=(\S+) T=(\S+) B=(\S+)$")


def prompt(ex: Exercise) -> str:
    return (f"Harmonise in four parts (SATB).\nKey: {ex.key}\nChords: {' '.join(ex.labels)}\n"
            f"Soprano: {' '.join(map(str, ex.soprano))}\nSATB:\n")


def step_line(ex: Exercise, t: int, v) -> str:
    s, a, tn, b = v
    return f"{t + 1} {ex.labels[t]}: S={s} A={a} T={tn} B={b}\n"


def response(ex: Exercise, sol) -> str:
    return "".join(step_line(ex, t, v) for t, v in enumerate(sol)) + "END"


def parse(ex: Exercise, text: str):
    """(voicings, line_starts, bad_step): the voicings of the leading well-formed lines, the char offset where each
    step's line starts (plus one past the last parsed line), and the index of the first malformed / missing step
    (None if every chord has a well-formed line)."""
    vs, starts, pos = [], [], 0
    lines = text.split("\n")
    for t in range(len(ex.labels)):
        if t >= len(lines):
            return vs, starts + [pos], t
        m = LINE_RE.match(lines[t].strip())
        notes = [parse_note(x) for x in m.groups()[2:]] if m else None
        if not m or int(m.group(1)) != t + 1 or m.group(2) != ex.labels[t] or None in notes:
            return vs, starts + [pos], t
        if str(notes[0]) != str(ex.soprano[t]):
            return vs, starts + [pos], t
        starts.append(pos)
        vs.append(tuple(notes))
        pos += len(lines[t]) + 1
    return vs, starts + [pos], None
