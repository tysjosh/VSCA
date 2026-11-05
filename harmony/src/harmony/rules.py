"""Exact four-part voice-leading checker, voicing enumeration and completability (the certifier).

A *step* is one chord voiced as (S, A, T, B). ``check_step`` returns the rules a voicing breaks, given the previous
voicing; ``certified`` lists every voicing at a step that breaks no rule **and** from which the rest of the exercise
can still be completed (exact, by dynamic programming over the remaining chords). There is no sampling anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from .theory import Chord, Key, Note, make_chord

RANGES = {"S": (60, 79), "A": (55, 74), "T": (48, 67), "B": (40, 60)}  # C4-G5, G3-D5, C3-G4, E2-C4
MAX_LEAP = {"S": 12, "A": 7, "T": 7, "B": 12}
VOICES = "SATB"
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


@dataclass(frozen=True)
class Exercise:
    key: Key
    labels: tuple[str, ...]
    soprano: tuple[Note, ...]

    @property
    def chords(self) -> tuple[Chord, ...]:
        return tuple(make_chord(l, self.key) for l in self.labels)


Voicing = tuple  # (S, A, T, B) Notes


def check_vertical(v: Voicing, chord: Chord, key: Key) -> list[str]:
    errs = []
    pcs = [n.pc for n in v]
    if any(chord.role_of(p) is None for p in pcs):
        errs.append("non_chord_tone")
        return errs
    roles = [chord.role_of(p) for p in pcs]
    if roles[3] != chord.bass_role:
        errs.append("wrong_inversion")
    need = {0, 1, 3} if chord.seventh else {0, 1, 2}
    if chord.seventh and chord.bass_role != 0:
        need = {0, 1, 2, 3}  # inverted seventh chords are complete
    if not need <= set(roles):
        errs.append("incomplete_chord")
    for name, n in zip(VOICES, v):
        lo, hi = RANGES[name]
        if not lo <= n.midi <= hi:
            errs.append(f"range_{name}")
    m = [n.midi for n in v]
    if not (m[0] > m[1] > m[2] > m[3]):
        errs.append("voice_crossing")
    if m[0] - m[1] > 12 or m[1] - m[2] > 12:
        errs.append("spacing")
    if chord.dominant and pcs.count(key.leading_tone) > 1:
        errs.append("doubled_leading_tone")
    if chord.seventh and roles.count(3) > 1:
        errs.append("doubled_seventh")
    return errs


def check_transition(p: Voicing, v: Voicing, pc: Chord, c: Chord, key: Key) -> list[str]:
    errs = []
    a, b = [n.midi for n in p], [n.midi for n in v]
    for i, name in enumerate(VOICES):
        if abs(b[i] - a[i]) > MAX_LEAP[name]:
            errs.append(f"leap_{name}")
    for i, j in PAIRS:
        i1, i2 = (a[i] - a[j]) % 12, (b[i] - b[j]) % 12
        moved = a[i] != b[i] and a[j] != b[j]
        if moved and i1 == i2 and i1 in (0, 7):
            errs.append("parallel_octaves" if i1 == 0 else "parallel_fifths")
    # direct (hidden) fifths / octaves in the outer voices with a soprano leap
    s_mv, b_mv = b[0] - a[0], b[3] - a[3]
    if s_mv * b_mv > 0 and abs(s_mv) > 2 and (b[0] - b[3]) % 12 in (0, 7):
        errs.append("direct_fifth_octave")
    # overlap: a voice may not pass the previous note of an adjacent voice
    for i in range(3):
        if b[i + 1] > a[i] or b[i] < a[i + 1]:
            errs.append("overlap")
            break
    # leading tone in the soprano resolves up to the tonic in a dominant -> tonic-function motion
    if pc.dominant and c.degree in (0, 5) and p[0].pc == key.leading_tone:
        if not (v[0].pc == key.degree(0) and 0 < b[0] - a[0] <= 2):
            errs.append("leading_tone_unresolved")
    # the chordal seventh resolves down by step
    if pc.seventh:
        for i in range(4):
            if pc.role_of(p[i].pc) == 3 and not (1 <= a[i] - b[i] <= 2):
                errs.append("seventh_unresolved")
                break
    return sorted(set(errs))


def check_step(prev: Voicing | None, v: Voicing, prev_chord: Chord | None, chord: Chord, key: Key) -> list[str]:
    errs = check_vertical(v, chord, key)
    if prev is not None and not errs:
        errs += check_transition(prev, v, prev_chord, chord, key)
    return errs


def _notes_in(pcs, lo: int, hi: int) -> list[Note]:
    out = []
    for letter, acc in pcs:
        for octv in range(1, 7):
            n = Note(letter, acc, octv)
            if lo <= n.midi <= hi:
                out.append(n)
    return out


@lru_cache(maxsize=200_000)
def candidate_voicings(chord: Chord, key: Key, soprano: Note | None) -> tuple:
    """Every vertically valid voicing of ``chord`` (with the given soprano, or any soprano if None)."""
    sops = [soprano] if soprano is not None else _notes_in(chord.members, *RANGES["S"])
    alts = _notes_in(chord.members, *RANGES["A"])
    tens = _notes_in(chord.members, *RANGES["T"])
    bass = _notes_in([chord.members[chord.bass_role]], *RANGES["B"])
    out = []
    for s, a, t, b in product(sops, alts, tens, bass):
        v = (s, a, t, b)
        if not check_vertical(v, chord, key):
            out.append(v)
    return tuple(out)


class Certifier:
    """Exact completability for one exercise: ``completable(t, v)`` is True iff voicing v at step t breaks no
    vertical rule and some rule-free continuation to the last chord exists."""

    def __init__(self, ex: Exercise, free_soprano: bool = False):
        self.ex, self.key, self.chords = ex, ex.key, ex.chords
        self.n = len(ex.labels)
        sop = (lambda t: None) if free_soprano else (lambda t: ex.soprano[t])
        self.cands = [candidate_voicings(self.chords[t], self.key, sop(t)) for t in range(self.n)]
        self._memo: dict = {}

    def completable(self, t: int, v: Voicing) -> bool:
        k = (t, v)
        if k not in self._memo:
            if t == self.n - 1:
                ok = True
            else:
                c, c2 = self.chords[t], self.chords[t + 1]
                ok = any(not check_transition(v, w, c, c2, self.key) and self.completable(t + 1, w)
                         for w in self.cands[t + 1])
            self._memo[k] = ok
        return self._memo[k]

    def certified(self, t: int, prev: Voicing | None) -> list:
        """Voicings at step t that follow ``prev`` without a rule violation and can still be completed."""
        out = []
        for v in self.cands[t]:
            if prev is not None and check_transition(prev, v, self.chords[t - 1], self.chords[t], self.key):
                continue
            if self.completable(t, v):
                out.append(v)
        return out

    def solvable(self) -> bool:
        return bool(self.certified(0, None))
