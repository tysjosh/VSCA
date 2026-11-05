"""Spelled pitches, keys and Roman-numeral chords for four-part (SATB) harmony.

A note is spelled: letter, accidental and octave (``F#4``, ``Bb3``). Chord tones are spelled from the key, so an
enharmonic misspelling (``Gb4`` for ``F#4`` in D major) is a wrong note, as in a theory exam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

LETTERS = "CDEFGAB"
NAT_PC = [0, 2, 4, 5, 7, 9, 11]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]  # natural minor; V, V7 and vii° raise degree 7
NOTE_RE = re.compile(r"^([A-G])(##|#|bb|b)?(-?\d)$")
ACC_STR = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


@dataclass(frozen=True, order=True)
class Note:
    letter: int  # 0..6 = C..B
    acc: int     # -2..2
    octave: int

    @property
    def midi(self) -> int:
        return 12 * (self.octave + 1) + NAT_PC[self.letter] + self.acc

    @property
    def pc(self) -> tuple[int, int]:
        """Spelled pitch class (letter, accidental)."""
        return (self.letter, self.acc)

    @property
    def diatonic(self) -> int:
        """Diatonic step number (letter + 7 * octave), for generic intervals."""
        return self.letter + 7 * self.octave

    def __str__(self) -> str:
        return LETTERS[self.letter] + ACC_STR[self.acc] + str(self.octave)


def parse_note(s: str):
    m = NOTE_RE.match(s.strip())
    if not m:
        return None
    acc = {None: 0, "#": 1, "##": 2, "b": -1, "bb": -2}[m.group(2)]
    return Note(LETTERS.index(m.group(1)), acc, int(m.group(3)))


def pc_name(pc: tuple[int, int]) -> str:
    return LETTERS[pc[0]] + ACC_STR[pc[1]]


@dataclass(frozen=True)
class Key:
    tonic: tuple[int, int]  # spelled pitch class
    mode: str               # "major" | "minor"

    def __str__(self) -> str:
        return f"{pc_name(self.tonic)} {self.mode}"

    def degree(self, d: int, raised7: bool = False) -> tuple[int, int]:
        """Spelled pitch class of scale degree d (0 = tonic); ``raised7`` raises degree 7 in minor."""
        steps = MAJOR if self.mode == "major" else MINOR
        letter = (self.tonic[0] + d) % 7
        target = (NAT_PC[self.tonic[0]] + self.tonic[1] + steps[d % 7]) % 12
        if raised7 and self.mode == "minor" and d % 7 == 6:
            target = (target + 1) % 12
        acc = (target - NAT_PC[letter] + 6) % 12 - 6
        return (letter, acc)

    @property
    def leading_tone(self) -> tuple[int, int]:
        return self.degree(6, raised7=True)


MAJOR_KEYS = [Key(p, "major") for p in [(0, 0), (4, 0), (1, 0), (5, 0), (2, 0), (6, 0), (3, 1), (1, -1), (5, -1),
                                         (2, -1), (6, -1), (3, 0)]]  # C G D A E B F# Db Ab Eb Bb F
MINOR_KEYS = [Key(p, "minor") for p in [(5, 0), (2, 0), (6, 0), (3, 1), (0, 1), (4, 1), (1, 0), (4, 0), (0, 0),
                                         (3, 0), (6, -1), (2, -1)]]  # a e b f# c# g# d g c f bb eb

# label -> (degree, seventh chord?, bass role 0=root 1=third 2=fifth 3=seventh, function)
_FIG = {"": (False, 0), "6": (False, 1), "7": (True, 0), "65": (True, 1), "43": (True, 2), "42": (True, 3)}
_ROMAN = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6}
_LABEL_RE = re.compile(r"^(VII|vii|VI|vi|IV|iv|V|v|III|iii|II|ii|I|i)(o)?(7|65|43|42|6)?$")
FUNCTION = {0: "T", 1: "PD", 2: "T", 3: "PD", 4: "D", 5: "T", 6: "D"}
ROLE_NAMES = ("root", "third", "fifth", "seventh")


@dataclass(frozen=True)
class Chord:
    label: str
    degree: int
    members: tuple       # spelled pitch classes, by role: root, third, fifth (, seventh)
    bass_role: int
    seventh: bool
    function: str
    dominant: bool       # V-type or vii°: the leading tone must resolve

    def role_of(self, pc) -> int | None:
        return self.members.index(pc) if pc in self.members else None


@lru_cache(maxsize=None)
def make_chord(label: str, key: Key) -> Chord:
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(f"bad chord label {label!r}")
    roman, _dim, fig = m.group(1), m.group(2), m.group(3) or ""
    deg = _ROMAN[roman.lower()]
    seventh, bass_role = _FIG[fig]
    dominant = deg in (4, 6)
    raise7 = key.mode == "minor" and dominant
    members = tuple(key.degree(deg + 2 * k, raised7=raise7) for k in range(4 if seventh else 3))
    return Chord(label, deg, members, bass_role, seventh, FUNCTION[deg], dominant)
