"""Generate solvable harmonisation exercises (key + Roman numerals + soprano) with one reference solution each.

Progressions follow the tonic -> predominant -> dominant -> tonic cycle and end with an authentic cadence. The
soprano is taken from a random rule-free SATB realisation (soprano steps <= 5 semitones), so every exercise is
solvable; the reference solution is that realisation. Other valid solutions usually exist.
"""

from __future__ import annotations

import random

from .rules import Certifier, Exercise, check_transition, candidate_voicings
from .theory import MAJOR_KEYS, MINOR_KEYS, Key, make_chord

VOCAB = {
    "major": {"T": ["I", "I6", "vi"], "PD": ["ii", "ii6", "IV", "IV6"], "D": ["V", "V6", "V7", "V65", "V43", "viio6"],
              "cad_D": ["V", "V7"], "end": ["I"]},
    "minor": {"T": ["i", "i6", "VI"], "PD": ["iio6", "iv", "iv6"], "D": ["V", "V6", "V7", "V65", "V43", "viio6"],
              "cad_D": ["V", "V7"], "end": ["i"]},
}
MAX_SOP_STEP = 5


def progression(rng: random.Random, mode: str, n: int) -> list[str]:
    v = VOCAB[mode]
    out = [v["end"][0]]
    while len(out) < n - 2:
        f = {"T": "PD", "PD": "D", "D": "T"}[_fn(out[-1])] if rng.random() < 0.8 else _fn(out[-1])
        nxt = rng.choice(v[f])
        if nxt != out[-1]:
            out.append(nxt)
    return out[: n - 2] + [rng.choice(v["cad_D"]), v["end"][0]]


def _fn(label: str) -> str:
    base = label.rstrip("0123456789").rstrip("o")
    return {"I": "T", "i": "T", "vi": "T", "VI": "T", "ii": "PD", "IV": "PD", "iv": "PD"}.get(base, "D")


def realise(rng: random.Random, key: Key, labels: list[str]):
    """A random rule-free SATB realisation with a singable soprano, or None."""
    chords = [make_chord(l, key) for l in labels]
    ex = Exercise(key, tuple(labels), tuple())
    cert = Certifier.__new__(Certifier)
    cert.ex, cert.key, cert.chords, cert.n, cert._memo = ex, key, tuple(chords), len(labels), {}
    cert.cands = [candidate_voicings(c, key, None) for c in chords]

    def ok(p, v, t):
        return abs(v[0].midi - p[0].midi) <= MAX_SOP_STEP and not check_transition(p, v, chords[t - 1], chords[t], key)

    # completability with the soprano-step constraint
    memo = {}

    def comp(t, v):
        if (t, v) not in memo:
            memo[(t, v)] = t == len(labels) - 1 or any(ok(v, w, t + 1) and comp(t + 1, w) for w in cert.cands[t + 1])
        return memo[(t, v)]

    starts = [v for v in cert.cands[0] if comp(0, v)]
    if not starts:
        return None
    seq = [rng.choice(starts)]
    for t in range(1, len(labels)):
        seq.append(rng.choice([w for w in cert.cands[t] if ok(seq[-1], w, t) and comp(t, w)]))
    return seq


def make_exercise(rng: random.Random, mode: str, n: int, keys=None):
    for _ in range(50):
        key = rng.choice(keys or (MAJOR_KEYS if mode == "major" else MINOR_KEYS))
        labels = progression(rng, mode, n)
        sol = realise(rng, key, labels)
        if sol is None:
            continue
        ex = Exercise(key, tuple(labels), tuple(v[0] for v in sol))
        if Certifier(ex).solvable():
            return ex, sol
    raise RuntimeError(f"no solvable exercise found ({mode}, {n} chords)")
