"""Write the harmony train / test splits (jsonl: id, key, labels, soprano, prompt, response).

Train: major keys except the held-out ones, 6-8 chords. Test sets (DIAGNOSTIC.md):
  test_id      same distribution as train
  test_long    major, train keys, 12-16 chords        (longer exercises)
  test_minor   minor keys, 6-8 chords                 (a mode never seen in training)
  test_keys    held-out major keys, 6-8 chords        (keys never seen in training)
  python scripts/make_data.py --out data [--n_train 12000 --n_test 300 --seed 0]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmony.generate import make_exercise  # noqa: E402
from harmony.textfmt import prompt, response  # noqa: E402
from harmony.theory import MAJOR_KEYS, MINOR_KEYS  # noqa: E402

HELD_OUT = {"F# major", "Db major", "B major"}


def write(path, items):
    with open(path, "w") as f:
        for i, (ex, sol) in enumerate(items):
            f.write(json.dumps({"id": f"{path.stem}-{i}", "key": str(ex.key), "labels": list(ex.labels),
                                "soprano": [str(n) for n in ex.soprano], "prompt": prompt(ex),
                                "response": response(ex, sol)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_keys = [k for k in MAJOR_KEYS if str(k) not in HELD_OUT]
    held = [k for k in MAJOR_KEYS if str(k) in HELD_OUT]
    specs = {
        "train": (a.n_train, "major", (6, 8), train_keys, a.seed),
        "test_id": (a.n_test, "major", (6, 8), train_keys, a.seed + 1),
        "test_long": (a.n_test, "major", (12, 16), train_keys, a.seed + 2),
        "test_minor": (a.n_test, "minor", (6, 8), MINOR_KEYS, a.seed + 3),
        "test_keys": (a.n_test, "major", (6, 8), held, a.seed + 4),
    }
    seen = set()
    for name, (n, mode, (lo, hi), keys, seed) in specs.items():
        rng = random.Random(seed)
        items = []
        while len(items) < n:
            ex, sol = make_exercise(rng, mode, rng.randint(lo, hi), keys)
            sig = (str(ex.key), ex.labels, tuple(map(str, ex.soprano)))
            if sig in seen:  # no exercise appears in two splits (or twice)
                continue
            seen.add(sig)
            items.append((ex, sol))
        write(out / f"{name}.jsonl", items)
        print(name, len(items))


if __name__ == "__main__":
    main()
