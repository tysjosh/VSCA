"""PROBE_V3.md experiment 3: re-solve every test problem as a Python program and use the program answer to choose
among the existing chain-of-thought samples. Runs on the GPU VM (vLLM). No gold answer is used for selection.

  python scripts/pal_check.py programs [--k 8]      # sample + execute programs -> v3probe/pal_programs.jsonl
  python scripts/pal_check.py select FILE...        # FILE = test_pass8v.jsonl; writes FILE.pmaj.jsonl (+ summary)
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs" / "Qwen2.5-1.5B" / "v3probe"
SHOTS = [
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n"
     "    answer = money_initial - money_spent\n    return answer"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf "
     "balls did he have at the end of wednesday?",
     "    golf_balls_initial = 58\n    lost_tuesday = 23\n    lost_wednesday = 2\n"
     "    answer = golf_balls_initial - lost_tuesday - lost_wednesday\n    return answer"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "    computers_initial = 9\n    per_day = 5\n    days = 4  # monday to thursday\n"
     "    answer = computers_initial + per_day * days\n    return answer"),
    ("A shirt costs $40. It is on sale for 25% off. Tom buys 3 shirts. How much does he pay?",
     "    price = 40\n    discount = 0.25\n    sale_price = price * (1 - discount)\n    shirts = 3\n"
     "    answer = sale_price * shirts\n    return answer"),
]
ALLOWED_IMPORTS = {"math", "fractions", "datetime", "itertools"}
BANNED = re.compile(r"__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\bglobals\s*\(|\blocals\s*\(|\binput\s*\(")


def prompt(q: str) -> str:
    shots = "".join(f"Question: {sq}\ndef solution():\n{body}\n\n" for sq, body in SHOTS)
    return shots + f"Question: {q.strip()}\ndef solution():\n"


def body_of(gen: str) -> str:
    lines = []
    for line in gen.split("\n"):
        if line.strip() and not line[0].isspace():
            break
        lines.append(line)
    return "\n".join(lines).rstrip()


def safe(code: str) -> bool:
    if BANNED.search(code):
        return False
    for m in re.finditer(r"^\s*(?:from\s+(\w+)|import\s+([\w, ]+))", code, re.M):
        mods = [m.group(1)] if m.group(1) else [x.strip().split(".")[0] for x in m.group(2).split(",")]
        if any(x not in ALLOWED_IMPORTS for x in mods):
            return False
    return True


def _limits():
    # memory, CPU seconds and no file writes; each best-effort (macOS rejects RLIMIT_AS; the Linux VM applies all)
    for lim, val in ((resource.RLIMIT_AS, 512 * 2**20), (resource.RLIMIT_CPU, 3), (resource.RLIMIT_FSIZE, 0)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass


def to_answer(s: str):
    try:
        v = float(s.strip().splitlines()[-1].replace(",", ""))
    except (ValueError, IndexError):
        return None
    if v != v or abs(v) > 1e12:
        return None
    r = round(v)
    return Fraction(r) if abs(v - r) < 1e-6 else Fraction(str(round(v, 6)))


def run_program(body: str):
    code = "def solution():\n" + body + "\nprint(solution())\n"
    if not body.strip() or not safe(code):
        return None
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, env={}, capture_output=True, text=True,
                               timeout=3, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return None
    return to_answer(r.stdout) if r.returncode == 0 else None


def majority(xs):
    c = Counter(x for x in xs if x is not None)
    if not c:
        return None
    first = {a: xs.index(a) for a in c}
    return max(c, key=lambda a: (c[a], -first[a]))


def cmd_programs(k: int) -> None:
    from vsca.config import load_config
    from vsca.data import load_split
    from vsca.modeling import load_tokenizer
    from vsca.rollout import VLLMBackend

    cfg = load_config([str(ROOT / "configs" / "base.yaml")], ["vllm_gpu_mem=0.85"])
    tok = load_tokenizer(cfg.model_name)
    recs = load_split(str(ROOT / cfg.data_dir) if not Path(cfg.data_dir).is_absolute() else cfg.data_dir,
                      "gsm8k_test", None)
    be = VLLMBackend(cfg, tok)
    be.set_snapshot(None)
    prefixes = [tok(prompt(r["question"]), add_special_tokens=True).input_ids for r in recs]
    seeds = [(17 * 1_000_003 + i) % (2**31) for i in range(len(recs))]
    outs = be.generate(prefixes, 256, do_sample=True, temperature=0.7, top_p=1.0, n=k, seeds=seeds)
    bodies = [body_of(tok.decode(g, skip_special_tokens=True)) for g in outs]
    with ThreadPoolExecutor(16) as ex:
        answers = list(ex.map(run_program, bodies))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n_ok, n_right = [], 0, 0
    for i, r in enumerate(recs):
        a = answers[i * k : (i + 1) * k]
        p = majority(a)
        n_ok += sum(x is not None for x in a)
        n_right += p is not None and p == Fraction(r["gold"])
        rows.append({"id": r["id"], "program_answers": [None if x is None else str(x) for x in a],
                     "p_star": None if p is None else str(p), "bodies": bodies[i * k : (i + 1) * k]})
    (OUT / "pal_programs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summ = {"k": k, "items": len(recs), "programs_ran": n_ok / (k * len(recs)), "program_majority_acc": n_right / len(recs)}
    (OUT / "pal_programs.summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ))


def cmd_select(files) -> None:
    pal = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "pal_programs.jsonl").read_text().splitlines()}
    from vsca.data import load_split

    for f in map(Path, files):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        gold = {r["id"]: Fraction(r["gold"]) for r in load_split(str(ROOT / "data"), "gsm8k_test", None)}
        out, n_cov, n_maj, n_p = [], 0, 0, 0
        for r in rows:
            ans = [None if a is None else Fraction(a) for a in r["answers"]]
            p = pal[r["id"]]["p_star"]
            p = None if p is None else Fraction(p)
            covered = p is not None and p in ans
            pick = p if covered else majority(ans)
            n_cov += covered
            n_maj += bool(r["maj_correct"])
            n_p += pick is not None and pick == gold[r["id"]]
            out.append({"id": r["id"], "correct": pick is not None and pick == gold[r["id"]], "covered": covered})
        f.with_suffix(".pmaj.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
        m = len(rows)
        s = {"file": str(f), "maj@8": n_maj / m, "pmaj@8": n_p / m, "coverage": n_cov / m}
        f.with_suffix(".pmaj.summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("programs")
    a.add_argument("--k", type=int, default=8)
    b = sp.add_parser("select")
    b.add_argument("files", nargs="+")
    args = ap.parse_args()
    cmd_programs(args.k) if args.cmd == "programs" else cmd_select(args.files)
