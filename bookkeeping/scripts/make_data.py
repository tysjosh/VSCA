"""Generate the bookkeeping splits with FinBalance's generator (clean records, no forced inconsistencies).

  python scripts/make_data.py --out data [--n_train 6000 --n_test 300]
Rows: id, prompt, response, record (the fields the checker needs; documents reduced to ids). Records whose prompt
exceeds --max_prompt tokens or whose response exceeds --max_response tokens are skipped (counts in data/stats.json).
Splits (DIAGNOSTIC.md): train / test_id (6 industries, month + quarter, levels 1-2); test_L3 (level 3);
test_ind (the 2 held-out industries); test_year (year periods, never seen).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bookkeeping.task import FB, prompt, response  # noqa: E402

from finbalance.generation.user_dataset import UserDatasetConfig, generate_user_dataset  # noqa: E402

TRAIN_IND = ("professional_services", "field_services", "retail", "wholesale_distribution", "manufacturing",
             "subscription_saas")
HELD_IND = ("healthcare_clinic", "property_management")
KEEP = ("record_id", "industry", "difficulty_level", "period_start", "period_end", "opening_balance",
        "allowed_accounts", "expected_entries", "expected_balance_sheet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--max_prompt", type=int, default=3300)
    ap.add_argument("--max_response", type=int, default=1100)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--entries_only", action="store_true", help="pilot format: entries then END (PILOT.md)")
    ap.add_argument("--train_levels", default="1,2", help="difficulty levels of the training split")
    ap.add_argument("--onpolicy", type=int, default=0, help="also write an `onpolicy` split of this many unseen records")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "train": (a.n_train, TRAIN_IND, ("month", "quarter"), tuple(int(x) for x in a.train_levels.split(",")), 100),
        "test_id": (a.n_test, TRAIN_IND, ("month", "quarter"), (1, 2), 101),
        "test_L3": (a.n_test, TRAIN_IND, ("month", "quarter"), (3,), 102),
        "test_ind": (a.n_test, HELD_IND, ("month", "quarter"), (1, 2), 103),
        "test_year": (a.n_test, TRAIN_IND, ("year",), (1, 2), 104),
    }
    if a.onpolicy:
        # PILOT.md run 2: records for on-policy sampling, from the training distribution but never seen in SFT
        # (generated last, so the other splits are byte-identical to a run without it)
        specs["onpolicy"] = (a.onpolicy, TRAIN_IND, ("month", "quarter"),
                             tuple(int(x) for x in a.train_levels.split(",")), 105)
    stats, seen = {}, set()
    for name, (n, inds, periods, levels, seed) in specs.items():
        tmp = out / f"_gen_{name}"
        # year and level-3 records are often over the length limit, so oversample generously
        generate_user_dataset(UserDatasetConfig(output_dir=str(tmp), dataset_name=name, records=int(n * 3) + 50,
                                                industries=inds, period_types=periods, levels=levels,
                                                negative_control_rate=0.0, seed=seed, overwrite=True))
        rows, skipped = [], 0
        for line in open(tmp / "records.jsonl"):
            r = json.loads(line)
            p, y = prompt(r, a.entries_only), response(r, a.entries_only)
            sig = (p, y)
            if sig in seen:
                continue
            if len(tok(p)["input_ids"]) > a.max_prompt or len(tok(y)["input_ids"]) > a.max_response:
                skipped += 1
                continue
            seen.add(sig)
            rec = {k: r[k] for k in KEEP}
            rec["documents"] = [{"doc_id": d["doc_id"]} for d in r["documents"]]
            rows.append({"id": f"{name}-{len(rows)}", "prompt": p, "response": y, "record": rec})
            if len(rows) == n:
                break
        if len(rows) < n:
            raise SystemExit(f"{name}: only {len(rows)} records within the length limits (skipped {skipped})")
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        tmp.rmdir()
        stats[name] = {"rows": len(rows), "skipped_too_long": skipped}
        print(name, stats[name])
    (out / "stats.json").write_text(json.dumps({"finbalance": str(FB), **stats}, indent=2))


if __name__ == "__main__":
    main()
