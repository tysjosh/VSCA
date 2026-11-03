"""Out-of-fold pool difficulty (Section 3.3): K class-stratified folds of D_beta, a CE fold model per fold (set
``oof``), and ``privred difficulty`` writing <budget dir>/difficulty.npz and difficulty_check.json."""

from __future__ import annotations

import numpy as np

from .io import align, budget_dir, load_npz, member_dir, read_json, save_npz, set_dir, write_json


def oof_folds(y, K: int, seed: int = 0) -> np.ndarray:
    """Fold id per record: per class (ascending), a permutation from default_rng([seed, 7]) dealt round-robin."""
    y = np.asarray(y, np.int64)
    rng = np.random.default_rng([int(seed), 7])
    fold = np.full(len(y), -1, np.int64)
    for c in np.unique(y):
        perm = rng.permutation(np.flatnonzero(y == c))
        fold[perm] = np.arange(len(perm)) % K
    return fold


def _rank_quartile(order) -> np.ndarray:
    """Quartile 0..3 by position in ``order`` (first quarter -> 0)."""
    n = len(order)
    q = np.empty(n, np.int64)
    q[order] = np.arange(n) * 4 // max(n, 1)
    return q


def confident_errors(probs, y) -> np.ndarray:
    """OOF argmax != y and OOF prob of the argmax class c >= t_c = mean OOF prob of class c over records labeled c."""
    probs = np.asarray(probs, np.float64)
    y = np.asarray(y, np.int64)
    C = probs.shape[1]
    t = np.full(C, np.inf)
    for c in np.unique(y):
        t[c] = probs[y == c, c].mean()
    am = probs.argmax(1)
    return (am != y) & (probs[np.arange(len(y)), am] >= t[am])


def _check(cfg, budget, fold_test) -> dict:
    """Spearman of the mean fold-model test p_true with each companion CE seed (CE-A rotation of
    sg.metrics.crossfit_folds) and overlap of the hard strata."""
    from scipy.stats import spearmanr
    from sg.metrics import crossfit_folds
    from sg.utils import read_jsonl

    from .config import ce_run

    seeds = {}
    for s in range(5):
        f = ce_run(cfg, budget, s) / "test.jsonl"
        if f.exists():
            seeds[s] = {r["id"]: r["p_true"] for r in read_jsonl(f)}
    if not seeds:
        return {"available": False, "note": f"no companion CE runs under {ce_run(cfg, budget, 0).parent}"}
    ids = sorted(set(fold_test).intersection(*[set(v) for v in seeds.values()]))
    mine = np.array([fold_test[i] for i in ids])
    ce = np.array([[seeds[s][i] for i in ids] for s in sorted(seeds)])
    n_hard = int(round(cfg.hard_frac * len(ids)))
    hard_mine = set(np.argsort(mine, kind="stable")[:n_hard].tolist())
    rows = []
    for fd in crossfit_folds(ce, cfg.hard_frac):
        s = sorted(seeds)[fd["ref"]]
        rho = spearmanr(mine, ce[fd["ref"]])[0] if len(ids) > 2 else float("nan")
        ov = len(hard_mine & set(fd["hard"].tolist())) / n_hard if n_hard else None
        rows.append({"ce_seed": s, "spearman": None if not np.isfinite(rho) else float(rho), "hard_overlap": ov})
    sp = [r["spearman"] for r in rows if r["spearman"] is not None]
    ov = [r["hard_overlap"] for r in rows if r["hard_overlap"] is not None]
    return {"available": True, "n_test": len(ids), "n_hard": n_hard, "per_seed": rows,
            "spearman_mean": float(np.mean(sp)) if sp else None, "hard_overlap_mean": float(np.mean(ov)) if ov else None}


def compute_difficulty(cfg, budget) -> dict:
    from scipy.special import softmax
    from sg.utils import read_jsonl

    sdir = set_dir(cfg, budget, "oof")
    man = read_json(sdir / "manifest.json")
    K = int(man["M"])
    ids = np.load(sdir / "pool_ids.npy")
    fold = np.load(sdir / "folds.npy")
    n = len(ids)
    d0, loss0, pd0, pur0 = (np.full(n, np.nan) for _ in range(4))
    oof_pred = np.full(n, -1, np.int64)
    y = np.full(n, -1, np.int64)
    probs = None
    aud_ids, aud_p, test_p = None, [], {}
    for k in range(K):
        md = member_dir(sdir, k)
        ex = load_npz(md / "examples_heldout.npz")
        pos = align(ids, ex["ids"])
        if not (fold[pos] == k).all():
            raise SystemExit(f"difficulty: {md} held-out rows do not match fold {k}")
        pdz = load_npz(md / "pd_heldout.npz")
        pp = align(pdz["ids"], ex["ids"])
        d0[pos] = ex["p_true"][:, 0]
        loss0[pos] = ex["loss"][:, 0]
        oof_pred[pos] = ex["pred"][:, 0]
        y[pos] = ex["y"]
        pd0[pos] = pdz["pd"][pp]
        pur0[pos] = pdz["purity8"][pp]
        pr = softmax(ex["logits"][:, 0].astype(np.float64), 1)
        if probs is None:
            probs = np.zeros((n, pr.shape[1]))
        probs[pos] = pr
        au = load_npz(md / "examples_audit.npz")
        if aud_ids is None:
            aud_ids = au["ids"]
        aud_p.append(au["p_true"][align(au["ids"], aud_ids), 0])
        for r in read_jsonl(md / "test.jsonl"):
            test_p.setdefault(r["id"], []).append(r["p_true"])
    if np.isnan(d0).any():
        raise SystemExit("difficulty: some pool records have no out-of-fold score")
    quartile = _rank_quartile(np.argsort(d0, kind="stable"))          # 0 = lowest d0 = hardest
    pd_quartile = _rank_quartile(np.lexsort((pur0, -loss0, -pd0)))    # PD desc, loss desc, purity asc
    flags = confident_errors(probs, y)
    audit_d = np.mean(aud_p, 0)
    cuts = np.quantile(d0, [0.25, 0.5, 0.75])
    audit_q = np.searchsorted(cuts, audit_d, side="right").astype(np.int64)
    out = budget_dir(cfg, budget)
    save_npz(out / "difficulty.npz", ids=ids, y=y, fold=fold, d0=d0, loss0=loss0, pd0=pd0.astype(np.int64),
             purity0=pur0, oof_pred=oof_pred, oof_correct=oof_pred == y, quartile=quartile, pd_quartile=pd_quartile,
             flag_label_error=flags, audit_ids=aud_ids, audit_d=audit_d, audit_quartile=audit_q)
    fold_test = {i: float(np.mean(v)) for i, v in test_p.items() if len(v) == K}
    check = _check(cfg, budget, fold_test)
    write_json(out / "difficulty_check.json", check)
    return {"out": str(out / "difficulty.npz"), "n": int(n), "K": K, "n_flagged": int(flags.sum()),
            "quartile_counts": np.bincount(quartile, minlength=4).tolist(),
            "audit_quartile_counts": np.bincount(audit_q, minlength=4).tolist(),
            "oof_acc": float((oof_pred == y).mean()), "check": check}
