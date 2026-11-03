"""Within-group similarity W(G) (Eq. 3) and true / random / spectrum partitions (Section III-D, D8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def cosine_matrix(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, np.float64)
    mu = mu / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
    return mu @ mu.T


def _as_S(assign, mu_or_S) -> np.ndarray:
    x = np.asarray(mu_or_S, np.float64)
    n = len(assign)
    if x.shape == (n, n) and np.allclose(x, x.T) and np.allclose(np.diag(x), 1.0):
        return x
    return cosine_matrix(x)


def within_group_similarity(assign, mu_or_S) -> float:
    """Eq. 3: mean over groups of the mean pairwise cosine within the group."""
    assign = np.asarray(assign)
    S = _as_S(assign, mu_or_S)
    vals = []
    for g in np.unique(assign):
        m = np.flatnonzero(assign == g)
        k = len(m)
        if k < 2:
            continue
        sub = S[np.ix_(m, m)]
        vals.append((sub.sum() - np.trace(sub)) / 2 / (k * (k - 1) / 2))
    return float(np.mean(vals))


def true_partition(coarse_of_fine) -> np.ndarray:
    return np.asarray(coarse_of_fine, np.int64).copy()


def random_partition(n_classes: int, n_groups: int, rng) -> np.ndarray:
    size = n_classes // n_groups
    assign = np.empty(n_classes, np.int64)
    assign[rng.permutation(n_classes)] = np.arange(n_classes) // size
    return assign


def is_balanced(assign, n_groups: int) -> bool:
    assign = np.asarray(assign)
    counts = np.bincount(assign, minlength=n_groups)
    return len(counts) == n_groups and bool((counts == len(assign) // n_groups).all()) and len(assign) % n_groups == 0


def local_search(S: np.ndarray, assign, mode: str, target=None, iters: int = 20000, rng=None, tol: float = 0.0):
    """Hill-climb with class swaps between groups (keeps balance).

    mode: 'max' / 'min' W, or 'target' (minimize |W - target|). The swap delta uses a
    class-to-group sum table T[c, g] = sum_{c' in g} S[c, c'], so each proposal costs O(1)
    and each accepted swap O(n_classes).
    Returns (assign, W).
    """
    rng = rng or np.random.default_rng()
    assign = np.asarray(assign, np.int64).copy()
    n = len(assign)
    G = int(assign.max()) + 1
    k = n // G
    norm = G * k * (k - 1) / 2
    onehot = np.zeros((n, G))
    onehot[np.arange(n), assign] = 1.0
    T = S @ onehot
    W = within_group_similarity(assign, S)

    def score(w):
        if mode == "max":
            return -w
        if mode == "min":
            return w
        return abs(w - target)

    cur = score(W)
    A = rng.integers(0, n, iters)
    B = rng.integers(0, n, iters)
    for a, b in zip(A.tolist(), B.tolist()):
        ga, gb = assign[a], assign[b]
        if ga == gb:
            continue
        d = (T[b, ga] - S[b, a]) - (T[a, ga] - S[a, a]) + (T[a, gb] - S[a, b]) - (T[b, gb] - S[b, b])
        w_new = W + d / norm
        s_new = score(w_new)
        if s_new < cur:
            diff = S[:, b] - S[:, a]
            T[:, ga] += diff
            T[:, gb] -= diff
            assign[a], assign[b] = gb, ga
            W, cur = w_new, s_new
            if mode == "target" and cur <= tol:
                break
    return assign, within_group_similarity(assign, S)


def spectrum_partitions(S: np.ndarray, n: int, n_groups: int, iters: int, restarts: int, rng) -> list:
    """n balanced partitions with W evenly spaced between the reachable min and max (D8).

    Returns [(assign, W)] sorted by W.
    """
    n_classes = S.shape[0]

    def best(mode, target=None, tol=0.0):
        cands = [local_search(S, random_partition(n_classes, n_groups, rng), mode, target, iters, rng, tol) for _ in range(restarts)]
        if mode == "max":
            return max(cands, key=lambda c: c[1])
        if mode == "min":
            return min(cands, key=lambda c: c[1])
        return min(cands, key=lambda c: abs(c[1] - target))

    lo, hi = best("min"), best("max")
    targets = np.linspace(lo[1], hi[1], n)
    tol = 1e-3 * (hi[1] - lo[1]) / max(n - 1, 1)
    parts = [lo] + [best("target", float(t), tol) for t in targets[1:-1]] + [hi]
    return sorted(parts, key=lambda p: p[1])


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def cover_subset(parts: list, k: int) -> list:
    """k partitions spread over the W range: sort by W, take round(linspace(0, len-1, k))."""
    key = (lambda p: p["W_train"] if p.get("W_train") is not None else p.get("W", 0.0)) if parts and isinstance(parts[0], dict) else (lambda p: p[1])
    srt = sorted(parts, key=key)
    idx = np.round(np.linspace(0, len(srt) - 1, k)).astype(int)
    return [srt[i] for i in idx]


def _entry(dataset, budget, name, kind, assign, W, true_assign) -> dict:
    pct = int(round(budget * 100))
    return {
        "id": f"{dataset}_b{pct:03d}_{name}", "name": name, "kind": kind, "dataset": dataset, "budget": budget,
        "assign": [int(v) for v in assign], "W_train": None if W is None else float(W), "ari_true": ari(assign, true_assign),
    }


def build_all(cfg, class_means: np.ndarray, coarse_of_fine, out_dir) -> dict:
    """Write true.json, random_XX.json, spectrum_XX.json and index.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = cosine_matrix(class_means)
    n_classes = S.shape[0]
    true = true_partition(coarse_of_fine)
    entries = [_entry(cfg.dataset, cfg.budget, "true", "true", true, within_group_similarity(true, S), true)]
    rng_r = np.random.default_rng(cfg.partition_seed)  # identical random partitions at every budget
    for i in range(cfg.n_random):
        a = random_partition(n_classes, cfg.n_groups, rng_r)
        entries.append(_entry(cfg.dataset, cfg.budget, f"random_{i:02d}", "random", a, within_group_similarity(a, S), true))
    rng_s = np.random.default_rng([cfg.partition_seed, int(round(cfg.budget * 100))])
    spec = spectrum_partitions(S, cfg.n_spectrum, cfg.n_groups, cfg.search_iters, cfg.search_restarts, rng_s)
    for i, (a, w) in enumerate(spec):
        entries.append(_entry(cfg.dataset, cfg.budget, f"spectrum_{i:02d}", "spectrum", a, w, true))
    for e in entries:
        assert is_balanced(e["assign"], cfg.n_groups) or e["kind"] == "true", e["id"]
        (out_dir / f"{e['name']}.json").write_text(json.dumps(e))
    index = {
        "dataset": cfg.dataset, "budget": cfg.budget, "W_min": float(spec[0][1]), "W_max": float(spec[-1][1]),
        "partitions": [{k: v for k, v in e.items() if k != "assign"} for e in entries],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def load_partition(path) -> dict:
    p = json.loads(Path(path).read_text())
    p["assign"] = np.asarray(p["assign"], np.int64)
    return p
