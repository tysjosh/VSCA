"""Study B removal sets (Section 3.9): cohorts, curated siblings, the six removal types with Delta-V matching,
support deltas and composition reports. ``privred deletion_sets`` writes, under <budget dir>/deletion/,
cohorts.json and per (cohort q, type) a directory q{q}_{type}/ with R.npy (sorted row ids of R), deltas.npz
(support on D_beta and on D_beta minus R for every retained record) and composition.json.

Cohort C_q: 4 fine classes from 4 different curated superclasses, classes drawn without replacement across
cohorts, rng default_rng([cohort_seed, pct]). T_q = {i in D_beta: y_i in C_q} is retained everywhere.
Y_sib = the 16 curated siblings of the cohort classes. |R| = round(eta n_beta), divisible by 16.
  rand: |R| uniform from D_beta minus T_q
  sem:  |R|/16 uniform from each class of Y_sib
  vis:  the 16 classes outside C_q and Y_sib whose phi-bar class means have the largest max-cosine to a cohort
        class mean, |R|/16 each: uniform start, then the Delta-V swap search towards mean Delta V(sem) over T_q
  hv:   the |R| records of D_beta minus T_q with the largest preliminary score Lbar^0 (Eq. lambda0), ties by the
        smaller row index
  near/far: from each class of Y_sib the |R|/16 records with the highest / lowest mean kappa to T_q
Every rng is default_rng([cohort_seed, q, crc32(type)]). The search uses only e and the cohort.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

from .io import align, budget_dir, load_npz, pool_indices, save_npz, set_dir, write_json
from .support import kappa, load_features, removal_stats

TYPES = ("rand", "sem", "vis", "hv", "near", "far")
N_PER_COHORT = 4
N_SIB = 16


# --- cohorts and class sets ------------------------------------------------------------------

def cohorts(assign, Q: int, seed: int, pct_: int, n_per: int = N_PER_COHORT) -> list:
    """Q cohorts of n_per classes, each from a different superclass; no class in two cohorts. Sequential draw in
    the order of a random permutation of the remaining classes, rejecting a class whose superclass is taken."""
    assign = np.asarray(assign, np.int64)
    rng = np.random.default_rng([int(seed), int(pct_)])
    avail = np.arange(len(assign))
    out = []
    for q in range(Q):
        chosen, sup = [], set()
        for c in rng.permutation(avail):
            if int(assign[c]) not in sup:
                chosen.append(int(c))
                sup.add(int(assign[c]))
            if len(chosen) == n_per:
                break
        if len(chosen) < n_per:
            raise SystemExit(f"deletion: cannot draw cohort {q} with {n_per} classes from different superclasses")
        out.append(sorted(chosen))
        avail = np.setdiff1d(avail, chosen)
    return out


def siblings(cohort, assign) -> list:
    """Y_sib: classes outside the cohort that share a curated superclass with a cohort class."""
    assign = np.asarray(assign, np.int64)
    groups = {int(assign[c]) for c in cohort}
    return [int(c) for c in range(len(assign)) if c not in cohort and int(assign[c]) in groups]


def vis_classes(cohort, sib, class_means, k: int = N_SIB) -> list:
    """The k classes outside cohort and Y_sib with the largest max cosine of class means to a cohort class mean
    (ties: smaller class index)."""
    from sg.partitions import cosine_matrix

    S = cosine_matrix(class_means)
    score = S[:, list(cohort)].max(1)
    excl = set(cohort) | set(sib)
    elig = [c for c in range(len(score)) if c not in excl]
    elig.sort(key=lambda c: (-score[c], c))
    if len(elig) < k:
        raise SystemExit(f"deletion: only {len(elig)} classes available for the visual-neighbour set")
    return sorted(elig[:k])


def removal_size(eta: float, n: int) -> int:
    R = int(round(float(eta) * n))
    if R <= 0 or R % N_SIB:
        raise SystemExit(f"deletion: |R| = round(eta * n_beta) = {R} must be a positive multiple of {N_SIB}")
    return R


def type_rng(seed: int, q: int, typ: str):
    return np.random.default_rng([int(seed), int(q), zlib.crc32(typ.encode())])


# --- Delta V on candidate lists ----------------------------------------------------------------

class DVIndex:
    """V_i(D minus R) for query records i (pool positions) from candidate lists: the top n_vnn + n_extra
    neighbours of i in D minus {i}. Removing at most n_extra records leaves the top n_vnn of D minus R minus {i}
    inside the list, so V after removal is exact."""

    def __init__(self, e, query, n_vnn: int, n_extra: int, kernel_t: float, device="cpu", chunk: int = 1024):
        e = torch.as_tensor(np.asarray(e), dtype=torch.float32)
        self.n = e.shape[0]
        self.query = np.asarray(query, np.int64)
        self.n_vnn = int(n_vnn)
        L = min(int(n_vnn) + int(n_extra), self.n - 1)
        E = e.to(device)
        idx, val = [], []
        for s in range(0, len(self.query), chunk):
            qi = torch.as_tensor(self.query[s:s + chunk], device=device)
            cos = E[qi] @ E.T
            cos[torch.arange(len(qi), device=device), qi] = float("-inf")
            v, i = cos.topk(L, dim=1)
            idx.append(i.cpu().numpy())
            val.append(v.double().cpu().numpy())
        self.cand = np.concatenate(idx) if idx else np.zeros((0, L), np.int64)
        self.kap = kappa(np.concatenate(val), kernel_t) if val else np.zeros((0, L))
        self.v0 = self.v_rows(np.arange(len(self.query)), np.zeros(self.n, bool))
        flat = self.cand.ravel()
        order = np.argsort(flat, kind="stable")
        self._inv_rows = (order // max(L, 1)).astype(np.int64)
        self._inv_start = np.searchsorted(flat[order], np.arange(self.n + 1))

    def v_rows(self, rows, removed) -> np.ndarray:
        """V of the given query rows with the records ``removed`` (bool over pool positions) taken out."""
        k = min(self.n_vnn, self.n - int(removed.sum()) - 1)
        if k <= 0:
            return np.full(len(rows), np.nan)
        keep = ~removed[self.cand[rows]]
        take = keep & (np.cumsum(keep, 1) <= k)
        return (self.kap[rows] * take).sum(1) / k

    def rows_with(self, p) -> np.ndarray:
        return self._inv_rows[self._inv_start[p]:self._inv_start[p + 1]]

    def mean_dv(self, removed) -> float:
        return float(np.mean(self.v_rows(np.arange(len(self.query)), removed) - self.v0))


def match_dv(index: DVIndex, removed, cls_members, target: float, tol: float, max_swaps: int, rng) -> dict:
    """Swap search on ``removed`` (bool over pool positions, modified in place): propose a removed <-> retained swap
    inside a random class of ``cls_members`` (list of position arrays) and accept iff |mean dV - target| decreases.
    Stops when the gap < tol |target| or after max_swaps proposals. Per-class counts never change."""
    nq = len(index.query)
    allrows = np.arange(nq)
    V = index.v_rows(allrows, removed)
    cur = float(np.mean(V - index.v0))
    gap0 = gap = abs(cur - target)
    thr = float(tol) * abs(float(target))
    props = acc = 0
    while gap >= thr and props < int(max_swaps):
        props += 1
        mem = cls_members[int(rng.integers(len(cls_members)))]
        rem, ret = mem[removed[mem]], mem[~removed[mem]]
        if not len(rem) or not len(ret):
            continue
        a, b = int(rem[rng.integers(len(rem))]), int(ret[rng.integers(len(ret))])
        aff = np.union1d(index.rows_with(a), index.rows_with(b))
        if not len(aff):
            continue
        removed[a], removed[b] = False, True
        v_new = index.v_rows(aff, removed)
        new = cur + float((v_new - V[aff]).sum()) / nq
        if abs(new - target) < gap:
            V[aff] = v_new
            cur, gap = new, abs(new - target)
            acc += 1
        else:
            removed[a], removed[b] = True, False
    return {"target": float(target), "mean_dV": cur, "gap_initial": gap0, "gap": gap, "tolerance": thr,
            "reached": bool(gap < thr), "proposals": props, "accepted": acc}


# --- removal sets ------------------------------------------------------------------------------

def _per_class(rng, members, k) -> np.ndarray:
    return np.concatenate([rng.choice(m, k, replace=False) for m in members])


def make_removals(y, e, assign, class_means, cohort, q: int, R_size: int, lbar0, kernel_t: float, n_vnn: int,
                  dv_tol: float, dv_max_swaps: int, seed: int, device="cpu") -> dict:
    """Removal sets for one cohort on a pool given by positions 0..n-1 (y, e aligned; lbar0 = Lbar^0 per position,
    NaN -> lowest). Returns {type: positions (sorted)}, the class lists, the cohort positions and the matching
    report for vis."""
    y = np.asarray(y, np.int64)
    n = len(y)
    sib = siblings(cohort, assign)
    vis = vis_classes(cohort, sib, class_means)
    if len(sib) != N_SIB:
        raise SystemExit(f"deletion: cohort {q} has {len(sib)} curated siblings, expected {N_SIB}")
    per = R_size // N_SIB
    T = np.flatnonzero(np.isin(y, cohort))
    nonT = np.flatnonzero(~np.isin(y, cohort))
    members = {c: np.flatnonzero(y == c) for c in set(sib) | set(vis)}
    for c, m in members.items():
        if len(m) < per:
            raise SystemExit(f"deletion: class {c} has {len(m)} pool records, fewer than |R|/16 = {per}")
    out = {}
    out["rand"] = type_rng(seed, q, "rand").choice(nonT, R_size, replace=False)
    out["sem"] = _per_class(type_rng(seed, q, "sem"), [members[c] for c in sib], per)
    # vis: uniform start, then Delta-V matching to mean Delta V(sem) over T_q
    index = DVIndex(e, T, n_vnn, R_size, kernel_t, device)
    rem_sem = np.zeros(n, bool)
    rem_sem[out["sem"]] = True
    target = index.mean_dv(rem_sem)
    rng = type_rng(seed, q, "vis")
    removed = np.zeros(n, bool)
    removed[_per_class(rng, [members[c] for c in vis], per)] = True
    match = match_dv(index, removed, [members[c] for c in vis], target, dv_tol, dv_max_swaps, rng)
    out["vis"] = np.flatnonzero(removed)
    # hv: top |R| of Lbar^0 outside T_q, ties by the smaller row (= position, the pool is sorted)
    lb = np.where(np.isfinite(lbar0), np.asarray(lbar0, np.float64), -np.inf)[nonT]
    out["hv"] = nonT[np.lexsort((nonT, -lb))[:R_size]]
    # near / far: per sibling class the |R|/16 records with the highest / lowest mean kappa to T_q
    Et = torch.as_tensor(np.asarray(e)[T], dtype=torch.float32)
    near, far = [], []
    for c in sib:
        m = members[c]
        mk = kappa(torch.as_tensor(np.asarray(e)[m], dtype=torch.float32) @ Et.T, kernel_t).double().mean(1).numpy()
        near.append(m[np.lexsort((m, -mk))[:per]])
        far.append(m[np.lexsort((m, mk))[:per]])
    out["near"], out["far"] = np.concatenate(near), np.concatenate(far)
    return {"R": {t: np.sort(v) for t, v in out.items()}, "siblings": sib, "vis_classes": vis, "T": T,
            "match": match, "dv_index": index}


# --- ``privred deletion_sets`` -------------------------------------------------------------------

def _quantiles(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {f"q{int(p * 100):02d}": float(np.quantile(x, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}


def build_deletion_sets(cfg, budget=None) -> dict:
    from sg.data import load_dataset
    from sg.partitions import true_partition
    from sg.plans import pct
    from sg.utils import pick_device

    b = float(cfg.studyB_budget if budget is None else budget)
    if cfg.studyB_budget is None or abs(b - float(cfg.studyB_budget)) > 1e-9:
        raise SystemExit(f"deletion_sets: Study B runs only at studyB_budget={cfg.studyB_budget}")
    ds = load_dataset(cfg)
    assign = true_partition(ds.coarse_of_fine)
    ids = pool_indices(cfg, ds, b)
    y = ds.y_train[ids]
    F = load_features(cfg)
    pos = align(F["ids"], ids)
    e = F["e"][pos]
    pre = set_dir(cfg, b, "prelim") / "scores.npz"
    if not pre.exists():
        raise SystemExit(f"deletion_sets: {pre} missing; plan, launch and score the prelim set first")
    P = load_npz(pre)
    lbar0 = P["Lbar_in"][align(P["ids"], ids)].astype(np.float64)
    bdir = budget_dir(cfg, b)
    d0 = None
    if (bdir / "difficulty.npz").exists():
        dz = load_npz(bdir / "difficulty.npz")
        d0 = dz["d0"][align(dz["ids"], ids)]
    R_size = removal_size(cfg.eta, len(ids))
    device = pick_device(cfg.device)
    coh = cohorts(assign, cfg.Q, cfg.cohort_seed, pct(b))
    out_dir = bdir / "deletion"
    # once any deletion set exists, removal sets are immutable: a rewrite must reproduce R.npy exactly
    trained = sorted(p.parent.name for p in (bdir / "sets").glob("del_*/manifest.json"))
    report = {"budget": b, "n_pool": int(len(ids)), "R_size": R_size, "per_class": R_size // N_SIB, "types": TYPES,
              "cohort_seed": cfg.cohort_seed, "cohorts": []}
    kw = dict(kernel_t=cfg.kernel_t, eps=cfg.support_eps, n_vnn=cfg.n_vnn, device=device)
    # pass 1: every removal set, validated and checked against the guard before anything is written
    built = []
    for q, cl in enumerate(coh):
        res = make_removals(y, e, assign, F["class_means"], cl, q, R_size, lbar0, cfg.kernel_t, cfg.n_vnn,
                            cfg.dv_tol, cfg.dv_max_swaps, cfg.cohort_seed, device)
        T = res["T"]
        for t in TYPES:
            Rp = res["R"][t]
            if len(Rp) != R_size or np.intersect1d(Rp, T).size or len(np.unique(Rp)) != R_size:
                raise SystemExit(f"deletion: q{q} {t}: |R| = {len(Rp)} or overlap with T_q")
            d = out_dir / f"q{q}_{t}"
            if trained and (d / "R.npy").exists() and not np.array_equal(np.load(d / "R.npy"), ids[Rp]):
                raise SystemExit(f"deletion_sets: {d / 'R.npy'} would change but deletion sets already exist "
                                 f"({', '.join(trained[:3])}{', ...' if len(trained) > 3 else ''}); delete them "
                                 f"under {bdir / 'sets'} or restore the inputs (prelim scores, features) first")
        built.append((q, cl, res))
    # pass 2: write R.npy, deltas.npz and composition.json
    for q, cl, res in built:
        T = res["T"]
        crow = {"q": q, "classes": cl, "superclasses": [int(assign[c]) for c in cl], "siblings": res["siblings"],
                "vis_classes": res["vis_classes"], "n_T": int(len(T)), "match_vis": res["match"], "sets": {}}
        Ek = torch.as_tensor(e[T], dtype=torch.float32)
        for t in TYPES:
            Rp = res["R"][t]
            keep = np.setdiff1d(np.arange(len(ids)), Rp)
            st = removal_stats(F["e"], F["y"], assign, pos[keep], pos, pos[Rp], **kw)
            inT = np.isin(y[keep], cl)
            d = out_dir / f"q{q}_{t}"
            new_R = ids[Rp]
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "R.npy", new_R)
            save_npz(d / "deltas.npz", ids=ids[keep], in_T=inT, **st)
            kap_T = kappa(torch.as_tensor(e[Rp], dtype=torch.float32) @ Ek.T, cfg.kernel_t).double().mean(1).numpy()
            yr = y[Rp]
            comp = {"q": q, "type": t, "R_size": int(len(Rp)), "classes": cl,
                    "class_counts": {int(c): int(k) for c, k in zip(*np.unique(yr, return_counts=True))},
                    "parent_counts": {int(g): int(k) for g, k in zip(*np.unique(assign[yr], return_counts=True))},
                    "n_curated_siblings_removed": int(np.isin(yr, res["siblings"]).sum()),
                    "mean_kappa_to_cohort": float(kap_T.mean()),
                    "d0_removed": None if d0 is None else _quantiles(d0[Rp]),
                    "retained_per_class": np.bincount(y[keep], minlength=ds.n_classes).tolist(),
                    "mean_dK_T": float(st["dK"][inT].mean()), "mean_dV_T": float(st["dV"][inT].mean()),
                    "mean_dU_T": float(st["dU"][inT].mean()), "mean_dN_T": float(st["dN"][inT].mean()),
                    "mean_sib_frac_T": float(st["sib_frac"][inT].mean()),
                    "match": res["match"] if t == "vis" else None}
            write_json(d / "composition.json", comp)
            crow["sets"][t] = {k: comp[k] for k in ("mean_dK_T", "mean_dV_T", "mean_kappa_to_cohort",
                                                    "n_curated_siblings_removed")}
        report["cohorts"].append(crow)
    write_json(out_dir / "cohorts.json", report)
    report["out"] = str(out_dir)
    return report
