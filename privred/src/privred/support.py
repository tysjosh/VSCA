"""Semantic neighbourhood support (Section 3.8) on the analysis features e_i = phi-bar(x_i) / ||phi-bar(x_i)||.

``privred features`` writes <out>/<dataset>/features.npz (ids, y, e, class_means) for the rows of every privacy
budget's pool D_beta and validation set V_beta and the audit pool A; phi-bar = companion simclr/b100 (trained on
exactly D_1.0 = P). ``privred support --budget B`` writes <budget dir>/support.npz for G_true on D_beta.

kappa(e, e') = exp((cos(e, e') - 1) / t). For a query record i and a pool D (positions into e):
  N_i  = #{j in D: j != i, G(y_j) = G(y_i), y_j != y_i}                     (Eq. siblings)
  U_i  = sum_{j in N_i} kappa(e_i, e_j) / (N_i + eps)                        (Eq. support)
  K_i  = sum_{j in N_i} kappa(e_i, e_j)                                      (Eq. mass)
  samecls_i = mean kappa to the other records of class y_i in D
  V_i  = mean kappa over the n_vnn records of D \\ {i} nearest in cosine     (Eq. vdens)
Deltas on D \\ R are recomputed, not updated; Delta K_i = -sum_{j in R cap N_i} kappa is a sanity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import align, audit_indices, budget_dir, load_npz, pool_indices, save_npz, val_indices

KEYS = ("N", "U", "K", "samecls", "V")


def kappa(cos, t: float):
    """kappa = exp((cos - 1) / t), works on numpy arrays and torch tensors."""
    return (torch.exp if isinstance(cos, torch.Tensor) else np.exp)((cos - 1.0) / t)


def features_path(cfg) -> Path:
    return Path(cfg.out_dir) / cfg.dataset / "features.npz"


# --- feature extraction ----------------------------------------------------------------------

def extract_features(cfg) -> dict:
    """phi-bar features of pool, validation and audit rows (L2-normalized) -> features.npz."""
    from sg.data import load_dataset
    from sg.simclr import extract_features as sg_extract
    from sg.simclr import load_backbone
    from sg.utils import ComputeLog, pick_device

    from .config import simclr_dir

    ext = simclr_dir(cfg, 1.0)
    if not (ext / "backbone.pt").exists():
        raise SystemExit(f"features: {ext}/backbone.pt missing (companion simclr at budget 1.0 is phi-bar)")
    ds = load_dataset(cfg)
    rows = [audit_indices(cfg, ds)]
    for b in cfg.privacy_budgets:
        rows += [pool_indices(cfg, ds, b), val_indices(cfg, ds, b)]
    ids = np.unique(np.concatenate(rows)).astype(np.int64)
    out = features_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    clog = ComputeLog(out.parent / "features_compute.json")
    with clog.phase("features"):
        model = load_backbone(ext, device)
        f = sg_extract(model, ds.x_train[ids], device, cfg.eval_batch_size, ds.mean, ds.std).astype(np.float64)
        e = (f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    clog.save()
    cm = np.load(ext / "class_means.npy")
    save_npz(out, ids=ids, y=ds.y_train[ids], e=e, class_means=cm)
    return {"out": str(out), "n": int(len(ids)), "dim": int(e.shape[1]), "extractor": str(ext)}


def load_features(cfg) -> dict:
    p = features_path(cfg)
    if not p.exists():
        raise SystemExit(f"{p} missing; run `privred features` first")
    return load_npz(p)


# --- support descriptors (vectorized) --------------------------------------------------------

def support(e, y, assign, query, ref, kernel_t: float, eps: float = 1e-6, n_vnn: int = 50, device="cpu",
            chunk: int = 1024) -> dict:
    """N, U, K, samecls, V of the records at positions ``query`` with respect to the pool at positions ``ref``
    (a query that is itself in ``ref`` is excluded from its own neighbourhood). Arrays [len(query)], float64
    (N int64); samecls is NaN without another same-class record, V averages over min(n_vnn, |D \\ {i}|)."""
    e = torch.as_tensor(np.asarray(e), dtype=torch.float32)
    y = np.asarray(y, np.int64)
    g = np.asarray(assign, np.int64)[y]
    query = np.asarray(query, np.int64)
    ref = np.asarray(ref, np.int64)
    E_ref = e[ref].to(device)
    y_ref = torch.as_tensor(y[ref], device=device)
    g_ref = torch.as_tensor(g[ref], device=device)
    # column of the query inside ref (or -1): self-exclusion by position, not by value
    self_col = np.full(len(query), -1, np.int64)
    if len(ref):
        order = np.argsort(ref, kind="stable")
        p = np.clip(np.searchsorted(ref[order], query), 0, len(ref) - 1)
        hit = ref[order][p] == query
        self_col[hit] = order[p][hit]
    k_nn = min(int(n_vnn), len(ref))
    out = {k: np.zeros(len(query), np.float64) for k in KEYS}
    out["N"] = np.zeros(len(query), np.int64)
    for s in range(0, len(query), chunk):
        qi = query[s:s + chunk]
        B = len(qi)
        cos = e[qi].to(device) @ E_ref.T                                       # [B, |D|]
        kap = kappa(cos, kernel_t).double()
        sc = torch.as_tensor(self_col[s:s + chunk], device=device)
        notself = torch.ones_like(cos, dtype=torch.bool)
        has = sc >= 0
        if has.any():
            notself[torch.nonzero(has).squeeze(1), sc[has]] = False
        yq = torch.as_tensor(y[qi], device=device)[:, None]
        gq = torch.as_tensor(g[qi], device=device)[:, None]
        sib = (g_ref[None, :] == gq) & (y_ref[None, :] != yq)
        same = (y_ref[None, :] == yq) & notself
        K = (kap * sib).sum(1)
        N = sib.sum(1)
        ns = same.sum(1)
        out["K"][s:s + B] = K.cpu().numpy()
        out["N"][s:s + B] = N.cpu().numpy()
        out["U"][s:s + B] = (K / (N.double() + eps)).cpu().numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out["samecls"][s:s + B] = ((kap * same).sum(1).cpu().numpy() / ns.cpu().numpy())
        if k_nn == 0:
            out["V"][s:s + B] = np.nan
            continue
        top = cos.masked_fill(~notself, float("-inf")).topk(k_nn, dim=1).values  # self (if any) sorts last
        fin = torch.isfinite(top)
        nv = fin.sum(1)
        v = torch.where(fin, kappa(top, kernel_t), torch.zeros_like(top)).double().sum(1) / nv.clamp_min(1)
        out["V"][s:s + B] = torch.where(nv > 0, v, torch.full_like(v, float("nan"))).cpu().numpy()
    return out


def removal_stats(e, y, assign, query, ref, removed, kernel_t: float, eps: float = 1e-6, n_vnn: int = 50,
                  device="cpu", chunk: int = 1024) -> dict:
    """Support on D and on D \\ R for the query records (which must not be in R), the deltas dN, dU, dK, dV,
    dsamecls by recomputation, sib_frac = |R cap N_i(D)| / |R| and the removed sibling mass sum_{R cap N_i} kappa."""
    ref = np.asarray(ref, np.int64)
    removed = np.asarray(removed, np.int64)
    if np.intersect1d(query, removed).size:
        raise ValueError("removal_stats: query records must be retained (not in R)")
    kw = dict(kernel_t=kernel_t, eps=eps, n_vnn=n_vnn, device=device, chunk=chunk)
    before = support(e, y, assign, query, ref, **kw)
    after = support(e, y, assign, query, np.setdiff1d(ref, removed), **kw)
    inR = support(e, y, assign, query, np.intersect1d(ref, removed), **kw)
    out = {f"{k}_before": before[k] for k in KEYS} | {f"{k}_after": after[k] for k in KEYS}
    for k in KEYS:
        out[f"d{k}"] = after[k].astype(np.float64) - before[k].astype(np.float64)
    out["sib_frac"] = inR["N"] / max(len(np.intersect1d(ref, removed)), 1)
    out["removed_sib_mass"] = inR["K"]
    return out


# --- brute-force references (sanity check only) -----------------------------------------------

def support_bruteforce(e, y, assign, query, ref, kernel_t: float, eps: float = 1e-6, n_vnn: int = 50) -> dict:
    """Loop version of ``support`` in float64."""
    e = np.asarray(e, np.float64)
    y = np.asarray(y, np.int64)
    a = np.asarray(assign, np.int64)
    out = {k: [] for k in KEYS}
    for i in query:
        D = [j for j in ref if j != i]
        cos = np.array([e[i] @ e[j] for j in D])
        kap = np.exp((cos - 1.0) / kernel_t)
        sib = np.array([a[y[j]] == a[y[i]] and y[j] != y[i] for j in D], bool)
        same = np.array([y[j] == y[i] for j in D], bool)
        N = int(sib.sum())
        K = float(kap[sib].sum())
        out["N"].append(N)
        out["K"].append(K)
        out["U"].append(K / (N + eps))
        out["samecls"].append(float(kap[same].mean()) if same.any() else np.nan)
        nn = np.argsort(-cos, kind="stable")[:n_vnn]
        out["V"].append(float(kap[nn].mean()) if len(nn) else np.nan)
    return {k: np.asarray(v, np.int64 if k == "N" else np.float64) for k, v in out.items()}


# --- ``privred support`` ----------------------------------------------------------------------

def compute_support(cfg, budget) -> dict:
    """Support descriptors for G_true on D_beta -> <budget dir>/support.npz (ids, y, N, U, K, samecls, V)."""
    from sg.data import load_dataset
    from sg.partitions import true_partition
    from sg.utils import pick_device

    F = load_features(cfg)
    ds = load_dataset(cfg)
    pool = pool_indices(cfg, ds, budget)
    pos = align(F["ids"], pool)
    assign = true_partition(ds.coarse_of_fine)
    s = support(F["e"], F["y"], assign, pos, pos, cfg.kernel_t, cfg.support_eps, cfg.n_vnn,
                device=pick_device(cfg.device))
    out = budget_dir(cfg, budget) / "support.npz"
    save_npz(out, ids=pool, y=ds.y_train[pool], **s)
    return {"out": str(out), "n": int(len(pool)), "partition": "true",
            **{f"mean_{k}": float(np.nanmean(s[k])) for k in KEYS}}
