"""Membership design (Section 3.5): membership matrices u in {0,1}^{|D| x M} with exactly M/2 IN models per record,
training sets D_m, and the seed registry (seeds disjoint across all model sets, budgets and datasets)."""

from __future__ import annotations

import zlib

import numpy as np
from sg.config import DATASETS
from sg.data import BUDGETS

SET_STRIDE = 1000         # seeds of one set: seed_base + m, m < M <= SET_STRIDE
BUDGET_STRIDE = 100_000   # at most 100 sets per budget
DATASET_STRIDE = 1_000_000


def name_seed(name: str) -> int:
    return zlib.crc32(name.encode())


def membership_matrix(n: int, M: int, name: str, seed: int = 0) -> np.ndarray:
    """bool [n, M]; row i has exactly M/2 ones at a uniformly random subset of the M models, independent across
    records. rng = default_rng([membership_seed, crc32(name)]); the M/2 IN models of a row are the first M/2
    entries of a uniformly random permutation (argsort of iid uniforms)."""
    if M % 2:
        raise ValueError(f"M must be even (got {M})")
    rng = np.random.default_rng([int(seed), name_seed(name)])
    perm = np.argsort(rng.random((n, M)), axis=1, kind="stable")
    u = np.zeros((n, M), bool)
    np.put_along_axis(u, perm[:, : M // 2], True, axis=1)
    return u


def train_rows(pool_ids, u, m) -> np.ndarray:
    """D_m = {i in D : u_im = 1} as rows of ds.x_train (sorted)."""
    return np.sort(np.asarray(pool_ids)[np.asarray(u)[:, m]])


def seed_base(cfg, budget, name: str) -> int:
    """seed_base = 1000 * (position of the set in plans.all_set_names(cfg, beta)) + 100000 * (index of beta in
    sg.data.BUDGETS) + 1000000 * (index of the dataset in sg.config.DATASETS). Model m uses seed_base + m."""
    from .plans import all_set_names

    names = all_set_names(cfg, budget)
    if name not in names:
        raise KeyError(f"seed_base: set {name!r} is not registered for budget {budget}")
    if len(names) * SET_STRIDE > BUDGET_STRIDE:
        raise ValueError(f"too many sets ({len(names)}) for the seed registry")
    b = [float(x) for x in BUDGETS].index(float(budget))
    return SET_STRIDE * names.index(name) + BUDGET_STRIDE * b + DATASET_STRIDE * DATASETS.index(cfg.dataset)
