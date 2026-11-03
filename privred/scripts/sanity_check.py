"""Identity and unit checks on synthetic data (no CIFAR, no companion runs needed).
Run: python scripts/sanity_check.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from privred import lira
from privred.config import load_config
from privred.deletion import TYPES as DEL_TYPES_ALL
from privred.deletion import cohorts as del_cohorts
from privred.deletion import make_removals, removal_size
from privred.prediction import SPLITS, class_folds, fit_ridge, make_splits, precision_top
from privred.prediction import _prep as pred_prep
from privred.prediction import metrics as pred_metrics
from privred.stats import holm_family, sign_flip_p
from privred.depth import exclude_self, pd_from_feats
from privred.difficulty import oof_folds
from privred.membership import membership_matrix, seed_base
from privred.plans import all_set_names, count_runs, subset_partitions
from privred.stats import residual_cov
from privred.studyA import pattern_codes, pattern_fractions
from privred.support import removal_stats, support, support_bruteforce
from privred.vulnerability import mem_scores
from sg.data import BUDGETS
from sg.stats import holm as sg_holm
from sg.stats import residual as sg_residual
from sg.train import _example_stats

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


def brute_fits(Q, U, per_record):
    """Loop version of the LOMO Gaussian fits (ddof=1)."""
    n, M = Q.shape
    mu = {s: np.zeros((n, M)) for s in ("in", "out")}
    var = {s: np.full((n, M), np.nan) for s in ("in", "out")}
    for i in range(n):
        for m in range(M):
            for s, side in (("in", U[i]), ("out", ~U[i])):
                ref = [j for j in range(M) if side[j] and j != m]
                mu[s][i, m] = np.mean(Q[i, ref])
                if len(ref) >= 2:
                    var[s][i, m] = np.var(Q[i, ref], ddof=1)
    if not per_record:
        for s in ("in", "out"):
            pooled = np.nanmean(var[s], axis=0)
            var[s] = np.broadcast_to(pooled, (n, M)).copy()
    return mu, var


# --- LiRA (Eq. lira) ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
for M, per in ((32, True), (16, False)):
    n = 12
    U = membership_matrix(n, M, f"toy{M}", 0)
    Q = rng.normal(size=(n, M)) + 2.0 * U
    f = lira.lomo_fits(Q, U, 15)
    mu, var = brute_fits(Q, U, per)
    L = lira.lira_scores(Q, f)
    closed = norm.logpdf(Q, mu["in"], np.sqrt(var["in"])) - norm.logpdf(Q, mu["out"], np.sqrt(var["out"]))
    check(f"(lira) M={M}: LOMO Gaussian fits equal brute force", np.allclose(f["mu_in"], mu["in"]) and
          np.allclose(f["var_in"], var["in"]) and np.allclose(f["mu_out"], mu["out"]) and np.allclose(f["var_out"], var["out"]))
    check(f"(lira) M={M}: Lambda equals the closed-form Gaussian log-likelihood ratio", np.allclose(L, closed))
    check(f"(lira) M={M}: variance rule -> {'per-record' if per else 'pooled'}", f["per_record"] is per)
    m = 3
    Q2 = Q.copy()
    Q2[:, m] += rng.normal(size=n) * 5
    f2 = lira.lomo_fits(Q2, U, 15)
    check(f"(lira) M={M}: LOMO never uses target m's own model (perturbing Q[:, m] leaves target-m fits unchanged)",
          all(np.allclose(f[k][:, m], f2[k][:, m]) for k in ("mu_in", "var_in", "mu_out", "var_out")))

# --- membership ----------------------------------------------------------------------------------
U = membership_matrix(4000, 32, "A_ce", 0)
col = U.sum(0)
check("(membership) every row has exactly M/2 IN models", (U.sum(1) == 16).all())
check("(membership) every column has about |D|/2 records (within 5 sd)", (np.abs(col - 2000) < 5 * np.sqrt(1000)).all())
check("(membership) deterministic per (seed, name), different across names",
      (membership_matrix(50, 8, "x", 0) == membership_matrix(50, 8, "x", 0)).all()
      and not (membership_matrix(50, 8, "x", 0) == membership_matrix(50, 8, "y", 0)).all())

# --- TPR at fixed FPR, AUC, r_i ------------------------------------------------------------------
non = np.arange(10.0)
mem = np.array([5.5, 8.5, 9.5, 10.0])
L = np.concatenate([non, mem])[None, :]
Um = np.concatenate([np.zeros(10, bool), np.ones(4, bool)])[None, :]
res = lira.attack_metrics(L, Um, [0.2], min_expected_fp=2)
a = res["alphas"]["0.2"]
check("(attack) hand case: tau = 7, FPR = 0.2, TPR = 0.75", a["tau"] == 7.0 and a["fpr"] == 0.2 and a["tpr"] == 0.75)
check("(attack) reportable iff alpha * #nonmember pairs >= min_expected_fp",
      a["reportable"] and not lira.attack_metrics(L, Um, [0.1], min_expected_fp=2)["alphas"]["0.1"]["reportable"])
check("(attack) global AUC hand case", abs(res["auc"] - 35 / 40) < 1e-12)
Lr = np.array([[3.0, 1.0, 2.0, 2.0], [0.0, 5.0, 1.0, 6.0]])
Ur = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], bool)
check("(vuln) r_i = fraction of IN models above tau", np.allclose(lira.vulnerability(Lr, Ur, 1.5), [1.0, 0.0]))
check("(vuln) per-example AUC hand case with a tie counted 1/2", np.allclose(lira.per_example_auc(Lr, Ur), [0.875, 0.0]))

# --- Mem (Eq. mem) ---------------------------------------------------------------------------------
P = np.array([[0.9, 0.1, 0.7, 0.3], [0.5, 0.5, 0.5, 0.5]])
check("(mem) hand case: mean IN p_true - mean OUT p_true", np.allclose(mem_scores(P, Ur), [0.6, 0.0]))

# --- sg.train._example_stats (Eq. q) --------------------------------------------------------------
z0 = torch.randn(64, 100) * 4
z1 = torch.randn(64, 100) * 4
yy = np.random.default_rng(1).integers(0, 100, 64)
st = _example_stats(np.arange(64), yy, z0, z1, keep_logits=True)
p = torch.softmax(torch.stack([z0, z1], 1).double(), 2).numpy()[np.arange(64), :, yy]
check("(q) sg.train._example_stats q == log(p / (1 - p))", st["q"].shape == (64, 2) and
      np.allclose(st["q"], np.log(p / (1 - p)), atol=1e-4))
check("(q) _example_stats loss, p_true, margin, conf, pred, float16 logits",
      np.allclose(st["p_true"], p, atol=1e-6) and np.allclose(st["loss"], -np.log(p), atol=1e-4)
      and st["pred"].dtype == np.int16 and (st["pred"][:, 0] == z0.argmax(1).numpy()).all()
      and np.allclose(st["conf"][:, 0], torch.softmax(z0, 1).max(1).values.numpy(), atol=1e-6)
      and (st["margin"][:, 0] >= 0).tolist() == (st["pred"][:, 0] == yy).tolist()
      and st["logits"].dtype == np.float16 and st["logits"].shape == (64, 2, 100))
check("(q) lira.q_from_logits agrees with _example_stats", np.allclose(lira.q_from_logits(z0.numpy(), yy), st["q"][:, 0], atol=1e-4))

# --- PD self-exclusion --------------------------------------------------------------------------
ref_ids = np.array([10, 11, 12, 13])
idx = torch.tensor([[1, 0, 2], [3, 2, 1]])
sim = torch.tensor([[1.0, 0.9, 0.8], [0.7, 0.6, 0.5]])
ki, ks = exclude_self(idx, sim, ref_ids, np.array([11, 99]), 2)
check("(pd) self-exclusion drops the query's own row, else the (k+1)-th neighbour",
      ki.tolist() == [[0, 2], [3, 2]] and np.allclose(ks.numpy(), [[0.9, 0.8], [0.7, 0.6]]))
feat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]]), dim=1)
ry = np.array([0, 1, 1, 2, 2])
q_ids = np.array([100, 101])
pd, pur = pd_from_feats([feat] * 8, ry, np.arange(200, 205), [feat[:1]] * 8, torch.tensor([1]), q_ids[:1], [0], 1, 3)
check("(pd) toy query outside the reference set keeps its nearest neighbour (PD 9 when it disagrees with the head)",
      pd.tolist() == [9])
pd, pur = pd_from_feats([feat] * 8, ry, np.array([100, 101, 102, 103, 104]), [feat[:1]] * 8, torch.tensor([1]),
                        np.array([100]), [0], 1, 3)
check("(pd) toy query in the reference set never votes for itself (PD 1, block-8 purity 0)",
      pd.tolist() == [1] and pur.tolist() == [0.0])

# --- OOF folds -----------------------------------------------------------------------------------
yf = np.repeat(np.arange(100), 40)
np.random.default_rng(2).shuffle(yf)
fold = oof_folds(yf, 5, 0)
check("(oof) folds partition D (every record in exactly one fold)", ((fold >= 0) & (fold < 5)).all() and len(fold) == 4000)
check("(oof) folds are class-stratified (8 per class per fold)",
      all((np.bincount(yf[fold == k], minlength=100) == 8).all() for k in range(5)))

# --- seed registry -------------------------------------------------------------------------------
seeds, n_sets = [], 0
for cfg_paths in (["configs/base.yaml"], ["configs/base.yaml", "configs/inat100.yaml"]):
    c = load_config([str(ROOT / p) for p in cfg_paths], [])
    for b in BUDGETS:
        names = all_set_names(c, b)
        for nm in names:
            M = c.K_folds if nm == "oof" else c.M if nm.startswith("A_") else c.M0 if nm == "prelim" else c.MB
            s0 = seed_base(c, b, nm)
            seeds += list(range(s0, s0 + M))
            n_sets += 1
check(f"(seeds) disjoint across all {n_sets} sets, every budget and both datasets", len(seeds) == len(set(seeds)))
check("(seeds) set names unique within a budget", len(set(all_set_names(c, 1.0))) == len(all_set_names(c, 1.0)))

# --- compute count (Table compute) -----------------------------------------------------------------
base = load_config([str(ROOT / "configs/base.yaml")], [])
cr = count_runs(base)
b1 = cr["budgets"]["1.0"]
comp = {k: (v["runs"], round(v["gpu_h"])) for k, v in b1["components"].items()}
check("(compute) count_runs(base) at beta=100%: 949 runs, round(GPU-h) = 326", b1["runs"] == 949 and round(b1["gpu_h"]) == 326)
check("(compute) components 384/134, 16/6, 5/3, 32/11, 384/129, 128/43",
      comp == {"studyA": (384, 134), "prelim": (16, 6), "oof": (5, 3), "baseB": (32, 11), "del_ce": (384, 129),
               "del_true_hier": (128, 43)})
check("(compute) Study A at beta=50% about 67 GPU-h", round(cr["budgets"]["0.5"]["components"]["studyA"]["gpu_h"]) == 67)
inat = count_runs(load_config([str(ROOT / "configs/base.yaml"), str(ROOT / "configs/inat100.yaml")], []))
check("(compute) iNat-100 Study A: 160 runs, no Study B",
      inat["budgets"]["1.0"]["components"]["studyA"]["runs"] == 160 and set(inat["budgets"]["1.0"]["components"]) == {"oof", "studyA"})

# --- support features (Eqs. siblings, support, mass, vdens) ---------------------------------------
rs = np.random.default_rng(3)
nf, C = 80, 10
ef = rs.normal(size=(nf, 6))
ef /= np.linalg.norm(ef, axis=1, keepdims=True)
ycl = np.arange(nf) % C
grp = np.repeat(np.arange(5), 2)            # 10 classes in 5 groups of 2
inref = rs.random(nf) < 0.8
ref, qry = np.flatnonzero(inref), np.arange(nf)  # some queries are in the pool, some are not
kw = dict(kernel_t=0.5, eps=1e-6, n_vnn=5)
sf = support(ef, ycl, grp, qry, ref, **kw)
bf = support_bruteforce(ef, ycl, grp, qry, ref, **kw)
n_sib = np.array([sum(1 for j in ref if j != i and grp[ycl[j]] == grp[ycl[i]] and ycl[j] != ycl[i]) for i in qry])
check("(support) sibling sets exclude the same class and self: N matches the definition, singleton groups give N = 0",
      (sf["N"] == n_sib).all() and (support(ef, ycl, np.arange(C), qry, ref, **kw)["N"] == 0).all()
      and all(np.allclose([support(ef, ycl, grp, [i], ref, **kw)[k][0] for k in ("N", "K", "samecls", "V")],
                          [support(ef, ycl, grp, [i], ref[ref != i], **kw)[k][0] for k in ("N", "K", "samecls", "V")])
              for i in ref[:10]))
check("(support) U, K, V, samecls match the brute-force loops",
      all(np.allclose(sf[k], bf[k], rtol=1e-5, atol=1e-7, equal_nan=True) for k in ("U", "K", "V", "samecls"))
      and (sf["N"] == bf["N"]).all())
Rm = ref[rs.permutation(len(ref))[:12]]
keep_q = np.setdiff1d(ref, Rm)
rst = removal_stats(ef, ycl, grp, keep_q, ref, Rm, **kw)
mass = np.array([sum(np.exp((ef[i] @ ef[j] - 1) / 0.5) for j in Rm if grp[ycl[j]] == grp[ycl[i]] and ycl[j] != ycl[i])
                 for i in keep_q])
check("(support) Delta K on D \\ R equals minus the summed kappa over the removed siblings",
      np.allclose(rst["dK"], -rst["removed_sib_mass"], atol=1e-6) and np.allclose(rst["dK"], -mass, atol=1e-6)
      and np.allclose(rst["sib_frac"] * len(Rm), -rst["dN"]))

# --- Study A subset selection (synthetic companion index.json) -------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    pdir = Path(tmp) / "cifar100" / "partitions" / "b100"
    pdir.mkdir(parents=True)
    Wspec = [0.0, 0.4, 0.5, 0.6, 10.0, 10.0, 10.0, 10.0, 10.0]
    parts = [{"name": "true", "kind": "true", "W_train": 1.0}]
    parts += [{"name": f"random_{i:02d}", "kind": "random", "W_train": 1.0 - 0.01 * i} for i in range(9)]
    parts += [{"name": f"spectrum_{8 - k:02d}", "kind": "spectrum", "W_train": w} for k, w in enumerate(Wspec)]
    (pdir / "index.json").write_text(json.dumps({"partitions": parts}))
    sub = subset_partitions(load_config([str(ROOT / "configs/base.yaml")], [f"companion_dir={tmp}"]), 1.0)
check("(studyA) subset: the 4 lowest-index random partitions (not the 4 lowest W)",
      [p for c, p in sub if c == "random_hier"] == ["random_00", "random_01", "random_02", "random_03"])
check("(studyA) subset: distinct spectrum partitions closest to the 10/37/63/90th W percentiles",
      [p for c, p in sub if c == "spectrum_hier"] == ["spectrum_07", "spectrum_05", "spectrum_00", "spectrum_01"])

# --- Study A four patterns -----------------------------------------------------------------------
dpd_h = np.array([-1.0, -1.0, -1.0, 0.0, 1.0, 0.5, np.nan, 0.0])
dsup_h = np.array([-0.2, 0.3, 0.0, 0.1, -0.1, 0.0, 0.2, 0.1])
check("(studyA) four-pattern classifier on a hand case (exact and with tolerance)",
      pattern_codes(dpd_h, dsup_h).tolist() == [0, 1, 2, 3, 3, 4, -1, 3]
      and pattern_codes(dpd_h, dsup_h, 0.0, 0.15).tolist() == [0, 1, 2, 4, 4, 4, -1, 4]
      and abs(pattern_fractions(pattern_codes(dpd_h, dsup_h))["pd_notdown_v_change"] - 3 / 7) < 1e-12)

# --- residual with covariates reduces to sg.stats.residual ------------------------------------------
Wr, Tr = rs.normal(size=8), rs.normal(size=8)
r_sg = sg_residual(Wr, Tr, 0.7, 1.3)
r_cv = residual_cov(Wr, Tr, 0.7, 1.3)
check("(stats) residual_cov without covariates equals sg.stats.residual (delta, PI, p, coef)",
      all(np.isclose(r_sg[k], r_cv[k]) for k in ("delta", "pi_low", "pi_high", "p_one_sided", "se_pred"))
      and np.allclose(r_sg["coef"], r_cv["coef"]) and r_cv["df"] == r_sg["df"])

# --- Study B removal sets (synthetic 100-class pool, CIFAR-like superclass map) ----------------------
rd = np.random.default_rng(5)
n_cls, per_cls = 100, 12
assign_c = np.arange(n_cls) // 5                      # 20 superclasses of 5 classes
yd = np.repeat(np.arange(n_cls), per_cls)
ed = rd.normal(size=(len(yd), 8))
ed /= np.linalg.norm(ed, axis=1, keepdims=True)
cm_d = np.stack([ed[yd == c].mean(0) for c in range(n_cls)])
pool_ids_d = np.arange(len(yd)) * 2                   # pool rows; the audit pool uses the odd rows
audit_ids_d = np.arange(len(yd)) * 2 + 1
lbar_d = np.round(rd.normal(size=len(yd)), 1)         # coarse values: many ties
try:
    removal_size(0.03, len(yd))                       # 36 is not a multiple of 16
    size_err = False
except SystemExit:
    size_err = True
coh_d = del_cohorts(assign_c, 3, 0, 100)
check("(deletion) cohorts: 4 classes from 4 different superclasses, no class in two cohorts; |R| % 16 enforced",
      all(len(c) == 4 and len({assign_c[k] for k in c}) == 4 for c in coh_d)
      and len(set(sum(coh_d, []))) == 12 and size_err and removal_size(32 / 1200, 1200) == 32)
R_d = 32
kwd = dict(kernel_t=0.5, n_vnn=5, dv_tol=0.05, dv_max_swaps=10000, seed=0)
rem = make_removals(yd, ed, assign_c, cm_d, coh_d[0], 0, R_d, lbar_d, **kwd)
Td = rem["T"]
sib_d, vis_d = rem["siblings"], rem["vis_classes"]
check("(deletion) every removal type has |R| distinct records, disjoint from T_q and from the audit pool",
      all(len(np.unique(rem["R"][t])) == R_d and not np.intersect1d(rem["R"][t], Td).size
          and not np.intersect1d(pool_ids_d[rem["R"][t]], audit_ids_d).size for t in DEL_TYPES_ALL))
check("(deletion) sem / vis / near / far remove |R|/16 records from each of their 16 classes",
      all(sorted(np.unique(yd[rem["R"][t]]).tolist()) == sorted(sib_d if t != "vis" else vis_d)
          and (np.bincount(yd[rem["R"][t]], minlength=n_cls)[sib_d if t != "vis" else vis_d] == R_d // 16).all()
          for t in ("sem", "vis", "near", "far")) and len(sib_d) == 16 and len(vis_d) == 16)
st_vis = removal_stats(ed, yd, assign_c, Td, np.arange(len(yd)), rem["R"]["vis"], kernel_t=0.5, n_vnn=5)
check("(deletion) vis removes no curated sibling and gives Delta K = 0 for every T_q record",
      not set(vis_d) & (set(sib_d) | set(coh_d[0])) and np.all(st_vis["dK"] == 0) and np.all(st_vis["dN"] == 0))
nonT_d = np.setdiff1d(np.arange(len(yd)), Td)
hv_ref = sorted(nonT_d.tolist(), key=lambda i: (-lbar_d[i], i))[:R_d]
check("(deletion) hv = top |R| of Lbar0 outside T_q with ties broken by the smaller row index",
      rem["R"]["hv"].tolist() == sorted(hv_ref))
idx_d = rem["dv_index"]
ok_dv = True
for t in DEL_TYPES_ALL:
    mask = np.zeros(len(yd), bool)
    mask[rem["R"][t]] = True
    bf_b = support_bruteforce(ed, yd, assign_c, Td, np.arange(len(yd)), kernel_t=0.5, n_vnn=5)["V"]
    bf_a = support_bruteforce(ed, yd, assign_c, Td, np.flatnonzero(~mask), kernel_t=0.5, n_vnn=5)["V"]
    ok_dv &= np.allclose(idx_d.v_rows(np.arange(len(Td)), mask) - idx_d.v0, bf_a - bf_b, atol=1e-6)
check("(deletion) candidate-list Delta V equals the brute-force recomputation on D \\ R for every type", ok_dv)
m = rem["match"]
mask_sem = np.zeros(len(yd), bool)
mask_sem[rem["R"]["sem"]] = True
mask_vis = np.zeros(len(yd), bool)
mask_vis[rem["R"]["vis"]] = True
gap_true = abs(idx_d.mean_dv(mask_vis) - idx_d.mean_dv(mask_sem))
rem2 = make_removals(yd, ed, assign_c, cm_d, coh_d[0], 0, R_d, lbar_d, kernel_t=0.5, n_vnn=5, dv_tol=1e-9,
                     dv_max_swaps=7, seed=0)
m2 = rem2["match"]
check("(deletion) Delta-V matching reaches the tolerance (or reports the gap) and keeps per-class counts",
      m["reached"] and m["gap"] < m["tolerance"] and abs(m["gap"] - gap_true) < 1e-9 and m["gap"] <= m["gap_initial"]
      and not m2["reached"] and m2["proposals"] == 7 and m2["gap"] <= m2["gap_initial"]
      and (np.bincount(yd[rem2["R"]["vis"]], minlength=n_cls)[vis_d] == 2).all())

# --- prediction splits, ridge, precision, Holm --------------------------------------------------------
rp_ = np.random.default_rng(6)
n_r = 600
coh_r = rp_.integers(0, 4, n_r)
exp_r = coh_r * 6 + rp_.integers(0, 6, n_r)          # experiments nested in cohorts
y_r = rp_.integers(0, 100, n_r)
par_r = y_r // 5
ok_split = True
for kind in SPLITS:
    sp_ = make_splits(kind, coh_r, exp_r, y_r, par_r, class_folds(100, 0))
    ok_split &= len(sp_) > 0 and all(not np.intersect1d(exp_r[a], exp_r[b]).size
                                     and not np.intersect1d(coh_r[a], coh_r[b]).size for a, b in sp_)
check("(predict) every split kind (cohort, experiment, class, parent) never shares an experiment or cohort", ok_split)
Xr = rp_.normal(size=(n_r, 5)) * np.array([1.0, 2.0, 0.5, 3.0, 1.0])
w_r = np.array([0.5, -1.0, 0.0, 0.3, 2.0])
yr_ = Xr @ w_r + 0.01 * rp_.normal(size=n_r)
tr_, te_ = np.flatnonzero(coh_r != 0), np.flatnonzero(coh_r == 0)
Xa, Xb_ = pred_prep(Xr[tr_], Xr[te_])
mdl, alpha_r = fit_ridge(Xa, yr_[tr_], coh_r[tr_], base.ridge_alphas, 3)
coef_raw = mdl.coef_ / Xr[tr_].std(0)
check("(predict) ridge with inner GroupKFold recovers a known linear signal (coef, held-out R^2 > 0.99)",
      np.allclose(coef_raw, w_r, atol=0.02) and pred_metrics(yr_[te_], mdl.predict(Xb_), 0.05)["r2"] > 0.99
      and alpha_r <= 1.0)
yt = np.arange(40, dtype=float)                       # k = ceil(0.05 * 40) = 2, true top-2 = {39, 38}
yh = np.zeros(40)
yh[39], yh[5] = 10.0, 9.0                             # predicted top-2 = rows 39 (hit) and 5 (miss)
check("(predict) precision of the predicted top 5% on a hand case (0.5; ties at the threshold count as hits)",
      precision_top(yt, yh, 0.05) == 0.5 and precision_top(np.r_[np.zeros(38), 1.0, 1.0], yh, 0.05) == 0.5
      and precision_top(np.ones(40), yh, 0.05) == 1.0)
praw = {"P1": 0.01, "P2": 0.04, "P3": 0.03, "P4": None, "P5": 0.2}
ph, rj = holm_family(praw, 0.05)
ref_h = sg_holm({k: v for k, v in praw.items() if v is not None})
check("(stats) Holm over the primary family equals sg.stats.holm (None stays None) and reject = p_holm <= 0.05",
      all(np.isclose(ph[k], ref_h[k]) for k in ref_h) and ph["P4"] is None and rj["P1"] and not rj["P5"]
      and np.isclose(ph["P1"], 0.04))
check("(stats) exact sign-flip p over cohorts on hand cases",
      sign_flip_p([1.0, 1.0]) == 0.5 and sign_flip_p([1.0, 2.0, 3.0, 4.0]) == 2 / 16)

raise SystemExit(0 if ok else 1)
