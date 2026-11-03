"""Identity and unit checks on synthetic data (no CIFAR needed).

Run: python scripts/sanity_check.py
"""

import itertools
from fractions import Fraction

import numpy as np
import torch

from sg.config import AUX_STAGES, CONDITIONS
from sg.data import BUDGETS, audit_split, budget_split, gpu_augment, normalize
from sg.model import ResNet18
from sg.partitions import (
    ari, cosine_matrix, is_balanced, random_partition, spectrum_partitions, within_group_similarity,
)
from sg.metrics import crossfit_folds, factorize, metric_value, stratum_deltas
from sg.probes import knn_predict, linear_probe, neighbor_purity, prediction_depth
from sg.stats import holm, rank_pvalue, residual
from sg.train import Calibrator, grad_norms

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# --- (a) Table II budgets ---------------------------------------------------
y = np.repeat(np.arange(100), 500)
np.random.default_rng(0).shuffle(y)
splits = {b: budget_split(y, b, 0.1, 1234, 100, 0.1) for b in BUDGETS}
check("(a) Table II train sizes", [len(splits[b][0]) for b in BUDGETS] == [4000, 10000, 20000, 40000])
check("(a) Table II val sizes", [len(splits[b][1]) for b in BUDGETS] == [500, 1250, 2500, 5000])
aud = audit_split(y, 0.1, 0.1, 1234, 100)
check("(a) audit pool has 5,000 images, 50 per class", len(aud) == 5000 and (np.bincount(y[aud], minlength=100) == 50).all())
check("(a) audit pool disjoint from every budget's train and val",
      all(not (set(aud) & (set(splits[b][0]) | set(splits[b][1]))) for b in BUDGETS))
check("(a) train nested across budgets",
      all(set(splits[a][0]) <= set(splits[b][0]) for a, b in zip(BUDGETS, BUDGETS[1:])))
check("(a) val nested across budgets",
      all(set(splits[a][1]) <= set(splits[b][1]) for a, b in zip(BUDGETS, BUDGETS[1:])))
check("(a) train/val disjoint within budget", all(not set(splits[b][0]) & set(splits[b][1]) for b in BUDGETS))
check("(a) stratified 10% train (40/class)", (np.bincount(y[splits[0.1][0]], minlength=100) == 40).all())

# --- (b) Eq. 3 vs brute force -------------------------------------------------
rng = np.random.default_rng(1)
mu = rng.normal(size=(100, 16))
assign = random_partition(100, 20, rng)
brute = 0.0
for g in range(20):
    m = [c for c in range(100) if assign[c] == g]
    s = 0.0
    for i, j in itertools.combinations(m, 2):
        s += float(mu[i] @ mu[j] / (np.linalg.norm(mu[i]) * np.linalg.norm(mu[j])))
    brute += s / 10
brute /= 20
check("(b) W(G) equals brute-force Eq. 3", abs(within_group_similarity(assign, mu) - brute) < 1e-10)
check("(b) W(G) from S equals W(G) from mu", abs(within_group_similarity(assign, cosine_matrix(mu)) - brute) < 1e-10)

# --- (c), (d) random and spectrum partitions -------------------------------------
S = cosine_matrix(mu)
rr = np.random.default_rng(0)
randoms = [random_partition(100, 20, rr) for _ in range(19)]
spec = spectrum_partitions(S, 19, 20, 3000, 3, np.random.default_rng(2))
check("(c) random partitions balanced", all(is_balanced(a, 20) for a in randoms))
check("(c) spectrum partitions balanced", len(spec) == 19 and all(is_balanced(a, 20) for a, _ in spec))
Wr = [within_group_similarity(a, S) for a in randoms]
Ws = np.array([w for _, w in spec])
check("(d) spectrum W strictly increasing", bool((np.diff(Ws) > 0).all()))
check("(d) spectrum W[0] <= min random W", Ws[0] <= min(Wr))
check("(d) spectrum W[-1] >= max random W", Ws[-1] >= max(Wr))
check("(d) spectrum reported W matches recomputed W", all(abs(w - within_group_similarity(a, S)) < 1e-10 for a, w in spec))

# --- (e) ARI ------------------------------------------------------------------
true = np.arange(100) // 5
relabel = np.random.default_rng(3).permutation(20)[true]
check("(e) ari(true, true) == 1", abs(ari(true, true) - 1) < 1e-12)
check("(e) ari(true, relabeled true) == 1", abs(ari(true, relabel) - 1) < 1e-12)

# --- (f) model shapes -------------------------------------------------------------
x = torch.randn(2, 3, 32, 32)
shapes_ok = True
for cond in CONDITIONS:
    st = AUX_STAGES[cond]
    n_aux = 100 if cond == "flat_aux" else 20
    m = ResNet18(100, st, n_aux).eval()
    with torch.no_grad():
        logits, aux, stages = m(x, return_stages=True)
    shapes_ok &= logits.shape == (2, 100) and set(aux) == set(st)
    shapes_ok &= all(aux[s].shape == (2, n_aux) for s in st)
    shapes_ok &= [tuple(stages[f"s{i}"].shape[1:2]) for i in range(1, 5)] == [(64,), (128,), (256,), (512,)]
check("(f) logits / aux head shapes for every condition", shapes_ok)
check("(f) Final-Hier has only an s4 head", list(ResNet18(100, AUX_STAGES["final_hier"], 20).heads) == ["s4"])
with torch.no_grad():
    bf = ResNet18(100).eval().block_features(x)
check("(f) 8 block features with channels 64,64,128,128,256,256,512,512",
      [f.shape[1] for f in bf] == [64, 64, 128, 128, 256, 256, 512, 512] and all(f.shape[0] == 2 for f in bf))

# --- (g) gpu_augment ----------------------------------------------------------------
xu8 = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
out = gpu_augment(xu8, 4, torch.Generator().manual_seed(0))
check("(g) gpu_augment keeps shape", out.shape == (16, 3, 8, 8) and out.dtype == torch.float32)
out0 = gpu_augment(xu8, 0, torch.Generator().manual_seed(0))
ref = normalize(xu8)
same = all(torch.allclose(out0[i], ref[i]) or torch.allclose(out0[i], ref[i].flip(-1)) for i in range(16))
n_flip = sum(not torch.allclose(out0[i], ref[i]) for i in range(16))
check("(g) pad 0 equals plain flip / no-flip of the input", same and 0 < n_flip < 16)

# --- (h) lambda calibration -----------------------------------------------------------
torch.manual_seed(0)
m = ResNet18(100, ("s2", "s3"), 20).train()
xb = torch.randn(8, 3, 16, 16)
yb = torch.randint(0, 100, (8,))
ab = yb // 5
rho = 0.5
cal = Calibrator(("s2", "s3"), rho)
norms, _ = grad_norms(m, xb, yb, ab, ("s2", "s3"))
cal.update(norms)
lam = cal.lambdas()
norms2, _ = grad_norms(m, xb, yb, ab, ("s2", "s3"))
ratios = {s: lam[s] * norms2[s][1] / norms2[s][0] for s in ("s2", "s3")}
check("(h) lambda_s * ||g_aux|| / ||g_fine|| == rho at s2 and s3", all(abs(r - rho) < 1e-5 for r in ratios.values()))

# --- (i), (j) factorization identities ---------------------------------------------------
rng = np.random.default_rng(4)
yt = rng.integers(0, 100, 2000)
true_a = np.arange(100) // 5
# mix of correct, in-group and out-of-group predictions
pr = np.where(rng.random(2000) < 0.4, yt, rng.integers(0, 100, 2000))
sib = true_a[yt] * 5 + rng.integers(0, 5, 2000)
pr = np.where(rng.random(2000) < 0.3, sib, pr)
rand_a = random_partition(100, 20, np.random.default_rng(5))
for label, a in (("true", true_a), ("random", rand_a)):
    f = factorize(yt, pr, a)
    exact = Fraction(f["n_in_group"], f["n"]) * Fraction(f["n_correct"], f["n_in_group"]) == Fraction(f["n_correct"], f["n"])
    check(f"(i) acc == C*S exactly ({label} partition)", exact and abs(f["acc"] - f["C"] * f["S"]) < 1e-12)
    check(f"(j) Eq. 2 F equals direct within-group error fraction ({label} partition)", abs(f["F"] - f["F_direct"]) < 1e-12)
    # independent per-item loop, not sharing factorize's in-group mask
    n_in_loop = sum(1 for yi, pi in zip(yt, pr) if a[pi] == a[yi])
    check(f"(i') C matches a per-item loop ({label} partition)", abs(f["C"] - n_in_loop / len(yt)) < 1e-12)

# hand-made case with known answers: 10 items, 3 correct, 2 in-group errors, 5 out-of-group errors
# -> C = 5/10 = 0.5, S = 3/5 = 0.6, acc = 0.3, F = 2/7. A wrong group mask cannot reproduce these.
hy = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
hp = np.array([0, 1, 2, 4, 3, 50, 60, 70, 80, 90])  # 0-2 correct; 3->4, 4->3 same group (0..4); rest other groups
hf = factorize(hy, hp, true_a)
check("(i'') hand-made case: C = 0.5, S = 0.6, acc = 0.3", abs(hf["C"] - 0.5) < 1e-12 and abs(hf["S"] - 0.6) < 1e-12
      and abs(hf["acc"] - 0.3) < 1e-12)
check("(j'') hand-made case: F = F_direct = 2/7", abs(hf["F"] - 2 / 7) < 1e-12 and abs(hf["F_direct"] - 2 / 7) < 1e-12)
check("(i'') all-in-group mask would give C = 1, so the case discriminates", hf["C"] != 1.0)

# --- (k) prediction depth -------------------------------------------------------------------
pd_rows = [[1] * 9, [2, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2, 1], [3] * 8 + [1]]
check("(k) prediction depth on hand-made rows == [1, 2, 3, 9, 9]", prediction_depth(pd_rows).tolist() == [1, 2, 3, 9, 9])

# --- (l) cross-fitted strata ------------------------------------------------------------------
N = 1001
ptrue = rng.random((5, N))
folds = crossfit_folds(ptrue, 0.25)
check("(l) crossfit_folds: round(0.25 N) hard items per fold", all(len(fd["hard"]) == round(0.25 * N) for fd in folds))
check("(l) hard stratum = lowest CE-A p_true",
      all(ptrue[fd["ref"], fd["hard"]].max() <= np.sort(ptrue[fd["ref"]])[round(0.25 * N) - 1] for fd in folds))
yl = rng.integers(0, 100, N)
same = np.tile(np.where(rng.random(N) < 0.5, yl, rng.integers(0, 100, N)), (5, 1))
same_p = np.tile(ptrue[0], (5, 1))
fn = lambda pred, idx: metric_value(yl, pred, idx, true_a, "acc")  # noqa: E731
sd = stratum_deltas(fn, same, None, crossfit_folds(same_p, 0.25))
check("(l) Delta_null == 0 when all CE seeds are identical", sd["delta_null"] == 0.0)
excl = crossfit_folds(ptrue, 0.25, exclude=set(folds[0]["hard"][:10].tolist()))
check("(l) excluded ids are removed from the hard stratum", len(excl[0]["hard"]) == round(0.25 * N) - 10)

# --- (m) k-NN on separated clusters -------------------------------------------------------------
c = torch.zeros(2, 8)
c[0, 0], c[1, 1] = 1.0, 1.0
ref_x = torch.cat([c[0] + 0.01 * torch.randn(30, 8), c[1] + 0.01 * torch.randn(30, 8)])
ref_lab = torch.tensor([0] * 30 + [1] * 30)
q_x = torch.cat([c[0] + 0.01 * torch.randn(10, 8), c[1] + 0.01 * torch.randn(10, 8)])
q_lab = torch.tensor([0] * 10 + [1] * 10)
kp, kidx = knn_predict(ref_x, ref_lab, q_x, k=20, n_classes=2)
check("(m) k-NN accuracy 1 and purity 1 on separated clusters",
      bool((kp == q_lab).all()) and neighbor_purity(ref_lab[kidx], q_lab) == 1.0)
lp = linear_probe(ref_x, ref_lab, q_x, 2, iters=50)
check("(m) linear probe accuracy 1 on separated clusters", bool((lp == q_lab).all()))

# --- (n), (o) rank p-value and Holm ----------------------------------------------------------------
r19 = list(range(1, 20))
check("(n) rank_pvalue(5, 1..19) == 0.8", rank_pvalue(5, r19) == 16 / 20)
check("(n) rank_pvalue(100, 1..19) == 0.05", rank_pvalue(100, r19) == 0.05)
check("(n) rank_pvalue(0, 1..19) == 1.0", rank_pvalue(0, r19) == 1.0)
hm = holm({"a": 0.01, "b": 0.04, "c": 0.03})
check("(o) holm({a:.01,b:.04,c:.03}) == {a:.03,c:.06,b:.06}",
      list(hm) == ["a", "c", "b"] and all(abs(hm[k] - v) < 1e-12 for k, v in {"a": 0.03, "c": 0.06, "b": 0.06}.items()))

# --- (p) Eq. 8 residual ----------------------------------------------------------------------------
Wp = np.linspace(0.0, 1.0, 10)
Tp = 1 + 2 * Wp + 1e-3 * np.sin(1.7 * np.arange(10))
r_pos = residual(Wp, Tp, 0.5, 1 + 2 * 0.5 + 0.5, degree=1)
check("(p) residual delta ~ 0.5 and pi_low > 0 for offset 0.5", abs(r_pos["delta"] - 0.5) < 1e-2 and r_pos["pi_low"] > 0)
r_zero = residual(Wp, Tp, 0.5, 1 + 2 * 0.5, degree=1)
check("(p) prediction interval contains 0 for offset 0", r_zero["pi_low"] < 0 < r_zero["pi_high"])
r_q = residual(Wp, Tp, 0.5, 2.5, degree=2)
check("(p) quadratic fit returns a finite delta", np.isfinite(r_q["delta"]) and np.isfinite(r_q["se_pred"]) and r_q["df"] == 7)

# --- power simulation matches the confirmatory H1 test (review issue 3) ----------------------
from sg.stats import ALPHA_ONE_SIDED, power_sim  # noqa: E402

ps = power_sim(1e-4, 1e-4, effect=0.01, Rs=(19,), seeds=(3,), n_sim=200)
row = ps["table"][0]
check("power_sim fits 2R partitions (random + spectrum)", row["fit_points"] == 38)
check("power_sim tests at alpha/3 of one-sided 0.025", abs(ps["alpha_per_test"] - ALPHA_ONE_SIDED / 3) < 1e-15)
check("rank test with R=19 cannot reach alpha/3 (min p = 0.05)", row["power_rank"] == 0.0)

raise SystemExit(0 if ok else 1)
