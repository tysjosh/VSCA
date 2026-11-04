"""Training objectives on flattened target positions.

All functions take ``logits`` [N, V] (float32) at the N supervised positions and
``labels`` [N]. ``seq`` [N] gives the batch row of each position so that
per-sequence selection rules can be applied.
"""

from __future__ import annotations

import torch

from .config import Config


def token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, -1) - logits.gather(-1, labels[:, None]).squeeze(-1)


def set_nll(logits: torch.Tensor, labels: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    """-log sum_{a in A_t} pi_t(a) with A_t = {gold} ∪ extra (Eq. 3). ``extra`` is padded with -1.

    With no extras this equals token_nll exactly (Proposition 3a).
    """
    acc = torch.cat([labels[:, None], extra], dim=1)
    valid = acc >= 0
    picked = logits.gather(-1, acc.clamp(min=0)).masked_fill(~valid, float("-inf"))
    return torch.logsumexp(logits, -1) - torch.logsumexp(picked, -1)


def fork_loss(logits: torch.Tensor, extra: torch.Tensor, bad: torch.Tensor, beta: float) -> torch.Tensor:
    """VSCA v2 loss at a first-error fork (Design 4.1 step 5, sign per the brief, plan Decision 7):

    -log sum_{a in A_t} pi(a|h) + beta * (-log(1 - pi(e|h))), with A_t = ``extra`` (padded with -1; the
    erroneous token e = ``bad`` is not in it). An empty A_t gives only the unlikelihood term (Design 5.3).
    Returned for every row; rows with ``bad < 0`` are not forks and are ignored by the caller.
    """
    lse = torch.logsumexp(logits, -1)
    valid = extra >= 0
    has_set = valid.any(-1)
    # rows without a set get a finite dummy (0 logits) so that no -inf/NaN reaches the backward pass
    picked = logits.gather(-1, extra.clamp(min=0)).masked_fill(~valid, float("-inf")).masked_fill(~has_set[:, None], 0.0)
    set_term = torch.where(has_set, lse - torch.logsumexp(picked, -1), torch.zeros_like(lse))
    p_bad = (logits.gather(-1, bad.clamp(min=0)[:, None]).squeeze(-1) - lse).exp()
    ul = -torch.log((1.0 - p_bad).clamp(min=1e-6))
    return set_term + beta * ul


def topk_per_seq(score: torch.Tensor, seq: torch.Tensor, rho: float) -> torch.Tensor:
    mask = torch.zeros_like(score, dtype=torch.bool)
    for b in seq.unique():
        idx = (seq == b).nonzero(as_tuple=True)[0]
        n = max(1, int(round(rho * idx.numel())))
        mask[idx[score[idx].topk(n).indices]] = True
    return mask


def minmax_per_seq(x: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for b in seq.unique():
        idx = (seq == b).nonzero(as_tuple=True)[0]
        v = x[idx]
        lo, hi = v.min(), v.max()
        out[idx] = (v - lo) / (hi - lo) if hi > lo else torch.zeros_like(v)
    return out


def expand_to_spans(mask: torch.Tensor, seq: torch.Tensor, claim_id: torch.Tensor) -> torch.Tensor:
    """Complete-span constraint of the earlier formulation: a selected claim token selects its whole claim."""
    out = mask.clone()
    hit = mask & (claim_id >= 0)
    for b, c in {(int(b), int(c)) for b, c in zip(seq[hit].tolist(), claim_id[hit].tolist())}:
        out |= (seq == b) & (claim_id == c)
    return out


def pairwise_cf_loss(s_gold: torch.Tensor, s_cf: torch.Tensor, tau: float) -> torch.Tensor:
    """-log sigma((S(z|h) - S(z~|h)) / tau), the earlier counterfactual contrastive term."""
    return -torch.nn.functional.logsigmoid((s_gold - s_cf) / tau).mean()


def compute_loss(cfg: Config, logits: torch.Tensor, labels: torch.Tensor, seq: torch.Tensor, aux: dict, gen: torch.Generator):
    """Loss for ``cfg.method`` plus logging statistics.

    ``aux`` may contain: ``claim`` (bool), ``claim_id`` (long), ``extra`` (long [N,K]),
    ``base_loss``, ``ref_loss``, ``attn``, ``cf_s`` (float).
    """
    N = labels.numel()
    method = cfg.method
    nll = token_nll(logits, labels)
    weights = None
    mask = None

    if method in ("sft", "rft"):
        per_tok = nll
    elif method == "vsca":
        per_tok = set_nll(logits, labels, aux["extra"])
    elif method == "unlikelihood":
        bad = aux["extra"]
        logp = torch.log_softmax(logits, -1)
        p_bad = logp.gather(-1, bad.clamp(min=0)).exp().masked_fill(bad < 0, 0.0)
        ul = -torch.log((1.0 - p_bad).clamp(min=1e-6)).masked_fill(bad < 0, 0.0).sum(-1)
        per_tok = nll + cfg.ul_lambda * ul
    elif method == "random":
        score = torch.rand(N, generator=gen, device="cpu").to(logits.device)
        mask = topk_per_seq(score, seq, cfg.rho)
        per_tok = nll
    elif method == "rho":
        mask = topk_per_seq(nll.detach() - aux["ref_loss"], seq, cfg.rho)
        per_tok = nll
    elif method == "token_cleaning":
        mask = topk_per_seq(aux["base_loss"] - aux["ref_loss"], seq, cfg.rho)
        per_tok = nll
    elif method == "sstoken":
        rel = aux["base_loss"] - nll.detach()
        score = cfg.sstoken_gamma * minmax_per_seq(rel, seq) + (1 - cfg.sstoken_gamma) * aux["attn"]
        mask = topk_per_seq(score, seq, cfg.rho)
        per_tok = nll
    elif method == "trimsft":
        with torch.no_grad():
            gold = logits.gather(-1, labels[:, None]).squeeze(-1)
            other = logits.scatter(-1, labels[:, None], float("-inf")).max(-1).values
            gap = gold - other
            weights = torch.exp(-((gap - cfg.trim_mu) ** 2) / (2 * cfg.trim_tau**2))
        per_tok = nll
    elif method == "claim_mask":
        noise = torch.rand(N, generator=gen, device="cpu").to(logits.device)
        mask = topk_per_seq(aux["claim"].float() * 2.0 + noise, seq, cfg.rho)
        per_tok = nll
    elif method == "claim_upweight":
        weights = 1.0 + cfg.upweight_beta * aux["claim"].float()
        per_tok = nll
    elif method == "cf_select":
        d = minmax_per_seq((nll.detach() - aux["ref_loss"]).clamp(min=0), seq)
        s = aux["cf_s"]
        q = cfg.cf_alpha * d + (1 - cfg.cf_alpha) * s + cfg.cf_gamma * d * s
        mask = expand_to_spans(topk_per_seq(q, seq, cfg.rho), seq, aux["claim_id"])
        per_tok = nll
    elif method in ("vsca2", "stepdpo_fork", "rft_verified"):
        # SFT on every supervised position except the first-error forks (bad >= 0), which get fork_loss.
        # rft_verified has no forks, so this is plain SFT on its examples.
        is_fork = aux["bad"] >= 0
        per_tok = torch.where(is_fork, fork_loss(logits, aux["extra"], aux["bad"], cfg.fork_beta), nll)
    else:
        raise ValueError(method)

    if mask is not None:
        weights = mask.float()
    if weights is None:
        weights = torch.ones_like(per_tok)
    denom = weights.sum().clamp(min=1.0) if cfg.resolved_norm() == "selected" else torch.tensor(float(N), device=per_tok.device)
    loss = (weights * per_tok).sum() / denom
    stats = {
        "nll": float(nll.detach().mean()),
        "n_tokens": N,
        "frac_selected": float((weights > 0).float().mean()),
        "mean_weight": float(weights.mean()),
    }
    if method == "vsca":
        with torch.no_grad():
            stats["certified_mass_gap"] = float((torch.exp(-per_tok.detach()) - torch.exp(-nll.detach())).mean())
            stats["frac_positions_with_alts"] = float((aux["extra"] >= 0).any(-1).float().mean())
    if method in ("vsca2", "stepdpo_fork", "rft_verified"):
        with torch.no_grad():
            is_fork = aux["bad"] >= 0
            n_forks = int(is_fork.sum())
            stats["n_forks"] = n_forks
            if n_forks:
                lg, bd = logits[is_fork], aux["bad"][is_fork]
                p_bad = torch.softmax(lg, -1).gather(-1, bd[:, None]).squeeze(-1)
                stats["frac_forks_with_set"] = float((aux["extra"][is_fork] >= 0).any(-1).float().mean())
                stats["fork_ul"] = float((-torch.log((1.0 - p_bad).clamp(min=1e-6))).mean())
            else:
                stats["frac_forks_with_set"] = 0.0
                stats["fork_ul"] = 0.0
    return loss, stats
