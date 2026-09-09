"""Loss functions for algorithms"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import torch
import torch.nn.functional as F

from minrl.training.estim import Sequence

Aux = Dict[str, float]


@dataclass
class Packed:
    """A micro-batch padded into tensors, shifted to the target frame."""

    ids: torch.Tensor       # [B, T]    padded token ids
    attn: torch.Tensor      # [B, T]    1 on real tokens
    targets: torch.Tensor   # [B, T-1]  ids shifted left
    mask: torch.Tensor      # [B, T-1]  1.0 on generated tokens
    old_logp: torch.Tensor  # [B, T-1]  behaviour-policy logprobs
    adv: torch.Tensor       # [B]


def pack(seqs: Seq[Sequence], device: torch.device) -> Packed:
    """Right-pad and shift to the target frame: index ``t`` describes token ``t+1``."""
    max_len = max(len(s.token_ids) for s in seqs)
    ids = _pad([s.token_ids for s in seqs], max_len, 0, torch.long, device)
    mask = _pad([s.action_mask for s in seqs], max_len, 0, torch.float32, device)
    # ``logprobs=None`` (offline data) pads up from empty; those positions are masked.
    old = _pad([s.logprobs or [] for s in seqs], max_len, 0.0, torch.float32, device)
    attn = _pad([[1] * len(s.token_ids) for s in seqs], max_len, 0, torch.long, device)
    return Packed(
        ids=ids, attn=attn, targets=ids[:, 1:], mask=mask[:, 1:], old_logp=old[:, 1:],
        adv=torch.tensor(
            [s.advantage for s in seqs], dtype=torch.float32, device=device
        ),
    )


def forward(model, p: Packed) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward pass -> ``(logits, logprob of each realized token)``."""
    logits = model(input_ids=p.ids, attention_mask=p.attn).logits[:, :-1]
    # -cross_entropy is the realized token's logprob, without a [B, T, V] log_softmax.
    return logits, -F.cross_entropy(
        logits.float().transpose(1, 2), p.targets, reduction="none"
    )


def logprobs(model, seqs: Seq[Sequence]) -> Tuple[Packed, torch.Tensor]:
    """Pack and run ``seqs`` -> ``(packed, per-token logprobs)``."""
    p = pack(seqs, device_of(model))
    return p, forward(model, p)[1]


def reduce(
    per_token: torch.Tensor,
    mask: torch.Tensor,
    *,
    agg: str,
    n_seqs: float,
    n_tokens: float,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Collapse a masked per-token quantity into this micro-batch's loss.

    Counts are step-wide, not per-micro-batch, which is what keeps micro-batching exact.

    ``seq-mean``   mean within each sequence, then across them -- GRPO's ``1/G sum 1/|o_i|``
    ``token-mean`` one mean over every action token in the step (DAPO, CISPO, SFT)
    ``seq-sum``    sum per trajectory, mean over trajectories (REINFORCE)
    ``budget``     sum over the constant ``n_seqs * max_tokens`` (Dr. GRPO)
    """
    x = per_token * mask
    if agg == "seq-mean":
        return x.sum(dim=1).div(mask.sum(dim=1).clamp(min=1.0)).sum() / n_seqs
    if agg == "token-mean":
        return x.sum() / n_tokens
    if agg == "seq-sum":
        return x.sum() / n_seqs
    if agg == "budget":
        if not max_tokens:
            raise ValueError('agg="budget" needs max_tokens')
        return x.sum() / (n_seqs * max_tokens)
    raise ValueError(f"unknown aggregation {agg!r}")


# One call convention for every loss below -- ``(model, items, *, agg, n_seqs,
# n_tokens, ref_model)``, hyper-parameters bound by ``algorithms.py``, ``**_``
# swallowing the rest -- so the Trainer invokes any of them identically.

def clipped_surrogate(
    model,
    seqs: Seq[Sequence],
    *,
    agg: str,
    n_seqs: float,
    n_tokens: float,
    clip_eps: float,
    ref_model=None,
    kl_coef: float = 0.0,
    max_tokens: Optional[int] = None,
    **_,
) -> Tuple[torch.Tensor, Aux]:
    """PPO's clipped objective, shared by GRPO and Dr. GRPO::

        rho_t = pi_theta(o_t) / pi_old(o_t)
        L = -E_t[ min(rho_t * A, clip(rho_t, 1-eps, 1+eps) * A)
                  - beta * KL(pi_theta || pi_ref) ]
    """
    p, logp = logprobs(model, seqs)
    ratio = torch.exp(logp - p.old_logp)
    a = p.adv[:, None]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)

    aux = {"clip_frac": _clip_frac(ratio, p.mask, n_tokens, clip_eps, clip_eps)}
    if kl_coef:
        if ref_model is None:
            raise ValueError("kl_coef > 0 needs ref_model=<frozen policy>")
        with torch.no_grad():
            ref_logp = forward(ref_model, p)[1]
        kl = _kl_k3(logp, ref_logp)
        surrogate = surrogate - kl_coef * kl
        aux["kl"] = _masked_mean(kl, p.mask, n_tokens)

    return -reduce(surrogate, p.mask, agg=agg, n_seqs=n_seqs, n_tokens=n_tokens,
                   max_tokens=max_tokens), aux


def cispo_surrogate(
    model,
    seqs: Seq[Sequence],
    *,
    agg: str,
    n_seqs: float,
    n_tokens: float,
    eps_low: float,
    eps_high: float,
    **_,
) -> Tuple[torch.Tensor, Aux]:
    """CISPO's clipped importance-weight objective::

        rho_t = pi_theta(o_t) / pi_old(o_t)
        L = -E_t[ sg(clip(rho_t, 1-eps_low, 1+eps_high)) * A * log pi_theta(o_t) ]

    The weight is clipped *and detached* -- a coefficient, never a path for
    gradients.  That is the whole idea: where PPO's clip deletes the gradient of
    any token that strays out of the trust region, CISPO keeps every token's
    gradient and only bounds how loudly it speaks.
    """
    p, logp = logprobs(model, seqs)
    ratio = torch.exp(logp - p.old_logp)
    surrogate = ratio.clamp(1 - eps_low, 1 + eps_high).detach() * p.adv[:, None] * logp

    aux = {"clip_frac": _clip_frac(ratio, p.mask, n_tokens, eps_low, eps_high)}
    return -reduce(surrogate, p.mask, agg=agg, n_seqs=n_seqs, n_tokens=n_tokens), aux


def score_function(
    model, seqs: Seq[Sequence], *, agg: str, n_seqs: float, n_tokens: float, **_
) -> Tuple[torch.Tensor, Aux]:
    """REINFORCE's score-function estimator::
        L = -E_t[ A_t * log pi_theta(a_t | s_t) ]
    """
    p, logp = logprobs(model, seqs)
    return -reduce(p.adv[:, None] * logp, p.mask, agg=agg, n_seqs=n_seqs,
                   n_tokens=n_tokens), {}


def cross_entropy(
    model, seqs: Seq[Sequence], *, agg: str, n_seqs: float, n_tokens: float, **_
) -> Tuple[torch.Tensor, Aux]:
    """SFT's masked cross-entropy -- maximize the demonstration's own tokens::
        L = -E_t[ log pi_theta(o_t | o_<t) ]
    """
    p = pack(seqs, device_of(model))
    logits, logp = forward(model, p)

    with torch.no_grad():
        hits = (logits.argmax(dim=-1) == p.targets).float()
        aux = {"token_acc": _masked_mean(hits, p.mask, n_tokens)}

    return -reduce(logp, p.mask, agg=agg, n_seqs=n_seqs, n_tokens=n_tokens), aux


def preference(
    model,
    pairs: List[Tuple[Sequence, Sequence]],
    *,
    n_seqs: float,
    beta: float,
    ref_model=None,
    **_,
) -> Tuple[torch.Tensor, Aux]:
    """DPO's pairwise preference objective::
        margin(y) = log pi_theta(y) - log pi_ref(y)
        L = -log sigmoid( beta * (margin(y_chosen) - margin(y_rejected)) )
    """
    if ref_model is None:
        raise ValueError("dpo needs ref_model=<the checkpoint the policy started from>")
    device = device_of(model)
    chosen = pack([c for c, _ in pairs], device)
    rejected = pack([r for _, r in pairs], device)

    policy = _seq_logp(model, chosen) - _seq_logp(model, rejected)
    with torch.no_grad():
        reference = _seq_logp(ref_model, chosen) - _seq_logp(ref_model, rejected)

    logits = beta * (policy - reference)
    # Sum over the step-wide pair count, not mean, so micro-batching stays exact.
    loss = -F.logsigmoid(logits).sum() / n_seqs

    with torch.no_grad():
        aux = {
            "accuracy": (logits > 0).float().sum().item() / n_seqs,
            "margin": logits.sum().item() / n_seqs,
        }
    return loss, aux


def device_of(model) -> torch.device:
    """The device ``model``'s parameters live on."""
    return next(model.parameters()).device


def _seq_logp(model, p: Packed) -> torch.Tensor:
    """Summed logprob of each sequence's action tokens. ``[B]``"""
    return (forward(model, p)[1] * p.mask).sum(dim=1)


def _kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimator of ``KL(pi_theta || pi_ref)``: unbiased, and
    non-negative sample by sample where the naive ``-d`` is not."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


def _clip_frac(
    ratio: torch.Tensor, mask: torch.Tensor, n_tokens: float, low: float, high: float
) -> float:
    """Fraction of action tokens whose ratio left ``[1-low, 1+high]``."""
    with torch.no_grad():
        strayed = ((ratio < 1 - low) | (ratio > 1 + high)).float()
    return _masked_mean(strayed, mask, n_tokens)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, n_tokens: float) -> float:
    """Mean over the step's action tokens, so micro-batch values sum to the step's."""
    return (x * mask).sum().item() / max(n_tokens, 1)


def _pad(rows, max_len: int, value, dtype, device) -> torch.Tensor:
    """Right-pad ``rows`` to ``max_len`` and stack into a single tensor."""
    return torch.tensor(
        [list(row) + [value] * (max_len - len(row)) for row in rows],
        dtype=dtype, device=device,
    )
