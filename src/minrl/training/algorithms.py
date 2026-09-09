"""Algorithms in minrl"""
from __future__ import annotations

import torch
from functools import partial
from minrl.types import Batch
from dataclasses import dataclass
from minrl.training import estim, loss
from typing import Callable, Dict, List, Optional, Tuple
Estimator = Callable[[Batch], Tuple[List, Dict[str, float]]]
Loss = Callable[..., Tuple[torch.Tensor, Dict[str, float]]]


@dataclass(frozen=True)
class Algorithm:
    """One algorithm have a name a estimator and a loss function(and me ofc)"""

    name: str
    estimator: Estimator
    loss: Loss

# --------------------------------------------------------------------------
# policy gradient
# --------------------------------------------------------------------------

def grpo(
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    group_size: Optional[int] = None,
) -> Algorithm:
    """Group Relative Policy Optimization (DeepSeekMath, arXiv:2402.03300)::

        J = E[ 1/G sum_i  1/|o_i| sum_t
               min(rho * A_i, clip(rho, 1-eps, 1+eps) * A_i) - beta * KL ]

    with ``A_i = (R_i - mean_group R) / std_group R``, and ``rho`` each token's
    importance ratio against the policy that sampled it.
    """
    return Algorithm(
        name="grpo",
        estimator=partial(estim.group_relative, group_size=group_size),
        loss=partial(
            loss.clipped_surrogate, agg="seq-mean",
            clip_eps=clip_eps, kl_coef=kl_coef,
        ),
    )


def dr_grpo(
    *,
    max_tokens: int,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    group_size: Optional[int] = None,
) -> Algorithm:
    """Dr. GRPO (arXiv:2503.20783).
    """
    return Algorithm(
        name="dr_grpo",
        estimator=partial(estim.group_relative, group_size=group_size, std=False),
        loss=partial(
            loss.clipped_surrogate, agg="budget", max_tokens=max_tokens,
            clip_eps=clip_eps, kl_coef=kl_coef,
        ),
    )


def cispo(
    *,
    eps_low: float = 0.2,
    eps_high: float = 4.0,
    group_size: Optional[int] = None,
) -> Algorithm:
    """Clipped IS-weight Policy Optimization (MiniMax-M1, arXiv:2506.13585)::
        J = E[ 1/sum_i|o_i| sum_i sum_t
               sg(clip(rho, 1-eps_low, 1+eps_high)) * A_i * log pi(o_it) ]
    """
    return Algorithm(
        name="cispo",
        estimator=partial(estim.group_relative, group_size=group_size),
        loss=partial(
            loss.cispo_surrogate, agg="token-mean",
            eps_low=eps_low, eps_high=eps_high,
        ),
    )


def reinforce(*, gamma: float = 1.0, baseline: bool = True) -> Algorithm:
    """Vanilla policy gradient with a Monte-Carlo return estimate::
        grad J = 1/N sum_i sum_t (G_t - b) grad log pi(a_t | s_t)
    """
    return Algorithm(
        name="reinforce",
        estimator=partial(estim.monte_carlo, gamma=gamma, baseline=baseline),
        loss=partial(loss.score_function, agg="seq-sum"),
    )


# --------------------------------------------------------------------------
# offline
# --------------------------------------------------------------------------

def sft() -> Algorithm:
    """ sft """
    return Algorithm(
        name="sft",
        estimator=estim.demonstrations,
        loss=partial(loss.cross_entropy, agg="token-mean"),
    )


def dpo(*, beta: float = 0.1) -> Algorithm:
    """Direct Preference Optimization (arXiv:2305.18290)."""
    return Algorithm(
        name="dpo",
        estimator=estim.preferences,
        loss=partial(loss.preference, beta=beta),
    )
