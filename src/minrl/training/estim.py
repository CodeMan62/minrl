"""Credit assignment for all algorithms"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence as Seq, Tuple
from minrl.types import Batch, Rollout, Sequence

EPS = 1e-6
#: what the Trainer hands to a loss, plus the metrics of this step
Assignment = Tuple[List, Dict[str, float]]

def monte_carlo(
    batch: Batch, *, gamma: float = 1.0, baseline: bool = True
) -> Assignment:
    """REINFORCE: credit step ``t`` with the discounted return that followed it::
        G_t = sum_{k>=t} gamma^(k-t) * r_k
        A_t = G_t - mean(G)     if baseline else G_t
    """
    if not batch.rollouts:
        raise ValueError("no batch of rollouts")
    rollouts = batch.rollouts
    weights = [_reward_to_go([s.reward for s in r.steps], gamma) for r in rollouts]
    if baseline:
        mean = torch.tensor([g for w in weights for g in w]).mean().item()
        weights = [[g - mean for g in w] for w in weights]
    return _assign(rollouts, weights)


def group_relative(
    batch: Batch, *, group_size: Optional[int] = None, std: bool = True
) -> Assignment:
    """GRPO family: credit every step of a rollout with its episode return
    measured against the other rollouts of the same prompt::
        A_i = (R_i - mean_g R) / (std_g R + eps)     std=True   GRPO, CISPO
        A_i =  R_i - mean_g R                        std=False  Dr. GRPO
    """
    if not batch.rollouts:
        raise ValueError("no batch of rollouts")
    rollouts = batch.rollouts 
    returns = torch.tensor([r.total_reward for r in rollouts], dtype=torch.float32)
    advantages = torch.cat(
        [_relative(g, std) for g in _groups(returns, group_size)]
    )
    weights = [[a] * len(r.steps) for a, r in zip(advantages.tolist(), rollouts)]
    return _assign(rollouts, weights)


def demonstrations(batch: Batch) -> Assignment:
    """SFT: sequences to imitate, every response token weighing the same."""
    seqs = [_from_example(ex) for ex in batch.examples]
    n_tokens = sum(sum(s.action_mask) for s in seqs)
    if not n_tokens:
        raise ValueError("batch has no unmasked response tokens.")
    return seqs, {"n_tokens": float(n_tokens)}


def preferences(batch: Batch) -> Assignment:
    """DPO: ``(chosen, rejected)`` pairs -- the signal is their ordering, so
    credit assignment is nothing more than pairing them up."""
    pairs = [
        (_from_example(ex["chosen"]), _from_example(ex["rejected"]))
        for ex in batch.examples
    ]
    n_tokens = sum(sum(c.action_mask) + sum(r.action_mask) for c, r in pairs)
    return pairs, {"n_tokens": float(n_tokens)}


def _reward_to_go(rewards: Seq[float], gamma: float) -> List[float]:
    """Discounted sum of each step's own reward and everything after it."""
    out, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    return out[::-1]

def _groups(returns: torch.Tensor, group_size: Optional[int]) -> List[torch.Tensor]:
    """Split returns into consecutive same-prompt groups; one group if unset."""
    n = returns.numel()
    if group_size is None or group_size >= n:
        return [returns]
    if n % group_size:
        raise ValueError(
            f"batch of {n} rollouts does not split into groups of {group_size}"
        )
    return list(returns.split(group_size))


def _relative(returns: torch.Tensor, std: bool) -> torch.Tensor:
    """Center one group's returns, optionally scaling by its spread."""
    if returns.numel() < 2:  # std of a single sample is undefined
        return torch.zeros_like(returns)
    advantages = returns - returns.mean()
    return advantages / (returns.std() + EPS) if std else advantages


def _assign(rollouts: Seq[Rollout], weights: Seq[Seq[float]]) -> Assignment:
    """Flatten rollouts into one sequence per generated turn, carrying its weight.

    Turns the agent generated nothing for are dropped.  Zero-weight sequences
    are kept -- they still count toward the loss normalizer -- and ``n_live``
    reports how many carry signal, so the Trainer can skip a step where none do.
    """
    seqs: List[Sequence] = []
    for rollout, ws in zip(rollouts, weights):
        for step, w in zip(rollout.steps, ws):
            if not step.token_ids or not any(step.action_mask or ()):
                continue
            seqs.append(
                Sequence(
                    token_ids=list(step.token_ids),
                    action_mask=list(step.action_mask),
                    logprobs=list(step.logprobs) if step.logprobs else None,
                    advantage=w,
                )
            )

    returns = torch.tensor([r.total_reward for r in rollouts], dtype=torch.float32)
    return seqs, {
        "mean_return": returns.mean().item(),
        "std_return": returns.std().item() if returns.numel() > 1 else 0.0,
        "n_tokens": float(sum(sum(s.action_mask) for s in seqs)),
        "n_live": float(sum(1 for s in seqs if s.advantage)),
    }


def _from_example(example: Dict[str, List[int]]) -> Sequence:
    """Offline example -> Sequence; no behaviour policy, no weight."""
    return Sequence(
        token_ids=list(example["token_ids"]),
        action_mask=list(example["action_mask"]),
    )
