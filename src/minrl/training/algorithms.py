import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.interaction import episode
from minrl.loggers import Logger
from minrl.types import Rollout
from minrl.training.loss import (
    _dpo_batch_loss,
    _grpo_microbatch_loss,
    _reinforce_microbatch_loss,
    _sft_batch_loss,
)

Example = Dict[str, List[int]]
# One preference pair: {"chosen": Example, "rejected": Example} -- prompt +
DPOExample = Dict[str, Example]


# --------------------------------------------------------------------------
# GRPO
# --------------------------------------------------------------------------

def grpo(
    model,
    agent: BaseAgent,
    env: env,
    *,
    iterations: int = 200,
    group_size: int = 8,
    max_episode_steps: int = 16,
    lr: float = 5e-6,
    clip_eps: float = 0.2,
    max_grad_norm: float = 1.0,
    micro_batch_size: int = 4,
    log_every: int = 1,
    logger: Optional[Logger] = None,
) -> Iterator[Tuple[List[Rollout], Dict[str, float]]]:
    """Train ``model`` with GRPO, yielding ``(group, metrics)`` per iteration."""
    logger = logger or Logger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(iterations):
        group = []
        for _ in range(group_size):
            agent.reset()
            group.append(episode(agent, env, max_episode_steps))
        metrics = _grpo_update(
            model, optimizer, group,
            clip_eps=clip_eps,
            max_grad_norm=max_grad_norm,
            micro_batch_size=micro_batch_size,
        )
        logger.log({f"train/{k}": v for k, v in metrics.items()}, step=i + 1)
        if log_every and (i % log_every == 0 or i == iterations - 1):
            print(
                f"[iter {i:>4}] return={metrics['mean_return']:+.3f} "
                f"loss={metrics['loss']:+.4f} tokens={metrics['n_tokens']:.0f}"
            )
        yield group, metrics


def _grpo_update(
    model,
    optimizer: torch.optim.Optimizer,
    group: List[Rollout],
    *,
    clip_eps: float,
    max_grad_norm: float,
    micro_batch_size: int,
) -> Dict[str, float]:
    returns = torch.tensor([r.total_reward for r in group], dtype=torch.float32)
    mean_r, std_r = returns.mean().item(), returns.std().item()
    base = {"mean_return": mean_r, "std_return": std_r}

    if std_r < 1e-6:
        # Every episode got the same return -> all advantages are zero.
        return {**base, "loss": 0.0, "n_tokens": 0.0, "skipped": 1.0}
    advantages = (returns - returns.mean()) / (returns.std() + 1e-6)

    # One training sequence per env step: (token_ids, behaviour logprobs,
    # action mask, episode advantage).
    seqs = []
    for adv, r in zip(advantages.tolist(), group):
        for s in r.steps:
            if s.token_ids and s.action_mask and any(s.action_mask):
                seqs.append((s.token_ids, s.logprobs, s.action_mask, adv))
    if not seqs:
        return {**base, "loss": 0.0, "n_tokens": 0.0, "skipped": 1.0}

    total_action_tokens = sum(sum(mask) for _, _, mask, _ in seqs)
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i in range(0, len(seqs), micro_batch_size):
        loss = _grpo_microbatch_loss(
            model, seqs[i : i + micro_batch_size], total_action_tokens, clip_eps
        )
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {**base, "loss": total_loss, "n_tokens": float(total_action_tokens),
            "skipped": 0.0}


# --------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------

def sft(
    model,
    dataset: List[Example],
    *,
    epochs: int = 3,
    micro_batch_size: int = 8,
    lr: float = 1e-5,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    shuffle: bool = True,
    seed: int = 0,
    logger: Optional[Logger] = None,
) -> Iterator[Dict[str, float]]:
    """Imitate a dataset of ``(token_ids, action_mask)`` demonstrations.

    Yields one metrics dict per optimizer step.
    """
    if not dataset:
        raise ValueError("sft got an empty dataset.")
    dataset = list(dataset)
    logger = logger or Logger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)

    step = 0
    for epoch in range(epochs):
        order = list(range(len(dataset)))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), micro_batch_size):
            batch = [dataset[j] for j in order[i : i + micro_batch_size]]
            metrics = _sft_update(model, optimizer, batch, max_grad_norm)
            step += 1
            metrics["epoch"] = float(epoch)
            logger.log({f"sft/{k}": v for k, v in metrics.items()}, step=step)
            if log_every and step % log_every == 0:
                print(
                    f"[epoch {epoch} step {step:>5}] "
                    f"loss={metrics['loss']:.4f} acc={metrics['token_acc']:.3f} "
                    f"tokens={metrics['n_tokens']:.0f}"
                )
            yield metrics


def _sft_update(
    model,
    optimizer: torch.optim.Optimizer,
    batch: List[Example],
    max_grad_norm: float,
) -> Dict[str, float]:
    """One optimizer step on a single micro-batch."""
    model.train()
    optimizer.zero_grad()
    loss, acc, n_tok = _sft_batch_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": loss.item(), "token_acc": acc, "n_tokens": float(n_tok)}


# --------------------------------------------------------------------------
# REINFORCE
# --------------------------------------------------------------------------

def reinforce(
    model,
    agent: BaseAgent,
    env: env,
    *,
    iterations: int = 200,
    batch_size: int = 8,
    max_episode_steps: int = 16,
    lr: float = 5e-6,
    max_grad_norm: float = 1.0,
    micro_batch_size: int = 4,
    log_every: int = 1,
    logger: Optional[Logger] = None,
) -> Iterator[Tuple[List[Rollout], Dict[str, float]]]:
    """Train ``model`` with vanilla REINFORCE (weight = episode return)."""
    yield from _reinforce_loop(
        model, agent, env,
        iterations=iterations,
        batch_size=batch_size,
        max_episode_steps=max_episode_steps,
        lr=lr,
        max_grad_norm=max_grad_norm,
        micro_batch_size=micro_batch_size,
        log_every=log_every,
        logger=logger,
        use_baseline=False,
    )


def reinforce_baseline(
    model,
    agent: BaseAgent,
    env: env,
    *,
    iterations: int = 200,
    batch_size: int = 8,
    max_episode_steps: int = 16,
    lr: float = 5e-6,
    max_grad_norm: float = 1.0,
    micro_batch_size: int = 4,
    log_every: int = 1,
    logger: Optional[Logger] = None,
) -> Iterator[Tuple[List[Rollout], Dict[str, float]]]:
    """REINFORCE with a batch-mean return baseline: weight = ``R - mean(R)``."""
    yield from _reinforce_loop(
        model, agent, env,
        iterations=iterations,
        batch_size=batch_size,
        max_episode_steps=max_episode_steps,
        lr=lr,
        max_grad_norm=max_grad_norm,
        micro_batch_size=micro_batch_size,
        log_every=log_every,
        logger=logger,
        use_baseline=True,
    )


def _reinforce_loop(
    model,
    agent: BaseAgent,
    env: env,
    *,
    iterations: int,
    batch_size: int,
    max_episode_steps: int,
    lr: float,
    max_grad_norm: float,
    micro_batch_size: int,
    log_every: int,
    logger: Optional[Logger],
    use_baseline: bool,
) -> Iterator[Tuple[List[Rollout], Dict[str, float]]]:
    logger = logger or Logger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(iterations):
        batch = []
        for _ in range(batch_size):
            agent.reset()
            batch.append(episode(agent, env, max_episode_steps))
        metrics = _reinforce_update(
            model, optimizer, batch,
            use_baseline=use_baseline,
            max_grad_norm=max_grad_norm,
            micro_batch_size=micro_batch_size,
        )
        logger.log({f"train/{k}": v for k, v in metrics.items()}, step=i + 1)
        if log_every and (i % log_every == 0 or i == iterations - 1):
            print(
                f"[iter {i:>4}] return={metrics['mean_return']:+.3f} "
                f"loss={metrics['loss']:+.4f} tokens={metrics['n_tokens']:.0f}"
            )
        yield batch, metrics


def _reinforce_update(
    model,
    optimizer: torch.optim.Optimizer,
    batch: List[Rollout],
    *,
    use_baseline: bool,
    max_grad_norm: float,
    micro_batch_size: int,
) -> Dict[str, float]:
    returns = torch.tensor([r.total_reward for r in batch], dtype=torch.float32)
    mean_r, std_r = returns.mean().item(), returns.std().item()
    base = {"mean_return": mean_r, "std_return": std_r}

    if use_baseline:
        if std_r < 1e-6:
            # Constant return across the batch -> zero advantages.
            return {**base, "loss": 0.0, "n_tokens": 0.0, "skipped": 1.0}
        weights = returns - returns.mean()
    else:
        weights = returns

    # One training sequence per env step: (token_ids, action_mask, weight).
    seqs = []
    for w, r in zip(weights.tolist(), batch):
        for s in r.steps:
            if s.token_ids and s.action_mask and any(s.action_mask):
                seqs.append((s.token_ids, s.action_mask, w))
    if not seqs:
        return {**base, "loss": 0.0, "n_tokens": 0.0, "skipped": 1.0}

    total_action_tokens = sum(sum(mask) for _, mask, _ in seqs)
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i in range(0, len(seqs), micro_batch_size):
        loss = _reinforce_microbatch_loss(
            model, seqs[i : i + micro_batch_size], total_action_tokens
        )
        loss.backward()
        total_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {**base, "loss": total_loss, "n_tokens": float(total_action_tokens),
            "skipped": 0.0}


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------

def dpo(
    model,
    ref_model,
    dataset: List[DPOExample],
    *,
    epochs: int = 1,
    micro_batch_size: int = 4,
    lr: float = 1e-6,
    beta: float = 0.1,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    shuffle: bool = True,
    seed: int = 0,
    logger: Optional[Logger] = None,
) -> Iterator[Dict[str, float]]:
    """Direct Preference Optimization over a dataset of preference pairs.

    Each example is ``{"chosen": {"token_ids": [...], "action_mask": [...]},
    "rejected": {...}}`` -- the same shape as an SFT example, once per side.
    ``ref_model`` should be a frozen copy of the policy DPO starts from (e.g.
    the SFT checkpoint); it is only ever read under ``torch.no_grad()`` and is
    never updated. Yields one metrics dict per optimizer step.
    """
    if not dataset:
        raise ValueError("dpo got an empty dataset.")
    dataset = list(dataset)
    logger = logger or Logger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    step = 0
    for epoch in range(epochs):
        order = list(range(len(dataset)))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), micro_batch_size):
            batch = [dataset[j] for j in order[i : i + micro_batch_size]]
            metrics = _dpo_update(
                model, ref_model, optimizer, batch, beta, max_grad_norm
            )
            step += 1
            metrics["epoch"] = float(epoch)
            logger.log({f"dpo/{k}": v for k, v in metrics.items()}, step=step)
            if log_every and step % log_every == 0:
                print(
                    f"[epoch {epoch} step {step:>5}] "
                    f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                    f"margin={metrics['margin']:+.3f}"
                )
            yield metrics


def _dpo_update(
    model,
    ref_model,
    optimizer: torch.optim.Optimizer,
    batch: List[DPOExample],
    beta: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """One optimizer step on a single micro-batch."""
    model.train()
    ref_model.eval()
    optimizer.zero_grad()
    loss, accuracy, margin = _dpo_batch_loss(model, ref_model, batch, beta)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": loss.item(), "accuracy": accuracy, "margin": margin}
