
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.interaction import episode
from minrl.loggers import Logger
from minrl.types import Rollout

Example = Dict[str, List[int]]


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


def _grpo_microbatch_loss(
    model, batch, total_action_tokens: int, clip_eps: float
) -> torch.Tensor:
    device = next(model.parameters()).device
    max_len = max(len(ids) for ids, _, _, _ in batch)

    ids = _pad([b[0] for b in batch], max_len, 0, torch.long, device)
    old_logp = _pad([b[1] for b in batch], max_len, 0.0, torch.float32, device)
    mask = _pad([b[2] for b in batch], max_len, 0.0, torch.float32, device)
    attn = _pad([[1] * len(b[0]) for b in batch], max_len, 0, torch.long, device)
    adv = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    # -cross_entropy == logprob of the realized token; avoids materializing
    # a full-vocab log_softmax.
    new_logp = -F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )
    tgt_mask = mask[:, 1:]          # mask/old_logp indexed like targets
    old = old_logp[:, 1:]

    ratio = torch.exp(new_logp - old)
    a = adv[:, None]
    surrogate = torch.minimum(
        ratio * a,
        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a,
    )
    # Sum here, normalize by the *global* action-token count so gradient
    # accumulation over micro-batches matches one big batch.
    return -(surrogate * tgt_mask).sum() / total_action_tokens


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


def _sft_batch_loss(
    model, batch: List[Example]
) -> Tuple[torch.Tensor, float, int]:
    device = next(model.parameters()).device
    max_len = max(len(ex["token_ids"]) for ex in batch)

    ids = _pad([ex["token_ids"] for ex in batch], max_len, 0, torch.long, device)
    mask = _pad([ex["action_mask"] for ex in batch], max_len, 0.0, torch.float32,
                device)
    attn = _pad([[1] * len(ex["token_ids"]) for ex in batch], max_len, 0,
                torch.long, device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    # mask/action_mask are indexed like ``ids``; shift to align with targets.
    tgt_mask = mask[:, 1:]
    # -cross_entropy == logprob of the realized token; avoids materializing
    # a full-vocab log_softmax (same trick as GRPO).
    logp = -F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )
    n_tok = tgt_mask.sum()
    denom = n_tok.clamp(min=1.0)
    loss = -(logp * tgt_mask).sum() / denom

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets).float() * tgt_mask).sum()
        acc = (correct / denom).item()
    return loss, acc, int(n_tok.item())


def _pad(rows, max_len: int, value, dtype, device) -> torch.Tensor:
    """Right-pad ``rows`` to ``max_len`` and stack into a single tensor."""
    return torch.tensor(
        [list(row) + [value] * (max_len - len(row)) for row in rows],
        dtype=dtype, device=device,
    )
