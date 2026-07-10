"""This still needs improvements"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.interaction import episode
from minrl.loggers import Logger
from minrl.types import Rollout


@dataclass
class GRPOConfig:
    iterations: int = 200
    group_size: int = 8          # episodes per update (the "G" in GRPO)
    max_episode_steps: int = 16
    lr: float = 5e-6
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    micro_batch_size: int = 4    # sequences per forward/backward pass
    log_every: int = 1


class GRPOTrainer:
    def __init__(
        self,
        model,
        agent: BaseAgent,
        env: env,
        cfg: GRPOConfig,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.history: List[Dict[str, float]] = []
        self.logger = logger or Logger()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def step(self) -> Tuple[List[Rollout], Dict[str, float]]:
        """One GRPO iteration: collect a group of episodes, update the policy."""
        group = []
        for _ in range(self.cfg.group_size):
            self.agent.reset()
            group.append(episode(self.agent, self.env, self.cfg.max_episode_steps))
        metrics = self.update(group)
        self.history.append(metrics)
        self.logger.log(
            {f"train/{k}": v for k, v in metrics.items()}, step=len(self.history)
        )
        return group, metrics

    def train(self) -> List[Dict[str, float]]:
        for i in range(self.cfg.iterations):
            _, metrics = self.step()
            if self.cfg.log_every and (
                i % self.cfg.log_every == 0 or i == self.cfg.iterations - 1
            ):
                print(
                    f"[iter {i:>4}] return={metrics['mean_return']:+.3f} "
                    f"loss={metrics['loss']:+.4f} tokens={metrics['n_tokens']:.0f}"
                )
        return self.history

    def update(self, group: List[Rollout]) -> Dict[str, float]:
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
        self.model.train()
        self.optimizer.zero_grad()
        mb = self.cfg.micro_batch_size
        for i in range(0, len(seqs), mb):
            loss = self._microbatch_loss(seqs[i : i + mb], total_action_tokens)
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return {**base, "loss": total_loss, "n_tokens": float(total_action_tokens),
                "skipped": 0.0}

    def _microbatch_loss(self, batch, total_action_tokens: int) -> torch.Tensor:
        device = self.device
        max_len = max(len(ids) for ids, _, _, _ in batch)

        def pad(rows, value=0.0, dtype=torch.float32):
            return torch.tensor(
                [list(row) + [value] * (max_len - len(row)) for row in rows],
                dtype=dtype, device=device,
            )

        ids = pad([b[0] for b in batch], 0, torch.long)
        old_logp = pad([b[1] for b in batch])
        mask = pad([b[2] for b in batch])
        attn = pad([[1] * len(b[0]) for b in batch], 0, torch.long)
        adv = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)

        logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1]
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
            torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a,
        )
        # Sum here, normalize by the *global* action-token count so gradient
        # accumulation over micro-batches matches one big batch.
        return -(surrogate * tgt_mask).sum() / total_action_tokens
