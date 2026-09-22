"""
`RolloutSource`: offline ones stream a fixed dataset
`DatasetSource`:  The Trainer only ever calls ``next_batch()``
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env as Env
from minrl.interaction import episode
from minrl.types import Batch, BatchSource, Rollout


class RolloutSource(BatchSource):
    """Fresh on-policy rollouts, ``batch_size`` per optimizer step."""

    def __init__(
        self,
        agent: BaseAgent,
        env: Env,
        *,
        batch_size: int = 8,
        max_episode_steps: int = 16,
        group_size: Optional[int] = None,
        group_seed: bool = False, # True resets each groupp of group_size
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if group_seed and (not group_size or batch_size % group_size):
            raise ValueError("group_seed needs a group_size that divides batch_size")
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.group_size = group_size
        self.group_seed = group_seed
        self.rng = random.Random(seed)

    def next_batch(self) -> Batch:
        rollouts: List[Rollout] = []
        seed: Optional[int] = None
        for i in range(self.batch_size):
            if self.group_seed and i % self.group_size == 0:
                seed = self.rng.randrange(2**31)
            self.agent.reset()
            rollouts.append(
                episode(self.agent, self.env, self.max_episode_steps, seed=seed)
            )
        return Batch(rollouts=rollouts)


class DatasetSource(BatchSource):
    """A fixed dataset, reshuffled each epoch, ``batch_size`` per step."""

    def __init__(
        self,
        examples: Sequence[Any],
        *,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not examples:
            raise ValueError("DatasetSource got an empty dataset.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.examples = list(examples)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.epoch = 0  # 1-based once the first batch is served
        self._order: List[int] = []

    def next_batch(self) -> Batch:
        picked: List[Any] = []
        while len(picked) < self.batch_size:
            if not self._order:
                self._order = list(range(len(self.examples)))
                if self.shuffle:
                    self.rng.shuffle(self._order)
                self.epoch += 1
            take = self._order[: self.batch_size - len(picked)]
            self._order = self._order[len(take) :]
            picked.extend(self.examples[j] for j in take)
        return Batch(examples=picked, meta={"epoch": self.epoch})
