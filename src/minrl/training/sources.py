"""
`RolloutSource`: offline ones stream a fixed dataset
`DatasetSource`:  The Trainer only ever calls ``next_batch()``
"""

from __future__ import annotations

import random
from typing import Any, List, Sequence

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
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps

    def next_batch(self) -> Batch:
        rollouts: List[Rollout] = []
        for _ in range(self.batch_size):
            self.agent.reset()
            rollouts.append(episode(self.agent, self.env, self.max_episode_steps))
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
