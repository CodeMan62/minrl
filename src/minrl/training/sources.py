"""Batch sources for the Trainer."""

from __future__ import annotations

from typing import List

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env as Env
from minrl.interaction import episode
from minrl.types import Batch, BatchSource, Rollout


class RolloutSource(BatchSource):

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
        group: List[Rollout] = []
        for _ in range(self.batch_size):
            self.agent.reset()
            group.append(episode(self.agent, self.env, self.max_episode_steps))
        return Batch(rollouts=group)


# TODO: add DatasetSource 
