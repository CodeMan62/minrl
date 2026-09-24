"""How agents and envs meet to produce experience."""
from abc import ABC, abstractmethod
from typing import List, Optional

from minrl.envs.env import env
from minrl.types import Rollout, Step
from minrl.agents.agent import BaseAgent


async def episode(
    agent: BaseAgent, env: env, max_steps: int = 100, *, seed: Optional[int] = None
) -> Rollout:
    """Act/step until the env ends, the agent runs out of context, or ``max_steps``."""
    agent.reset()
    r = Rollout(index=0, steps=[], total_reward=0, terminated=False, truncated=False, info={})
    obs, _ = env.reset(seed=seed)
    for step in range(max_steps):
        action = await agent.act(obs)
        out = env.step(action)
        span = getattr(agent, "last_span", None)
        r.steps.append(Step(
            index=step,
            prev_obs=obs,
            action=action,
            next_obs=out.obs,
            reward=out.reward,
            terminated=out.terminated,
            truncated=out.truncated,
            info=out.info,
            token_ids=None if span else getattr(agent, "last_token_ids", None),
            logprobs=None if span else getattr(agent, "last_logprobs", None),
            action_mask=None if span else getattr(agent, "last_action_mask", None),
            span=span))
        r.total_reward += out.reward
        r.terminated = out.terminated
        r.truncated = out.truncated
        r.info = out.info
        obs = out.obs
        if out.terminated or out.truncated:
            break
        if getattr(agent, "truncated", False):
            r.truncated = True
            break
    r.index = len(r.steps)
    if r.steps and r.steps[-1].span is not None:
        r.token_ids = list(agent.last_token_ids)
        r.logprobs = list(agent.last_logprobs)
        r.action_mask = list(agent.last_action_mask)
    return r


class InteractionProtocol(ABC):
    """Defines *how* agents and an env interact to produce experience.

    ``run(seed)`` plays one episode and returns one :class:`Rollout` per
    learning-agent perspective: a single-agent setup returns a list of length
    1, self-play one rollout per player. A protocol owns its agent(s) and env,
    both stateful, so one protocol instance never runs two episodes at once.
    """

    @abstractmethod
    async def run(self, seed: Optional[int] = None) -> List[Rollout]:
        ...


class SingleAgentProtocol(InteractionProtocol):
    """One agent playing one env for at most ``max_steps`` per episode."""

    def __init__(self, agent: BaseAgent, env: env, max_steps: int = 100):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps

    async def run(self, seed: Optional[int] = None) -> List[Rollout]:
        return [await episode(self.agent, self.env, self.max_steps, seed=seed)]
