from abc import ABC, abstractmethod
from typing import List, Optional

from minrl.envs.env import env
from minrl.types import Rollout, Step
from minrl.agents.agent import BaseAgent


# interaction between agent and environment
def episode(
    agent: BaseAgent, env: env, max_steps: int = 100, *, seed: Optional[int] = None
) -> Rollout:
    """Act/step until the env ends, the agent runs out of context, or ``max_steps``."""
    r = Rollout(index=0, steps=[], total_reward=0, terminated=False, truncated=False, info={})
    obs, _ = env.reset(seed=seed)
    for step in range(max_steps):
        action = agent.act(obs)
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

    ``run()`` returns one :class:`Rollout` per learning-agent perspective, so
    single-agent setups return a list of length 1 while self-play returns one
    rollout per player. The trainer only calls ``run()`` and stays agnostic to
    the interaction style (single-turn, multi-turn, self-play, ...).
    """

    @abstractmethod
    def run(self) -> List[Rollout]:
        ...


class SingleAgentProtocol(InteractionProtocol):
    """One agent interacting with one env for ``num_steps`` (auto-resetting)."""

    def __init__(self, env: env, agent: BaseAgent, num_steps: int):
        self.env = env
        self.agent = agent
        self.num_steps = num_steps

    def run(self) -> List[Rollout]:
        self.agent.reset()
        return [episode(self.agent, self.env, self.num_steps)]
