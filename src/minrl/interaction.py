from abc import ABC, abstractmethod
from typing import List

from minrl.envs.env import env
from minrl.types import Rollout, Step
from minrl.agents.agent import BaseAgent


# interaction between agent and environment
def rollout(agent: BaseAgent, env: env, num_steps) -> Rollout:
  rollout = Rollout(index=0, steps=[], total_reward=0, terminated=False, truncated=False, info={})
  obs, info = env.reset()
  for step in range(num_steps):
    action = agent.act(obs)
    out = env.step(action)
    rollout.steps.append(Step(
      index=step,
      prev_obs=obs,
      action=action,
      next_obs=out.obs,
      reward=out.reward,
      terminated=out.terminated,
      truncated=out.truncated,
      info=out.info,
      token_ids=getattr(agent, "last_token_ids", None),
      logprobs=getattr(agent, "last_logprobs", None),
      action_mask=getattr(agent, "last_action_mask", None)))
    rollout.total_reward += out.reward
    rollout.terminated = out.terminated
    rollout.truncated = out.truncated
    rollout.info = out.info
    obs = out.obs
    if out.terminated or out.truncated:
      obs, info = env.reset()
    rollout.index = len(rollout.steps)
  return rollout


def episode(agent: BaseAgent, env: env, max_steps: int = 100) -> Rollout:
  r = Rollout(index=0, steps=[], total_reward=0, terminated=False, truncated=False, info={})
  obs, _ = env.reset()
  for step in range(max_steps):
    action = agent.act(obs)
    out = env.step(action)
    r.steps.append(Step(
      index=step,
      prev_obs=obs,
      action=action,
      next_obs=out.obs,
      reward=out.reward,
      terminated=out.terminated,
      truncated=out.truncated,
      info=out.info,
      token_ids=getattr(agent, "last_token_ids", None),
      logprobs=getattr(agent, "last_logprobs", None),
      action_mask=getattr(agent, "last_action_mask", None)))
    r.total_reward += out.reward
    r.terminated = out.terminated
    r.truncated = out.truncated
    r.info = out.info
    obs = out.obs
    if out.terminated or out.truncated:
      break
  r.index = len(r.steps)
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
        return [rollout(self.agent, self.env, self.num_steps)]
