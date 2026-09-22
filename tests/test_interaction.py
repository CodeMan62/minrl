"""Tests for the interaction layer (rollout + InteractionProtocol).

Uses minimal fake env/agent so the tests pin down the protocol's behaviour
independently of any concrete environment.
"""

from typing import Optional, Tuple

from minrl.agents.agent import BaseAgent
from minrl.envs.singleagent import SingleAgentEnv
from minrl.interaction import (
    InteractionProtocol,
    SingleAgentProtocol,
    episode,
)
from minrl.types import Info, Observation, Rollout, StepOutPut


class CountingEnv(SingleAgentEnv):
    """Deterministic env: reward 1.0 per step, terminates after `episode_len`."""

    def __init__(self, episode_len: int = 3):
        self.episode_len = episode_len
        self.t = 0

    def env_reset(self, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self.t = 0
        return self.get_obs(), {}

    def env_step(self, action: int) -> StepOutPut:
        self.t += 1
        terminated = self.t >= self.episode_len
        return StepOutPut(
            obs=self.get_obs(),
            reward=1.0,
            terminated=terminated,
            truncated=False,
            info={},
        )

    def get_obs(self) -> Observation:
        return f"t={self.t}"

    def sys_prompt(self) -> str:
        return "Counting env: reward 1 per step."


class ConstantAgent(BaseAgent):
    def act(self, obs: Observation) -> int:
        return 0


class TracingAgent(BaseAgent):
    """Simulates an LLM agent that exposes a token trace for its last action."""

    def act(self, obs: Observation) -> int:
        self.last_token_ids = [1, 2, 3]
        self.last_logprobs = [-0.1, -0.2, -0.3]
        self.last_action_mask = [1, 1, 1]
        return 0


def test_run_returns_single_rollout_list():
    proto = SingleAgentProtocol(CountingEnv(3), ConstantAgent(), num_steps=6)
    rollouts = proto.run()
    assert isinstance(proto, InteractionProtocol)
    assert isinstance(rollouts, list)
    assert len(rollouts) == 1
    assert isinstance(rollouts[0], Rollout)


def test_rollout_step_count_and_reward():
    # one episode: the env ends it after 3 steps even though 6 are allowed
    r = SingleAgentProtocol(CountingEnv(3), ConstantAgent(), num_steps=6).run()[0]
    assert len(r.steps) == 3
    assert r.total_reward == 3.0  # 1.0 reward per step


def test_termination_ends_the_episode():
    r = SingleAgentProtocol(CountingEnv(3), ConstantAgent(), num_steps=6).run()[0]
    assert [s.terminated for s in r.steps] == [False, False, True]
    assert r.terminated and not r.truncated
    assert r.steps[-1].next_obs == "t=3"


def test_max_steps_caps_the_episode():
    r = SingleAgentProtocol(CountingEnv(10), ConstantAgent(), num_steps=4).run()[0]
    assert len(r.steps) == 4
    assert not r.terminated and not r.truncated


def test_token_trace_defaults_none_for_non_llm_agent():
    r = SingleAgentProtocol(CountingEnv(3), ConstantAgent(), num_steps=3).run()[0]
    assert all(s.token_ids is None for s in r.steps)
    assert all(s.logprobs is None for s in r.steps)
    assert all(s.action_mask is None for s in r.steps)


def test_token_trace_captured_from_agent():
    r = SingleAgentProtocol(CountingEnv(3), TracingAgent(), num_steps=3).run()[0]
    assert r.steps[0].token_ids == [1, 2, 3]
    assert r.steps[0].logprobs == [-0.1, -0.2, -0.3]
    assert r.steps[0].action_mask == [1, 1, 1]


def test_free_episode_function_matches_protocol():
    # SingleAgentProtocol.run() should wrap the same episode() body.
    r = episode(ConstantAgent(), CountingEnv(3), 4)
    assert len(r.steps) == 3
    assert r.index == 3
