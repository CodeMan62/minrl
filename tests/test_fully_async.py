"""FullyAsyncRolloutSource keeps every trained-on group within ``max_staleness`` policy versions."""
import asyncio
import random

import pytest

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.training.fully_async import FullyAsyncRolloutSource
from minrl.types import StepOutPut


class CoinAgent(BaseAgent):
    """Takes a random while to answer, and answers 0 or 1."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    async def act(self, obs):
        await asyncio.sleep(self.rng.uniform(0, 0.004))
        return self.rng.randint(0, 1)


class OneShot(env):
    async def env_reset(self, seed):
        return "q", {}

    async def env_step(self, action):
        return StepOutPut(obs="", reward=float(action), terminated=True, truncated=False, info={})

    def get_obs(self):
        return ""

    def sys_prompt(self):
        return ""


def _source(**kw):
    rng = random.Random(0)
    kw = {"batch_size": 8, "group_size": 2, "max_steps": 1, "concurrency": 8, **kw}
    return FullyAsyncRolloutSource(lambda: (CoinAgent(rng), OneShot()), **kw)


def _train(src, steps):
    """What Trainer.train does to its source: draw a batch, step, push, report the version."""
    metas = []
    for step in range(1, steps + 1):
        metas.append(src.next_batch().meta)
        src.on_policy_update(step)
    return metas


@pytest.mark.parametrize("max_staleness", [0, 1, 3])
@pytest.mark.parametrize("drop_uniform", [False, True])
def test_staleness_never_exceeds_the_bound(max_staleness, drop_uniform):
    with _source(max_staleness=max_staleness, drop_uniform=drop_uniform) as src:
        metas = _train(src, 12)
    assert max(m["staleness_max"] for m in metas) <= max_staleness


def test_zero_staleness_is_on_policy():
    with _source(max_staleness=0) as src:
        metas = _train(src, 6)
    assert all(m["staleness"] == 0 and m["version_span"] == 0 for m in metas)


def test_runs_ahead_when_allowed():
    with _source(max_staleness=2, concurrency=64) as src:
        metas = _train(src, 8)
    assert max(m["staleness_max"] for m in metas) > 0


def test_fails_loudly_when_weights_are_never_pushed():
    with _source(max_staleness=1) as src:
        src.next_batch()
        src.next_batch()  # batches 0 and 1 fit the budget of version 0
        with pytest.raises(RuntimeError, match="none may start"):
            src.next_batch()


def test_resume_waits_for_the_runtime_version():
    with _source(max_staleness=1) as src:
        _train(src, 3)
        state = src.state_dict()
    assert state["served"] == 3
    with _source(max_staleness=1) as src:
        src.load_state_dict(state)
        assert not src.engine._requests  # nothing starts on unknown weights
        src.on_policy_update(3)  # what Trainer.load does after its push
        meta = src.next_batch().meta
    assert meta["staleness_max"] <= 1
