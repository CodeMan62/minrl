"""RolloutSource(drop_uniform=True): zero-advantage groups are swapped for fresh ones."""
import asyncio
from types import SimpleNamespace

from minrl.training.sources import RolloutSource


def _group(*returns):
    return SimpleNamespace(rollouts=[SimpleNamespace(total_reward=r) for r in returns], staleness=0)


class FakeEngine:
    def __init__(self, groups):
        self.groups = list(groups)

    async def generate(self):
        return self.groups.pop(0)


def _source(groups, **kw):
    src = RolloutSource.__new__(RolloutSource)  # skip the real engine and its event loop
    src.groups_per_batch, src.drop_uniform, src.max_redraws = 2, True, kw.get("max_redraws", 4)
    src.engine = FakeEngine(groups)
    return src


def test_uniform_groups_are_replaced():
    src = _source([_group(1, 1), _group(0, 1), _group(0, 0), _group(1, 0)])
    kept, dropped = asyncio.run(src._groups(2))
    assert [[r.total_reward for r in g.rollouts] for g in kept] == [[0, 1], [1, 0]]
    assert len(dropped) == 2


def test_gives_up_after_max_redraws_and_keeps_batch_shape():
    src = _source([_group(0, 0)] * 2 + [_group(0, 1)], max_redraws=1)
    kept, dropped = asyncio.run(src._groups(2))
    assert len(kept) == 2 and not dropped
