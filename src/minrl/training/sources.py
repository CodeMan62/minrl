"""Where a trainer step's data comes from. The Trainer only calls ``next_batch()``."""
from __future__ import annotations

import asyncio
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from minrl.rollout_engine import Actor, Group, RolloutEngine
from minrl.types import Batch, BatchSource


class RolloutSource(BatchSource):
    """Trainer-facing adapter: runs a :class:`~minrl.rollout_engine.RolloutEngine`
    on its own background event loop -- the only synchronous thing about
    this class -- and packs each :meth:`next_batch` call's ``batch_size //
    group_size`` groups into one :class:`Batch`, with ``meta["staleness"]``
    and ``["staleness_max"]`` read straight off the groups' own staleness.

    ``drop_uniform`` is DAPO's dynamic sampling: a group whose rollouts all
    got the same return has zero group-relative advantage, so it is dropped
    and another generated in its place (``meta["dropped_groups"]``). Without
    it such groups still count in the loss normalizer and only dilute the
    step. After ``max_redraws`` times the requested groups it gives up and
    serves what it has, so a too-easy or too-hard dataset cannot hang training.
    """

    def __init__(
        self,
        make: Callable[[], Actor],
        *,
        batch_size: int,
        max_steps: int = 100,
        group_size: int = 1,
        concurrency: int = 1,
        max_queued: Optional[int] = None,
        seed: int = 0,
        drop_uniform: bool = False,
        max_redraws: int = 4,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size % group_size:
            raise ValueError(f"batch_size={batch_size} is not a multiple of group_size={group_size}")
        if max_redraws < 1:
            raise ValueError("max_redraws must be at least 1")
        self.groups_per_batch = batch_size // group_size
        self.drop_uniform = drop_uniform
        self.max_redraws = max_redraws
        self.engine = self._engine(
            make, max_steps=max_steps, group_size=group_size, concurrency=concurrency,
            max_queued=max_queued, seed=seed,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._call(self.engine.__aenter__())

    def _engine(self, make: Callable[[], Actor], **kwargs) -> RolloutEngine:
        """The engine to run; subclasses swap in their own scheduling."""
        return RolloutEngine(make, **kwargs)

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _groups(self, n: int) -> Tuple[List[Group], List[Group]]:
        """The groups to train on, and the uniform ones dropped on the way."""
        if not self.drop_uniform:
            return [await self.engine.generate() for _ in range(n)], []
        kept: List[Group] = []
        uniform: List[Group] = []
        for _ in range(n * self.max_redraws):
            group = await self.engine.generate()
            (kept if _varies(group) else uniform).append(group)
            if len(kept) == n:
                return kept, uniform
        # Out of draws: top up with uniform groups, which carry no signal but keep the batch shape.
        short = n - len(kept)
        return kept + uniform[:short], uniform[short:]

    def next_batch(self) -> Batch:
        groups, dropped = self._call(self._groups(self.groups_per_batch))
        staleness = [g.staleness for g in groups]
        # Kept groups are biased towards ~50% success; this is the policy's real rate.
        drawn = [r.total_reward for g in groups + dropped for r in g.rollouts]
        return Batch(
            rollouts=[r for g in groups for r in g.rollouts],
            meta={
                "staleness": sum(staleness) / len(staleness),
                "staleness_max": float(max(staleness)),
                "dropped_groups": float(len(dropped)),
                "sampled_return": sum(drawn) / len(drawn),
            },
        )

    async def _cancel(self, request_id: str) -> bool:
        return self.engine.cancel(request_id)

    def cancel(self, request_id: str) -> bool:
        """Cancel one in-flight request by id; the hop lands Task.cancel() on the engine's loop."""
        return self._call(self._cancel(request_id))

    async def _state(self) -> Dict[str, int]:
        return self.engine.state_dict()

    def state_dict(self) -> Dict[str, int]:
        return self._call(self._state())

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self._call(self.engine.restore(state))

    def close(self) -> None:
        """Cancel every in-flight request and stop the background loop."""
        self._call(self.engine.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def __enter__(self) -> "RolloutSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _varies(group: Group) -> bool:
    """True if the group's rollouts did not all earn the same return."""
    returns = [r.total_reward for r in group.rollouts]
    return max(returns) != min(returns)


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

    def state_dict(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "order": list(self._order), "rng": self.rng.getstate()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.epoch, self._order = state["epoch"], list(state["order"])
        self.rng.setstate(state["rng"])
