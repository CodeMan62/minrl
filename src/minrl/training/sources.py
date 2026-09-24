"""Where a trainer step's data comes from. The Trainer only calls ``next_batch()``."""
from __future__ import annotations

import asyncio
import random
import threading
from typing import Any, Callable, List, Optional, Sequence

from minrl.rollout_engine import Actor, RolloutEngine
from minrl.types import Batch, BatchSource


class RolloutSource(BatchSource):
    """Trainer-facing adapter: runs a :class:`~minrl.rollout_engine.RolloutEngine`
    on its own background event loop -- the only synchronous thing about
    this class -- and packs each :meth:`next_batch` call's ``batch_size //
    group_size`` groups into one :class:`Batch`, with ``meta["staleness"]``
    and ``["staleness_max"]`` read straight off the groups' own staleness.
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
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size % group_size:
            raise ValueError(f"batch_size={batch_size} is not a multiple of group_size={group_size}")
        self.groups_per_batch = batch_size // group_size
        self.engine = RolloutEngine(
            make, max_steps=max_steps, group_size=group_size, concurrency=concurrency,
            max_queued=max_queued, seed=seed,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._call(self.engine.__aenter__())

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def next_batch(self) -> Batch:
        groups = self._call(self.engine.generate_batch(self.groups_per_batch))
        staleness = [g.staleness for g in groups]
        return Batch(
            rollouts=[r for g in groups for r in g.rollouts],
            meta={
                "staleness": sum(staleness) / len(staleness),
                "staleness_max": float(max(staleness)),
            },
        )

    def cancel(self, request_id: str) -> bool:
        """Cancel one in-flight rollout request by id (see :meth:`RolloutEngine.cancel`)."""
        return self._call(self.engine.cancel(request_id))

    def close(self) -> None:
        """Cancel every in-flight request and stop the background loop."""
        self._call(self.engine.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


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
