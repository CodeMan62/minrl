"""Where a trainer step's data comes from. The Trainer only calls ``next_batch()``."""
from __future__ import annotations

import asyncio
import itertools
import random
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

from minrl.interaction import InteractionProtocol
from minrl.types import Batch, BatchSource, Rollout

Group = Tuple[int, List[Rollout]]  # (batches served when it started, its rollouts)


class RolloutSource(BatchSource):
    """.
    Async BatchSource
    """

    def __init__(
        self,
        make: Callable[[], InteractionProtocol],
        *,
        batch_size: int,
        group_size: int = 1,
        concurrency: int = 1,
        max_queued: Optional[int] = None,
        seed: int = 0,
    ):
        if batch_size <= 0 or group_size <= 0 or concurrency <= 0:
            raise ValueError("batch_size, group_size and concurrency must be positive")
        if batch_size % group_size:
            raise ValueError(f"batch_size={batch_size} is not a multiple of group_size={group_size}")
        self.groups_per_batch = batch_size // group_size
        self.served = 0
        self._seeds = itertools.count(seed)
        self._queue: asyncio.Queue[Group] = asyncio.Queue(maxsize=max_queued or concurrency)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        protocols = [[make() for _ in range(group_size)] for _ in range(concurrency)]
        self._workers = self._call(self._spawn(protocols))

    async def _spawn(self, protocols: List[List[InteractionProtocol]]) -> List[asyncio.Task]:
        return [asyncio.create_task(self._worker(group)) for group in protocols]

    async def _worker(self, protocols: List[InteractionProtocol]) -> None:
        """Play groups forever; block when the queue is full."""
        while True:
            seed, started = next(self._seeds), self.served
            runs = await asyncio.gather(*(p.run(seed) for p in protocols))
            await self._queue.put((started, [r for run in runs for r in run]))

    async def _take(self) -> List[Group]:
        """Oldest groups for one batch; re-raises if a worker died instead."""
        take = asyncio.ensure_future(_collect(self._queue, self.groups_per_batch))
        done, _ = await asyncio.wait({take, *self._workers}, return_when=asyncio.FIRST_COMPLETED)
        if take not in done:
            take.cancel()
        for task in done:
            task.result()
        return take.result()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def next_batch(self) -> Batch:
        groups = self._call(self._take())
        staleness = [self.served - started for started, _ in groups]
        self.served += 1
        return Batch(
            rollouts=[r for _, rollouts in groups for r in rollouts],
            meta={
                "staleness": sum(staleness) / len(staleness),
                "staleness_max": float(max(staleness)),
            },
        )

    async def _shutdown(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def close(self) -> None:
        """Cancel in-flight episodes and stop the loop."""
        self._call(self._shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def __enter__(self) -> "RolloutSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()


async def _collect(queue: asyncio.Queue, n: int) -> list:
    return [await queue.get() for _ in range(n)]


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
