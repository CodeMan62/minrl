"""Fully-async rollouts: generation runs ahead of the trainer by at most ``max_staleness`` policy versions.

Select it in a training script with::

    --rollout-source-path minrl.training.fully_async.FullyAsyncRolloutSource

Versions count optimizer steps: version ``v`` is the policy after ``v`` steps,
which is what vLLM serves once the trainer pushes it. The trainer reports
each push through :meth:`FullyAsyncRolloutSource.on_policy_update`; without
pushes the version never moves and the source stops admitting work.

Batch ``n`` is trained on by the policy at version ``n``, so a group started
under version ``v`` is ``n - v`` versions stale when it is trained on. Three
things keep that at or below ``max_staleness``:

- **admission** -- a group is only started while
  ``taken + outstanding < (version + max_staleness + 1) * groups_per_batch``,
  so it is due for a batch at most ``max_staleness`` versions on;
- **drop on arrival** -- a finished group that drops and redraws pushed
  past that is discarded (``meta["stale_dropped"]``);
- **cancel in flight** -- a group still generating that can no longer make
  it is cancelled, and its slot goes to a fresh one.

``max_staleness=0`` is synchronous on-policy RL: the next batch's groups
only start once the policy that will train on them is being served.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List

from minrl.rollout_engine import Actor, RolloutEngine
from minrl.training.sources import RolloutSource, _varies
from minrl.types import Batch, Rollout, RolloutRequest


@dataclass(frozen=True)
class VersionedGroup:
    request_id: str
    rollouts: List[Rollout]
    version_start: int  # policy version served when the group started
    version_end: int  # and when it finished; later if a push landed mid-generation


class FullyAsyncEngine(RolloutEngine):
    """A :class:`RolloutEngine` whose admission is bounded by policy version, not queue size."""

    def __init__(
        self,
        make: Callable[[], Actor],
        *,
        max_staleness: int,
        groups_per_batch: int,
        **kwargs,
    ):
        super().__init__(make, **kwargs)
        if max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        self.max_staleness = max_staleness
        self.groups_per_batch = groups_per_batch
        self.version = 0  # policy version the runtime serves
        self.served = 0  # batches handed to the trainer; the next one is batch ``served``
        self.taken = 0  # groups handed to the trainer, counting the batch being drawn
        self._starts: Dict[str, int] = {}  # in-flight request id -> version it started under
        self._held = False  # after a restore, until the runtime reports its version
        # Unbounded: admission is the bound, and a put must never stall a finished group.
        self._queue = asyncio.Queue()

    # ---- admission -------------------------------------------------------

    def _limit(self) -> int:
        return (self.version + self.max_staleness + 1) * self.groups_per_batch

    def _top_up(self) -> None:
        """Start requests until the concurrency or staleness budget is spent."""
        while (
            not self._closed and not self._held
            and len(self._requests) < self.concurrency
            and self.taken + len(self._requests) + self._queue.qsize() < self._limit()
        ):
            self._submit()

    def set_version(self, version: int) -> None:
        """The runtime now serves ``version``; that opens new admission slots."""
        self.version, self._held = version, False
        self._top_up()

    def release(self) -> None:
        """Give back a group's slot: it was drawn but will not be trained on."""
        self.taken -= 1
        self._top_up()

    def cancel_stale(self) -> int:
        """Cancel in-flight requests too old to be trained on from batch ``served`` on."""
        stale = [rid for rid, v in self._starts.items() if self.served - v > self.max_staleness]
        for rid in stale:
            self._starts.pop(rid)
            self.cancel(rid)
        return len(stale)

    # ---- scheduling ------------------------------------------------------

    async def __aenter__(self) -> "FullyAsyncEngine":
        self._top_up()
        return self

    def _submit(self) -> str:
        request_id, seed = uuid.uuid4().hex, self.next_seed
        self.next_seed += 1
        requests = [RolloutRequest(*self.make(), seed=seed) for _ in range(self.group_size)]
        task = asyncio.ensure_future(self._play(request_id, requests, self.version))
        self._requests[request_id] = task
        self._starts[request_id] = self.version
        task.add_done_callback(lambda _t, rid=request_id: self._on_done(rid))
        return request_id

    async def _play(self, request_id: str, requests: List[RolloutRequest], started: int) -> None:
        try:
            rollouts = await self.generate_rollout(requests)
        except Exception as e:  # noqa: BLE001 -- handed to the consumer, not swallowed
            self._queue.put_nowait(e)
            return
        self._queue.put_nowait(VersionedGroup(request_id, rollouts, started, self.version))

    def _on_done(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        self._starts.pop(request_id, None)
        self._top_up()

    async def generate(self) -> VersionedGroup:
        """The oldest finished group, counted as taken; :meth:`release` it to drop it."""
        while self._queue.empty():
            self._top_up()
            if not self._requests:
                raise RuntimeError(
                    "no rollouts in flight and none may start: the runtime is still at "
                    f"version {self.version} while batch {self.served} is due. Push weights "
                    "every step, or raise max_staleness to at least sync_every - 1."
                )
            await asyncio.wait(list(self._requests.values()), return_when=asyncio.FIRST_COMPLETED)
        item = self._queue.get_nowait()
        if isinstance(item, BaseException):
            raise item
        self.taken += 1  # the queue shrank by as much, so no slot opens
        return item

    async def aclose(self) -> None:
        await super().aclose()
        self._starts.clear()

    # ---- checkpointing ---------------------------------------------------

    def state_dict(self) -> Dict[str, int]:
        """Seed cursor and batch count. Saved between steps, so ``taken`` is ``served`` batches' worth."""
        return {"next_seed": self.next_seed, "served": self.served}

    async def restore(self, state: Dict[str, int]) -> None:
        """Drop everything in flight and wait for the runtime's version before admitting again."""
        await self.aclose()
        while not self._queue.empty():
            self._queue.get_nowait()
        self.next_seed, self.served = state["next_seed"], state["served"]
        self.taken = self.served * self.groups_per_batch
        # Until set_version, the runtime's weights are unknown; anything started now would be mislabelled.
        self.version, self._held, self._closed = self.served, True, False


class FullyAsyncRolloutSource(RolloutSource):
    """:class:`RolloutSource` with policy-version staleness control. See the module docstring.

    ``meta`` adds ``stale_dropped`` (groups discarded for staleness),
    ``version_span`` (mean pushes that landed while a kept group generated),
    ``wait_s`` (seconds the trainer blocked on this batch), and ``in_flight``
    and ``queued`` (the engine's load once the batch was drawn). ``staleness``
    is in policy versions.
    """

    def __init__(self, make: Callable[[], Actor], *, max_staleness: int = 1, **kwargs):
        self.max_staleness = max_staleness
        super().__init__(make, **kwargs)

    def _engine(self, make: Callable[[], Actor], **kwargs) -> FullyAsyncEngine:
        kwargs.pop("max_queued", None)  # admission bounds the queue instead
        return FullyAsyncEngine(
            make, max_staleness=self.max_staleness, groups_per_batch=self.groups_per_batch, **kwargs,
        )

    async def _batch(self) -> Batch:
        engine, n = self.engine, self.groups_per_batch
        engine.cancel_stale()
        kept: List[VersionedGroup] = []
        uniform: List[VersionedGroup] = []
        stale = draws = 0
        while len(kept) < n and draws < n * self.max_redraws:
            group = await engine.generate()
            if engine.served - group.version_start > self.max_staleness:
                engine.release()
                stale += 1
                continue
            draws += 1
            if not self.drop_uniform or _varies(group):
                kept.append(group)
            else:
                # Released at once: holding its slot while redrawing could starve admission.
                engine.release()
                uniform.append(group)
        # Out of draws: top up with uniform groups, which carry no signal but keep the batch shape.
        short = n - len(kept)
        groups, dropped = kept + uniform[:short], uniform[short:]
        engine.taken += short

        staleness = [engine.served - g.version_start for g in groups]
        span = [g.version_end - g.version_start for g in groups]
        # Kept groups are biased towards ~50% success; this is the policy's real rate.
        drawn = [r.total_reward for g in groups + dropped for r in g.rollouts]
        engine.served += 1
        return Batch(
            rollouts=[r for g in groups for r in g.rollouts],
            meta={
                "staleness": sum(staleness) / len(staleness),
                "staleness_max": float(max(staleness)),
                "version_span": sum(span) / len(span),
                "dropped_groups": float(len(dropped)),
                "stale_dropped": float(stale),
                "sampled_return": sum(drawn) / len(drawn),
                "in_flight": float(len(engine._requests)),
                "queued": float(engine._queue.qsize()),
            },
        )

    def next_batch(self) -> Batch:
        start = time.monotonic()
        batch = self._call(self._batch())
        batch.meta["wait_s"] = time.monotonic() - start
        return batch

    async def _set_version(self, version: int) -> None:
        self.engine.set_version(version)

    def on_policy_update(self, version: int) -> None:
        """Called by the trainer once the runtime serves the policy after ``version`` steps."""
        self._call(self._set_version(version))
