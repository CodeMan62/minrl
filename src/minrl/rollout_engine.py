"""RolloutEngine: plays rollouts, and schedules many of them at once."""
from __future__ import annotations

import asyncio
import itertools
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.types import Rollout, Step, RolloutRequest, Group

Actor = Tuple[BaseAgent, env]  # one fresh, stateful (agent, env) pair



class RolloutEngine:
    """Plays ``make()``-built (agent, env) pairs, ``concurrency`` requests at once.

    A request is ``group_size`` rollouts under one shared seed. It carries a
    unique id, can be cancelled while in flight, and is replaced the moment it
    ends, so generation never idles waiting on a consumer.
    """

    def __init__(
        self,
        make: Callable[[], Actor],
        *,
        max_steps: int = 100,
        group_size: int = 1,
        concurrency: int = 1,
        max_queued: Optional[int] = None,
        seed: int = 0,
    ):
        if min(max_steps, group_size, concurrency) <= 0:
            raise ValueError("max_steps, group_size and concurrency must be positive")
        self.make = make
        self.max_steps = max_steps
        self.group_size = group_size
        self.concurrency = concurrency
        self.tick = 0
        self._seeds = itertools.count(seed)
        self._queue: "asyncio.Queue[tuple]" = asyncio.Queue(maxsize=max_queued or concurrency)
        self._requests: Dict[str, asyncio.Task] = {}
        self._closed = False

    # ---- playing ---------------------------------------------------------

    async def one_rollout(self, request: RolloutRequest) -> Rollout:
        """Act/step until the env ends, the agent runs out of context, or ``max_steps``."""
        agent, environment = request.agent, request.env
        agent.reset()
        r = Rollout(index=0, steps=[], total_reward=0.0, terminated=False, truncated=False, info={})
        obs, _ = await environment.env_reset(seed=request.seed)
        for step in range(self.max_steps):
            action = await agent.act(obs)
            out = await environment.env_step(action)
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

    async def generate_rollout(self, requests: List[RolloutRequest]) -> List[Rollout]:
        """Play every request concurrently."""
        return list(await asyncio.gather(*(self.one_rollout(r) for r in requests)))

    # ---- scheduling ------------------------------------------------------

    async def __aenter__(self) -> "RolloutEngine":
        for _ in range(self.concurrency):
            self._submit()
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    def _submit(self) -> str:
        """Launch one request; returns its id without waiting for it."""
        request_id, seed = uuid.uuid4().hex, next(self._seeds)
        requests = [RolloutRequest(*self.make(), seed=seed) for _ in range(self.group_size)]
        task = asyncio.ensure_future(self._play(request_id, requests, self.tick))
        self._requests[request_id] = task
        # A done callback, not a try/finally inside _play: a task cancelled
        # before its first step never runs its own body, but this always fires.
        task.add_done_callback(lambda _t, rid=request_id: self._on_done(rid))
        return request_id

    async def _play(self, request_id: str, requests: List[RolloutRequest], started: int) -> None:
        """Play one request and queue its result; the put is what applies backpressure."""
        try:
            rollouts = await self.generate_rollout(requests)
        except Exception as e:  # noqa: BLE001 -- handed to the consumer, not swallowed
            await self._queue.put((request_id, started, None, e))
            return
        await self._queue.put((request_id, started, rollouts, None))

    def _on_done(self, request_id: str) -> None:
        """Drop a finished request and start its replacement."""
        self._requests.pop(request_id, None)
        if not self._closed:
            self._submit()

    def cancel(self, request_id: str) -> bool:
        """Cancel an in-flight request; ``False`` if it already finished."""
        task = self._requests.get(request_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def generate(self) -> Group:
        """The oldest finished request, blocking until one is ready."""
        request_id, started, rollouts, error = await self._queue.get()
        if error is not None:
            raise error
        staleness = self.tick - started
        self.tick += 1
        return Group(request_id, rollouts, staleness)

    async def aclose(self) -> None:
        """Cancel every in-flight request and stop replacing them."""
        self._closed = True
        tasks = list(self._requests.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
