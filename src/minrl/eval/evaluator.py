from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Callable, List, Optional

from minrl.eval.config import EvalConfig
from minrl.eval.result import EvalResult, Label, Success, generated_tokens, was_truncated
from minrl.loggers import Logger
from minrl.rollout_engine import Actor, RolloutEngine
from minrl.types import Rollout, RolloutRequest


def _solved(r: Rollout) -> bool:
    return r.total_reward > 0


class Eval:
    """Plays a fixed set of episodes, then reports to the console, a logger, and disk.

    Builds its own actors with ``make`` and never touches training state or RNG,
    so it is safe to run between training steps.
    """

    def __init__(
        self,
        config: EvalConfig,
        make: Callable[[], Actor],
        *,
        success: Success = _solved,
        label: Optional[Label] = None,
        logger: Optional[Logger] = None,
    ):
        self.config = config
        self.make = make
        self.success = success
        self.label = label
        self.logger = logger
        self._seeds = config.episode_seeds()  # validates the config up front

    def evaluate_sync(self, step: Optional[int] = None) -> EvalResult:
        return asyncio.run(self.evaluate(step))

    async def evaluate(self, step: Optional[int] = None) -> EvalResult:
        cfg = self.config
        engine = RolloutEngine(self.make, max_steps=cfg.max_steps)  # only for one_rollout
        limit = asyncio.Semaphore(cfg.concurrency)

        async def play(seed: int) -> Rollout:
            async with limit:  # build the actor inside, so only `concurrency` exist at once
                agent, environment = self.make()
                return await engine.one_rollout(RolloutRequest(agent, environment, seed))

        k = cfg.samples_per_episode
        start = time.perf_counter()
        flat = await asyncio.gather(*(play(s) for s in self._seeds for _ in range(k)))
        seconds = time.perf_counter() - start

        result = EvalResult.from_rollouts(
            [flat[i * k:(i + 1) * k] for i in range(len(self._seeds))],
            step=step, model=cfg.model, seconds=seconds,
            success=self.success, label=self.label,
        )

        if cfg.console:
            print(result.summary(cfg.name), flush=True)
        if self.logger is not None:
            self.logger.log({f"{cfg.name}/{k}": v for k, v in result.metrics().items()},
                            step=result.step)
        if cfg.dump_dir:
            os.makedirs(cfg.dump_dir, exist_ok=True)
            suffix = f"_step{step:08d}" if step is not None else ""
            with open(os.path.join(cfg.dump_dir, f"{cfg.name}{suffix}.jsonl"), "w") as f:
                for seed, per_seed in zip(self._seeds, result.rollouts):
                    for sample, r in enumerate(per_seed):
                        f.write(json.dumps({
                            "seed": seed,
                            "sample": sample,
                            "reward": r.total_reward,
                            "success": self.success(r),
                            "label": self.label(r) if self.label else None,
                            "tokens": generated_tokens(r),
                            "truncated": was_truncated(r),
                            "steps": [
                                {"obs": s.prev_obs, "action": s.action, "reward": s.reward,
                                 "finish_reason": s.finish_reason, "info": s.info}
                                for s in r.steps
                            ],
                        }, default=str) + "\n")
        return result
