from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalConfig:
    name: str = "eval"                  # metric prefix: eval/success, eval/reward_mean, ...
    model: str = ""                     # recorded in results and logs
    num_episodes: int = 100             # plays seeds 0..num_episodes-1
    seeds: Optional[List[int]] = None   # an explicit list; overrides num_episodes
    samples_per_episode: int = 1        # >1 averages over samples; needs temperature > 0
    max_steps: int = 1
    concurrency: int = 32
    console: bool = True                # print a summary after each eval
    dump_dir: Optional[str] = None      # write every rollout as jsonl, one file per eval

    def episode_seeds(self) -> List[int]:
        seeds = list(range(self.num_episodes)) if self.seeds is None else list(self.seeds)
        if not seeds:
            raise ValueError("eval needs at least one episode")
        if self.samples_per_episode < 1 or self.concurrency < 1 or self.max_steps < 1:
            raise ValueError("samples_per_episode, concurrency and max_steps must be positive")
        return seeds
