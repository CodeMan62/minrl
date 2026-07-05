from dataclasses import dataclass
from typing import Dict, List, Optional

from minrl.agents.agent import BaseAgent
from minrl.envs.env import env
from minrl.interaction import rollout as collect_rollout
from minrl.types import Rollout


@dataclass
class OrchestratorConfig:
    iterations: int = 100
    max_steps: int = 100
    log_every: int = 10
    seed: Optional[int] = None


def summarize(rollout: Rollout) -> Dict[str, float]:
    counts = {"win": 0, "loss": 0, "draw": 0, "illegal": 0}
    episodes = 0
    for step in rollout.steps:
        if step.terminated or step.truncated:
            episodes += 1
            if step.info.get("illegal_move"):
                counts["illegal"] += 1
            else:
                label = step.info.get("result")
                if label in counts:
                    counts[label] += 1
    return {
        "steps": float(len(rollout.steps)),
        "episodes": float(episodes),
        "total_reward": float(rollout.total_reward),
        "avg_return": rollout.total_reward / episodes if episodes else 0.0,
        "n_win": float(counts["win"]),
        "n_loss": float(counts["loss"]),
        "n_draw": float(counts["draw"]),
        "n_illegal": float(counts["illegal"]),
    }


class Orchestrator:
    """Minimal training loop: collect experience, let the agent learn, repeat."""

    def __init__(self, env: env, agent: BaseAgent, cfg: OrchestratorConfig):
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.history: List[Dict[str, float]] = []

    def train(self) -> List[Rollout]:
        rollouts: List[Rollout] = []
        for i in range(self.cfg.iterations):
            self.agent.reset()
            r = collect_rollout(self.agent, self.env, self.cfg.max_steps)
            self.agent.update(r)
            rollouts.append(r)

            stats = summarize(r)
            self.history.append(stats)
            if self._should_log(i):
                self._log(i, stats)
        return rollouts

    def _should_log(self, i: int) -> bool:
        if not self.cfg.log_every:
            return False
        return i % self.cfg.log_every == 0 or i == self.cfg.iterations - 1

    def _log(self, i: int, stats: Dict[str, float]) -> None:
        print(
            f"[iter {i:>4}] "
            f"reward={stats['total_reward']:+.1f} "
            f"episodes={stats['episodes']:.0f} "
            f"W/L/D/illegal="
            f"{stats['n_win']:.0f}/{stats['n_loss']:.0f}/"
            f"{stats['n_draw']:.0f}/{stats['n_illegal']:.0f}"
        )
