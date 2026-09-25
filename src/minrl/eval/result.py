from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from minrl.types import Rollout

Success = Callable[[Rollout], bool]
Label = Callable[[Rollout], str]


def generated_tokens(r: Rollout) -> int:
    """Tokens the policy sampled in this episode; prompts and observations excluded."""
    if r.action_mask is not None:  # multi-turn: one sequence for the episode
        return sum(r.action_mask)
    return sum(sum(s.action_mask or ()) for s in r.steps)


def was_truncated(r: Rollout) -> bool:
    """Cut off by a length limit: the sampler's token cap or the agent's context."""
    return r.truncated or any(s.finish_reason == "length" for s in r.steps)


@dataclass
class EvalResult:
    step: Optional[int]
    model: str
    success: float
    reward_mean: float
    reward_std: float
    steps_mean: float
    tokens_mean: float
    tokens_max: int
    truncated: float
    seconds: float
    tokens_per_s: float
    labels: Dict[str, float] = field(default_factory=dict)
    rollouts: List[List[Rollout]] = field(default_factory=list, repr=False)  # per seed

    @classmethod
    def from_rollouts(
        cls, rollouts: List[List[Rollout]], *, step: Optional[int], model: str,
        seconds: float, success: Success, label: Optional[Label] = None,
    ) -> "EvalResult":
        flat = [r for per_seed in rollouts for r in per_seed]
        n = len(flat)
        rewards = [r.total_reward for r in flat]
        tokens = [generated_tokens(r) for r in flat]
        return cls(
            step=step,
            model=model,
            success=sum(map(success, flat)) / n,
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards),
            steps_mean=statistics.fmean(len(r.steps) for r in flat),
            tokens_mean=statistics.fmean(tokens),
            tokens_max=max(tokens),
            truncated=sum(map(was_truncated, flat)) / n,
            seconds=seconds,
            tokens_per_s=sum(tokens) / seconds if seconds > 0 else 0.0,
            labels={k: v / n for k, v in sorted(Counter(map(label, flat)).items())} if label else {},
            rollouts=rollouts,
        )

    def metrics(self) -> Dict[str, float]:
        """Flat scalars, the form loggers take."""
        out = {
            "success": self.success,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "steps_mean": self.steps_mean,
            "tokens_mean": self.tokens_mean,
            "tokens_max": float(self.tokens_max),
            "truncated": self.truncated,
            "seconds": self.seconds,
            "tokens_per_s": self.tokens_per_s,
        }
        out.update({f"label/{k}": v for k, v in self.labels.items()})
        return out

    def summary(self, name: str = "eval") -> str:
        """A few human-readable lines for the console."""
        n_seeds = len(self.rollouts)
        samples = len(self.rollouts[0]) if self.rollouts else 0
        where = f" @ step {self.step}" if self.step is not None else ""
        model = f"{self.model} · " if self.model else ""
        lines = [
            f"[{name}{where}] {model}{n_seeds} episodes × {samples} · {self.seconds:.1f}s",
            f"  success {self.success:6.1%}   reward {self.reward_mean:.3f} ± {self.reward_std:.3f}"
            f"   steps {self.steps_mean:.1f}",
            f"  tokens  {self.tokens_mean:.0f} mean / {self.tokens_max} max"
            f"   truncated {self.truncated:.1%}   {self.tokens_per_s:.0f} tok/s",
        ]
        if self.labels:
            lines.append("  " + "   ".join(f"{k} {v:.1%}" for k, v in self.labels.items()))
        return "\n".join(lines)
