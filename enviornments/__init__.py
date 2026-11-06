"""The environments the examples train on, and the registry ``train.py`` picks from.

An :class:`EnvSpec` is everything the training script needs to know about an
environment that the environment class itself does not say: how to build one
for a split, which parser turns the model's text into an action, the system
prompt, how many steps an episode can take, and what counts as success.
Register a new environment by adding one entry to ``ENVS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from minrl.envs.env import env
from minrl.inference.parser import MoveParser, Parser, TextParser
from minrl.types import Rollout

from enviornments.dapo_math import DAPOMath
from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT
from enviornments.tic_tac_toe import TicTacToe

#: ``make(split, *, limit, shuffle, seed, **options)`` -> a fresh env.
#: ``options`` are the ``--env-arg key=value`` pairs from the command line.
MakeEnv = Callable[..., env]


@dataclass(frozen=True)
class EnvSpec:
    make: MakeEnv
    parser: Callable[[], Parser]
    max_steps: int
    system_prompt: Optional[str] = None
    success: Callable[[Rollout], bool] = lambda r: r.total_reward > 0
    label: Optional[Callable[[Rollout], str]] = None  # per-episode outcome, logged as eval/label/<x>
    metric: str = "accuracy"  # what ``success`` measures, for the console summary


def _solved(r: Rollout) -> bool:
    return r.total_reward > 0


TIC_TAC_TOE_SYSTEM_PROMPT = (
    "You are playing Tic-Tac-Toe against an opponent. Cells are numbered 0-8, "
    "left to right, top to bottom. Pick an empty cell. Reply with only the "
    "cell number you play, nothing else."
)


def tic_tac_toe_outcome(r: Rollout) -> str:
    """How a game ended, from the final step's info."""
    info = r.steps[-1].info
    return "illegal" if info.get("illegal_move") else info.get("result", "loss")


ENVS: Dict[str, EnvSpec] = {
    "gsm8k": EnvSpec(
        make=lambda split, *, limit, shuffle, seed, **_: GSM8K(
            split, limit=limit, shuffle=shuffle, seed=seed
        ),
        parser=TextParser,
        max_steps=1,
        system_prompt=GSM8K_SYSTEM_PROMPT,
        success=_solved,
    ),
    "dapo_math": EnvSpec(
        # Options: dataset, config, eval_dataset (see enviornments/dapo_math.py).
        make=lambda split, *, limit, shuffle, seed, **options: DAPOMath(
            split, limit=limit, shuffle=shuffle, seed=seed, **options
        ),
        parser=TextParser,
        max_steps=1,
        system_prompt=None,  # the task prompt carries the format instruction
        success=_solved,
    ),
    "tic_tac_toe": EnvSpec(
        make=lambda split, **_: TicTacToe(),  # no dataset: every seed is a game
        parser=MoveParser,
        max_steps=9,
        system_prompt=TIC_TAC_TOE_SYSTEM_PROMPT,
        success=lambda r: tic_tac_toe_outcome(r) == "win",
        label=tic_tac_toe_outcome,
        metric="win rate",
    ),
}

__all__ = ["ENVS", "EnvSpec", "DAPOMath", "GSM8K", "TicTacToe", "tic_tac_toe_outcome"]
