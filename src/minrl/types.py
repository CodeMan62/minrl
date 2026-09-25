from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # both import types.py at runtime; eager imports here are a cycle
    from minrl.agents.agent import BaseAgent
    from minrl.envs.env import env

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


Observation = str
Info = Dict[str, JSON]
Span = Tuple[int, int]  # [start, end) of a step's tokens in the rollout's sequence
# Agent level types
@dataclass
class Step:
    index: int
    prev_obs: Observation
    action: str
    next_obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Info
    # per-step mode: this step's own sequence
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    action_mask: Optional[List[int]] = None
    # multi-turn mode: this step's slice of the rollout's sequence
    span: Optional[Span] = None
    finish_reason: Optional[str] = None  # the sampler's, e.g. "stop" or "length"
@dataclass
class Rollout:
    index: int
    steps: List[Step]
    total_reward: float
    terminated: bool
    truncated: bool
    info: Info
    # multi-turn mode: the whole episode as one sequence
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    action_mask: Optional[List[int]] = None
# Env level types
@dataclass
class StepOutPut:
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Info

#-------------------inference types-----------------------
@dataclass
class ChatResponse:
    text: str
    token_idx: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    finish_reason: Optional[str] = None


#-------------------training types-----------------------

@dataclass
class Batch:
    """One trainer step's worth of data.

    Online RL fills ``rollouts``. Offline SFT/DPO fills ``examples``.
    """
    rollouts: Optional[List[Rollout]] = None
    examples: Optional[List[Any]] = None
    meta: Dict[str, JSON] = field(default_factory=dict)


class BatchSource:
    def next_batch(self) -> Batch:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Whatever a resume needs to continue the data stream; nothing by default."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None
#Training types
@dataclass
class Sequence:
    token_ids: List[int]
    action_mask: List[int]
    logprobs: Optional[List[float]] = None
    advantages: Union[float, List[float]] = 1.0  # one per sequence, or one per token

@dataclass(frozen=True)
class RolloutRequest:
    agent: BaseAgent
    env: env
    seed: Optional[int] = None

@dataclass(frozen=True)
class Group:
    request_id: str
    rollouts: List[Rollout]
    staleness: int