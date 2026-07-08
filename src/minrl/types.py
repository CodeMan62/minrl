from dataclasses import dataclass, field
from typing import Dict, Union, Any, List, Optional

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


Observation = str
Info = Dict[str, JSON]
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
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    action_mask: Optional[List[int]] = None
@dataclass
class Rollout:
    index: int
    steps: List[Step]
    total_reward: float
    terminated: bool
    truncated: bool
    info: Info
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



