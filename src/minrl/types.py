from dataclasses import dataclass
from typing import Dict, Union, Any, List
import json

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


