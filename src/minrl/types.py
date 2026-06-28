from dataclasses import dataclass
from typing import Dict, Union, Any, List
import json

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


Observation = str
Info = Dict[str, JSON]

# Env level types
@dataclass
class StepOutPut:
    obs: str
    reward: float
    terminated: bool
    truncated: bool
    info: str


