from abc import ABC, abstractmethod 
from typing import Optional, Dict
from minrl.types import StepOutPut


class env(ABC):
    """Base class for all envs"""
    @abstractmethod
    def reset(self, seed: Optional[int]) -> Dict[Tuple[]]:
        """"""
        ...
    @abstractmethod
    def step(self, action) -> StepOutPut:
        """"""
        ...
    @abstractmethod
    def get_obs(self) -> Dict[]:
        """"""
        raise NotImplementedError