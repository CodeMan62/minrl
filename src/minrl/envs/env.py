from abc import ABC, abstractmethod 
from typing import Optional, Dict, Tuple
from minrl.types import StepOutPut, Observation, Info



class env(ABC):
    """Base class for all envs"""
    @abstractmethod
    def reset(self, seed: Optional[int]) -> Tuple[Observation, Info]:
        """"""
        ...
    @abstractmethod
    def step(self, action) -> StepOutPut:
        """"""
        ...
    @abstractmethod
    def get_obs(self):
        """"""
        raise NotImplementedError
    @abstractmethod
    def sys_prompt(self) -> str:
        """A prompt for the system to use when interacting with the environment."""
        ...
