from minrl.envs.env import env 
from abc import abstractmethod
from typing import Optional
from minrl.types import StepOutPut, Observation



class SingleAgentEnv(env):
    @abstractmethod
    def env_reset(self, seed: Optional[int]=None):
        ...
    @abstractmethod
    def env_step(self, action: int) -> StepOutPut:
        ...
    @abstractmethod
    def get_obs(self)->Observation:
        ...
    def reset(self, ):
        pass
    def step(action):
        pass
