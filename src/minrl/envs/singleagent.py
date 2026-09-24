from minrl.envs.env import env
from abc import abstractmethod
from typing import Optional
from minrl.types import StepOutPut, Observation



class SingleAgentEnv(env):
    """``reset``/``step`` are async so an env's own logic can await I/O (a
    judge-model call, a tool, a subprocess) without blocking every other
    rollout sharing the event loop it runs on. ``env_reset``/``env_step`` are
    where that logic goes; most envs need no ``await`` inside them at all.
    """

    @abstractmethod
    async def env_reset(self, seed: Optional[int]=None):
        ...
    @abstractmethod
    async def env_step(self, action: int) -> StepOutPut:
        ...
    @abstractmethod
    def get_obs(self)->Observation:
        ...