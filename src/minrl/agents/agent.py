from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs):
        ...


