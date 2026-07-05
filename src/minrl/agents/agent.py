from abc import ABC, abstractmethod

from minrl.types import Rollout


class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs):
        """Choose an action for the current observation."""
        ...

    def update(self, rollout: Rollout) -> dict:
        """Learn from a collected rollout. Returns a dict of metrics.

        Default is a no-op so reactive agents (random/scripted) work unchanged.
        Learning agents override this to update their policy/value estimates.
        """
        return {}

    def reset(self) -> None:
        """Hook called at the start of each rollout. Override if stateful."""
        return None
