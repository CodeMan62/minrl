
from minrl.interaction import rollout
from minrl.types import Rollout
from typing import List


class OrchestratorConfig:
    episodes: int
    max_steps: int


class Orchestrator:
    """
    dumbest orchestrator ever
    """
    def __init__(self, env, agent, cfg: OrchestratorConfig):
        self.env = env
        self.agent = agent
        self.cfg = cfg
    async def train(self) -> List[Rollout]:
        rolouts = []
        for _ in range(self.cfg.episodes):
            env = self.env
            agent = self.agent
            rollout = await rollout(
                    agent=agent,
                    env=env,
                    num_steps=self.cfg.max_steps
                    )
            rolouts.append(rollout)
        return rolouts

