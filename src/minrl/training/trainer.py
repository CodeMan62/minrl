from typing import Dict, Iterator, Optional

from torch import nn
import torch.optim as optim
from minrl.loggers import Logger
from minrl.training.algorithms import Algorithm
from minrl.training.config import TrainerConfig
from minrl.types import BatchSource


class Trainer:
    """Main trainer"""
    def __init__(
        self,
        model: nn.Module,
        *,
        algorithm: Algorithm,
        source: BatchSource,
        config: TrainerConfig,
        logger: Optional[Logger] = None,
        ref_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.algorithm = algorithm
        self.source = source
        self.cfg = config
        self.logger = logger
        self.ref_model = ref_model

        self.optimizer = self.initialize_optimizer()
    def initialize_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.model.parameters(),
         lr=self.cfg.lr,
         weight_decay=self.cfg.weight_decay,
         betas=(self.cfg.beta1, self.cfg.beta2),
         eps=self.cfg.eps
        )

    def train_step(self) -> Dict[str, float]:
        batch = self.source.next_batch()
        self.model.train()
        return self.algorithm.update(
            self.model,
            self.optimizer,
            batch,
            ref_model=self.ref_model,
            max_grad_norm=self.cfg.max_grad_norm,
            micro_batch_size=self.cfg.micro_batch_size,
            accum_steps=self.cfg.accum_steps,
        )

    def train(self, num_steps: int) -> Iterator[Dict[str, float]]:
        prefix = self.cfg.log_prefix
        for step in range(1, num_steps + 1):
            stats = self.train_step()
            stats = {**stats, "step": float(step)}
            if self.logger and self.cfg.log_every and step % self.cfg.log_every == 0:
                self.logger.log(
                    {f"{prefix}/{k}": v for k, v in stats.items() if k != "step"},
                    step=step,
                )
            yield stats

    def save(self, path: str) -> None:
        # TODO: implement save
        ...
