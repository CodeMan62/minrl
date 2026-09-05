from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from typing import Sequence
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncWorkerExtension:
    group: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None
    def init_communicator(
        self,
        host: str,
        port: int,
        world_size: str
    ) -> None:
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self._group = PyNcclCommunicator(
            pg, device=self.device
        )
        self.client_rank = world_size - 1

    def sync(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        dtype_x = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=dtype_x, device=self.device)
        self.group.broadcast(self, weight, src=self.client_rank)
        self.model_runner.model.load_weights(weight=[(name,weight)])
    def close(self) -> None:
        if self.group is not None:
            del self.group
            self.client_rank = None
