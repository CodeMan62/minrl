from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup

# (name, dtype without the "torch." prefix, shape): what to allocate for each broadcast.
ParamMeta = Tuple[str, str, Sequence[int]]


class WeightSyncWorkerExtension:
    weight_sync_group: Optional[PyNcclCommunicator] = None

    def init_weight_sync(self, host: str, port: int, world_size: int) -> None:
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=1 + get_world_group().rank, world_size=world_size
        )
        self.weight_sync_group = PyNcclCommunicator(pg, device=self.device)

    def receive_weights(self, weights: List[ParamMeta]) -> None:
        group = self.weight_sync_group
        if group is None:
            raise RuntimeError("weight sync not initialized; the trainer must call init_weight_sync")
        model = self.get_model()
        stream = torch.cuda.current_stream(self.device)
        for name, dtype, shape in weights:
            tensor = torch.empty(tuple(shape), dtype=getattr(torch, dtype), device=self.device)
            group.broadcast(tensor, src=0, stream=stream)
            model.load_weights([(name, tensor)])  # same stream, so ordered after the broadcast
        stream.synchronize()

    def close_weight_sync(self) -> None:
        self.weight_sync_group = None
