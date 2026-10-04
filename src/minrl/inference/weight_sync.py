from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup

DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
DEFAULT_PACKED_NUM_BUFFERS = 2


class WeightSyncWorkerExtension:
    group: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None
    def init_communicator(
        self,
        host: str,
        port: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self.group = PyNcclCommunicator(pg, device=self.device())
        self.device = device
        self.client_rank = 0

    def sync(
        self,
        weights: Sequence[tuple[str, str, Sequence[int]]],
        model: Any,
    ) -> None:
        if self.group is None or self.device is None:
            raise RuntimeError("weight communicator is not initialized")
        stream = torch.cuda.current_stream(self.device)
        received: list[tuple[str, torch.Tensor]] = []
        for name, dtype_name, shape in weights:
            dtype = getattr(torch, dtype_name.rsplit(".", 1)[-1], None)
            if dtype is None or not isinstance(dtype, torch.dtype):
                raise ValueError(f"unknown torch dtype {dtype_name!r}")
            tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            self.group.broadcast(tensor, src=self.client_rank, stream=stream)
            received.append((name, tensor))
        model.load_weights(received)
        stream.synchronize()

    def close(self) -> None:
        self.group = None
        self.client_rank = None
