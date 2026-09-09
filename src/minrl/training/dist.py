"""for distributed training"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def setup(backend: Optional[str] = None) -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    return device

def teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main() -> bool:
    return rank() == 0

# --------------------------------------------------------------------------
# sharding
# --------------------------------------------------------------------------

def shard(
    model: nn.Module,
    *,
    strategy: str = "auto",
    mixed_precision: str = "bf16",
    cpu_offload: bool = False,
    reshard_after_forward: Optional[bool] = None,
) -> nn.Module:
    if strategy == "none" or (strategy == "auto" and world_size() == 1):
        return model
    if strategy not in ("auto", "fsdp"):
        raise ValueError(f"unknown strategy {strategy!r}; use auto | none | fsdp")
    if world_size() == 1:
        raise ValueError("strategy='fsdp' needs more than one rank; launch with torchrun")

    device = next(model.parameters()).device
    kwargs = dict(
        mesh=init_device_mesh(device.type, (world_size(),)),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=DTYPES[mixed_precision], reduce_dtype=torch.float32
        ),
    )
    if cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()

    blocks = set(getattr(model, "_no_split_modules", None) or ())
    for module in model.modules():
        if type(module).__name__ in blocks:
            fully_shard(module, **kwargs)
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward
    fully_shard(model, **kwargs)
    return model

# --------------------------------------------------------------------------
# collectives
# --------------------------------------------------------------------------
def sum_all(*values: float) -> list[float]:
    if not dist.is_initialized():
        return list(values)
    t = torch.tensor(values, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()

def mean_all(metrics: Dict[str, float]) -> Dict[str, float]:
    if not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    t = torch.tensor([metrics[k] for k in keys], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return dict(zip(keys, t.tolist()))

def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()

# --------------------------------------------------------------------------
# checkpoint state: full (unsharded) on rank 0, whatever the wrapping
# --------------------------------------------------------------------------
def full_state(model: nn.Module, optimizer: torch.optim.Optimizer):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return (
        get_model_state_dict(model, options=options),
        get_optimizer_state_dict(model, optimizer, options=options),
    )

def load_state(model: nn.Module, optimizer: torch.optim.Optimizer, model_state, optim_state) -> None:
    options = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=dist.is_initialized()
    )
    set_model_state_dict(model, model_state, options=options)
    set_optimizer_state_dict(model, optimizer, optim_state, options=options)
