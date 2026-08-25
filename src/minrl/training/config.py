from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    # --- optim ---
    optim_name: str = "adamw"
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # --- train loop ---
    micro_batch_size: int = 4
    accum_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 0

    # --- dist (FSDP / DDP / none) ---
    strategy: str = "auto"  # auto | none | ddp | fsdp
    backend: str = "nccl"
    sharding: str = "full_shard"  # full_shard | shard_grad_op | hybrid_shard | no_shard
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32
    wrap: str = "transformer"  # transformer | size | none
    transformer_layer_cls: Optional[str] = None  # None = detect Qwen/Llama/... blocks
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    collect: str = "summon"  # summon | sharded | external
    timeout_min: int = 30

    # --- logging ---
    log_prefix: str = "train"
    log_every: int = 1
