from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    # --- optim (AdamW) ---
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # --- train loop ---
    # memory knob only: one source batch is always exactly one optimizer step
    micro_batch_size: int = 4
    max_grad_norm: float = 1.0

    # --- dist (FSDP2; see dist.shard) ---
    strategy: str = "auto"  # auto | none | fsdp
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32: compute dtype under FSDP
    cpu_offload: bool = False
    activation_checkpointing: bool = False

    # --- checkpointing ---
    ckpt_dir: Optional[str] = None
    ckpt_every: int = 2  # a crash loses at most one finished step; 0 = only explicit saves
    keep_last: int = 3  # step_* checkpoints kept under ckpt_dir; 0 = all

    # --- weight sync into the inference runtime (Trainer.push_weights_to_runtime) ---
    sync_every: int = 1  # optimizer steps between pushes; 0 = never
    sync_host: str = "127.0.0.1"  # NCCL rendezvous the vLLM workers connect to
    sync_port: int = 29501
    sync_pause: Optional[str] = "keep"

    # --- logging ---
    log_prefix: str = "train"
    log_every: int = 1
