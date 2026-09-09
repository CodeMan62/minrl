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
    ckpt_every: int = 0  # 0 = only on explicit trainer.save()

    # --- logging ---
    log_prefix: str = "train"
    log_every: int = 1
