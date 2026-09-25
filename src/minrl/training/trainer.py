"""The one update loop."""
from __future__ import annotations

import os
import warnings
from dataclasses import asdict
from typing import Dict, Iterator, Optional
import torch
import torch.optim as optim
from torch import nn
from minrl.loggers import Logger
from minrl.training import checkpoint, dist
from minrl.training.algorithms import Algorithm
from minrl.training.config import TrainerConfig
from minrl.types import BatchSource


class Trainer:
    """Trainer of minrl"""
    def __init__(
        self,
        model: nn.Module,
        *,
        algorithm: Algorithm,
        source: BatchSource,
        config: TrainerConfig,
        logger: Optional[Logger] = None,
        ref_model: Optional[nn.Module] = None,
        weight_synchronizer=None,
        sync_every: int = 1,
    ):
        self.algorithm = algorithm
        self.source = source
        self.cfg = config
        self.logger = logger
        self.weight_synchronizer = weight_synchronizer
        self.sync_every = sync_every
        self.step = 0
        self.tokens = 0  # action tokens trained on so far, across every rank
        self.sequences = 0

        if config.activation_checkpointing:
            model.gradient_checkpointing_enable()
        self.model = self._shard(model)
        self.ref_model = ref_model
        if ref_model is not None:
            # Frozen once here, not re-frozen on every step.
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            # No backward ever follows a ref forward: reshard its root too.
            self.ref_model = self._shard(ref_model, reshard_after_forward=True)
        self.optimizer = self.initialize_optimizer()

    def _shard(self, model: nn.Module, **kwargs) -> nn.Module:
        return dist.shard(
            model,
            strategy=self.cfg.strategy,
            mixed_precision=self.cfg.mixed_precision,
            cpu_offload=self.cfg.cpu_offload,
            **kwargs,
        )

    def initialize_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def train_step(self) -> Dict[str, float]:
        algo = self.algorithm
        # 1. collect experience -- fresh rollouts, or the next slice of a dataset
        batch = self.source.next_batch()

        # 2. credit assignment -- rewards become the weight each token carries
        items, metrics = algo.estimator(batch)
        metrics = {**{k: float(v) for k, v in batch.meta.items() if isinstance(v, (int, float))}, **metrics}

        # Counts over the whole step, across every rank.
        world = dist.world_size()
        n_seqs, n_tokens, n_live = dist.sum_all(
            len(items), metrics["n_tokens"], metrics.get("n_live", len(items))
        )
        metrics["n_tokens"], metrics["n_seqs"] = n_tokens, n_seqs
        if not n_live:
            # Nothing to learn from: an all-zero-advantage batch. Stepping
            # anyway would still drift the weights via momentum/weight decay.
            return {**metrics, "loss": 0.0, "grad_norm": 0.0, "skipped": 1.0}
        if not items:
            raise RuntimeError(
                f"rank {dist.rank()} has no trainable sequences while others do; "
                "every rank must forward under FSDP"
            )

        # 3. loss + backward, one micro-batch at a time.  FSDP *averages*
        #    gradients across ranks, so each rank normalizes by its 1/world
        #    share of the global counts: the average then equals the gradient
        #    of one loss over every sequence on every GPU.
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        aux: Dict[str, float] = {}
        size = self.cfg.micro_batch_size
        for i in range(0, len(items), size):
            loss, extra = algo.loss(
                self.model, items[i : i + size],
                n_seqs=n_seqs / world, n_tokens=n_tokens / world,
                ref_model=self.ref_model,
            )
            loss.backward()
            total_loss += loss.item()
            # Every metric is already divided by a step-wide count, so summing
            # over micro-batches gives its value for the whole step.
            for k, v in extra.items():
                aux[k] = aux.get(k, 0.0) + v

        # 4. update the policy
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()

        return {
            **metrics,
            **dist.mean_all({**aux, "loss": total_loss}),
            "grad_norm": float(grad_norm),
            "skipped": 0.0,
        }

    def train(self, num_steps: int) -> Iterator[Dict[str, float]]:
        """Run until ``self.step == num_steps``; resumes from a loaded checkpoint."""
        cfg = self.cfg
        while self.step < num_steps:
            self.step += 1
            stats = self.train_step()
            self.tokens += int(stats["n_tokens"])
            self.sequences += int(stats["n_seqs"])
            if self.weight_synchronizer is not None and self.step % self.sync_every == 0:
                self.weight_synchronizer.send_weights()
            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save()
            stats = {**stats, "step": float(self.step), "tokens": float(self.tokens)}
            if self.logger and dist.is_main() and cfg.log_every and self.step % cfg.log_every == 0:
                self.logger.log(
                    {f"{cfg.log_prefix}/{k}": v for k, v in stats.items() if k != "step"},
                    step=self.step,
                )
            yield stats

    def close(self) -> None:
        """Release the batch source; a RolloutSource stops generating and joins its loop."""
        close = getattr(self.source, "close", None)
        if close is not None:
            close()

    # ---- checkpointing --------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Checkpoint everything a bit-exact resume needs. Every rank must call it."""
        path = path or os.path.join(self.cfg.ckpt_dir or "ckpts", f"step_{self.step:08d}")
        model_state, optim_state = dist.full_state(self.model, self.optimizer)
        checkpoint.save(
            path,
            model_state=model_state,
            optim_state=optim_state,
            trainer_state={
                "step": self.step,
                "tokens": self.tokens,
                "sequences": self.sequences,
                "algorithm": self.algorithm.name,
                "config": asdict(self.cfg),
                "topology": {
                    "world_size": dist.world_size(),
                    "strategy": self.cfg.strategy,
                    "mixed_precision": self.cfg.mixed_precision,
                    "cpu_offload": self.cfg.cpu_offload,
                },
                "torch": torch.__version__,
            },
            rank_state={"rng": checkpoint.rng_state(), "source": self.source.state_dict()},
            keep_last=self.cfg.keep_last,
        )

    def load(self, path: str, *, strict: bool = True) -> None:
        """Resume from ``path``: one checkpoint, or a ckpt_dir whose ``latest`` picks one."""
        ck = checkpoint.load(path, strict=strict)
        saved = ck.trainer
        if saved["algorithm"] != self.algorithm.name:
            raise ValueError(
                f"checkpoint trained {saved['algorithm']!r}, this trainer runs "
                f"{self.algorithm.name!r}; optimizer state would be meaningless"
            )
        changed = {k: (v, asdict(self.cfg).get(k)) for k, v in saved["config"].items()
                   if asdict(self.cfg).get(k) != v}
        if changed and dist.is_main():
            warnings.warn(f"config differs from the checkpoint's: {changed}", stacklevel=2)
        dist.load_state(self.model, self.optimizer, ck.model, ck.optimizer)
        self.step, self.tokens, self.sequences = saved["step"], saved["tokens"], saved["sequences"]
        if "rng" in ck.rank:
            checkpoint.set_rng_state(ck.rank["rng"])
        if "source" in ck.rank:
            self.source.load_state_dict(ck.rank["source"])
