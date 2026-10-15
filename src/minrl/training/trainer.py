"""The one update loop."""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, Iterator, Optional
import torch
import torch.optim as optim
from torch import nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from minrl.inference.vllm_client import VLLMClient
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
        runtime: Optional[VLLMClient] = None,  # where the policy is served; pushed to after each step
    ):
        self.algorithm = algorithm
        self.source = source
        self.cfg = config
        self.logger = logger
        self.runtime = runtime
        self._runtime_link: Optional[PyNcclCommunicator] = None  # rank 0's NCCL group, opened on first push
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
            if self.runtime is not None and cfg.sync_every and self.step % cfg.sync_every == 0:
                self.push_weights_to_runtime()
            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save()
            stats = {**stats, "step": float(self.step), "tokens": float(self.tokens)}
            if self.logger and dist.is_main() and cfg.log_every and self.step % cfg.log_every == 0:
                self.logger.log(
                    {f"{cfg.log_prefix}/{k}": v for k, v in stats.items() if k != "step"},
                    step=self.step,
                )
            yield stats

    # ---- weight sync ----------------------------------------------------

    def push_weights_to_runtime(self) -> None:
        if self.runtime is None:
            return
        params = list(self.model.named_parameters())
        if not dist.is_main():
            with torch.no_grad():
                for _, p in params:
                    dist.full_tensor(p)
            return

        if self._runtime_link is None:
            self._runtime_link = self._open_runtime_link()
        link, cfg = self._runtime_link, self.cfg
        meta = [(name, str(p.dtype).removeprefix("torch."), list(p.shape)) for name, p in params]
        if cfg.sync_pause:
            self.runtime.pause(cfg.sync_pause)
        # The workers sit in receive_weights until the last broadcast lands,
        # so that request runs on a thread while we broadcast. If a broadcast
        # fails they stay stuck in NCCL, so never join the thread on the way out.
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(self.runtime.update_weights, meta, str(self.step))
            stream = torch.cuda.current_stream(link.device)
            with torch.no_grad():
                for _, p in params:
                    link.broadcast(dist.full_tensor(p).contiguous(), src=0, stream=stream)
            stream.synchronize()
            future.result()
        finally:
            exe.shutdown(wait=False)
        if cfg.sync_pause:
            self.runtime.resume()

    def _open_runtime_link(self) -> PyNcclCommunicator:
        """Rank 0 forms one NCCL group with the runtime's workers: itself as rank 0, them after."""
        cfg = self.cfg
        world_size = 1 + int(self.runtime.wait_ready()["world_size"])
        device = torch.device("cuda", torch.cuda.current_device())
        # The workers block inside init_weight_sync until rank 0 joins the
        # rendezvous, so the request runs on a thread while we join it.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(self.runtime.init_weight_sync, cfg.sync_host, cfg.sync_port, world_size)
            pg = StatelessProcessGroup.create(
                host=cfg.sync_host, port=cfg.sync_port, rank=0, world_size=world_size
            )
            link = PyNcclCommunicator(pg, device=device)
            future.result()
        return link

    def close(self) -> None:
        """Release the batch source and the runtime link."""
        close = getattr(self.source, "close", None)
        if close is not None:
            close()
        self._runtime_link = None

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
        # The runtime still holds the base weights; serve the resumed policy.
        self.push_weights_to_runtime()
