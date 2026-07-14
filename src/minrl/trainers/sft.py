import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from minrl.loggers import Logger

Example = Dict[str, List[int]]


@dataclass
class SFTConfig:
    epochs: int = 3
    micro_batch_size: int = 8
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    log_every: int = 10
    shuffle: bool = True
    seed: int = 0


class SFTTrainer:
    """Imitate a dataset of ``(token_ids, action_mask)`` demonstrations."""

    def __init__(
        self,
        model,
        dataset: List[Example],
        cfg: SFTConfig,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.dataset: List[Example] = list(dataset)
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.logger = logger or Logger()
        self.history: List[Dict[str, float]] = []
        self._rng = random.Random(cfg.seed)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self) -> List[Dict[str, float]]:
        if not self.dataset:
            raise ValueError("SFTTrainer got an empty dataset.")
        step = 0
        mb = self.cfg.micro_batch_size
        for epoch in range(self.cfg.epochs):
            order = list(range(len(self.dataset)))
            if self.cfg.shuffle:
                self._rng.shuffle(order)
            for i in range(0, len(order), mb):
                batch = [self.dataset[j] for j in order[i : i + mb]]
                metrics = self.update(batch)
                step += 1
                metrics["epoch"] = float(epoch)
                self.history.append(metrics)
                self.logger.log(
                    {f"sft/{k}": v for k, v in metrics.items()}, step=step
                )
                if self.cfg.log_every and step % self.cfg.log_every == 0:
                    print(
                        f"[epoch {epoch} step {step:>5}] "
                        f"loss={metrics['loss']:.4f} acc={metrics['token_acc']:.3f} "
                        f"tokens={metrics['n_tokens']:.0f}"
                    )
        return self.history

    def update(self, batch: List[Example]) -> Dict[str, float]:
        """One optimizer step on a single micro-batch."""
        self.model.train()
        self.optimizer.zero_grad()
        loss, acc, n_tok = self._batch_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()
        return {"loss": loss.item(), "token_acc": acc, "n_tokens": float(n_tok)}

    def _batch_loss(
        self, batch: List[Example]
    ) -> Tuple[torch.Tensor, float, int]:
        device = self.device
        max_len = max(len(ex["token_ids"]) for ex in batch)

        def pad(rows, value, dtype):
            return torch.tensor(
                [list(row) + [value] * (max_len - len(row)) for row in rows],
                dtype=dtype, device=device,
            )

        ids = pad([ex["token_ids"] for ex in batch], 0, torch.long)
        mask = pad([ex["action_mask"] for ex in batch], 0, torch.float32)
        attn = pad([[1] * len(ex["token_ids"]) for ex in batch], 0, torch.long)

        logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        targets = ids[:, 1:]
        # mask/action_mask are indexed like ``ids``; shift to align with targets.
        tgt_mask = mask[:, 1:]
        # -cross_entropy == logprob of the realized token; avoids materializing
        # a full-vocab log_softmax (same trick as GRPO).
        logp = -F.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        )
        n_tok = tgt_mask.sum()
        denom = n_tok.clamp(min=1.0)
        loss = -(logp * tgt_mask).sum() / denom

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == targets).float() * tgt_mask).sum()
            acc = (correct / denom).item()
        return loss, acc, int(n_tok.item())
