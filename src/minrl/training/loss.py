import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
# GRPO LOSS
def _grpo_microbatch_loss(
    model, batch, total_action_tokens: int, clip_eps: float
) -> torch.Tensor:
    device = next(model.parameters()).device
    max_len = max(len(ids) for ids, _, _, _ in batch)

    ids = _pad([b[0] for b in batch], max_len, 0, torch.long, device)
    old_logp = _pad([b[1] for b in batch], max_len, 0.0, torch.float32, device)
    mask = _pad([b[2] for b in batch], max_len, 0.0, torch.float32, device)
    attn = _pad([[1] * len(b[0]) for b in batch], max_len, 0, torch.long, device)
    adv = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    # -cross_entropy == logprob of the realized token; avoids materializing
    # a full-vocab log_softmax.
    new_logp = -F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )
    tgt_mask = mask[:, 1:]          # mask/old_logp indexed like targets
    old = old_logp[:, 1:]

    ratio = torch.exp(new_logp - old)
    a = adv[:, None]
    surrogate = torch.minimum(
        ratio * a,
        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a,
    )
    # Sum here, normalize by the *global* action-token count so gradient
    # accumulation over micro-batches matches one big batch.
    return -(surrogate * tgt_mask).sum() / total_action_tokens


def _reinforce_microbatch_loss(
    model, batch, total_action_tokens: int
) -> torch.Tensor:
    """On-policy REINFORCE: ``-E[A * log π(a|s)]`` over action tokens."""
    device = next(model.parameters()).device
    max_len = max(len(ids) for ids, _, _ in batch)

    ids = _pad([b[0] for b in batch], max_len, 0, torch.long, device)
    mask = _pad([b[1] for b in batch], max_len, 0.0, torch.float32, device)
    attn = _pad([[1] * len(b[0]) for b in batch], max_len, 0, torch.long, device)
    adv = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    logp = -F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )
    tgt_mask = mask[:, 1:]
    return -(logp * adv[:, None] * tgt_mask).sum() / total_action_tokens


def _sft_batch_loss(
    model, batch: List[Dict[str, List[int]]]
) -> Tuple[torch.Tensor, float, int]:
    device = next(model.parameters()).device
    max_len = max(len(ex["token_ids"]) for ex in batch)

    ids = _pad([ex["token_ids"] for ex in batch], max_len, 0, torch.long, device)
    mask = _pad([ex["action_mask"] for ex in batch], max_len, 0.0, torch.float32,
                device)
    attn = _pad([[1] * len(ex["token_ids"]) for ex in batch], max_len, 0,
                torch.long, device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
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

def _pad(rows, max_len: int, value, dtype, device) -> torch.Tensor:
    """Right-pad ``rows`` to ``max_len`` and stack into a single tensor."""
    return torch.tensor(
        [list(row) + [value] * (max_len - len(row)) for row in rows],
        dtype=dtype, device=device,
    )
