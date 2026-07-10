from typing import List, Optional

import torch

from minrl.inference.client import InferenceClient
from minrl.types import ChatResponse


class HFClient(InferenceClient):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def complete_tokens(
        self,
        prompt_token_ids: List[int],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> ChatResponse:
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            # Override the model's bundled generation_config (Qwen3 ships
            # top_k=20), which would otherwise warp the sampling distribution.
            gen_kwargs["top_k"] = 0
        if stop_token_ids:
            gen_kwargs["eos_token_id"] = stop_token_ids

        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model.generate(
                input_ids, attention_mask=attention_mask, **gen_kwargs
            )
        finally:
            if was_training:
                self.model.train()

        completion_ids = out.sequences[0, len(prompt_token_ids):].tolist()
        # ``scores`` are the (warped) logits actually sampled from, one row per
        # generated token; log-softmax + gather recovers the sample's logprob.
        logprobs = [
            torch.log_softmax(score[0].float(), dim=-1)[tok].item()
            for score, tok in zip(out.scores, completion_ids)
        ]

        stops = set(stop_token_ids or [])
        finished = bool(completion_ids) and completion_ids[-1] in stops
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return ChatResponse(
            text=text,
            token_idx=completion_ids,
            logprobs=logprobs,
            finish_reason="stop" if finished else "length",
        )
