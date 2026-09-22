from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}

@dataclass
class TemplateOutput:
    prompt_ids: List[int]
    prompt_text: str

class HFChatTemplate:
    """chat template using HF tokenizer
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        template_kwargs: Optional[Dict[str, object]] = None,
    ):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
    def apply(self, messages: List[Message], add_generation_prompt: bool = True) -> TemplateOutput:
        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )
        prompt_ids = list(getattr(out, "input_ids", out))
        prompt_text = self.tokenizer.decode(prompt_ids)
        return TemplateOutput(
            prompt_ids=prompt_ids,
            prompt_text=prompt_text
        )

    def user_turn_ids(self, content: str) -> List[int]:
        """Ids that close the assistant turn, add a user message, reopen the assistant turn."""
        return self._turn_suffix([{"role": "user", "content": content}])

    def tool_turn_ids(self, responses: List[str]) -> List[int]:
        """Same as ``user_turn_ids``, with one tool message per response."""
        return self._turn_suffix([{"role": "tool", "content": r} for r in responses])

    def _turn_suffix(self, messages: List[Message]) -> List[int]:
        """Ids to append after sampled tokens, starting with the eos the sampler dropped.

        Read off the template by diffing two renders that differ only in the new
        message's content. Both include a follower, since Qwen3 renders an
        assistant turn differently once something follows it.
        """
        eos = self.tokenizer.eos_token_id
        base = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        role = messages[0]["role"]
        probes = [
            self.apply(base + [{"role": role, "content": c}], add_generation_prompt=True).prompt_ids
            for c in ("0", "1")
        ]
        head = _common_prefix(*probes)  # ends inside the new turn's header
        full = self.apply(base + messages, add_generation_prompt=True).prompt_ids
        if full[: len(head)] != head or eos not in head:
            raise ValueError("chat template does not render this turn as a token suffix")
        cut = len(head) - 1 - head[::-1].index(eos)  # eos closing the assistant turn
        return list(full[cut:])


def _common_prefix(a: List[int], b: List[int]) -> List[int]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return list(a[:n])
