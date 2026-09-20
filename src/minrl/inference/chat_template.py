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

    def tool_turn_ids(self, responses: List[str]) -> List[int]:
        """Token ids that close the assistant turn, add one ``tool`` message per
        response, and reopen the assistant turn -- appended verbatim after the
        sampled ids, so the model's own tokens are never re-tokenized.

        The sampler drops the eos it stopped on, so the segment starts with it.
        Everything after it is read off the model's own template by diffing two
        renders that share a dummy prefix.
        """
        eos = self.tokenizer.eos_token_id
        base = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        tools = [{"role": "tool", "content": r} for r in responses]
        prefix = self.apply(base, add_generation_prompt=False).prompt_ids
        full = self.apply(base + tools, add_generation_prompt=True).prompt_ids
        if full[: len(prefix)] != prefix or eos not in prefix:
            raise ValueError("chat template does not render tool turns as a token suffix")
        # prefix ends `...<eos>` plus whatever the template puts after it (`\n` on Qwen)
        cut = len(prefix) - 1 - prefix[::-1].index(eos)
        return list(prefix[cut:]) + list(full[len(prefix):])

