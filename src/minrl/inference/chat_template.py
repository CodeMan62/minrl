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

