from typing import Dict, List, Optional

from transformers import AutoTokenizer

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class ChatTemplate:
    """Applies an HF chat template locally and returns **token ids** (token-in).

    Rendering + tokenizing on the client (rather than sending text to the server)
    keeps train and inference in lockstep: the same tokenizer produces the prompt
    ids we send to the sampler and the ids we later train on, so there is no
    template/tokenizer drift. Pair with :meth:`InferenceClient.complete_tokens`.
    """

    def __init__(
        self,
        model: str,
        *,
        trust_remote_code: bool = False,
        template_kwargs: Optional[Dict[str, object]] = None,
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        # Extra kwargs forwarded to ``apply_chat_template`` on every call,
        # e.g. ``{"enable_thinking": False}`` for Qwen3.
        self.template_kwargs = dict(template_kwargs or {})

    def encode(
        self, messages: List[Message], *, add_generation_prompt: bool = True
    ) -> List[int]:
        """Render ``messages`` through the chat template and tokenize to ids."""
        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )
        # Newer transformers returns a BatchEncoding; older returns a plain list.
        input_ids = getattr(out, "input_ids", out)
        return list(input_ids)

    def render(
        self, messages: List[Message], *, add_generation_prompt: bool = True
    ) -> str:
        """The templated prompt as a string (handy for debugging / logging)."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def stop_token_ids(self) -> List[int]:
        """Ids that should halt sampling (passed to ``complete_tokens``)."""
        return [self.eos_token_id] if self.eos_token_id is not None else []
