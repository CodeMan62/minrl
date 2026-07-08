from abc import ABC, abstractmethod
from typing import List, Optional

from minrl.types import ChatResponse


class InferenceClient(ABC):
    """Token-in / token-out inference against a running model server.

    We send **token ids** (not text) and get back the sampled token ids +
    logprobs, so the exact tokens the sampler saw are the ones we train on.
    This avoids chat-template / tokenizer drift between inference and training.
    """

    @abstractmethod
    def complete_tokens(
        self,
        prompt_token_ids: List[int],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> ChatResponse:
        """Sample a completion from ``prompt_token_ids`` (a flat list of ids)."""
        ...
