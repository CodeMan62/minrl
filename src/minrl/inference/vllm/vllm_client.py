from typing import List, Optional, Tuple

from openai import OpenAI

from minrl.inference.client import InferenceClient
from minrl.types import ChatResponse

# vLLM reports each logprob token as this string when the server is started
# with ``--return-tokens-as-token-ids``; that is how we recover the sampled ids.
_TOKEN_ID_PREFIX = "token_id:"


class VLLMClient(InferenceClient):
    """:class:`InferenceClient` backed by a vLLM OpenAI-compatible server.

    Hits the ``/v1/completions`` endpoint with a token-id prompt (token-in) and
    parses the sampled token ids back out of the per-token logprobs (token-out).

    The server **must** be launched with ``--return-tokens-as-token-ids`` so the
    ``logprobs.tokens`` field carries ``"token_id:<id>"`` strings instead of the
    decoded text; otherwise we cannot recover the exact ids to train on. e.g.::

        vllm serve Qwen/Qwen3-0.6B --return-tokens-as-token-ids
    """

    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete_tokens(
        self,
        prompt_token_ids: List[int],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> ChatResponse:
        # Passing a list[int] as ``prompt`` makes vLLM treat it as token ids.
        extra_body = {"stop_token_ids": stop_token_ids} if stop_token_ids else None
        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1,  # need per-token logprobs + the token strings
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        token_ids, logprobs = _extract_tokens(choice.logprobs)
        return ChatResponse(
            text=choice.text,
            token_idx=token_ids,
            logprobs=logprobs,
            finish_reason=choice.finish_reason,
        )


def _extract_tokens(logprobs) -> Tuple[List[int], List[float]]:
    """Pull ``(token_ids, logprobs)`` out of an OpenAI completion logprobs object."""
    if logprobs is None:
        return [], []
    token_ids: List[int] = []
    for tok in logprobs.tokens:
        if not tok.startswith(_TOKEN_ID_PREFIX):
            raise ValueError(
                f"expected '{_TOKEN_ID_PREFIX}<id>' tokens but got {tok!r}; start "
                "the vLLM server with --return-tokens-as-token-ids."
            )
        token_ids.append(int(tok[len(_TOKEN_ID_PREFIX):]))
    return token_ids, list(logprobs.token_logprobs)
