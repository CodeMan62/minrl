import asyncio
import time
import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from openai import AsyncOpenAI

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

        uv run vllm_server Qwen/Qwen3-0.6
    """

    def __init__(self, server_url: str, model: str, api_key: str = "local", timeout: float = 300.0):
        # The bare server URL; a trailing /v1 from older callers is tolerated.
        self.server_url = server_url.rstrip("/").removesuffix("/v1")
        self.base_url = f"{self.server_url}/v1"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # One AsyncOpenAI per event loop: its connection pool is bound to the loop
        # that first used it, and training and eval each run on their own loop.
        self._clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = (
            weakref.WeakKeyDictionary()
        )

    @property
    def client(self) -> AsyncOpenAI:
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._clients[loop]

    async def complete_tokens(
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
        resp = await self.client.completions.create(
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

    # ---- server routes the trainer calls between steps (blocking) ------------

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(f"{self.server_url}/{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kwargs) -> Dict[str, Any]:
        r = requests.post(f"{self.server_url}/{path}", timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        """Model, dtype, worker world size and weight version, from ``/minrl/health``."""
        return self._get("minrl/health")

    def wait_ready(self, timeout: float = 600.0, poll: float = 2.0) -> Dict[str, Any]:
        """Block until the server answers :meth:`health`, or raise."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.health()
            except requests.RequestException as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"vLLM server at {self.server_url} not ready after {timeout}s") from e
                time.sleep(poll)

    def init_weight_sync(self, host: str, port: int, world_size: int) -> None:
        """Have every worker join the trainer's NCCL group; returns once they all have."""
        self._post("minrl/init_weight_sync", json={"host": host, "port": port, "world_size": world_size})

    def update_weights(self, weights: Sequence[Tuple[str, str, Sequence[int]]],
                       version: Optional[str] = None) -> None:
        """Have the workers receive one broadcast per ``(name, dtype, shape)`` and load them."""
        self._post("minrl/update_weights", json={"weights": weights, "version": version})

    def pause(self, mode: str = "keep") -> None:
        """Stop scheduling generation: ``keep`` freezes, ``wait`` drains, ``abort`` drops requests."""
        self._post("pause", params={"mode": mode})

    def resume(self) -> None:
        self._post("resume")


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

