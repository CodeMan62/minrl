from typing import List, Optional

from minrl.agents.agent import BaseAgent
from minrl.inference.chat_template import ChatTemplate, Message
from minrl.inference.client import InferenceClient
from minrl.inference.parser import Parser


class LLMAgent(BaseAgent):
    def __init__(
        self,
        client: InferenceClient,
        template: ChatTemplate,
        parser: Parser,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        keep_history: bool = False,
    ):
        self.client = client
        self.template = template
        self.parser = parser
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.keep_history = keep_history
        self.reset()

    def reset(self) -> None:
        self.history: List[Message] = []
        self.last_text: Optional[str] = None
        self.last_move: Optional[int] = None
        self.last_token_ids: Optional[List[int]] = None
        self.last_logprobs: Optional[List[float]] = None
        self.last_action_mask: Optional[List[int]] = None

    def _build_messages(self, obs: str) -> List[Message]:
        messages: List[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": obs})
        return messages

    def act(self, obs: str) -> Optional[int]:
        """Sample a move for ``obs`` and record its token trace.

        Returns the parsed cell (``0..8``) or ``None`` when the completion has no
        valid move — the env / reward treats ``None`` as an illegal move.
        """
        messages = self._build_messages(obs)
        prompt_ids = self.template.encode(messages, add_generation_prompt=True)

        resp = self.client.complete_tokens(
            prompt_ids,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_token_ids=self.template.stop_token_ids,
        )
        completion_ids = resp.token_idx

        # Token trace over the full sequence; loss only on completion tokens.
        self.last_text = resp.text
        self.last_token_ids = list(prompt_ids) + list(completion_ids)
        self.last_logprobs = [0.0] * len(prompt_ids) + list(resp.logprobs)
        self.last_action_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        if self.keep_history:
            self.history.append({"role": "user", "content": obs})
            self.history.append({"role": "assistant", "content": resp.text})

        self.last_move = self.parser.parse(resp.text)
        return self.last_move
