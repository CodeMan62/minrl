import json
import re
from typing import Callable, Dict, List, Optional

from minrl.agents.agent import BaseAgent
from minrl.inference.chat_template import HFChatTemplate, Message
from minrl.inference.client import InferenceClient
from minrl.inference.parser import Parser
from minrl.types import Span

Tool = Callable[..., object]
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class LLMAgent(BaseAgent):
    """Token-exact LLM policy.

    Per-step (default): each ``act`` is a fresh prompt and its own sequence.
    ``multi_turn=True``: the episode is one sequence. Later observations are
    appended as user turns (mask 0) and generation continues from there.
    ``last_span`` is what this ``act`` appended; ``truncated`` is set once
    ``max_seq_len`` is reached.

    With ``tools``, ``<tool_call>{"name", "arguments"}</tool_call>`` blocks are
    answered in-sequence as tool turns (mask 0), up to ``max_tool_calls`` rounds.
    Pass the tool schemas via ``template_kwargs={"tools": [...]}``.
    """

    def __init__(
        self,
        client: InferenceClient,
        template: HFChatTemplate,
        parser: Parser,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        multi_turn: bool = False,
        max_seq_len: Optional[int] = None,
        tools: Optional[Dict[str, Tool]] = None,
        max_tool_calls: int = 8,
    ):
        if max_seq_len is not None and max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.client = client
        self.template = template
        self.parser = parser
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.multi_turn = multi_turn
        self.max_seq_len = max_seq_len
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls
        self.reset()

    def reset(self) -> None:
        self.history: List[Message] = []
        self.last_text: Optional[str] = None
        self.last_move: Optional[int] = None
        self.last_token_ids: Optional[List[int]] = None
        self.last_logprobs: Optional[List[float]] = None
        self.last_action_mask: Optional[List[int]] = None
        self.last_span: Optional[Span] = None
        self.last_tool_calls: int = 0
        self.truncated: bool = False

    def _build_messages(self, obs: str) -> List[Message]:
        messages: List[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": obs})
        return messages

    def _budget(self, ids: List[int]) -> int:
        """Tokens the next generation may add."""
        if self.max_seq_len is None:
            return self.max_tokens
        return min(self.max_tokens, self.max_seq_len - len(ids))

    def _generate(self, ids: List[int], max_tokens: int):
        eos_token_id = self.template.tokenizer.eos_token_id
        return self.client.complete_tokens(
            ids,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
        )

    def _call_tool(self, raw: str) -> str:
        # Bad JSON, unknown tool or a tool crash come back as text, not an exception.
        try:
            call = json.loads(raw)
            fn = self.tools[call["name"]]
            args = call.get("arguments") or {}
            out = fn(**args) if isinstance(args, dict) else fn(args)
            return out if isinstance(out, str) else json.dumps(out)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    def _start_turn(self, obs: str):
        """Sequence to generate from, plus where this act's slice starts."""
        if self.multi_turn and self.last_token_ids is not None:
            ids, logprobs, mask = self.last_token_ids, self.last_logprobs, self.last_action_mask
            start = len(ids)
            seg = self.template.user_turn_ids(obs)
            ids += seg
            logprobs += [0.0] * len(seg)
            mask += [0] * len(seg)
            return ids, logprobs, mask, start
        messages = self._build_messages(obs)
        ids = list(self.template.apply(messages, add_generation_prompt=True).prompt_ids)
        if self.max_seq_len is not None and len(ids) >= self.max_seq_len:
            raise ValueError(
                f"prompt is {len(ids)} tokens; max_seq_len={self.max_seq_len} "
                "leaves no room to generate"
            )
        return ids, [0.0] * len(ids), [0] * len(ids), len(ids)

    def act(self, obs: str) -> Optional[int]:
        ids, logprobs, mask, start = self._start_turn(obs)
        turns: List[Message] = []
        self.last_tool_calls = 0
        text = ""

        while True:
            budget = self._budget(ids)
            if budget <= 0:
                break  # observation filled the context; nothing sampled
            resp = self._generate(ids, budget)
            ids += list(resp.token_idx)
            logprobs += list(resp.logprobs)
            mask += [1] * len(resp.token_idx)
            turns.append({"role": "assistant", "content": resp.text})
            text = resp.text

            calls = _TOOL_CALL.findall(resp.text) if self.tools else []
            if not calls or self.last_tool_calls >= self.max_tool_calls or resp.finish_reason == "length":
                break
            self.last_tool_calls += len(calls)
            results = [self._call_tool(c) for c in calls]
            seg = self.template.tool_turn_ids(results)
            ids += seg
            logprobs += [0.0] * len(seg)
            mask += [0] * len(seg)
            turns += [{"role": "tool", "content": r} for r in results]

        # Full sequence; loss only on mask-1 tokens.
        self.last_text = text
        self.last_token_ids = ids
        self.last_logprobs = logprobs
        self.last_action_mask = mask
        self.last_span = (start, len(ids)) if self.multi_turn else None
        self.history.append({"role": "user", "content": obs})
        self.history.extend(turns)
        if self.max_seq_len is not None and len(ids) >= self.max_seq_len:
            self.truncated = True

        self.last_move = self.parser.parse(text)
        return self.last_move
