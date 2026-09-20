import json
import re
from typing import Callable, Dict, List, Optional

from minrl.agents.agent import BaseAgent
from minrl.inference.chat_template import HFChatTemplate, Message
from minrl.inference.client import InferenceClient
from minrl.inference.parser import Parser

Tool = Callable[..., object]
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class LLMAgent(BaseAgent):
    """One env step = one token-exact sequence.

    With ``tools``, a completion containing ``<tool_call>{"name", "arguments"}
    </tool_call>`` blocks is answered in-sequence: each tool's result is appended
    as a ``tool`` turn (action_mask 0) and generation resumes from the extended
    ids, up to ``max_tool_calls`` rounds. Pass the tool schemas the model should
    see via ``template_kwargs={"tools": [...]}`` on the template.
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
        keep_history: bool = False,
        tools: Optional[Dict[str, Tool]] = None,
        max_tool_calls: int = 8,
    ):
        self.client = client
        self.template = template
        self.parser = parser
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.keep_history = keep_history
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
        self.last_tool_calls: int = 0

    def _build_messages(self, obs: str) -> List[Message]:
        messages: List[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": obs})
        return messages

    def _generate(self, ids: List[int]):
        eos_token_id = self.template.tokenizer.eos_token_id
        return self.client.complete_tokens(
            ids,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
        )

    def _call_tool(self, raw: str) -> str:
        # Model output is untrusted: bad JSON / unknown tool / tool crash all
        # come back as text so the policy can recover instead of the rollout dying.
        try:
            call = json.loads(raw)
            fn = self.tools[call["name"]]
            args = call.get("arguments") or {}
            out = fn(**args) if isinstance(args, dict) else fn(args)
            return out if isinstance(out, str) else json.dumps(out)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    def act(self, obs: str) -> Optional[int]:
        messages = self._build_messages(obs)
        ids = list(self.template.apply(messages, add_generation_prompt=True).prompt_ids)
        logprobs = [0.0] * len(ids)
        mask = [0] * len(ids)
        turns: List[Message] = []
        self.last_tool_calls = 0

        while True:
            resp = self._generate(ids)
            ids += list(resp.token_idx)
            logprobs += list(resp.logprobs)
            mask += [1] * len(resp.token_idx)
            turns.append({"role": "assistant", "content": resp.text})

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

        # Token trace over the full sequence; loss only on sampled tokens.
        self.last_text = resp.text
        self.last_token_ids = ids
        self.last_logprobs = logprobs
        self.last_action_mask = mask

        if self.keep_history:
            self.history.append({"role": "user", "content": obs})
            self.history.extend(turns)

        self.last_move = self.parser.parse(resp.text)
        return self.last_move
