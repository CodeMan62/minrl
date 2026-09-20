"""Tool-call turns stay token-exact: sampled ids untouched, mask 1 only on them."""
from transformers import AutoTokenizer

from minrl.agents.llm_agent import LLMAgent
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.parser import TextParser
from minrl.types import ChatResponse

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class ScriptedClient:
    """Replays canned completions; records every prompt it was given."""

    def __init__(self, tok, texts):
        self.tok, self.texts, self.prompts = tok, list(texts), []

    def complete_tokens(self, prompt_token_ids, **_):
        self.prompts.append(list(prompt_token_ids))
        ids = self.tok.encode(self.texts.pop(0), add_special_tokens=False)
        return ChatResponse(text=self.tok.decode(ids), token_idx=ids,
                            logprobs=[-0.5] * len(ids), finish_reason="stop")


def test_tool_call_roundtrip():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tmpl = HFChatTemplate(tok)
    call = '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>'
    client = ScriptedClient(tok, [call, "The answer is 5."])
    agent = LLMAgent(client, tmpl, TextParser(), tools={"add": lambda a, b: a + b})

    out = agent.act("what is 2+3?")

    assert out == "The answer is 5."
    assert agent.last_tool_calls == 1
    ids, mask, lp = agent.last_token_ids, agent.last_action_mask, agent.last_logprobs
    assert len(ids) == len(mask) == len(lp)
    # second generation was fed exactly the running sequence so far
    assert client.prompts[1] == ids[: len(client.prompts[1])]
    # sampled tokens are the only ones with mask 1 and a real logprob
    sampled = [i for i, m in zip(ids, mask) if m]
    assert sampled == tok.encode(call, add_special_tokens=False) + tok.encode("The answer is 5.", add_special_tokens=False)
    assert all((lp[i] == -0.5) == bool(mask[i]) for i in range(len(ids)))
    # the injected tool turn reads back as a real Qwen tool message
    text = tok.decode(ids)
    assert "<|im_end|>\n<|im_start|>user\n<tool_response>\n5\n</tool_response><|im_end|>\n<|im_start|>assistant\n" in text
    assert text.endswith("The answer is 5.")


def test_bad_tool_call_is_text_not_crash():
    tok = AutoTokenizer.from_pretrained(MODEL)
    client = ScriptedClient(tok, ["<tool_call>not json</tool_call>", "ok"])
    agent = LLMAgent(client, HFChatTemplate(tok), TextParser(), tools={"add": lambda: 0})
    agent.act("x")
    assert "error: JSONDecodeError" in tok.decode(agent.last_token_ids)


if __name__ == "__main__":
    test_tool_call_roundtrip()
    test_bad_tool_call_is_text_not_crash()
    print("ok")
