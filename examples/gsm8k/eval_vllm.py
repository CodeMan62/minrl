"""Evaluate the model currently served by vLLM on GSM8K.

Run before and after ``train_grpo.py`` while the same server stays up::

    python examples/gsm8k/eval_vllm.py --samples 200
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

from minrl.agents.llm_agent import LLMAgent
from minrl.eval import Eval, EvalConfig
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.parser import TextParser
from minrl.inference.vllm_client import VLLMClient

# ``enviornments`` lives at the repo root and is not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                   help="vLLM server URL without /v1")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=16, help="questions in flight at once")
    p.add_argument("--dump", default=None, help="dir for per-rollout jsonl dumps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = VLLMClient(args.vllm_url, args.model)
    template = HFChatTemplate(tokenizer)

    # Greedy decoding; a fresh (agent, env) pair per question since both are stateful.
    def make():
        agent = LLMAgent(
            client, template, TextParser(), system_prompt=GSM8K_SYSTEM_PROMPT,
            max_tokens=args.max_new_tokens, temperature=0.0,
        )
        return agent, GSM8K(split=args.split, limit=args.limit)

    Eval(
        EvalConfig(model=args.model, num_episodes=args.samples, max_steps=1,
                   concurrency=args.concurrency, dump_dir=args.dump),
        make,
    ).evaluate_sync()


if __name__ == "__main__":
    main()
