"""Evaluate the model currently served by vLLM on GSM8K.

Run before and after ``train_grpo.py`` while the same server stays up::

    python examples/env/gsm8k/eval_vllm.py --samples 200
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT, evaluate_gsm8k  # noqa: E402
from minrl.agents.llm_agent import LLMAgent  # noqa: E402
from minrl.inference.chat_template import HFChatTemplate  # noqa: E402
from minrl.inference.parser import TextParser  # noqa: E402
from minrl.inference.vllm_client import VLLMClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                   help="vLLM server URL without /v1")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    agent = LLMAgent(
        VLLMClient(f"{args.vllm_url.rstrip('/')}/v1", args.model),
        HFChatTemplate(tokenizer),
        TextParser(),
        system_prompt=GSM8K_SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    env = GSM8K(split=args.split, limit=args.limit, repeat=1)

    score = evaluate_gsm8k(agent, env, args.samples)
    print(f"accuracy: {score['accuracy']:.1%} ({int(score['correct'])}/{args.samples})")


if __name__ == "__main__":
    main()
