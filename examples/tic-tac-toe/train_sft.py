import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from enviornments import TicTacToe  # noqa: E402

from minrl.agents.llm_agent import LLMAgent  # noqa: E402
from minrl.inference.chat_template import ChatTemplate  # noqa: E402
from minrl.inference.hf import HFClient  # noqa: E402
from minrl.inference.parser import MoveParser  # noqa: E402
from minrl.interaction import episode  # noqa: E402
from minrl.trainers import SFTConfig, SFTTrainer  # noqa: E402

SYSTEM_PROMPT = (
    "You are playing Tic-Tac-Toe against an opponent. Cells are numbered 0-8, "
    "left to right, top to bottom. Pick an empty cell. Reply with only the "
    "cell number you play, nothing else."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "sft_data.jsonl"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_dataset(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run generate_sft_data.py first to create it."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(agent: LLMAgent, env: TicTacToe, games: int) -> dict:
    """Play ``games`` full games greedily; report win/draw/loss/illegal rates."""
    counts = {"win": 0, "draw": 0, "loss": 0, "illegal": 0}
    for _ in range(games):
        agent.reset()
        r = episode(agent, env, max_steps=9)
        last = r.steps[-1]
        if last.info.get("illegal_move"):
            counts["illegal"] += 1
        else:
            counts[last.info.get("result", "loss")] += 1
    return {k: v / games for k, v in counts.items()}


def fmt_eval(tag: str, e: dict) -> str:
    return (
        f"{tag}: win {e['win']:.0%} | draw {e['draw']:.0%} | "
        f"loss {e['loss']:.0%} | illegal {e['illegal']:.0%}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.data)
    print(f"loaded {len(dataset)} SFT demonstrations from {args.data}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = HFClient(model, tokenizer)
    template = ChatTemplate(args.model, template_kwargs={"enable_thinking": False})
    parser = MoveParser()
    env = TicTacToe()
    eval_agent = LLMAgent(
        client, template, parser, system_prompt=SYSTEM_PROMPT,
        max_tokens=8, temperature=0.0,
    )

    baseline = evaluate(eval_agent, env, args.eval_games)
    print(fmt_eval("[eval] before SFT", baseline))

    trainer = SFTTrainer(
        model, dataset,
        SFTConfig(
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
    )
    trainer.train()

    final = evaluate(eval_agent, env, args.eval_games)
    print(fmt_eval("[eval] after SFT", final))
    delta = final["win"] - baseline["win"]
    print(f"win rate change: {baseline['win']:.0%} -> {final['win']:.0%} ({delta:+.0%})")


if __name__ == "__main__":
    main()
