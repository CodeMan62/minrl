"""Rejection-sampling SFT data for TicTacToe.

Sample full games from the model against the env's random opponent, KEEP only
the games the model won (optionally draws too), and emit every move in those
games as an SFT demonstration (prompt board -> the move the model played).
Because the whole game reached a good outcome, each of its moves is "good
enough" to imitate. This is STaR / best-of-N style rejection sampling: the
policy generates its own supervision, filtered by outcome.

Each accepted step becomes one JSONL line:

    {"token_ids": [...], "action_mask": [...], "text": "4", "result": "win"}

``token_ids`` / ``action_mask`` are copied verbatim from what ``LLMAgent``
recorded during the rollout, so the SFT target is a real tokenization with no
detokenize/retokenize drift -- exactly the sequence the sampler saw.

Run (needs a GPU box to be quick; CPU works but is slow):

    python examples/tic-tac-toe/generate_sft_data.py --num-examples 500

Then train on it with examples/tic-tac-toe/train_sft.py (minrl.algorithms.sft).
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root on sys.path so the top-level ``enviornments`` package resolves
# regardless of the cwd this script is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from enviornments import TicTacToe  # noqa: E402

from minrl.agents.llm_agent import LLMAgent  # noqa: E402
from minrl.inference.chat_template import ChatTemplate  # noqa: E402
from minrl.inference.hf import HFClient  # noqa: E402
from minrl.inference.parser import MoveParser  # noqa: E402
from minrl.interaction import episode  # noqa: E402

SYSTEM_PROMPT = (
    "You are playing Tic-Tac-Toe against an opponent. Cells are numbered 0-8, "
    "left to right, top to bottom. Pick an empty cell. Reply with only the "
    "cell number you play, nothing else."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "sft_data.jsonl"))
    p.add_argument("--num-examples", type=int, default=500,
                   help="stop once this many demonstration moves are collected")
    p.add_argument("--max-games", type=int, default=5000,
                   help="hard cap on games sampled (safety valve)")
    p.add_argument("--keep-draws", action="store_true",
                   help="also accept drawn games, not only wins")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature; >0 gives the diversity rejection "
                        "sampling relies on")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def accept(rollout, keep_draws: bool) -> bool:
    """A game is accepted if the model won (or drew, when --keep-draws)."""
    last = rollout.steps[-1]
    if last.info.get("illegal_move"):
        return False
    result = last.info.get("result")
    return result == "win" or (keep_draws and result == "draw")


def examples_from(rollout, tokenizer):
    """Every scored move in an accepted game is one demonstration."""
    out = []
    for s in rollout.steps:
        if not (s.token_ids and s.action_mask and any(s.action_mask)):
            continue
        completion_ids = [t for t, m in zip(s.token_ids, s.action_mask) if m]
        out.append({
            "token_ids": list(s.token_ids),
            "action_mask": list(s.action_mask),
            "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
            "result": rollout.steps[-1].info.get("result"),
        })
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = HFClient(model, tokenizer)
    template = ChatTemplate(args.model, template_kwargs={"enable_thinking": False})
    parser = MoveParser()
    env = TicTacToe()
    # Sample at T>0 so different games explore different lines -- that diversity
    # is exactly what rejection sampling filters down to the good ones.
    agent = LLMAgent(
        client, template, parser, system_prompt=SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    dataset = []
    games = wins = draws = 0
    while len(dataset) < args.num_examples and games < args.max_games:
        agent.reset()
        r = episode(agent, env, max_steps=9)
        games += 1
        if accept(r, args.keep_draws):
            result = r.steps[-1].info.get("result")
            wins += result == "win"
            draws += result == "draw"
            dataset.extend(examples_from(r, tokenizer))
        if games % 25 == 0:
            print(
                f"games={games} accepted(win/draw)={wins}/{draws} "
                f"examples={len(dataset)}/{args.num_examples}"
            )

    with open(args.out, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")

    accept_rate = (wins + draws) / games if games else 0.0
    print(
        f"\ndone: {len(dataset)} demonstrations from {wins + draws} accepted "
        f"games (accept rate {accept_rate:.0%} over {games} sampled).\n"
        f"wrote {args.out}"
    )
    if len(dataset) < args.num_examples:
        print(
            f"NOTE: hit --max-games ({args.max_games}) before reaching "
            f"--num-examples ({args.num_examples}); the base model wins rarely. "
            f"Raise --max-games or --temperature, or pass --keep-draws."
        )


if __name__ == "__main__":
    main()
