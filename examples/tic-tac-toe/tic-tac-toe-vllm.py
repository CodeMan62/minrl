"""Run a vLLM server first (the flag is required so we can recover the sampled
token *ids*, not just text):

    vllm serve Qwen/Qwen3-0.6B \
        --port 8000 \
        --return-tokens-as-token-ids

Then:

    python examples/tic-tac-toe/tic-tac-toe-vllm.py
    # or point at a different server / model:
    MINRL_BASE_URL=http://localhost:8000/v1 MINRL_MODEL=Qwen/Qwen3-0.6B \
        python examples/tic-tac-toe/tic-tac-toe-vllm.py
"""

import os

from minrl.agents.llm_agent import LLMAgent
from minrl.inference.chat_template import ChatTemplate
from minrl.inference.parser import MoveParser
from minrl.inference.vllm.vllm_client import VLLMClient

BASE_URL = os.environ.get("MINRL_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MINRL_MODEL", "Qwen/Qwen3-0.6B")

SYSTEM_PROMPT = (
    "You are playing TicTacToe as X. Cells are numbered 0-8 left-to-right, "
    "top-to-bottom. Reply with the single cell number you play, nothing else."
)

# A mid-game TicTacToe board. Cells are numbered 0..8; '.' is empty.
BOARD = [
    "X", ".", "O",
    ".", "X", ".",
    ".", ".", "O",
]


def render_board(board):
    rows = [" | ".join(board[i : i + 3]) for i in range(0, 9, 3)]
    return "\n---------\n".join(rows)


def build_observation(board):
    legal = [i for i, c in enumerate(board) if c == "."]
    return (
        f"Current board:\n{render_board(board)}\n\n"
        f"Empty cells: {legal}\n"
        "Your move (a single number 0-8):"
    )


def main():
    print(f"model    : {MODEL}")
    print(f"base_url : {BASE_URL}\n")

    # Wire the harness: server client + local tokenizer + output parser.
    agent = LLMAgent(
        client=VLLMClient(base_url=BASE_URL, model=MODEL),
        template=ChatTemplate(MODEL),
        parser=MoveParser(),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=64,
        temperature=0.7,
    )

    obs = build_observation(BOARD)

    try:
        move = agent.act(obs)
    except Exception as exc:  # connection refused / server misconfigured
        print(f"!! inference call failed: {exc}\n")
        print("Is the vLLM server running with --return-tokens-as-token-ids?")
        print(f"Expected an OpenAI-compatible server at {BASE_URL}")
        return

    # The token trace the trainer consumes (via interaction.rollout -> Step).
    n_prompt = agent.last_action_mask.count(0)
    n_action = agent.last_action_mask.count(1)
    completion_ids = agent.last_token_ids[n_prompt:]
    completion_logprobs = agent.last_logprobs[n_prompt:]

    # Real output from the completion.
    print("=== completion ===")
    print(f"text          : {agent.last_text!r}")
    print(f"sampled ids   : {completion_ids}")
    print(f"logprobs (gen): {[round(lp, 3) for lp in completion_logprobs]}\n")
    print("=== token trace ===")
    print(f"total tokens  : {len(agent.last_token_ids)}  "
          f"(prompt {n_prompt} + action {n_action})")
    print(f"action_mask   : sums to {n_action} trainable tokens")
    print(f"logprobs len  : {len(agent.last_logprobs)} (aligned to token_ids)\n")

    # The parsed action handed back to the env.
    legal = [i for i, c in enumerate(BOARD) if c == "."]
    print("=== parsed action ===")
    if move is None:
        print("move: None  (unparseable -> illegal move penalty downstream)")
    elif move in legal:
        print(f"move: {move}  (legal)")
    else:
        print(f"move: {move}  (ILLEGAL -> penalty downstream; legal were {legal})")


if __name__ == "__main__":
    main()
