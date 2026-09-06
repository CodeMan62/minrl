# TicTacToe examples

minrl using an LLM playing `TicTacToe` (from the top-level
`enviornments/` package) against a random opponent.

| File | What it does |
|---|---|
| `train_grpo.py` | Trains Qwen3-0.6B with GRPO so its win rate vs the random opponent goes up. |
| `tic-tac-toe-vllm.py` | Single inference call against a vLLM server; prints the token trace (ids, logprobs, action mask) the trainer consumes. Sanity-check for the vLLM path, no training. |

## Prerequisites

- Python >= 3.12 with the repo installed (`torch`, `transformers`, `openai`).
- Internet access on first run to download `Qwen/Qwen3-0.6B` (~1.5 GB) from
  the HF Hub.

**Hardware for `train_grpo.py`:** two GPUs. One serves the model with vLLM
for rollouts; the other trains. After every optimizer step the trainer pushes
its weights into the server over NCCL, so the server and trainer must share a
host.

`tic-tac-toe-vllm.py` needs whatever GPU your vLLM server runs on; the script
itself is just a client.

## Train with GRPO

```bash
CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.7 --weight-transfer-config '{"backend":"nccl"}'
CUDA_VISIBLE_DEVICES=1 python examples/env/tic-tac-toe/train_grpo.py
```

Defaults: 150 iterations, 8 episodes per GRPO group, lr 5e-6, win-rate eval
(greedy decoding, 50 games) before training, every 25 iterations, and at the
end. Expect roughly 15–30 min on a modern GPU.

Common knobs:

```bash
python examples/env/tic-tac-toe/train_grpo.py \
    --iterations 150 --group-size 8 --lr 5e-6 \
    --eval-every 25 --eval-games 50
```

If the win rate climbs too slowly, try `--lr 1e-5` and/or `--group-size 16`.

Notes:

- Rollouts come from the vLLM server; weights are synced after every step,
  so each group is sampled from the current policy.
- Rewards: +1 win, 0 draw, -1 loss, -1 illegal/unparseable move (the episode
  ends on an illegal move).
- A `loss=0 ... skipped` iteration means every episode in the group got the
  same return, so all advantages are zero — normal at small group sizes.

## Tracking training with W&B

W&B logging is built into the library: `minrl.loggers.WandbLogger` wraps a
W&B run, and any logger passed to `grpo()` gets the per-iteration
`train/*` metrics automatically:

```python
from minrl.algorithms import grpo
from minrl.loggers import WandbLogger

logger = WandbLogger(project="minrl-tictactoe")   # kwargs go to wandb.init
history = [metrics for _, metrics in grpo(model, agent, env, logger=logger)]
logger.finish()
```

`train_grpo.py` does exactly this (plus extra eval metrics) when `wandb` is
installed; otherwise it prints a note and trains without it.

One-time setup:

```bash
pip install wandb        # or: pip install -e ".[examples]"
wandb login              # paste your API key from https://wandb.ai/authorize
```

Then just run training as usual — the run URL is printed at start and end:

```bash
python examples/env/tic-tac-toe/train_grpo.py
# W&B run: https://wandb.ai/<your-entity>/minrl-tictactoe/runs/<run-id>
```

Viewing the logs:

- Open the printed URL, or browse all runs at
  `https://wandb.ai/<your-entity>/minrl-tictactoe`.
- **`eval/win_rate`** is the headline chart — win rate vs the random opponent
  at iteration 0, every `--eval-every` iterations, and at the end. It should
  trend up; `eval/illegal_rate` should trend down.
- `train/*` charts show per-iteration signals — `grpo()` logs
  `mean_return`, `std_return`, `loss`, `n_tokens`, and `skipped` (1 when a
  zero-variance group skipped the update); the example script adds
  `win_rate` / `illegal_rate` within each training group.
- The run summary records `win_rate_before` / `win_rate_after` /
  `win_rate_delta` for quick comparison across runs.

Useful variants:

```bash
# name the run / use a different project
python examples/env/tic-tac-toe/train_grpo.py \
    --wandb-run-name qwen3-0.6b-lr5e-6 --wandb-project my-project

# no internet on the training box: log offline, sync later
WANDB_MODE=offline python examples/env/tic-tac-toe/train_grpo.py
wandb sync wandb/offline-run-*        # from the same directory, once online

# disable W&B entirely
python examples/env/tic-tac-toe/train_grpo.py --no-wandb
```

## vLLM inference check

Start a server (the flag is required so the sampled token *ids* can be
recovered for training):

```bash
uv run vllm_server Qwen/Qwen3-0.6B
```

Then:

```bash
python examples/env/tic-tac-toe/tic-tac-toe-vllm.py
# or point elsewhere:
MINRL_BASE_URL=http://localhost:8000/v1 MINRL_MODEL=Qwen/Qwen3-0.6B \
    python examples/env/tic-tac-toe/tic-tac-toe-vllm.py
```
