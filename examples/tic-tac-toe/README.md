# TicTacToe examples

Two examples of using minrl with an LLM playing `TicTacToe` (from the
top-level `enviornments/` package) against a random opponent.

| File | What it does |
|---|---|
| `train_grpo.py` | Trains Qwen3-0.6B with GRPO so its win rate vs the random opponent goes up. |
| `tic-tac-toe-vllm.py` | Single inference call against a vLLM server; prints the token trace (ids, logprobs, action mask) the trainer consumes. Sanity-check for the vLLM path, no training. |

## Prerequisites

- Python >= 3.12 with the repo installed (`torch`, `transformers`, `openai`).
- Internet access on first run to download `Qwen/Qwen3-0.6B` (~1.5 GB) from
  the HF Hub.

**Hardware for `train_grpo.py`:** a GPU box. The model is loaded in bf16
(~1.2 GB weights) plus AdamW states and activations — any 16 GB+ GPU is
comfortable. There is no multi-GPU support; it runs on a single device.
CPU works only for a tiny "does it move" smoke run (see below) — actual
training on CPU is far too slow.

`tic-tac-toe-vllm.py` needs whatever GPU your vLLM server runs on; the script
itself is just a client.

## Train with GRPO

```bash
python examples/tic-tac-toe/train_grpo.py
```

Defaults: 150 iterations, 8 episodes per GRPO group, lr 5e-6, win-rate eval
(greedy decoding, 50 games) before training, every 25 iterations, and at the
end. Expect roughly 15–30 min on a modern GPU.

Common knobs:

```bash
python examples/tic-tac-toe/train_grpo.py \
    --iterations 150 --group-size 8 --lr 5e-6 \
    --eval-every 25 --eval-games 50 --device cuda
```

If the win rate climbs too slowly, try `--lr 1e-5` and/or `--group-size 16`.

CPU smoke run (minutes, just to verify the loop executes):

```bash
python examples/tic-tac-toe/train_grpo.py \
    --iterations 2 --group-size 2 --eval-games 4 --device cpu
```

Notes:

- Inference runs in-process (`HFClient`), so the sampled model is literally
  the trained model — fully on-policy, no weight syncing.
- Rewards: +1 win, 0 draw, -1 loss, -1 illegal/unparseable move (the episode
  ends on an illegal move).
- A `loss=0 ... skipped` iteration means every episode in the group got the
  same return, so all advantages are zero — normal at small group sizes.

## Tracking training with W&B

W&B logging is built into the library: `minrl.loggers.WandbLogger` wraps a
W&B run, and any logger passed to `GRPOTrainer` gets the per-iteration
`train/*` metrics automatically:

```python
from minrl.loggers import WandbLogger
from minrl.trainers import GRPOConfig, GRPOTrainer

logger = WandbLogger(project="minrl-tictactoe")   # kwargs go to wandb.init
trainer = GRPOTrainer(model, agent, env, cfg, logger=logger)
trainer.train()
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
python examples/tic-tac-toe/train_grpo.py
# W&B run: https://wandb.ai/<your-entity>/minrl-tictactoe/runs/<run-id>
```

Viewing the logs:

- Open the printed URL, or browse all runs at
  `https://wandb.ai/<your-entity>/minrl-tictactoe`.
- **`eval/win_rate`** is the headline chart — win rate vs the random opponent
  at iteration 0, every `--eval-every` iterations, and at the end. It should
  trend up; `eval/illegal_rate` should trend down.
- `train/*` charts show per-iteration signals — the trainer logs
  `mean_return`, `std_return`, `loss`, `n_tokens`, and `skipped` (1 when a
  zero-variance group skipped the update); the example script adds
  `win_rate` / `illegal_rate` within each training group.
- The run summary records `win_rate_before` / `win_rate_after` /
  `win_rate_delta` for quick comparison across runs.

Useful variants:

```bash
# name the run / use a different project
python examples/tic-tac-toe/train_grpo.py \
    --wandb-run-name qwen3-0.6b-lr5e-6 --wandb-project my-project

# no internet on the training box: log offline, sync later
WANDB_MODE=offline python examples/tic-tac-toe/train_grpo.py
wandb sync wandb/offline-run-*        # from the same directory, once online

# disable W&B entirely
python examples/tic-tac-toe/train_grpo.py --no-wandb
```

## vLLM inference check

Start a server (the flag is required so the sampled token *ids* can be
recovered for training):

```bash
vllm serve Qwen/Qwen3-0.6B --port 8000 --return-tokens-as-token-ids
```

Then:

```bash
python examples/tic-tac-toe/tic-tac-toe-vllm.py
# or point elsewhere:
MINRL_BASE_URL=http://localhost:8000/v1 MINRL_MODEL=Qwen/Qwen3-0.6B \
    python examples/tic-tac-toe/tic-tac-toe-vllm.py
```
