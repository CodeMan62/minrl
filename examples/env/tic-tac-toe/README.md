# Tic-tac-toe

An LLM learns to beat a random opponent at tic-tac-toe with GRPO. The env
lives in the top-level `enviornments/` package.

## Hardware

Two GPUs on one host. One serves the policy with vLLM for rollouts, the other
trains. After every optimizer step the trainer pushes its weights into the
server over NCCL, which is why both must share a host.

## Train
Run vllm server in terminal 1:-
```bash
CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.7 --weight-transfer-config '{"backend":"nccl"}'
```
Run training on terminal 2:-

```bash
CUDA_VISIBLE_DEVICES=1 python examples/env/tic-tac-toe/train_grpo.py
```

Or on GPUs 1-3 with FSDP2, one rank per GPU:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 examples/env/tic-tac-toe/train_grpo.py
```

Defaults: 150 steps, 8 games per GRPO group, lr 5e-6, a checkpoint every
2 steps, and an eval of 50 greedy games before training, every 25 steps, and
at the end.

## Knobs

| Flag | Default | What it does |
|---|---|---|
| `--iterations` | 150 | optimizer steps |
| `--group-size` | 8 | games per group; GRPO advantages are relative within a group |
| `--lr` | 5e-6 | if the win rate climbs slowly, try 1e-5 or a bigger group |
| `--multi-turn` | off | one sequence per game, so the policy sees every earlier board and its own moves |
| `--concurrency` | 2 | groups generating at once while the trainer steps |
| `--ent-coef` | 0 | entropy bonus; 0 still logs `train/entropy` |
| `--ckpt-every` | 2 | checkpoint every N steps; 0 = only at the end |
| `--keep-last` | 3 | `step_*` checkpoints kept; older ones are deleted after a good save |
| `--resume` | none | a checkpoint path, or `auto` for the newest under `--ckpt-dir` |
| `--eval-every` / `--eval-games` | 25 / 50 | when and how many games to eval |
| `--eval-concurrency` | 16 | eval games played at once |
| `--eval-dump` | none | directory for per-game jsonl, to read what the model actually played |

## Crash and resume

Restart with the same command plus `--resume auto`. Training continues from
the newest checkpoint with the same weights, optimizer state, RNG, step and
token counters, and rollout cursor. With the default `--ckpt-every 2` a crash
costs at most one finished step.

## W&B

```bash
pip install wandb && wandb login
python examples/env/tic-tac-toe/train_grpo.py --wandb
```

Logging is off by default; `--wandb` turns it on, and the run URL is printed
at the start. Useful charts:

- **`eval/success`**, the headline: win rate against the random opponent.
- **`eval/label/illegal`** should fall as the policy learns the rules.
- `eval/label/win`, `draw`, `loss` break down the rest.
- `train/mean_return`, `loss`, `entropy`, `clip_frac`, `grad_norm`,
  `staleness`, `tokens`.

The run summary records `win_rate_before`, `win_rate_after`, and
`win_rate_delta` for comparing runs. For offline boxes, run with
`WANDB_MODE=offline` and `wandb sync wandb/offline-run-*` later.

## Eval on its own

Any saved policy that vLLM is serving can be evaluated without training:

```python
from minrl.eval import Eval, EvalConfig

Eval(EvalConfig(num_episodes=50, max_steps=9), make,
     success=lambda r: outcome(r) == "win", label=outcome).evaluate_sync()
```

`outcome` is the function in `train_grpo.py`. Build `make` the way
`make_actor(0.0)` does there, returning a fresh `(LLMAgent, TicTacToe())` pair.
