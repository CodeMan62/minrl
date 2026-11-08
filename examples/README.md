# Examples

One Python entry point, `train.py` at the repo root, and one self-contained
shell script per example. Each script sets the flags in named arrays, starts
the vLLM server on the rollout GPU, waits for it, runs `train.py` on the
training GPU(s), and stops the server when training ends.

```
train.py                        the training loop glue; never edited per experiment
examples/dapo-math/run.sh       Qwen3.5-0.8B, GRPO + TIS on DAPO-Math-17k, eval on MATH-500
examples/gsm8k/run.sh           Qwen2.5-0.5B-Instruct, GRPO on GSM8K
examples/tic-tac-toe/run.sh     Qwen3-0.6B, GRPO against a random opponent
enviornments/__init__.py        ENVS: the registry --env picks from
```

## Run

```bash
examples/dapo-math/run.sh                 # GPU 0 rollouts, GPU 1 trains
examples/dapo-math/run.sh --eval-only     # score whatever vLLM serves now
examples/dapo-math/run.sh --no-wandb --iterations 5 --eval-samples 20   # smoke test
```

Anything appended to a script is passed to `train.py` after the arrays, so it
overrides them. Each script reads a few environment variables:

| Variable | Default | What it does |
|---|---|---|
| `ROLLOUT_GPU` | `0` | GPU that serves the policy with vLLM |
| `TRAIN_GPUS` | `1` | training GPU(s); `1,2,3` launches torchrun with FSDP2, one rank per GPU |
| `VLLM_PORT` | `8000` | server port (`VLLM_HOST` is `127.0.0.1`) |
| `WEIGHT_SYNC_PORT` | `29501` | NCCL rendezvous the trainer opens for the server's workers |
| `SKIP_SERVER` | unset | set to anything to attach to a server already running |
| `PYTHON` | `uv run python` | command that runs the interpreter; falls back to `python` when `uv` is unavailable, override with e.g. `PYTHON=.venv/bin/python` |
| `LOG_DIR` | `logs/<example>` | `vllm_server.log`, `train.log` and eval dumps land here |

Both GPUs must share a host: the weights go from trainer to server over NCCL.

## Add an example

Copy a `run.sh` into a new folder and change the arrays. A new environment is
a class under `enviornments/` plus one `EnvSpec` entry in
`enviornments/__init__.py`: how to build it for a split, which parser turns
the model's text into an action, the system prompt, the max steps per
episode, and what counts as success. Options the environment needs beyond
split, limit, shuffle and seed come in as `--env-arg key=value`.

## Knobs

`python train.py --help` lists everything. The ones that matter most:

| Flag | What it does |
|---|---|
| `--group-size` | episodes per group; they share a seed (a prompt), and GRPO's advantages are relative within a group |
| `--groups-per-step` | groups per optimizer step; the batch is `group_size * groups_per_step` sequences |
| `--concurrency` | groups generating at once while the trainer steps; `concurrency * group_size` sequences in flight on vLLM |
| `--max-queued` | finished groups buffered ahead of the trainer; bounds how stale a batch can get |
| `--max-new-tokens` | sampler cap per generation; truncation is read from vLLM's `finish_reason` |
| `--algorithm` | `grpo`, `dr_grpo`, `cispo`, `reinforce`; see `src/minrl/training/algorithms.py` |
| `--tis` / `--icepop` | correct for the rollouts being off-policy: truncated importance weights, or two-sided rejection |
| `--kl-coef` | KL to a frozen copy of the start model; > 0 loads that second copy |
| `--dtype` | dtype the policy is loaded and stepped in; see "precision" below |
| `--micro-batch-size` | memory only; one step is always the whole batch, micro-batching is exact |
| `--sync-every` / `--sync-pause` | how often weights go into vLLM and what in-flight requests do meanwhile (`keep` freezes them) |
| `--eval-every` / `--eval-samples` | when and how many episodes to eval; `--eval-samples 0` disables eval |
| `--ckpt-every` / `--keep-last` / `--resume auto` | checkpoint cadence, retention, and restart from the newest |

## Precision

With one training GPU nothing is sharded and no autocast runs, so the model
computes and is stepped in `--dtype`. In bf16 the optimizer's `lr * update`
is often below the weights' resolution and rounds away. `--dtype fp32` is the
safe choice when the memory allows it (a 0.8B model needs ~13 GB for weights,
gradients and Adam state in fp32). Under FSDP, `--mixed-precision` sets the
compute dtype while the sharded weights stay in `--dtype`.

## Crash and resume

Rerun the same script with `--resume auto` in `CKPT_ARGS`. Training continues
from the newest checkpoint with the same weights, optimizer state, RNG, step
and token counters, and rollout cursor; the resumed policy is pushed into vLLM
before the first step.

## W&B

`--wandb` in `WANDB_ARGS` turns logging on (`pip install wandb && wandb login`).
Useful charts: `eval/success` (the headline), `eval/truncated`,
`train/mean_return`, `train/n_live` (sequences with non-zero advantage),
`train/staleness`, `train/clip_frac`, `train/tis_frac`, `train/entropy`,
`train/grad_norm`. The run summary records `<metric>_before`, `_after` and
`_delta`. For offline boxes run with `WANDB_MODE=offline` and
`wandb sync wandb/offline-run-*` later.
