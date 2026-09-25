# minrl - A RL library to experience RL

minrl is a simple RL-LLM post-training library which i built experience all RL-LLM cocepts I learn.
there are already alot of big RL frameworks out there with much better design and much more features.
minrl is tiny repo for me to experiment around RL-LLM concepts and I wanted to share it.

## Design decisions

**One training loop, swappable math.** An algorithm is two functions: an
*estimator* that turns rewards into a weight for each token, and a *loss* that
turns those weights into a gradient. The `Trainer` runs the same loop for every
algorithm. Trying a new idea means writing its math in `estim.py` or `loss.py`
and pairing the two in `algorithms.py`. The trainer never changes.

**Train on exactly the tokens that were sampled.** The inference client sends
token ids to vLLM and gets token ids and logprobs back, so nothing is
re-tokenized between sampling and training. Multi-turn episodes and tool calls
are stitched into one sequence with an action mask, so the loss only touches
tokens the model actually chose.

**Inference, rollouts and training are separate pieces.** vLLM does inference.
The `RolloutEngine` is an async scheduler that plays episodes: every request has
a unique id and can be cancelled, and it keeps generating while the trainer
computes gradients. The `Trainer` only knows that `next_batch()` returns a
batch. Each piece can be read, tested and replaced on its own.

**Environments and agents are small, async interfaces.** An environment
implements `env_reset(seed)` and `env_step(action)`, both async, so it can await
a judge model or a tool without stalling other episodes. An agent implements one
async method, `act(obs)`. `LLMAgent` is the agent that talks to the model: it
applies the chat template, samples through vLLM, runs tool calls, and uses a
parser to turn the model's text into an action the env understands. Both hold
per-episode state, so training takes a `make()` function that builds a fresh
`(agent, env)` pair for every episode. Episodes in one GRPO group share a seed,
so they face the same problem and their rewards can be compared.

**One training step, end to end.** The pieces above meet like this:

```
RolloutEngine  (background event loop, always generating)
  keeps `concurrency` requests in flight; each plays `group_size` episodes
  under one seed against vLLM, and queues the finished group
        │
        ▼
RolloutSource.next_batch()   takes the oldest groups, reports how stale they are
        │
        ▼
Trainer.train_step()
  estimator  →  per-token weights    (e.g. GRPO's group-normalized advantage)
  loss       →  gradient             (e.g. PPO clipped surrogate + entropy)
  optimizer step  →  push new weights into vLLM  →  checkpoint every N steps
```

**Correct before fast.** Micro-batching is exact, because every loss
normalizes by counts across the whole step and every GPU. Resuming from a
checkpoint is bit-exact. Truncation is read from vLLM's own `finish_reason`,
not guessed. The speed comes from async generation, not from shortcuts.

**Checkpoints you can bet a run on.** Each checkpoint saves the weights, the
optimizer, every RNG stream, the step and token counters, the distributed
topology, and the rollout cursor. Writes are atomic and checksummed, and they
are verified on load. It saves every 2 steps by default, and `--resume auto`
picks up from the newest one.

**Log what you would need to debug.** Entropy, clip fraction, staleness,
gradient norm, token counts and truncation are logged on every run, because
each of them explains some kind of broken training.

## Repo layout

```
src/minrl/      the library; the rollout engine and shared types sit at its top level
  agents/       the agent interface, and LLMAgent for multi-turn and tool calls
  envs/         environment base classes, and a ready-made env for question answering
  inference/    the vLLM server and client, chat templates, and weight sync
  training/     the trainer, algorithms, losses, data sources, checkpoints, FSDP2
  eval/         evaluation on a fixed set of episodes
enviornments/   the environments the examples train on: GSM8K and tic-tac-toe
examples/       runnable training and eval scripts, one folder per environment
docs/           longer notes, starting with entropy in RL
tests/          the test suite, which runs on CPU in about a minute
```

## Examples

| Example | Status | What it teaches |
|---|---|---|
| **GSM8K** | ready | GRPO on grade-school math: one answer per episode, scored by an exact-match check. `examples/env/gsm8k/` |
| **Tic-tac-toe** | ready | GRPO against a random opponent, either one prompt per move or the whole game as one multi-turn sequence. `examples/env/tic-tac-toe/` |
| **Wordle** | in progress | Multi-turn with a reward at every step, and whether episode-level credit is too coarse for it. |
| **Alignment tool env** | in progress | A model answers requests using a policy-lookup tool and is scored by a judge: tool calls, a reward that isn't a simple rule, and reward hacking in one env. |
| **Multi-agent** | in progress | Self-play, where one episode produces a rollout per player. Tic-tac-toe first. |

## Algorithms

| Algorithm | Estimator | Loss | Paper |
|---|---|---|---|
| `grpo` | group-normalized advantage | PPO clipped surrogate, averaged per sequence | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| `dr_grpo` | group-centered advantage, no std | clipped surrogate over a fixed token budget | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| `cispo` | group-normalized advantage | clipped, detached importance weight | [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) |
| `reinforce` | Monte Carlo return, optional baseline | score function | Williams, 1992 |
| `sft` | every demonstration token weighs 1 | cross-entropy | |
| `dpo` | chosen and rejected pairs | logistic preference loss against a reference | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |

The policy-gradient algorithms also take `kl_coef`, which needs a frozen
reference model, and `ent_coef` for an entropy bonus. Entropy is logged either
way; [`docs/entropy.md`](docs/entropy.md) explains why it matters.

## TODOs

- **On-policy distillation (OPD).** Train a student on its own samples, with a
  teacher model scoring every token.
- **Tinker API integration.** Run minrl's algorithms on Tinker's hosted
  training API as well as on local GPUs.
- **Post-train a model on too many gpu's(kinda dumb).**
- **Env registry**


## Acknowledgements

A lot of the code in this repo is inspired by
[ludic](https://github.com/hallerite/ludic). thanks to @hallerite 
