"""Train Qwen3-0.6B to play TicTacToe with GRPO, using only minrl.

The model plays full games against the env's random opponent. Episode return
is +1 win / 0 draw / -1 loss / -1 illegal move, GRPO normalizes returns within
each group of episodes, and the win rate vs the random opponent is evaluated
(greedy decoding) before, during, and after training.

Inference runs *in-process* through ``HFClient`` — the sampled model IS the
trained model, so every rollout is exactly on-policy with no weight syncing.

By default the model is wrapped in a LoRA adapter (peft) covering every linear
layer: the base weights stay frozen, so gradients and AdamW state exist only
for the adapter (~1% of params) and checkpoints are a few MB. Pass
--full-finetune to train all weights instead.

Run (needs a GPU box; ~1.2GB of bf16 weights, adapter-only optimizer states):

    python examples/tic-tac-toe/train_grpo.py

Useful knobs:

    python examples/tic-tac-toe/train_grpo.py \
        --iterations 150 --group-size 8 --lr 1e-4 \
        --lora-r 16 --lora-alpha 32 \
        --eval-every 25 --eval-games 50 \
        --save-dir runs/ttt-lora

A CPU smoke run (tiny + slow, just to see it move): --iterations 2
--group-size 2 --eval-games 4 --device cpu
"""

import argparse
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
from minrl.algorithms import grpo  # noqa: E402
from minrl.loggers import WandbLogger  # noqa: E402

SYSTEM_PROMPT = (
    "You are playing Tic-Tac-Toe against an opponent. Cells are numbered 0-8, "
    "left to right, top to bottom. Pick an empty cell. Reply with only the "
    "cell number you play, nothing else."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: 1e-4 with LoRA, 5e-6 with "
                        "--full-finetune)")
    p.add_argument("--full-finetune", action="store_true",
                   help="train all weights instead of a LoRA adapter")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA scaling (effective scale is alpha/r)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-dir", default=None,
                   help="save the trained adapter (or full model) here")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true",
                   help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", default="minrl-tictactoe")
    p.add_argument("--wandb-run-name", default=None,
                   help="optional run name (W&B generates one if omitted)")
    return p.parse_args()


def make_logger(args: argparse.Namespace):
    """Start a W&B run, or return None (with a note) if unavailable/disabled."""
    if args.no_wandb:
        return None
    try:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    except ImportError:
        print("wandb not installed — skipping W&B logging (pip install wandb)")
        return None
    print(f"W&B run: {logger.url}")
    return logger


def wrap_lora(model, args: argparse.Namespace):
    """Freeze the base model and inject a trainable LoRA adapter."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA training requires peft (pip install peft); "
            "or pass --full-finetune to train all weights."
        ) from e
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA r={args.lora_r} alpha={args.lora_alpha}: "
        f"{trainable:,} trainable params ({trainable / total:.2%} of {total:,})"
    )
    return model


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

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if not args.full_finetune:
        model = wrap_lora(model, args)
    if args.lr is None:
        # LoRA adapters train well at much higher learning rates than full FT.
        args.lr = 5e-6 if args.full_finetune else 1e-4

    client = HFClient(model, tokenizer)
    # enable_thinking=False: Qwen3 answers directly instead of spending the
    # token budget on a <think> block.
    template = ChatTemplate(args.model, template_kwargs={"enable_thinking": False})
    parser = MoveParser()
    env = TicTacToe()

    # Training samples at T=1.0 so behaviour logprobs match the raw policy;
    # eval decodes greedily (T=0) against the same random opponent.
    train_agent = LLMAgent(
        client, template, parser, system_prompt=SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens, temperature=1.0,
    )
    eval_agent = LLMAgent(
        client, template, parser, system_prompt=SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens, temperature=0.0,
    )

    logger = make_logger(args)

    # grpo() logs train/* metrics (mean_return, loss, ...) each step by
    # itself; this script adds the env-specific extras and eval/* on top.
    training = grpo(
        model, train_agent, env,
        iterations=args.iterations,
        group_size=args.group_size,
        max_episode_steps=9,          # a TicTacToe game is at most 5 own moves
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        log_every=0,                  # this script prints its own line per iter
        logger=logger,
    )

    def log_eval(step: int, e: dict) -> None:
        if logger:
            logger.log({f"eval/{k}_rate": v for k, v in e.items()}, step=step)

    baseline = evaluate(eval_agent, env, args.eval_games)
    print(fmt_eval("[eval] before training", baseline))
    log_eval(0, baseline)

    evals = [(0, baseline)]
    for i, (group, metrics) in enumerate(training):
        wins = sum(1 for r in group if r.steps[-1].info.get("result") == "win")
        illegal = sum(1 for r in group if r.steps[-1].info.get("illegal_move"))
        print(
            f"[iter {i + 1:>4}/{args.iterations}] "
            f"return={metrics['mean_return']:+.2f} loss={metrics['loss']:+.4f} "
            f"train W/illegal={wins}/{illegal} of {len(group)}"
        )
        if logger:
            logger.log(
                {
                    "train/win_rate": wins / len(group),
                    "train/illegal_rate": illegal / len(group),
                },
                step=i + 1,
            )
        if (i + 1) % args.eval_every == 0 and (i + 1) < args.iterations:
            e = evaluate(eval_agent, env, args.eval_games)
            evals.append((i + 1, e))
            print(fmt_eval(f"[eval] after iter {i + 1}", e))
            log_eval(i + 1, e)

    final = evaluate(eval_agent, env, args.eval_games)
    evals.append((args.iterations, final))
    log_eval(args.iterations, final)

    print("\n==== win rate vs random opponent ====")
    for it, e in evals:
        print(f"  iter {it:>4}: {e['win']:.0%} (illegal {e['illegal']:.0%})")
    delta = final["win"] - baseline["win"]
    print(f"\n{fmt_eval('final', final)}")
    print(f"win rate change: {baseline['win']:.0%} -> {final['win']:.0%} ({delta:+.0%})")

    if args.save_dir:
        # PeftModel.save_pretrained writes only the adapter (a few MB);
        # with --full-finetune this saves the whole model instead.
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        kind = "model" if args.full_finetune else "LoRA adapter"
        print(f"saved {kind} to {args.save_dir}")
    if logger:
        logger.log_summary(
            {
                "win_rate_before": baseline["win"],
                "win_rate_after": final["win"],
                "win_rate_delta": delta,
            }
        )
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")


if __name__ == "__main__":
    main()
