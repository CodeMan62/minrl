"""Train Qwen3-0.6B on GSM8K with GRPO, using only minrl.

Each GRPO group answers the same GSM8K question ``group_size`` times, so a
question every sample in the group gets right (or wrong) contributes zero
advantage and is skipped -- GRPO only learns from questions the group's
answers disagree on. Pass@1 accuracy on the held-out test split (greedy
decoding) is checked before, during, and after training.

Inference runs *in-process* through ``HFClient`` -- the sampled model IS the
trained model, so every rollout is exactly on-policy with no weight syncing.

Run (needs a GPU box; ~1.2GB of bf16 weights + optimizer states):

    python examples/gsm8k/train_gsm8k.py

Useful knobs:

    python examples/gsm8k/train_gsm8k.py \
        --iterations 500 --group-size 8 --lr 5e-6 \
        --eval-every 50 --eval-questions 200

A CPU smoke run (tiny + slow, just to see it move):

    python examples/gsm8k/train_gsm8k.py \
        --iterations 2 --group-size 2 --train-limit 8 --eval-questions 4 \
        --device cpu --no-wandb
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# Repo root on sys.path so the top-level ``enviornments`` package resolves
# regardless of the cwd this script is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from enviornments import GSM8K  # noqa: E402

from minrl.agents.llm_agent import LLMAgent  # noqa: E402
from minrl.inference.chat_template import ChatTemplate  # noqa: E402
from minrl.inference.hf import HFClient  # noqa: E402
from minrl.inference.parser import TextParser  # noqa: E402
from minrl.interaction import episode  # noqa: E402
from minrl.training.algorithms import grpo  # noqa: E402
from minrl.loggers import WandbLogger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-questions", type=int, default=200)
    p.add_argument("--train-limit", type=int, default=None,
                   help="cap the number of training questions (debugging)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true",
                   help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", default="minrl-gsm8k")
    p.add_argument("--wandb-run-name", default=None,
                   help="optional run name (W&B generates one if omitted)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--save-dir", default=None,
                   help="if set, save the LoRA adapter + tokenizer here after training")
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


def evaluate(agent: LLMAgent, env: GSM8K, questions: int) -> dict:
    """Answer ``questions`` GSM8K problems greedily; report pass@1 accuracy."""
    correct = 0
    for _ in range(questions):
        agent.reset()
        r = episode(agent, env, max_steps=1)
        if r.steps[-1].info.get("correct"):
            correct += 1
    return {"accuracy": correct / questions}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base_model.to(args.device)
    lora_config = LoraConfig(
        r=args.lora_rank,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = HFClient(model, tokenizer)
    # enable_thinking=False: Qwen3 answers directly instead of spending the
    # token budget on a <think> block.
    template = ChatTemplate(args.model, template_kwargs={"enable_thinking": False})
    parser = TextParser()

    # repeat=group_size: every GRPO group answers the same question, so the
    # group mean is a meaningful per-question baseline.
    train_env = GSM8K(
        split="train", limit=args.train_limit, repeat=args.group_size,
        shuffle=True, seed=args.seed,
    )
    eval_env = GSM8K(split="test", limit=args.eval_questions)

    # Training samples at T=1.0 so behaviour logprobs match the raw policy;
    # eval decodes greedily (T=0) on the held-out split.
    train_agent = LLMAgent(
        client, template, parser, system_prompt=train_env.sys_prompt(),
        max_tokens=args.max_new_tokens, temperature=1.0,
    )
    eval_agent = LLMAgent(
        client, template, parser, system_prompt=eval_env.sys_prompt(),
        max_tokens=args.max_new_tokens, temperature=0.0,
    )

    logger = make_logger(args)

    # grpo() logs train/* metrics (mean_return, loss, ...) each step by
    # itself; this script adds the env-specific extras and eval/* on top.
    training = grpo(
        model, train_agent, train_env,
        iterations=args.iterations,
        group_size=args.group_size,
        max_episode_steps=1,          # QAEnv scores the completion in one step
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        log_every=0,                  # this script prints its own line per iter
        logger=logger,
    )

    def log_eval(step: int, e: dict) -> None:
        if logger:
            logger.log({f"eval/{k}": v for k, v in e.items()}, step=step)

    baseline = evaluate(eval_agent, eval_env, args.eval_questions)
    print(f"[eval] before training: accuracy {baseline['accuracy']:.1%}")
    log_eval(0, baseline)

    evals = [(0, baseline)]
    for i, (group, metrics) in enumerate(training):
        train_acc = sum(
            1 for r in group if r.steps[-1].info.get("correct")
        ) / len(group)
        print(
            f"[iter {i + 1:>4}/{args.iterations}] "
            f"return={metrics['mean_return']:+.2f} loss={metrics['loss']:+.4f} "
            f"train acc={train_acc:.0%} tokens={metrics['n_tokens']:.0f}"
        )
        if logger:
            logger.log({"train/accuracy": train_acc}, step=i + 1)
        if (i + 1) % args.eval_every == 0 and (i + 1) < args.iterations:
            e = evaluate(eval_agent, eval_env, args.eval_questions)
            evals.append((i + 1, e))
            print(f"[eval] after iter {i + 1}: accuracy {e['accuracy']:.1%}")
            log_eval(i + 1, e)

    final = evaluate(eval_agent, eval_env, args.eval_questions)
    evals.append((args.iterations, final))
    log_eval(args.iterations, final)

    print("\n==== accuracy on GSM8K test split ====")
    for it, e in evals:
        print(f"  iter {it:>4}: {e['accuracy']:.1%}")
    delta = final["accuracy"] - baseline["accuracy"]
    print(f"\naccuracy change: {baseline['accuracy']:.1%} -> {final['accuracy']:.1%} ({delta:+.1%})")

    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"saved LoRA adapter to {args.save_dir}")

    if logger:
        logger.log_summary(
            {
                "accuracy_before": baseline["accuracy"],
                "accuracy_after": final["accuracy"],
                "accuracy_delta": delta,
            }
        )
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")


if __name__ == "__main__":
    main()
