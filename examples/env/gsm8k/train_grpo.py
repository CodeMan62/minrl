"""Train Qwen3-0.6B on GSM8K with GRPO, using only minrl.
Single-GPU::

    python examples/env/gsm8k/train_grpo.py
Useful knobs::

    python examples/env/gsm8k/train_grpo.py \\
        --iterations 100 --group-size 8 --lr 5e-6 \\
        --eval-every 25 --eval-samples 50
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root + src on sys.path so top-level ``enviornments`` and ``minrl`` (src-layout)
# resolve regardless of cwd or `uv run` / `pip -e` state.
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT  # noqa: E402

from minrl.agents.llm_agent import LLMAgent  # noqa: E402
from minrl.config import LoRAConfig  # noqa: E402
from minrl.inference.chat_template import HFChatTemplate  # noqa: E402
from minrl.inference.hf import HFClient  # noqa: E402
from minrl.inference.parser import TextParser  # noqa: E402
from minrl.interaction import episode  # noqa: E402
from minrl.loggers import WandbLogger, make_logger  # noqa: E402
from minrl.training.algorithms import Algorithm  # noqa: E402
from minrl.training.config import TrainerConfig  # noqa: E402
from minrl.training.sources import RolloutSource  # noqa: E402
from minrl.training.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--eval-split", default="test", choices=["train", "test"])
    p.add_argument("--train-split", default="train", choices=["train", "test"])
    p.add_argument("--train-limit", type=int, default=None,
                   help="limit train dataset to first N examples (default: all)")
    p.add_argument("--eval-limit", type=int, default=None,
                   help="limit eval pool to first N examples (default: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true",
                   help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", default="minrl-gsm8k")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--wandb-run-name", default=None,
                   help="optional run name (W&B generates one if omitted)")
    return p.parse_args()

def evaluate(agent: LLMAgent, env, n: int) -> dict:
    """Score ``n`` questions greedily; report accuracy."""
    correct = 0
    for _ in range(n):
        agent.reset()
        r = episode(agent, env, max_steps=1)
        # QAEnv reward is 1.0 / 0.0; also available as info["correct"]
        if r.total_reward > 0:
            correct += 1
    acc = correct / n if n else 0.0
    return {"accuracy": acc, "correct": correct, "total": n}

def fmt_eval(tag: str, e: dict) -> str:
    return f"{tag}: accuracy {e['accuracy']:.1%} ({e['correct']}/{e['total']})"

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading {args.model} on {args.device} ({dtype}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base_model.to(args.device)
    model = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    ).apply(base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = HFClient(model, tokenizer)
    # enable_thinking=False: Qwen3 answers directly instead of spending the
    # token budget on a <think> block.
    template = HFChatTemplate(tokenizer, template_kwargs={"enable_thinking": False})
    parser = TextParser()

    # ``repeat=group_size`` so each GRPO group shares one question — the
    # setting where group-normalized advantages are meaningful.
    train_env = GSM8K(
        split=args.train_split,
        limit=args.train_limit,
        repeat=args.group_size,
        shuffle=False,
        seed=args.seed,
    )
    # Eval uses a separate env instance so its resets don't desync the
    # training env's repeat cycling (see QAEnv docstring).
    eval_env = GSM8K(
        split=args.eval_split,
        limit=args.eval_limit,
        repeat=1,
        shuffle=False,
        seed=args.seed,
    )

    # Training samples at T=1.0 so behaviour logprobs match the raw policy;
    # eval decodes greedily (T=0).
    train_agent = LLMAgent(
        client, template, parser, system_prompt=GSM8K_SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens, temperature=1.0,
    )
    eval_agent = LLMAgent(
        client, template, parser, system_prompt=GSM8K_SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens, temperature=0.0,
    )

    logger = make_logger(args)

    cfg = TrainerConfig(
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        strategy="none",
        mixed_precision="bf16",
        log_prefix="train",
        log_every=1,
    )

    source = RolloutSource(
        train_agent,
        train_env,
        batch_size=args.group_size,
        max_episode_steps=1,
    )
    trainer = Trainer(
        model,
        algorithm=Algorithm("grpo", clip_eps=args.clip_eps),
        source=source,
        config=cfg,
        logger=logger,
    )

    def log_eval(step: int, e: dict) -> None:
        if logger:
            logger.log({f"eval/{k}": v for k, v in e.items() if k != "total"}, step=step)

    baseline = evaluate(eval_agent, eval_env, args.eval_samples)
    print(fmt_eval("[eval] before training", baseline))
    log_eval(0, baseline)

    evals = [(0, baseline)]
    for metrics in trainer.train(num_steps=args.iterations):
        step = int(metrics["step"])
        print(
            f"[iter {step:>4}/{args.iterations}] "
            f"return={metrics.get('mean_return', float('nan')):+.2f} "
            f"loss={metrics.get('loss', float('nan')):+.4f} "
            f"tokens={metrics.get('n_tokens', 0):.0f}"
        )
        if step % args.eval_every == 0 and step < args.iterations:
            e = evaluate(eval_agent, eval_env, args.eval_samples)
            evals.append((step, e))
            print(fmt_eval(f"[eval] after iter {step}", e))
            log_eval(step, e)

    final = evaluate(eval_agent, eval_env, args.eval_samples)
    evals.append((args.iterations, final))
    log_eval(args.iterations, final)

    print("\n==== accuracy on held-out split ====")
    for it, e in evals:
        print(f"  iter {it:>4}: {e['accuracy']:.1%}")
    delta = final["accuracy"] - baseline["accuracy"]
    print(f"\n{fmt_eval('final', final)}")
    print(f"accuracy change: {baseline['accuracy']:.1%} -> {final['accuracy']:.1%} ({delta:+.1%})")

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
