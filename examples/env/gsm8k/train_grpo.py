"""
Start the server on GPU 0 first::

    CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen2.5-0.5B-Instruct \\
        --gpu-memory-utilization 0.7 \\
        --weight-transfer-config '{"backend":"nccl"}'

Then train on GPU 1::

    CUDA_VISIBLE_DEVICES=1 python examples/env/gsm8k/train_grpo.py

or on GPUs 1-3 with FSDP2 (one rank per GPU, each collecting its own group)::

    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 examples/env/gsm8k/train_grpo.py

After every optimizer step the trainer broadcasts its weights to vLLM, so the
next group is sampled from the updated policy.
"""

import argparse
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minrl.agents.llm_agent import LLMAgent
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.parser import TextParser
from minrl.inference.vllm_client import VLLMClient, vllm_weight_synchronizer
from minrl.interaction import episode
from minrl.loggers import make_logger
from minrl.training import dist
from minrl.training.algorithms import grpo
from minrl.training.config import TrainerConfig
from minrl.training.sources import RolloutSource
from minrl.training.trainer import Trainer

# ``enviornments`` lives at the repo root and is not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                   help="vLLM server URL without /v1")
    p.add_argument("--weight-transfer-host", default="127.0.0.1")
    p.add_argument("--weight-transfer-port", type=int, default=29501)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--train-split", default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="ckpts/gsm8k")
    p.add_argument("--ckpt-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--resume", default=None, help="checkpoint dir to continue from")
    p.add_argument("--wandb", dest="no_wandb", action="store_false")
    p.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    p.add_argument("--wandb-project", default="minrl-gsm8k")
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(no_wandb=True)
    return p.parse_args()


def evaluate_gsm8k(agent: LLMAgent, env: GSM8K, samples: int) -> dict[str, float]:
    """Evaluate ``samples`` questions with the agent's current decode settings."""
    if samples < 1:
        raise ValueError("samples must be positive")
    env.rewind()
    correct = 0
    truncated = 0
    for _ in range(samples):
        agent.reset()
        rollout = episode(agent, env, max_steps=1)
        if rollout.total_reward > 0:
            correct += 1
        step = rollout.steps[-1]
        if sum(step.action_mask or []) >= agent.max_tokens:
            truncated += 1
    return {
        "accuracy": correct / samples,
        "correct": float(correct),
        "truncated": truncated / samples,
    }


def fmt_eval(tag: str, e: dict) -> str:
    return (
        f"{tag}: accuracy={e['accuracy']:.1%} "
        f"({int(e['correct'])} correct) truncated={e['truncated']:.1%}"
    )


def main() -> None:
    args = parse_args()
    device = dist.setup()
    if device.type != "cuda":
        raise SystemExit("This example needs a CUDA training GPU.")
    rank, main = dist.rank(), dist.is_main()
    # Per-rank seed: each GPU walks the questions in its own order, so ranks
    # train on different prompts instead of the same one in lockstep.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if main:
        print(f"loading {args.model} on {dist.world_size()} GPU(s) (float16) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = VLLMClient(f"{args.vllm_url.rstrip('/')}/v1", args.model)
    template = HFChatTemplate(tokenizer)
    parser = TextParser()

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
    # repeat=group_size makes each source batch one question sampled group_size
    # times -- the prompt-sharing GRPO advantages are computed over.
    env = GSM8K(
        split=args.train_split, limit=args.train_limit,
        repeat=args.group_size, shuffle=True, seed=args.seed + rank,
    )
    eval_env = GSM8K(split="test", repeat=1)

    synchronizer = vllm_weight_synchronizer(
        model, args.vllm_url, host=args.weight_transfer_host, port=args.weight_transfer_port,
        rank=rank,
    )
    logger = make_logger(args) if main else None
    trainer = Trainer(
        model,
        algorithm=grpo(clip_eps=args.clip_eps, group_size=args.group_size),
        source=RolloutSource(train_agent, env, batch_size=args.group_size, max_episode_steps=1),
        config=TrainerConfig(
            lr=args.lr,
            eps=args.adam_eps,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            mixed_precision="fp16",
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
        ),
        logger=logger,
        weight_synchronizer=synchronizer,
    )

    if args.resume:
        trainer.load(args.resume)

    # Eval is rank 0's job: it talks to vLLM, which already holds the synced
    # policy, and the other ranks just wait at their next collective.
    def log_eval(step: int, e: dict) -> None:
        if logger:
            logger.log({f"eval/{k}": v for k, v in e.items()}, step=step)

    evals = []
    if main:
        baseline = evaluate_gsm8k(eval_agent, eval_env, args.eval_samples)
        print(fmt_eval("[eval] before training", baseline))
        log_eval(trainer.step, baseline)
        evals.append((trainer.step, baseline))

    for metrics in trainer.train(num_steps=args.iterations):
        step = int(metrics["step"])
        if not main:
            continue
        print(
            f"[iter {step:>4}/{args.iterations}] "
            f"return={metrics.get('mean_return', float('nan')):+.2f} "
            f"loss={metrics.get('loss', float('nan')):+.4f} "
            f"tokens={metrics.get('n_tokens', 0):.0f}"
        )
        if step % args.eval_every == 0 and step < args.iterations:
            e = evaluate_gsm8k(eval_agent, eval_env, args.eval_samples)
            evals.append((step, e))
            print(fmt_eval(f"[eval] after iter {step}", e))
            log_eval(step, e)

    trainer.save(os.path.join(args.ckpt_dir, "final"))
    synchronizer.shutdown()
    if main:
        final = evaluate_gsm8k(eval_agent, eval_env, args.eval_samples)
        evals.append((args.iterations, final))
        log_eval(args.iterations, final)

        print("\n==== GSM8K accuracy on test split ====")
        for it, e in evals:
            print(f"  iter {it:>4}: {e['accuracy']:.1%} (truncated {e['truncated']:.0%})")
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
    dist.teardown()


if __name__ == "__main__":
    main()
