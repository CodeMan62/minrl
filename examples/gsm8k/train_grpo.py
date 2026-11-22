"""
Start the server on GPU 0 first::

    CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen2.5-0.5B-Instruct \\
        --gpu-memory-utilization 0.7

Then train on GPU 1::

    CUDA_VISIBLE_DEVICES=1 python examples/gsm8k/train_grpo.py

or on GPUs 1-3 with FSDP2 (one rank per GPU, each collecting its own group)::

    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 examples/gsm8k/train_grpo.py

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
from minrl.eval import Eval, EvalConfig
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.parser import TextParser
from minrl.inference.vllm_client import VLLMClient
from minrl.loggers import make_logger
from minrl.training import dist
from minrl.training.algorithms import grpo
from minrl.training.config import TrainerConfig
from minrl.training.sources import RolloutSource
from minrl.training.trainer import Trainer

# ``enviornments`` lives at the repo root and is not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT, load_pairs  # noqa: E402


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
    p.add_argument("--concurrency", type=int, default=2,
                   help="groups generating at once while the trainer steps")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--eval-concurrency", type=int, default=16)
    p.add_argument("--eval-dump", default=None, help="dir for per-rollout jsonl dumps")
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

    client = VLLMClient(args.vllm_url, args.model)
    template = HFChatTemplate(tokenizer)

    # Load both splits now, so the first rollout does not wait on a download.
    load_pairs(args.train_split, args.train_limit)
    load_pairs("test")

    # Each ``make()`` builds a fresh (agent, env) pair -- both stateful, and a
    # group's rollouts run concurrently. The train env is shuffled per rank and
    # ``reset(seed)`` serves question ``seed % len``, so one group (one seed)
    # answers one question. Training samples at T=1.0 so behaviour logprobs
    # match the raw policy; eval decodes greedily (T=0) on the test split.
    def make_actor(temperature: float, make_env):
        def make():
            agent = LLMAgent(
                client, template, TextParser(), system_prompt=GSM8K_SYSTEM_PROMPT,
                max_tokens=args.max_new_tokens, temperature=temperature,
            )
            return agent, make_env()
        return make

    make_train = make_actor(1.0, lambda: GSM8K(
        split=args.train_split, limit=args.train_limit, shuffle=True, seed=args.seed + rank,
    ))
    make_eval = make_actor(0.0, lambda: GSM8K(split="test"))

    logger = make_logger(args) if main else None
    trainer = Trainer(
        model,
        algorithm=grpo(clip_eps=args.clip_eps, group_size=args.group_size),
        # One group per step: group_size answers to one question, the set the
        # group-relative advantages are computed over.
        source=RolloutSource(
            make_train, batch_size=args.group_size, max_steps=1, group_size=args.group_size,
            concurrency=args.concurrency,
        ),
        config=TrainerConfig(
            lr=args.lr,
            eps=args.adam_eps,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            mixed_precision="fp16",
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            sync_host=args.weight_transfer_host,
            sync_port=args.weight_transfer_port,
        ),
        logger=logger,
        runtime=client,  # after every step the new weights are pushed into vLLM
    )

    if args.resume:
        trainer.load(args.resume)

    # Eval is rank 0's job: it talks to vLLM, which already holds the synced
    # policy, and the other ranks just wait at their next collective.
    evaluator = Eval(
        EvalConfig(model=args.model, num_episodes=args.eval_samples, max_steps=1,
                   concurrency=args.eval_concurrency, dump_dir=args.eval_dump),
        make_eval, logger=logger,
    ) if main else None
    evals = [evaluator.evaluate_sync(step=trainer.step)] if main else []

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
            evals.append(evaluator.evaluate_sync(step=step))

    trainer.save(os.path.join(args.ckpt_dir, "final"))
    trainer.close()  # stops the rollout engine and drops the NCCL link to vLLM
    if main:
        evals.append(evaluator.evaluate_sync(step=trainer.step))
        baseline, final = evals[0], evals[-1]
        print("\n==== GSM8K accuracy on test split ====")
        for e in evals:
            print(f"  step {e.step:>4}: {e.success:.1%} (truncated {e.truncated:.0%})")
        delta = final.success - baseline.success
        print(f"accuracy change: {baseline.success:.1%} -> {final.success:.1%} ({delta:+.1%})")

    if logger:
        logger.log_summary({
            "accuracy_before": baseline.success,
            "accuracy_after": final.success,
            "accuracy_delta": delta,
        })
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")
    dist.teardown()


if __name__ == "__main__":
    main()
