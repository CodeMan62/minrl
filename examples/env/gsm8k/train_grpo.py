"""Start the server on GPU 0 first::

    CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen2.5-0.5B-Instruct \\
        --gpu-memory-utilization 0.7 \\
        --weight-transfer-config '{"backend":"nccl"}'

Then train on GPU 1::

    CUDA_VISIBLE_DEVICES=1 python examples/env/gsm8k/train_grpo.py

"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minrl.agents.llm_agent import LLMAgent
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.parser import TextParser
from minrl.inference.vllm_client import VLLMClient, vllm_weight_synchronizer
from minrl.loggers import make_logger
from minrl.training.algorithms import Algorithm
from minrl.training.config import TrainerConfig
from minrl.training.sources import RolloutSource
from minrl.training.trainer import Trainer

# ``enviornments`` lives at the repo root and is not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from enviornments.gsm8k import GSM8K, GSM8K_SYSTEM_PROMPT, evaluate_gsm8k  # noqa: E402


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
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--train-split", default="train", choices=["train", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", dest="no_wandb", action="store_false")
    p.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    p.add_argument("--wandb-project", default="minrl-gsm8k")
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(no_wandb=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("This example needs a CUDA training GPU.")
    device = torch.device(args.device)
    if device.index is not None:
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"loading {args.model} on {device} (float16) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = VLLMClient(f"{args.vllm_url.rstrip('/')}/v1", args.model)
    template = HFChatTemplate(tokenizer)
    agent = LLMAgent(
        client,
        template,
        TextParser(),
        system_prompt=GSM8K_SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens,
        temperature=1.0,
    )
    env = GSM8K(
        split=args.train_split,
        limit=args.train_limit,
        repeat=args.group_size,
        shuffle=False,
        seed=args.seed,
    )
    eval_env = GSM8K(split="test", repeat=1)
    eval_agent = LLMAgent(
        client,
        template,
        TextParser(),
        system_prompt=GSM8K_SYSTEM_PROMPT,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    synchronizer = vllm_weight_synchronizer(
        model, args.vllm_url, host=args.weight_transfer_host, port=args.weight_transfer_port,
    )
    logger = make_logger(args)
    trainer = Trainer(
        model,
        algorithm=Algorithm("grpo", clip_eps=args.clip_eps),
        source=RolloutSource(agent, env, batch_size=args.group_size, max_episode_steps=1),
        config=TrainerConfig(
            lr=args.lr,
            eps=args.adam_eps,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            strategy="none",
            mixed_precision="fp16",
            log_prefix="train",
            log_every=1,
        ),
        logger=logger,
        weight_synchronizer=synchronizer,
    )

    def evaluate(step: int) -> dict:
        score = evaluate_gsm8k(eval_agent, eval_env, args.eval_samples)
        if logger:
            logger.log({f"eval/{k}": v for k, v in score.items()}, step=step)
        return score

    baseline = evaluate(0)
    print(f"[eval    0] accuracy={baseline['accuracy']:.1%} "
          f"({int(baseline['correct'])}/{args.eval_samples}) truncated={baseline['truncated']:.0%}")
    for metrics in trainer.train(num_steps=args.iterations):
        step = int(metrics["step"])
        print(
            f"[iter {step:>4}/{args.iterations}] "
            f"return={metrics.get('mean_return', float('nan')):+.2f} "
            f"loss={metrics.get('loss', float('nan')):+.4f} "
            f"tokens={metrics.get('n_tokens', 0):.0f}"
        )
        if step % args.eval_every == 0 and step < args.iterations:
            score = evaluate(step)
            delta = score["accuracy"] - baseline["accuracy"]
            print(f"[eval {step:>4}] accuracy={score['accuracy']:.1%} ({delta:+.1%}) "
                  f"truncated={score['truncated']:.0%}")

    synchronizer.shutdown()
    final = evaluate(args.iterations)
    delta = final["accuracy"] - baseline["accuracy"]
    print(
        f"\nGSM8K accuracy: {baseline['accuracy']:.1%} -> "
        f"{final['accuracy']:.1%} ({delta:+.1%})"
    )
    if logger:
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")


if __name__ == "__main__":
    main()
