"""
Start the server on GPU 0 first::

    CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen3-0.6B \\
        --gpu-memory-utilization 0.7 \\
        --weight-transfer-config '{"backend":"nccl"}'

Then train on GPU 1::

    CUDA_VISIBLE_DEVICES=1 python examples/env/tic-tac-toe/train_grpo.py

or on GPUs 1-3 with FSDP2 (one rank per GPU, each playing its own games)::

    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 examples/env/tic-tac-toe/train_grpo.py

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
from minrl.inference.parser import MoveParser
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
from enviornments import TicTacToe  # noqa: E402

SYSTEM_PROMPT = (
    "You are playing Tic-Tac-Toe against an opponent. Cells are numbered 0-8, "
    "left to right, top to bottom. Pick an empty cell. Reply with only the "
    "cell number you play, nothing else."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                   help="vLLM server URL without /v1")
    p.add_argument("--weight-transfer-host", default="127.0.0.1")
    p.add_argument("--weight-transfer-port", type=int, default=29501)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="ckpts/tic-tac-toe")
    p.add_argument("--ckpt-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--resume", default=None, help="checkpoint dir to continue from")
    p.add_argument("--wandb", dest="no_wandb", action="store_false")
    p.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    p.add_argument("--wandb-project", default="minrl-tictactoe")
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(no_wandb=True)
    return p.parse_args()


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
    device = dist.setup()
    if device.type != "cuda":
        raise SystemExit("This example needs a CUDA training GPU.")
    rank, main = dist.rank(), dist.is_main()
    # Per-rank seed: each GPU plays its own games against the random opponent.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if main:
        print(f"loading {args.model} on {dist.world_size()} GPU(s) (float16) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    client = VLLMClient(f"{args.vllm_url.rstrip('/')}/v1", args.model)
    # enable_thinking=False: Qwen3 answers directly instead of spending the
    # token budget on a <think> block.
    template = HFChatTemplate(tokenizer, template_kwargs={"enable_thinking": False})
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

    synchronizer = vllm_weight_synchronizer(
        model, args.vllm_url, host=args.weight_transfer_host, port=args.weight_transfer_port,
        rank=rank,
    )
    logger = make_logger(args) if main else None
    trainer = Trainer(
        model,
        algorithm=grpo(clip_eps=args.clip_eps, group_size=args.group_size),
        source=RolloutSource(train_agent, env, batch_size=args.group_size, max_episode_steps=9),
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

    # Eval is rank 0's job: it plays through vLLM, which already holds the
    # synced policy, and the other ranks just wait at their next collective.
    def log_eval(step: int, e: dict) -> None:
        if logger:
            logger.log({f"eval/{k}_rate": v for k, v in e.items()}, step=step)

    evals = []
    if main:
        baseline = evaluate(eval_agent, env, args.eval_games)
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
            e = evaluate(eval_agent, env, args.eval_games)
            evals.append((step, e))
            print(fmt_eval(f"[eval] after iter {step}", e))
            log_eval(step, e)

    trainer.save(os.path.join(args.ckpt_dir, "final"))
    synchronizer.shutdown()
    if main:
        final = evaluate(eval_agent, env, args.eval_games)
        evals.append((args.iterations, final))
        log_eval(args.iterations, final)

        print("\n==== win rate vs random opponent ====")
        for it, e in evals:
            print(f"  iter {it:>4}: {e['win']:.0%} (illegal {e['illegal']:.0%})")
        delta = final["win"] - baseline["win"]
        print(f"\n{fmt_eval('final', final)}")
        print(f"win rate change: {baseline['win']:.0%} -> {final['win']:.0%} ({delta:+.0%})")

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
    dist.teardown()


if __name__ == "__main__":
    main()
