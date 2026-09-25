"""
Start the server on GPU 0 first::

    CUDA_VISIBLE_DEVICES=0 vllm_server Qwen/Qwen3-0.6B \\
        --gpu-memory-utilization 0.7 \\
        --weight-transfer-config '{"backend":"nccl"}'

Then train on GPU 1::

    CUDA_VISIBLE_DEVICES=1 python examples/env/tic-tac-toe/train_grpo.py

or on GPUs 1-3 with FSDP2 (one rank per GPU, each playing its own games)::

    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 examples/env/tic-tac-toe/train_grpo.py
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
from minrl.eval import Eval, EvalConfig
from minrl.inference.vllm_client import VLLMClient, vllm_weight_synchronizer
from minrl.loggers import make_logger
from minrl.training import checkpoint, dist
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
    p.add_argument("--multi-turn", action="store_true",
                   help="one sequence per game: the policy sees every earlier board "
                        "and its own moves, instead of a fresh prompt per move")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="context budget per game in --multi-turn mode")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="entropy bonus; 0 logs entropy without steering it")
    p.add_argument("--concurrency", type=int, default=2,
                   help="groups playing at once while the trainer steps")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--eval-concurrency", type=int, default=16)
    p.add_argument("--eval-dump", default=None, help="dir for per-rollout jsonl dumps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="ckpts/tic-tac-toe")
    p.add_argument("--ckpt-every", type=int, default=2,
                   help="checkpoint every N steps; 0 = only at the end")
    p.add_argument("--resume", default=None,
                   help="checkpoint to continue from, or 'auto' for the newest under --ckpt-dir")
    p.add_argument("--keep-last", type=int, default=3, help="step_* checkpoints to keep; 0 = all")
    p.add_argument("--wandb", dest="no_wandb", action="store_false")
    p.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    p.add_argument("--wandb-project", default="minrl-tictactoe")
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(no_wandb=True)
    return p.parse_args()


def outcome(r) -> str:
    """How a game ended, from the final step's info."""
    info = r.steps[-1].info
    return "illegal" if info.get("illegal_move") else info.get("result", "loss")


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

    # Each ``make()`` builds a fresh (agent, env) pair -- both stateful, and
    # games run concurrently. Training samples at T=1.0 so behaviour logprobs
    # match the raw policy; eval decodes greedily (T=0) against the same
    # random opponent.
    def make_actor(temperature: float):
        def make():
            agent = LLMAgent(
                client, template, MoveParser(), system_prompt=SYSTEM_PROMPT,
                max_tokens=args.max_new_tokens, temperature=temperature,
                multi_turn=args.multi_turn,
                max_seq_len=args.max_seq_len if args.multi_turn else None,
            )
            return agent, TicTacToe()
        return make

    make_train, make_eval = make_actor(1.0), make_actor(0.0)

    synchronizer = vllm_weight_synchronizer(
        model, args.vllm_url, host=args.weight_transfer_host, port=args.weight_transfer_port,
        rank=rank,
    )
    logger = make_logger(args) if main else None
    trainer = Trainer(
        model,
        algorithm=grpo(clip_eps=args.clip_eps, ent_coef=args.ent_coef, group_size=args.group_size),
        # One group per step: group_size games from one seed (same opponent
        # moves, same side), the set the group-relative advantages are over.
        source=RolloutSource(
            make_train, batch_size=args.group_size, max_steps=9, group_size=args.group_size,
            concurrency=args.concurrency, seed=args.seed + rank * 1_000_000,
        ),
        config=TrainerConfig(
            lr=args.lr,
            eps=args.adam_eps,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            mixed_precision="fp16",
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            keep_last=args.keep_last,
        ),
        logger=logger,
        weight_synchronizer=synchronizer,
    )

    if args.resume == "auto":
        args.resume = checkpoint.latest(args.ckpt_dir)  # None on a fresh run
    if args.resume:
        trainer.load(args.resume)

    # Eval is rank 0's job: it plays through vLLM, which already holds the
    # synced policy, and the other ranks just wait at their next collective.
    evaluator = Eval(
        EvalConfig(model=args.model, num_episodes=args.eval_games, max_steps=9,
                   concurrency=args.eval_concurrency, dump_dir=args.eval_dump),
        make_eval, success=lambda r: outcome(r) == "win", label=outcome, logger=logger,
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
    trainer.close()  # stop the rollout engine before tearing down weight transfer
    synchronizer.shutdown()
    if main:
        evals.append(evaluator.evaluate_sync(step=trainer.step))
        baseline, final = evals[0], evals[-1]
        print("\n==== win rate vs random opponent ====")
        for e in evals:
            print(f"  step {e.step:>4}: {e.success:.0%} (illegal {e.labels.get('illegal', 0):.0%})")
        delta = final.success - baseline.success
        print(f"win rate change: {baseline.success:.0%} -> {final.success:.0%} ({delta:+.0%})")

    if logger:
        logger.log_summary({
            "win_rate_before": baseline.success,
            "win_rate_after": final.success,
            "win_rate_delta": delta,
        })
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")
    dist.teardown()


if __name__ == "__main__":
    main()
