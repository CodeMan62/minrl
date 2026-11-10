#!/usr/bin/env python
"""minrl's one training entry point.

Every example is this file plus a self-contained ``examples/<name>/run.sh``
that sets the flags, the way slime does it: the Python never changes between
experiments, only the config does. The script starts the vLLM server on the
rollout GPU, waits for it, runs this on the training GPU(s), and tears the
server down afterwards::

    examples/dapo-math/run.sh                 # GPU 0 rollouts, GPU 1 trains
    examples/dapo-math/run.sh --eval-only     # score whatever vLLM serves

Done by hand, the same thing is::

    CUDA_VISIBLE_DEVICES=0 python -m minrl.inference.vllm_server Qwen/Qwen3.5-0.8B
    CUDA_VISIBLE_DEVICES=1 python train.py --model Qwen/Qwen3.5-0.8B --env dapo_math ...

or with FSDP2 on several training GPUs, one rank per GPU, each collecting
its own groups::

    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 train.py ...

The environment comes from ``environments.ENVS`` by ``--env``; anything an
environment wants beyond split/limit/shuffle/seed goes through
``--env-arg key=value``. Generation runs on vLLM the whole time: the rollout
engine keeps ``--concurrency`` groups in flight while the trainer steps, and
after every optimizer step the new weights are pushed into vLLM over NCCL.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minrl.agents.llm_agent import LLMAgent
from minrl.eval import Eval, EvalConfig, EvalResult
from minrl.inference.chat_template import HFChatTemplate
from minrl.inference.vllm_client import VLLMClient
from minrl.loggers import Logger, make_logger
from minrl.training import algorithms, checkpoint, dist
from minrl.training.algorithms import Algorithm
from minrl.training.config import TrainerConfig
from minrl.training.sources import RolloutSource
from minrl.training.trainer import Trainer

# ``environments`` lives next to this file and is not an installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments import ENVS, EnvSpec  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------
# flags
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("model")
    g.add_argument("--model", required=True, help="HF id or path; vLLM must serve the same one")
    g.add_argument("--dtype", default="bf16", choices=sorted(DTYPES),
                   help="dtype the policy is loaded and stepped in (see examples/README.md)")
    g.add_argument("--activation-checkpointing", action="store_true")
    g.add_argument("--template-kwargs", default=None,
                   help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": false}\'')

    g = p.add_argument_group("inference runtime (the vLLM server)")
    g.add_argument("--vllm-url", default="http://127.0.0.1:8000", help="server URL without /v1")
    g.add_argument("--server-timeout", type=float, default=900.0,
                   help="seconds to wait for the server to answer /minrl/health")
    g.add_argument("--weight-transfer-host", default="127.0.0.1")
    g.add_argument("--weight-transfer-port", type=int, default=29501)
    g.add_argument("--sync-every", type=int, default=1,
                   help="optimizer steps between weight pushes into vLLM; 0 = never")
    g.add_argument("--sync-pause", default="keep", choices=["keep", "wait", "abort", "none"],
                   help="what vLLM does with in-flight requests during a push: freeze, drain, drop, nothing")

    g = p.add_argument_group("environment and data")
    g.add_argument("--env", required=True, choices=sorted(ENVS), help="entry in environments.ENVS")
    g.add_argument("--env-arg", action="append", metavar="KEY=VALUE",
                   help="environment option, JSON-parsed when it parses; repeatable")
    g.add_argument("--train-split", default="train")
    g.add_argument("--eval-split", default="test")
    g.add_argument("--train-limit", type=int, default=None, help="use only the first N train items")

    g = p.add_argument_group("rollouts")
    g.add_argument("--group-size", type=int, default=8,
                   help="episodes per group; they share a seed, so a prompt, and GRPO compares them")
    g.add_argument("--groups-per-step", type=int, default=1,
                   help="groups per optimizer step; group_size * groups_per_step sequences")
    g.add_argument("--concurrency", type=int, default=2,
                   help="groups generating at once while the trainer steps")
    g.add_argument("--max-queued", type=int, default=None,
                   help="finished groups buffered ahead of the trainer; default = --concurrency")
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=1.0,
                   help="keep 1.0 unless vLLM runs with --logprobs-mode processed_logprobs")
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--multi-turn", action="store_true",
                   help="one sequence per episode: later observations are appended as user turns")
    g.add_argument("--max-seq-len", type=int, default=None, help="context budget per episode in --multi-turn")

    g = p.add_argument_group("algorithm")
    g.add_argument("--algorithm", default="grpo", choices=["grpo", "dr_grpo", "cispo", "reinforce"])
    g.add_argument("--clip-eps", type=float, default=0.2, help="grpo / dr_grpo")
    g.add_argument("--eps-low", type=float, default=0.2, help="cispo")
    g.add_argument("--eps-high", type=float, default=4.0, help="cispo")
    g.add_argument("--kl-coef", type=float, default=0.0,
                   help="KL to a frozen copy of the start model; > 0 loads that second copy")
    g.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus; entropy is logged either way")
    g.add_argument("--tis", action="store_true",
                   help="truncated importance sampling against the rollout logprobs")
    g.add_argument("--icepop", action="store_true",
                   help="IcePop's two-sided rejection of off-policy tokens")
    g.add_argument("--gamma", type=float, default=1.0, help="reinforce")
    g.add_argument("--no-baseline", action="store_true", help="reinforce: raw returns, no mean baseline")

    g = p.add_argument_group("optimizer and training loop")
    g.add_argument("--iterations", type=int, default=100, help="optimizer steps; resume counts from the checkpoint")
    g.add_argument("--lr", type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.999)
    g.add_argument("--adam-eps", type=float, default=1e-8)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--micro-batch-size", type=int, default=2,
                   help="memory knob only; one step is always the whole batch")
    g.add_argument("--strategy", default="auto", choices=["auto", "none", "fsdp"])
    g.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="compute dtype: FSDP's cast policy, or autocast over fp32 weights on one GPU "
                        "(fp16 adds loss scaling)")
    g.add_argument("--cpu-offload", action="store_true")

    g = p.add_argument_group("eval")
    g.add_argument("--eval-every", type=int, default=20, help="steps between evals; 0 = only first and last")
    g.add_argument("--eval-samples", type=int, default=200, help="episodes per eval; 0 disables eval")
    g.add_argument("--eval-concurrency", type=int, default=32)
    g.add_argument("--eval-temperature", type=float, default=0.0)
    g.add_argument("--eval-max-new-tokens", type=int, default=None, help="default: --max-new-tokens")
    g.add_argument("--eval-dump", default=None, help="dir for per-episode jsonl dumps")
    g.add_argument("--eval-only", action="store_true",
                   help="one eval of the policy vLLM currently serves, no training, no model load")

    g = p.add_argument_group("checkpoints")
    g.add_argument("--ckpt-dir", default=None, help="default: ckpts/<env>")
    g.add_argument("--ckpt-every", type=int, default=2, help="0 = only the final checkpoint")
    g.add_argument("--keep-last", type=int, default=3, help="step_* checkpoints kept; 0 = all")
    g.add_argument("--resume", default=None, help="a checkpoint path, or 'auto' for the newest in --ckpt-dir")

    g = p.add_argument_group("logging")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--wandb", dest="no_wandb", action="store_false")
    g.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    g.add_argument("--wandb-project", default="minrl")
    g.add_argument("--wandb-run-name", default=None)
    p.set_defaults(no_wandb=True)

    args = p.parse_args(argv)
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join("ckpts", args.env)
    if args.multi_turn and args.max_seq_len is None:
        p.error("--multi-turn needs --max-seq-len")
    if args.tis and args.icepop:
        p.error("--tis and --icepop are two corrections for the same mismatch; pick one")
    return args


def parse_env_args(items: Optional[List[str]]) -> Dict[str, Any]:
    """``--env-arg limit=100 --env-arg config=en`` -> ``{"limit": 100, "config": "en"}``."""
    opts: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"--env-arg expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # a bare string, e.g. a dataset id
        opts[key.strip().replace("-", "_")] = value
    return opts


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def load_model(name: str, dtype: torch.dtype, device: torch.device):
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except ValueError:
        # A vision-language checkpoint (Qwen3.5 ships as one) is not registered
        # under the causal-LM head. Its text tower still trains on text alone.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.use_cache = False
    return model.to(device)


def build_algorithm(args: argparse.Namespace) -> Algorithm:
    common = dict(ent_coef=args.ent_coef, tis=args.tis, icepop=args.icepop)
    if args.algorithm == "grpo":
        return algorithms.grpo(clip_eps=args.clip_eps, kl_coef=args.kl_coef,
                               group_size=args.group_size, **common)
    if args.algorithm == "dr_grpo":
        return algorithms.dr_grpo(max_tokens=args.max_new_tokens, clip_eps=args.clip_eps,
                                  kl_coef=args.kl_coef, group_size=args.group_size, **common)
    if args.algorithm == "cispo":
        return algorithms.cispo(eps_low=args.eps_low, eps_high=args.eps_high,
                                group_size=args.group_size, **common)
    if args.algorithm == "reinforce":
        return algorithms.reinforce(gamma=args.gamma, baseline=not args.no_baseline, **common)
    raise ValueError(args.algorithm)


def actor_factory(
    args: argparse.Namespace, spec: EnvSpec, client: VLLMClient, template: HFChatTemplate,
    *, temperature: float, max_tokens: int, make_env: Callable[[], Any],
):
    """A ``make()`` that builds a fresh (agent, env) pair: both are stateful,
    and the episodes of one group run concurrently."""
    def make():
        agent = LLMAgent(
            client, template, spec.parser(), system_prompt=spec.system_prompt,
            max_tokens=max_tokens, temperature=temperature, top_p=args.top_p,
            multi_turn=args.multi_turn, max_seq_len=args.max_seq_len if args.multi_turn else None,
        )
        return agent, make_env()
    return make


def report(spec: EnvSpec, args: argparse.Namespace, evals: List[EvalResult], logger: Optional[Logger]) -> None:
    baseline, final = evals[0], evals[-1]
    print(f"\n==== {args.env}: {spec.metric} on the {args.eval_split} split ====")
    for e in evals:
        labels = "".join(f" {k} {v:.0%}" for k, v in e.labels.items())
        print(f"  step {e.step:>4}: {e.success:.1%} (truncated {e.truncated:.0%}){labels}")
    delta = final.success - baseline.success
    print(f"{spec.metric} change: {baseline.success:.1%} -> {final.success:.1%} ({delta:+.1%})")
    if logger:
        key = spec.metric.replace(" ", "_")
        logger.log_summary({
            f"{key}_before": baseline.success,
            f"{key}_after": final.success,
            f"{key}_delta": delta,
        })


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    spec = ENVS[args.env]
    env_opts = parse_env_args(args.env_arg)
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else {}

    device = dist.setup()
    rank, main_rank = dist.rank(), dist.is_main()
    # Per-rank seed: each GPU walks its own order of prompts / games.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = HFChatTemplate(tokenizer, template_kwargs=template_kwargs)
    client = VLLMClient(args.vllm_url, args.model)
    if main_rank:
        served = client.wait_ready(timeout=args.server_timeout)
        if served.get("model") != args.model:
            warnings.warn(f"vLLM serves {served.get('model')!r}, training {args.model!r}", stacklevel=1)
        print(f"vLLM at {args.vllm_url}: {served}")
    dist.barrier()

    # Build each env once now, so the first rollout does not wait on a download.
    def make_train_env():
        return spec.make(args.train_split, limit=args.train_limit, shuffle=True,
                         seed=args.seed + rank, **env_opts)

    def make_eval_env():
        return spec.make(args.eval_split, limit=None, shuffle=False, seed=0, **env_opts)

    make_train_env()
    make_eval_env()

    # Eval is rank 0's job: it talks to vLLM, which holds the synced policy,
    # and the other ranks wait at their next collective.
    logger = make_logger(args) if main_rank else None
    evaluator = None
    if main_rank and args.eval_samples > 0:
        make_eval = actor_factory(
            args, spec, client, template, temperature=args.eval_temperature,
            max_tokens=args.eval_max_new_tokens or args.max_new_tokens, make_env=make_eval_env,
        )
        evaluator = Eval(
            EvalConfig(model=args.model, num_episodes=args.eval_samples, max_steps=spec.max_steps,
                       concurrency=args.eval_concurrency, dump_dir=args.eval_dump),
            make_eval, success=spec.success, label=spec.label, logger=logger,
        )

    if args.eval_only:
        if main_rank:
            if evaluator is None:
                raise SystemExit("--eval-only needs --eval-samples > 0")
            evaluator.evaluate_sync()
            if logger:
                logger.finish()
        dist.teardown()
        return

    if device.type != "cuda":
        raise SystemExit("training needs a CUDA GPU")
    if main_rank:
        print(f"loading {args.model} on {dist.world_size()} GPU(s) ({args.dtype}) ...")
    model = load_model(args.model, DTYPES[args.dtype], device)
    ref_model = load_model(args.model, DTYPES[args.dtype], device) if args.kl_coef > 0 else None

    # Training samples at --temperature (1.0 by default) so the behaviour
    # logprobs match the policy; eval decodes at --eval-temperature.
    make_train = actor_factory(
        args, spec, client, template, temperature=args.temperature,
        max_tokens=args.max_new_tokens, make_env=make_train_env,
    )
    trainer = Trainer(
        model,
        algorithm=build_algorithm(args),
        source=RolloutSource(
            make_train,
            batch_size=args.group_size * args.groups_per_step,
            max_steps=spec.max_steps,
            group_size=args.group_size,
            concurrency=args.concurrency,
            max_queued=args.max_queued,
            seed=args.seed + rank * 1_000_000,
        ),
        config=TrainerConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            micro_batch_size=args.micro_batch_size,
            max_grad_norm=args.max_grad_norm,
            strategy=args.strategy,
            mixed_precision=args.mixed_precision,
            cpu_offload=args.cpu_offload,
            activation_checkpointing=args.activation_checkpointing,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            keep_last=args.keep_last,
            sync_every=args.sync_every,
            sync_host=args.weight_transfer_host,
            sync_port=args.weight_transfer_port,
            sync_pause=None if args.sync_pause == "none" else args.sync_pause,
        ),
        logger=logger,
        ref_model=ref_model,
        runtime=client,  # after every step the new weights are pushed into vLLM
    )

    resume = checkpoint.latest(args.ckpt_dir) if args.resume == "auto" else args.resume
    if resume:
        trainer.load(resume)  # also pushes the resumed policy into vLLM

    evals = [evaluator.evaluate_sync(step=trainer.step)] if evaluator else []
    for metrics in trainer.train(num_steps=args.iterations):
        step = int(metrics["step"])
        if not main_rank:
            continue
        print(
            f"[step {step:>4}/{args.iterations}] "
            f"return={metrics.get('mean_return', float('nan')):+.3f} "
            f"loss={metrics.get('loss', float('nan')):+.4f} "
            f"tokens={metrics.get('n_tokens', 0):.0f} "
            f"live={metrics.get('n_live', 0):.0f}/{metrics.get('n_seqs', 0):.0f} "
            f"staleness={metrics.get('staleness', 0):.1f}"
            + (" skipped" if metrics.get("skipped") else "")
        )
        if evaluator and args.eval_every and step % args.eval_every == 0 and step < args.iterations:
            evals.append(evaluator.evaluate_sync(step=step))

    trainer.save(os.path.join(args.ckpt_dir, "final"))
    trainer.close()  # stops the rollout engine and drops the NCCL link to vLLM
    if evaluator:
        evals.append(evaluator.evaluate_sync(step=trainer.step))
        report(spec, args, evals, logger)
    if logger:
        url = logger.url
        logger.finish()
        print(f"W&B logs: {url}")
    dist.teardown()


if __name__ == "__main__":
    main()
