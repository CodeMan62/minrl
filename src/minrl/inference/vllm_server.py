"""minrl vLLM server: vLLM's OpenAI-compatible app, plus our own routes.

We build the app ourselves instead of calling ``vllm serve`` so that minrl
can add endpoints (see ``router``) and hook the engine as training needs grow.
Usage::

    vllm_server Qwen/Qwen3-0.6B --port 8000
"""

from __future__ import annotations

import os
import signal
import sys
from argparse import Namespace
from typing import List, Optional, Sequence

import uvloop
from fastapi import APIRouter, Request
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.launchers.launcher import serve_http, setup_server
from vllm.utils.argparse_utils import FlexibleArgumentParser

_TOKEN_ID_FLAG = "--return-tokens-as-token-ids"
_NO_TOKEN_ID_FLAG = "--no-return-tokens-as-token-ids"

# minrl's own endpoints live under /minrl;
router = APIRouter(prefix="/minrl")


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness plus what is being served, for trainers waiting on startup."""
    state = request.app.state
    return {
        "status": "ok",
        "model": state.args.model,
        "dtype": str(state.vllm_config.model_config.dtype),
        "world_size": state.vllm_config.parallel_config.world_size,
        "weight_version": await state.engine_client.get_weight_version(),
    }


@router.post("/init_weight_sync")
async def init_weight_sync(request: Request) -> dict:
    """Workers join the trainer's NCCL group; returns once the rendezvous completes."""
    body = await request.json()
    await request.app.state.engine_client.collective_rpc(
        "init_weight_sync",
        kwargs={"host": body["host"], "port": int(body["port"]), "world_size": int(body["world_size"])},
    )
    return {"status": "ok"}


@router.post("/update_weights")
async def update_weights(request: Request) -> dict:
    """Workers receive one broadcast per parameter from the trainer and load them."""
    body = await request.json()
    engine = request.app.state.engine_client
    await engine.collective_rpc("receive_weights", kwargs={"weights": body["weights"]})
    if body.get("version") is not None:
        await engine.update_weight_version(str(body["version"]))
    return {"status": "ok", "weight_version": await engine.get_weight_version()}


def build_serve_argv(model: str, extra: Sequence[str] = ()) -> List[str]:
    """Build the ``vllm serve`` argv, injecting minrl's required flag."""
    extra = list(extra)
    argv = ["vllm", "serve", model]
    if _TOKEN_ID_FLAG not in extra and _NO_TOKEN_ID_FLAG not in extra:
        argv.append(_TOKEN_ID_FLAG)
    argv.extend(extra)
    return argv


async def run_server(args: Namespace) -> None:
    """Serve ``args.model`` with vLLM's app extended by ``router``."""
    listen_address, sock = setup_server(args, reuse_port=False)

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine:
        supported_tasks = await engine.get_supported_tasks()
        app = build_app(args, supported_tasks, engine.model_config)
        app.include_router(router)
        await init_app_state(engine, app.state, args, supported_tasks)
        app.state.args = args
        print(f"minrl: serving {args.model} on {listen_address}", flush=True)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )
    try:
        await shutdown_task
    finally:
        sock.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raise SystemExit("usage: vllm_server <model> [vllm opts...]")

    os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")

    serve_argv = build_serve_argv(raw[0], raw[1:])
    sys.argv = [serve_argv[0], *serve_argv[2:]]
    args = make_arg_parser(
        FlexibleArgumentParser(description="minrl vLLM server")
    ).parse_args()
    validate_parsed_serve_args(args)
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    args.return_tokens_as_token_ids = True
    args.worker_extension_cls = "minrl.inference.weight_sync.WeightSyncWorkerExtension"

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
