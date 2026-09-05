"""minrl vLLM server: OpenAI-compatible API with weight synchronization.
Usage::

    vllm_server Qwen/Qwen3-0.6B --port 8000
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from typing import List, Optional, Sequence

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client,
    init_app_state,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger("minrl.inference.server")

_TOKEN_ID_FLAG = "--return-tokens-as-token-ids"
_NO_TOKEN_ID_FLAG = "--no-return-tokens-as-token-ids"


def build_serve_argv(model: str, extra: Sequence[str] = ()) -> List[str]:
    """Build the ``vllm serve`` argv, injecting minrl's required flag."""
    extra = list(extra)
    argv = ["vllm", "serve", model]
    if _TOKEN_ID_FLAG not in extra and _NO_TOKEN_ID_FLAG not in extra:
        argv.append(_TOKEN_ID_FLAG)
    argv.extend(extra)
    return argv


async def run_server(args: Namespace) -> None:
    """Serve ``args.model`` with vLLM's OpenAI-compatible app."""
    listen_address, sock = setup_server(args, reuse_port=False)
    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        app = build_app(args, supported_tasks, engine_client.model_config)

        router = APIRouter()

        @router.get("/minrl/health")
        async def minrl_health(request: Request):
            try:
                await request.app.state.engine_client.check_health()
            except Exception as e:  # noqa: BLE001 - report as JSON
                return JSONResponse(
                    content={"status": "unhealthy", "error": str(e)},
                    status_code=503,
                )
            return {"status": "ok"}

        @router.get("/minrl/weight_version")
        async def minrl_weight_version(request: Request):
            version = await request.app.state.engine_client.get_weight_version()
            return {"weight_version": version}

        @router.post("/minrl/reset_prefix_cache")
        async def minrl_reset_prefix_cache(request: Request):
            success = await request.app.state.engine_client.reset_prefix_cache()
            return {"success": bool(success)}

        @router.post("/minrl/init_weight_sync")
        async def minrl_init_weight_sync(request: Request):
            body = await request.json()
            await request.app.state.engine_client.collective_rpc(
                "init_weight_sync",
                kwargs={
                    "master_address": body["master_address"],
                    "master_port": int(body["master_port"]),
                    "world_size": int(body["world_size"]),
                },
            )
            return {"status": "ok"}

        @router.post("/minrl/update_weights")
        async def minrl_update_weights(request: Request):
            body = await request.json()
            await request.app.state.engine_client.collective_rpc(
                "update_weights",
                kwargs={
                    "name": body["name"],
                    "dtype": body["dtype"],
                    "shape": body["shape"],
                },
            )
            return {"status": "ok"}

        app.include_router(router)

        await init_app_state(engine_client, app.state, args, supported_tasks)
        logger.info("minrl: starting server on %s", listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
        )
    await shutdown_task
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
    args.return_tokens_as_token_ids = True
    args.worker_extension_cls = "minrl.inference.worker_extension.WeightSyncWorkerExtension"

    print(f"minrl: serving {args.model} on {args.host}:{args.port}", flush=True)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
