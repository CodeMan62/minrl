"""minrl vLLM server: OpenAI-compatible API with weight synchronization.
Usage::

    vllm_server Qwen/Qwen3-0.6B --port 8000
"""

from __future__ import annotations

import os
import sys
import signal
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
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext
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
    def signal_handler(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    app = build_app(args)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}


    await init_app_state(engine, app.state, args)
    shutdown_task = await serve_http(
        app,
        sock=sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
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
    args.worker_extension_cls = "minrl.inference.weight_sync.WeightSyncWorkerExtension"

    print(f"minrl: serving {args.model} on {args.host}:{args.port}", flush=True)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
