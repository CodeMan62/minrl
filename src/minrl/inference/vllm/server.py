# this thing is still work in progress my idea is something else someday i will find time and implement it
from __future__ import annotations

import sys
from typing import List, Optional, Sequence

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    serve_argv = build_serve_argv(args[0], args[1:])
    try:
        from vllm.entrypoints.cli.main import main as vllm_main
    except ImportError:
        sys.stderr.write(
            "vLLM is not installed. Install the extra, then retry:\n"
            "  uv sync --extra vllm\n"
            f"  uv run {_USAGE}\n"
        )
        sys.exit(1)

    print(f"minrl: {' '.join(serve_argv)}", flush=True)
    sys.argv = serve_argv
    vllm_main()


if __name__ == "__main__":
    main()
