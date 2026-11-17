# Source this before running examples on a machine set up by scripts/setup_v100.sh:
#     source scripts/v100_env.sh
export PATH="$HOME/.local/bin:$PATH"
# `uv run` would re-sync .venv to uv.lock and put the CUDA 13 torch back (no V100 kernels).
export UV_NO_SYNC=1
# Triton compiles a small C helper at runtime and needs Python.h; uv's Python ships it.
_minrl_inc=$(dirname "$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)")
if [[ -f "$_minrl_inc/Python.h" ]]; then
    export C_INCLUDE_PATH="$_minrl_inc${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
else
    echo "v100_env.sh: Python.h not found; run scripts/setup_v100.sh first" >&2
fi
unset _minrl_inc
