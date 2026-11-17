#!/usr/bin/env bash
# One-shot environment setup for V100 (sm_70) machines.
#
#   scripts/setup_v100.sh                          # full setup, builds vLLM from source (~30 min, 12 cores)
#   VLLM_WHEEL=/path/vllm-0.33.0+cu129-*.whl scripts/setup_v100.sh   # reuse a saved wheel: ~5 min
#
# Why this exists: uv.lock pins torch 2.14 + vLLM 0.33 built for CUDA 13, and
# CUDA 13 dropped Volta. On a V100 the vLLM server dies with "no kernel image
# is available for execution on the device". The fix, done here:
#   1. torch 2.14.0 from the cu126 index (its wheels still ship sm_70 kernels)
#   2. NCCL >= 2.29.7 (vLLM 0.33 needs ncclCommSuspend; torch cu126 ships 2.29.3)
#   3. Python headers without sudo (Triton JIT-compiles a C helper at runtime)
#   4. vLLM 0.33.0 compiled for sm_70 against that torch with a CUDA 12.x nvcc;
#      its CMakeLists drops 7.0 for CUDA >= 12.8, so 7.0 is patched back in
# The built wheel is copied to $WHEEL_DIR so the next machine skips step 4.
#
# Afterwards `source scripts/v100_env.sh` before training: it sets UV_NO_SYNC=1
# so `uv run` (the default in examples/*/run.sh) does not re-sync the venv
# back to the CUDA 13 torch from uv.lock.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

TORCH_VERSION=2.14.0
VLLM_VERSION=0.33.0
NCCL_VERSION=2.30.7
PY_MINOR=3.12
WHEEL_DIR=${WHEEL_DIR:-$HOME/wheels}
BUILD_DIR=${BUILD_DIR:-$HOME/.cache/minrl-vllm-build}
MAX_JOBS=${MAX_JOBS:-$(nproc)}
# A wheel saved by an earlier run is picked up automatically.
if [[ -z "${VLLM_WHEEL:-}" ]]; then
    VLLM_WHEEL=$(ls -t "$WHEEL_DIR"/vllm-"$VLLM_VERSION"+cu12*.whl 2>/dev/null | head -1 || true)
fi

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 0. preflight ------------------------------------------------------------
log "preflight"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; no NVIDIA driver?"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
command -v gcc >/dev/null || die "gcc not found (install build-essential)"

if [[ -z "${VLLM_WHEEL:-}" ]]; then
    # nvcc must be CUDA 12.x: CUDA 13 cannot target sm_70 at all.
    CUDA_HOME=${CUDA_HOME:-}
    if [[ -z "$CUDA_HOME" ]]; then
        for d in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12; do
            [[ -x "$d/bin/nvcc" ]] && { CUDA_HOME=$d; break; }
        done
    fi
    [[ -n "$CUDA_HOME" && -x "$CUDA_HOME/bin/nvcc" ]] \
        || die "no CUDA 12.x nvcc under /usr/local/cuda-12*; set CUDA_HOME or pass VLLM_WHEEL"
    "$CUDA_HOME/bin/nvcc" --version | grep -q "release 12\." \
        || die "$CUDA_HOME/bin/nvcc is not CUDA 12.x; CUDA 13 cannot build sm_70"
    echo "nvcc: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*'))"
fi

# ---- 1. uv + project venv ---------------------------------------------------------
log "uv + project dependencies (uv sync)"
if ! command -v uv >/dev/null; then
    [[ -x "$HOME/.local/bin/uv" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
PY=$ROOT/.venv/bin/python
# A sync reverts torch/vLLM to uv.lock's CUDA 13 builds, so only on a fresh venv
# (or FORCE_SYNC=1, e.g. after a dependency change); steps 3-4 then redo the swap.
if [[ ! -x "$PY" || -n "${FORCE_SYNC:-}" ]]; then
    uv sync --all-extras
else
    echo ".venv exists; skipping uv sync (FORCE_SYNC=1 to force)"
fi
pip_install() { uv pip install --python "$PY" "$@"; }

# ---- 2. Python headers without sudo ---------------------------------------------------
log "Python $PY_MINOR headers"
uv python install "$PY_MINOR" >/dev/null
PY_INCLUDE=$(dirname "$(find "$HOME/.local/share/uv/python" -name Python.h -path "*python$PY_MINOR*" | head -1)")
[[ -f "$PY_INCLUDE/Python.h" ]] || die "could not find Python.h from uv's Python $PY_MINOR"
echo "headers: $PY_INCLUDE"
export C_INCLUDE_PATH=$PY_INCLUDE CPLUS_INCLUDE_PATH=$PY_INCLUDE

# ---- 3. torch cu126 (has sm_70) + new-enough NCCL ------------------------------------------
log "torch $TORCH_VERSION+cu126"
if "$PY" -c "import torch,sys; sys.exit(0 if 'sm_70' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
    echo "already has sm_70 kernels"
else
    pip_install --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
        "torch==$TORCH_VERSION" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi
log "nvidia-nccl-cu12 $NCCL_VERSION"
pip_install --no-deps "nvidia-nccl-cu12==$NCCL_VERSION"

# ---- 4. vLLM for sm_70 --------------------------------------------------------------------
log "vLLM $VLLM_VERSION for sm_70"
vllm_ok() { "$PY" - <<'EOF' 2>/dev/null
import importlib.metadata as m, sys
v = m.version("vllm")
sys.exit(0 if "+cu12" in v else 1)   # the PyPI wheel is CUDA 13 and reports a plain version
EOF
}
if vllm_ok; then
    echo "already installed: $("$PY" -c 'import importlib.metadata as m; print(m.version("vllm"))')"
elif [[ -n "${VLLM_WHEEL:-}" ]]; then
    pip_install --no-deps --reinstall-package vllm "$VLLM_WHEEL"
else
    pip_install cmake ninja "setuptools>=78.1.1,<81" setuptools_scm setuptools_rust wheel packaging jinja2 regex build
    mkdir -p "$BUILD_DIR"; SRC=$BUILD_DIR/vllm-$VLLM_VERSION
    if [[ ! -d "$SRC" ]]; then
        url=$(curl -s "https://pypi.org/pypi/vllm/$VLLM_VERSION/json" | "$PY" -c \
            'import json,sys; print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')
        curl -sL "$url" -o "$BUILD_DIR/vllm-$VLLM_VERSION.tar.gz"
        tar xzf "$BUILD_DIR/vllm-$VLLM_VERSION.tar.gz" -C "$BUILD_DIR"
    fi
    # vLLM lists 7.0 only for CUDA < 12.8; nvcc 12.8/12.9 can still target it.
    sed -i 's/set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")/set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")/' \
        "$SRC/CMakeLists.txt"
    grep -q '"7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0' "$SRC/CMakeLists.txt" \
        || die "CMakeLists.txt patch did not apply; vLLM's arch list changed"
    echo "building in $SRC with $MAX_JOBS jobs; log: $BUILD_DIR/build.log (resumable: rerun this script)"
    (
        cd "$SRC"
        export CUDA_HOME PATH="$CUDA_HOME/bin:$ROOT/.venv/bin:$PATH" TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS NVCC_THREADS=2 \
               VLLM_TARGET_DEVICE=cuda CMAKE_ARGS="-DPython_INCLUDE_DIR=$PY_INCLUDE"
        # A wheel, not a direct install, so it can be kept and reused.
        "$PY" -m build --wheel --no-isolation --skip-dependency-check --outdir "$BUILD_DIR/dist" . >"$BUILD_DIR/build.log" 2>&1
    ) || die "vLLM build failed; tail $BUILD_DIR/build.log"
    whl=$(ls -t "$BUILD_DIR"/dist/vllm-"$VLLM_VERSION"*.whl | head -1)
    pip_install --no-deps --reinstall-package vllm "$whl"
    mkdir -p "$WHEEL_DIR"; cp "$whl" "$WHEEL_DIR/"
    echo "saved $WHEEL_DIR/$(basename "$whl"); next time: VLLM_WHEEL=$WHEEL_DIR/$(basename "$whl") $0"
fi

# ---- 5. self-check ------------------------------------------------------------------------
log "self-check"
CHECK=$(mktemp -d)
cat >"$CHECK/check.py" <<'EOF'
import torch, triton, triton.language as tl
assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.randn(256, 256, device="cuda", dtype=torch.float16); (x @ x).sum().item()
print(f"torch {torch.__version__}: fp16 matmul on {torch.cuda.get_device_name()} ok")

@triton.jit
def double(X, Y, N: tl.constexpr):
    o = tl.arange(0, N); tl.store(Y + o, tl.load(X + o) * 2)
a = torch.randn(1024, device="cuda"); b = torch.empty_like(a); double[(1,)](a, b, 1024)
assert torch.allclose(b, a * 2); print(f"triton {triton.__version__}: kernel ok")

from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
lib = NCCLLibrary(); assert lib.has_symbol("ncclCommSuspend")
print("nccl: vLLM can load it (ncclCommSuspend present)")
import vllm, vllm._C_stable_libtorch  # noqa: F401  -- the compiled kernels, the part that broke
print(f"vllm {vllm.__version__}: compiled extension imports")
EOF
"$PY" -I "$CHECK/check.py" 2>&1 | grep -v -i deprecat
rm -rf "$CHECK"

log "done"
cat <<EOF
Before training, in every new shell:
    source scripts/v100_env.sh
then run examples as usual, e.g.
    examples/gsm8k/run.sh
EOF
