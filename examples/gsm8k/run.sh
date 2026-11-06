#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct on GSM8K with GRPO: one answer per episode, scored by
# an exact numeric match. GPU 0 serves with vLLM, GPU 1 trains.
#
#   examples/gsm8k/run.sh
#   examples/gsm8k/run.sh --eval-only
#   TRAIN_GPUS=1,2,3 examples/gsm8k/run.sh          # FSDP2 on three training GPUs
#
# Anything appended is passed to train.py and overrides the arrays below.
#
# Environment knobs:
#   ROLLOUT_GPU=0  TRAIN_GPUS=1  VLLM_PORT=8000  WEIGHT_SYNC_PORT=29501
#   SKIP_SERVER=1 (attach to a running server)  PYTHON=python  LOG_DIR=logs/gsm8k
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
LOG_DIR=${LOG_DIR:-logs/gsm8k}
ROLLOUT_GPU=${ROLLOUT_GPU:-0}
TRAIN_GPUS=${TRAIN_GPUS:-1}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
WEIGHT_SYNC_PORT=${WEIGHT_SYNC_PORT:-29501}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}

# ---- model: a plain dense model, no thinking mode
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

MODEL_ARGS=(
    --model "$MODEL"
    --dtype bf16
)

VLLM_ARGS=(
    --dtype bfloat16
    --max-model-len 4096
    --gpu-memory-utilization 0.7
    --logprobs-mode processed_logprobs
)

# ---- experiment
ENV_ARGS=(
    --env gsm8k
    --train-split train
    --eval-split test
)

ROLLOUT_ARGS=(
    --group-size 8
    --groups-per-step 1
    --concurrency 2
    --max-new-tokens 1024
    --temperature 1.0
)

ALGO_ARGS=(
    --algorithm grpo
    --clip-eps 0.2
    --kl-coef 0.0
    --ent-coef 0.0
)

OPTIMIZER_ARGS=(
    --lr 1e-6
    --adam-eps 1e-4
    --max-grad-norm 1.0
)

TRAIN_ARGS=(
    --iterations 100
    --micro-batch-size 2
    --seed 0
)

EVAL_ARGS=(
    --eval-every 20
    --eval-samples 200
    --eval-concurrency 16
    --eval-temperature 0.0
)

CKPT_ARGS=(
    --ckpt-dir ckpts/gsm8k
    --ckpt-every 0             # only the final checkpoint
)

WANDB_ARGS=(
    --no-wandb
    --wandb-project minrl-gsm8k
)

# ---- launch: vLLM on the rollout GPU, wait for it, train, stop the server
mkdir -p "$LOG_DIR"
SERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "stopping vLLM (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
if [[ -z "${SKIP_SERVER:-}" ]]; then
    echo "starting vLLM on GPU $ROLLOUT_GPU, log: $LOG_DIR/vllm_server.log"
    CUDA_VISIBLE_DEVICES=$ROLLOUT_GPU $PYTHON -m minrl.inference.vllm_server "$MODEL" "${VLLM_ARGS[@]}" \
        --host "$VLLM_HOST" --port "$VLLM_PORT" >"$LOG_DIR/vllm_server.log" 2>&1 &
    SERVER_PID=$!
    trap stop_server EXIT INT TERM
fi

HEALTH="http://$VLLM_HOST:$VLLM_PORT/minrl/health"
deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
until curl -sf "$HEALTH" >/dev/null 2>&1; do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM server exited before it was ready; see $LOG_DIR/vllm_server.log" >&2; exit 1
    fi
    if (( $(date +%s) >= deadline )); then
        echo "vLLM server not ready after ${SERVER_TIMEOUT}s; see $LOG_DIR/vllm_server.log" >&2; exit 1
    fi
    sleep 5
done
echo "vLLM ready: $(curl -s "$HEALTH")"

NGPU=$(tr ',' '\n' <<<"$TRAIN_GPUS" | grep -c .)
LAUNCHER=("$PYTHON")
if (( NGPU > 1 )); then
    LAUNCHER=(torchrun --standalone --nproc_per_node="$NGPU")
fi
echo "training on GPU(s) $TRAIN_GPUS, log: $LOG_DIR/train.log"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS "${LAUNCHER[@]}" train.py \
    --vllm-url "http://$VLLM_HOST:$VLLM_PORT" \
    --weight-transfer-host "$VLLM_HOST" \
    --weight-transfer-port "$WEIGHT_SYNC_PORT" \
    "${MODEL_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_DIR/train.log"
