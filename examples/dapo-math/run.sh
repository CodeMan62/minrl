#!/usr/bin/env bash
# Qwen3.5-0.8B on DAPO-Math-17k with GRPO, two GPUs, fully asynchronous:
# GPU 0 serves the policy with vLLM and never stops generating; GPU 1 trains
# on the groups as they finish and pushes new weights into vLLM after every
# optimizer step. Rollouts in flight during a push continue under the new
# weights, which is what --tis corrects for. Eval is MATH-500.
#
#   examples/dapo-math/run.sh                       # train
#   examples/dapo-math/run.sh --eval-only           # score the served policy
#   examples/dapo-math/run.sh --no-wandb --iterations 5 --eval-samples 20   # smoke test
#
# Anything appended is passed to train.py and overrides the arrays below.
#
# Environment knobs:
#   ROLLOUT_GPU=0  TRAIN_GPUS=1 ("1,2,3" = torchrun + FSDP2)  VLLM_PORT=8000
#   WEIGHT_SYNC_PORT=29501  SKIP_SERVER=1 (attach to a running server)
#   PYTHON="uv run python"  LOG_DIR=logs/dapo-math
set -euo pipefail
cd "$(dirname "$0")/../.."

# Use the uv-managed project environment by default; PYTHON overrides it.
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=($PYTHON)
elif [[ -f uv.lock ]] && command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
else
    PYTHON_CMD=(python)
fi
LOG_DIR=${LOG_DIR:-logs/dapo-math}
ROLLOUT_GPU=${ROLLOUT_GPU:-0}
TRAIN_GPUS=${TRAIN_GPUS:-1}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
WEIGHT_SYNC_PORT=${WEIGHT_SYNC_PORT:-29501}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}

# ---- model: Qwen3.5-0.8B, a hybrid Gated-DeltaNet / attention model with a
# 248k vocabulary, shipped as a vision-language checkpoint, thinking on by default.
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}

MODEL_ARGS=(
    --model "$MODEL"
    # bf16 weights stepped directly: with lr ~1e-6 many updates round to zero.
    # --dtype fp32 is the safe choice when the training GPU has the memory
    # (see examples/README.md, "precision").
    --dtype bf16
    --activation-checkpointing
    # Thinking on: the model reasons in a <think> block before "Answer:".
    # Set false for a first, much cheaper run without reasoning.
    --template-kwargs '{"enable_thinking": true}'
)

VLLM_ARGS=(
    --dtype bfloat16
    --max-model-len 8192           # prompt (~200 tokens) + --max-new-tokens, with headroom
    --max-num-seqs 256             # concurrency * group_size in flight, plus eval
    --gpu-memory-utilization 0.85
    # Logprobs of the distribution actually sampled from, so --temperature != 1
    # still gives correct importance ratios. Harmless at temperature 1.
    --logprobs-mode processed_logprobs
    # Text only. Uncomment if the server spends startup memory profiling images:
    # --limit-mm-per-prompt '{"image": 0, "video": 0}'
)

# ---- experiment
ENV_ARGS=(
    --env dapo_math
    --env-arg dataset=open-r1/DAPO-Math-17k-Processed   # the DAPO data, deduplicated
    --env-arg config=en
    --env-arg eval_dataset=HuggingFaceH4/MATH-500
    --train-split train
    --eval-split test
)

ROLLOUT_ARGS=(
    --group-size 16            # answers per problem; advantages are relative within these
    --groups-per-step 4        # 4 problems x 16 answers = 64 sequences per optimizer step
    --concurrency 8            # groups generating at once: 128 sequences in flight on vLLM
    --max-queued 8             # finished groups buffered ahead of the trainer (~2 steps)
    --max-new-tokens 4096      # thinking + answer; watch eval/truncated
    --temperature 1.0
    --top-p 1.0
)

ALGO_ARGS=(
    --algorithm grpo
    --clip-eps 0.2
    --kl-coef 0.0              # DAPO-style: no KL, no reference model
    --ent-coef 0.0
    --tis                      # truncated importance sampling: the rollouts are off-policy
)

OPTIMIZER_ARGS=(
    --lr 1e-6
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --max-grad-norm 1.0
)

TRAIN_ARGS=(
    --iterations 300
    # Logits are [micro_batch, 4096+, 248k] in fp32: ~4 GB per sequence per
    # copy, and the loss holds a few. 1 fits a 24 GB card; 2-4 need 80 GB.
    --micro-batch-size 1
    --seed 0
)

EVAL_ARGS=(
    --eval-every 25
    --eval-samples 500         # all of MATH-500
    --eval-concurrency 64
    --eval-temperature 0.6     # thinking models loop at temperature 0
    --eval-max-new-tokens 4096
    --eval-dump "$LOG_DIR/eval-dumps"
)

CKPT_ARGS=(
    --ckpt-dir ckpts/dapo-math
    --ckpt-every 10
    --keep-last 2
    --resume auto              # a fresh run ignores it; a restart continues
)

WANDB_ARGS=(
    --wandb                    # swap for --no-wandb to log nothing
    --wandb-project minrl-dapo-math
    --wandb-run-name "qwen3.5-0.8b-grpo-tis"
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
    CUDA_VISIBLE_DEVICES=$ROLLOUT_GPU "${PYTHON_CMD[@]}" -m minrl.inference.vllm_server "$MODEL" "${VLLM_ARGS[@]}" \
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
LAUNCHER=("${PYTHON_CMD[@]}")
if (( NGPU > 1 )); then
    LAUNCHER=("${PYTHON_CMD[@]}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU")
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
