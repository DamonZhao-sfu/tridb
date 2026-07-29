#!/usr/bin/env bash
set -euo pipefail

# Launch one component of the LongMemEval local-model stack. This script never
# terminates or replaces an existing service; it fails if the requested port is
# already listening.

component="${1:-}"
vllm_bin="${TRIDB_LME_VLLM_BIN:-vllm}"

case "$component" in
  answer)
    model="${TRIDB_LME_ANSWER_ARTIFACT:-Qwen/Qwen3-32B-FP8}"
    served_model="${TRIDB_LME_ANSWER_MODEL:-Qwen/Qwen3-32B}"
    port="${TRIDB_LME_ANSWER_PORT:-8000}"
    cuda_devices="${TRIDB_LME_ANSWER_GPU:-0}"
    gpu_util="${TRIDB_LME_ANSWER_GPU_UTIL:-0.90}"
    runner="generate"
    max_model_len="${TRIDB_LME_ANSWER_MAX_MODEL_LEN:-40960}"
    dtype="auto"
    extra_args=(--generation-config vllm)
    ;;
  embedding)
    model="${TRIDB_LME_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
    served_model="$model"
    port="${TRIDB_LME_EMBEDDING_PORT:-8001}"
    cuda_devices="${TRIDB_LME_EMBEDDING_GPU:-1}"
    gpu_util="${TRIDB_LME_EMBEDDING_GPU_UTIL:-0.20}"
    runner="pooling"
    max_model_len="${TRIDB_LME_EMBEDDING_MAX_MODEL_LEN:-32768}"
    dtype="bfloat16"
    extra_args=()
    ;;
  *)
    echo "usage: $0 answer|embedding" >&2
    exit 2
    ;;
esac

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
  echo "port $port is already in use; refusing to replace the existing service" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$cuda_devices"
exec "$vllm_bin" serve "$model" \
  --served-model-name "$served_model" \
  --runner "$runner" \
  --port "$port" \
  --tensor-parallel-size 1 \
  --dtype "$dtype" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization "$gpu_util" \
  "${extra_args[@]}"
