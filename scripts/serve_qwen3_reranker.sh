#!/usr/bin/env bash
# Serve the ORIGINAL Qwen/Qwen3-Reranker-0.6B as a vLLM pooling (score / rerank) model for
# trimem.experiments.rank_impls.CrossEncoderRank (spec 2026-11-03 execution-engine extensions §2).
#
# Usage: scripts/serve_qwen3_reranker.sh [PORT] [GPU]
#   PORT  default 8050 (= rank_impls.CROSS_ENCODER_BASE)
#   GPU   default 0 (sets CUDA_VISIBLE_DEVICES)
# Env overrides:
#   VLLM_BIN        default /localhome/hza214/miniconda3/envs/vllm/bin/vllm (vLLM 0.15.0)
#   GPU_MEM_UTIL    default 0.05 (fraction of the GPU's TOTAL memory; must be free at start-up)
#   MAX_MODEL_LEN   default 8192 (a card is <= 12000 chars; query + card + template must fit)
#   MODEL           default Qwen/Qwen3-Reranker-0.6B (also the served model name)
#   CHAT_TEMPLATE   default scripts/qwen3_reranker.jinja (vLLM examples/pooling/score/template)
#
# The hf_overrides convert the causal-LM checkpoint into Qwen3ForSequenceClassification at load time
# (score = softmax over the "no" / "yes" logits = P(yes)); the chat template wraps each (query, document)
# pair in the official Qwen3-Reranker prompt. Without the template vLLM concatenates query + document raw.
#
# Endpoints: POST /v1/rerank {"model", "query", "documents": [...]}
#              -> {"results": [{"index", "relevance_score", ...}], "usage": {...}}
#            POST /score (pairwise).
# Readiness: GET /health returns 200 once the engine is up (no body); GET /v1/models lists MODEL.
#   until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do sleep 5; done
set -euo pipefail

PORT=${1:-8050}
GPU=${2:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_BIN=${VLLM_BIN:-/localhome/hza214/miniconda3/envs/vllm/bin/vllm}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.05}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MODEL=${MODEL:-Qwen/Qwen3-Reranker-0.6B}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$HERE/qwen3_reranker.jinja}

[[ -x $VLLM_BIN ]] || { echo "vllm not found: $VLLM_BIN" >&2; exit 1; }
[[ -f $CHAT_TEMPLATE ]] || { echo "chat template not found: $CHAT_TEMPLATE" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template "$CHAT_TEMPLATE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16
