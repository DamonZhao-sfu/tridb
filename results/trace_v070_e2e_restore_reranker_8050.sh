#!/bin/bash
# Restore the TRACE cross-encoder server (stopped 2026-11-05 to free GPU1 for the AdaMEM second embedding server).
systemd-run --user --unit=trace-qwen3-reranker-8050 --slice=trimem-eval.slice --setenv=HF_HOME=/localhome/hza214/hf-cache \
  --setenv=HF_HUB_OFFLINE=1 --working-directory=/localhome/hza214/tridb /usr/bin/bash /localhome/hza214/tridb/scripts/serve_qwen3_reranker.sh 8050 1
