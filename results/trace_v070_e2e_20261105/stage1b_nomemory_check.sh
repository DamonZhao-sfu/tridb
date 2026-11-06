#!/bin/bash
# Rerun of the ALF NoMemory baseline-only live check after the run_dir canonicalization fix (2026-11-05).
cd /localhome/hza214/tridb
Q=results/trace_v070_e2e_20261105; LOG=$Q/queue.log; say() { echo "$(date -Is) $*" >> $LOG; }
export PYTHONPATH=/localhome/hza214/tridb TRIMEM_ALFWORLD_MEMORY_DB=trace_ablation_sci_48d26f8d18eba20f TRIMEM_SCIWORLD_MEMORY_DB=trace_ablation_sci_48d26f8d18eba20f TRIMEM_WEBSHOP_MEMORY_DB=trace_ablation_web_22467f614f8be112
say "ALF init NoMemory baseline-only live check start (after path fix)"
.venv-optimizer/bin/python -m tools.trace_v070_live --run-dir results/trace_v070_alfworld_init_20261105 --baseline-only >> $Q/alfworld_baseline_check.log 2>&1
say "ALF NoMemory baseline-only ended rc=$?"
