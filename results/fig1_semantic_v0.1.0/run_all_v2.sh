#!/usr/bin/env bash
# Figure 1 EvoTrace experiments (v2, after adversarial review fixes). Sequential on purpose.
set -u
cd /localhome/hza214/tridb
DSN="postgresql://$USER@127.0.0.1:55440/fig1?gssencmode=disable"
SLICE="--tasks ale:ahc008,ale:ahc015,math:circle_packing,math:heilbronn_triangle --max-sessions-per-task 3 --max-nodes-per-session 60"
R=results/fig1_semantic_v0.1.0
.venv/bin/python -m bench.agent_memory.fig1.run --dsn "$DSN" $SLICE --llm real --c-llm 0.085 \
  --tag realv2 --replay-check --trace 40 --progress 200 --out $R/real_v2 > $R/real_v2/run.log 2>&1
.venv/bin/python -m bench.agent_memory.fig1.run --dsn "$DSN" $SLICE --llm real --c-llm 0.085 \
  --tag realv2s --shared-cache --replay-check --progress 200 --out $R/real_v2_shared > $R/real_v2_shared/run.log 2>&1
.venv/bin/python -m bench.agent_memory.fig1.run --dsn "$DSN" --llm mock --c-llm 0.085 \
  --tag mockfull --replay-check --progress 1000 --out $R/mock_full > $R/mock_full/run.log 2>&1
.venv/bin/python -m bench.agent_memory.fig1.report $R/real_v2 $R/real_v2_shared $R/mock_full > /dev/null
echo ALL_DONE
