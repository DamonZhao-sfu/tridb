#!/bin/bash
# User 2026-11-05 "没问题 请启动": after stage 1 (AdaMEM paused, reranker restored, ALF NoMemory baseline run),
# check the NoMemory receipts, then start the TRACE v0.7.0 queue (ALF -> Sci -> WebShop).
cd /localhome/hza214/tridb
Q=results/trace_v070_e2e_20261105; LOG=$Q/queue.log; I=results/trace_v070_alfworld_init_20261105
say() { echo "$(date -Is) $*" >> $LOG; }
while systemctl --user is-active -q trace-v070-stage1-20261105; do sleep 30; done
grep -Eq '^[0-9T:+-]+ ALF NoMemory baseline-only ended rc=0$' $LOG || { say "stage2: baseline-only did not end rc=0; queue NOT started"; exit 1; }
python3 - "$I" >> $LOG 2>&1 <<'PY' || { echo "$(date -Is) stage2: NoMemory receipt check FAILED; queue NOT started" >> results/trace_v070_e2e_20261105/queue.log; exit 2; }
import json, sqlite3, statistics as st, sys
run = sys.argv[1]
db = sqlite3.connect(f"file:{run}/baselines/nomemory/ledger.sqlite?mode=ro", uri=True)
states = dict(db.execute("SELECT state, count(*) FROM episodes GROUP BY state"))
rs = [json.loads(r) for (r,) in db.execute("SELECT receipt FROM episodes WHERE state='done'")]
assert states == {"done": 100}, states
assert all(r["retrieval_tokens"] == 0 and r["total_tokens"] > 0 and 0 < r["latency_seconds"] < 3600 for r in rs)
print(json.dumps({"stage2_nomemory_check": "ok", "n": len(rs), "sr": st.mean(r["raw_score"] >= 1 for r in rs),
                  "tokens": st.mean(r["total_tokens"] for r in rs), "latency": st.mean(r["latency_seconds"] for r in rs),
                  "steps": st.mean(r["agent_steps"] for r in rs)}))
PY
say "stage2: NoMemory check ok; starting TRACE v0.7.0 queue"
systemd-run --user --unit=trace-v070-e2e-20261105 --slice=trimem-eval.slice --collect \
  --setenv=PATH=/local-scratch/localhome/hza214/local/node-v22.14.0-linux-x64/bin:/usr/local/bin:/usr/bin:/bin \
  -p WorkingDirectory=/localhome/hza214/tridb \
  -p StandardOutput=append:/localhome/hza214/tridb/$Q/service.log -p StandardError=append:/localhome/hza214/tridb/$Q/service.log \
  /localhome/hza214/tridb/$Q/run_v070_queue.sh >> $LOG 2>&1
