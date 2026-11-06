#!/bin/bash
# User 2026-11-05: when TRACE-latent v0.7.0 is ready, pause AdaMEM and run TRACE-latent first.
# Stage 1: let the nearly finished AdaMEM ALF TEST complete, stop AdaMEM (stop, not freeze), clean leftover envs,
# restore the TRACE cross-encoder server (8050), then run the NoMemory TRAIN baseline of the ALF init as a live check.
cd /localhome/hza214/tridb
Q=results/trace_v070_e2e_20261105; LOG=$Q/queue.log; say() { echo "$(date -Is) $*" >> $LOG; }
D=results/memp_adamem_c16_fixed_20261103
st() { python3 -c "import json;print(json.load(open('$D/status.json'))['stage'])"; }
until [ $(ls $D/adamem/alfworld/test/episodes | wc -l) -ge 200 ] || [ "$(st)" != "c16-adamem-alfworld-test" ]; do sleep 20; done
say "AdaMEM PAUSE (user): ALF TEST $(ls $D/adamem/alfworld/test/episodes | wc -l)/200, stage $(st); stopping units (stop, not freeze)"
for u in $(systemctl --user list-units 'memp-adamem-c16-adamem-*' --state=active --no-legend | awk '{print $1}') memp-adamem-c16-webshop-fix.service; do
  systemctl --user stop "$u"; say "stopped $u"; done
sleep 5
for p in 36001 36011 36002 36012 36003 36013; do
  python3 - "$p" >> $LOG 2>&1 <<'PY'
import json, sys, urllib.request
p = sys.argv[1]; base = f"http://127.0.0.1:{p}"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    s = json.load(op.open(base + "/managed_stats", timeout=10))
except Exception as e:
    print(p, "no stats", e); sys.exit(0)
field = "env_idx" if s.get("env") == "webshop" else "id"
for i in s.get("idle_s", {}):
    r = op.open(urllib.request.Request(base + "/close", json.dumps({field: int(i)}).encode(), {"Content-Type": "application/json"}), timeout=30).read()
    print(p, "closed", i, r)
print(p, "live_envs", s.get("live_envs"))
PY
done
systemctl --user is-active -q trace-qwen3-reranker-8050 || bash results/trace_v070_e2e_restore_reranker_8050.sh
for i in $(seq 1 60); do curl -s --noproxy '*' http://127.0.0.1:8050/v1/models | grep -q Reranker && break; sleep 10; done
curl -s --noproxy '*' http://127.0.0.1:8050/v1/models | grep -q Reranker && say "reranker 8050 restored" || { say "reranker 8050 NOT healthy; stop"; exit 2; }
say "ALF init NoMemory baseline-only live check start"
export PYTHONPATH=/localhome/hza214/tridb TRIMEM_ALFWORLD_MEMORY_DB=trace_ablation_sci_48d26f8d18eba20f TRIMEM_SCIWORLD_MEMORY_DB=trace_ablation_sci_48d26f8d18eba20f TRIMEM_WEBSHOP_MEMORY_DB=trace_ablation_web_22467f614f8be112
.venv-optimizer/bin/python -m tools.trace_v070_live --run-dir results/trace_v070_alfworld_init_20261105 --baseline-only >> $Q/alfworld_baseline_check.log 2>&1
say "ALF NoMemory baseline-only ended rc=$?"
