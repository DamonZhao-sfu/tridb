#!/bin/bash
# TRACE v0.7.0 per-dataset queue (user 2026-11-05, plan docs/superpowers/plans/2026-11-05-trace-space-v070.md, Amendment A).
# NOT STARTED by the implementer. Order ALFWorld -> ScienceWorld -> WebShop (override: DATASETS="...").
# Per dataset:
#   0. hash guard (no file hashed by a live / paused run changed)
#   1. prepare the shared init (fresh v0.7.0 space; per-configuration stratified 100-task TRAIN samples; NoMemory ledger);
#      completion = <run>/prepared.json (written last; an interrupted preparation is set aside and replayed)
#   2. init live: NoMemory TRAIN (100, c16) -> 5 Sobol -> 5 LLM (each on its own sample) -> test_deferred
#   3. readiness R1-R5 of the VAE checkpoint on the init space (stop unless R1>=.95, R2>=.90, R3=1, R4=0)
#   4. freeze-affine on the 10 init configurations (R6 logged)
#   5. prepare the latent arm (imports 10 init + NoMemory; staged pruning 20..100; <= 80 configs; 5,000 episodes;
#      latency cap at the same stage barriers: factor ALF 2.0 / Sci 3.0 / WebShop 2.0, NoMemory in the front)
#   6. arm live: TRAIN -> automatic TEST on (3-objective Pareto U BRANE fuzzy Pareto) U NoMemory
#   7. hash guard (fails closed)
# Episodes wait inside tools.trace_v070_live for the shared evaluation lease + idle GPUs (resource gate).
set -u
cd /localhome/hza214/tridb
PY=.venv-optimizer/bin/python
export PYTHONPATH=/localhome/hza214/tridb
Q=results/trace_v070_e2e_20261105; LOG=$Q/queue.log
VAE=results/trace_v070_seqvae_20261105
CK=${CK:-$VAE/seqvae_v2_d16.pt}          # Q7: d16 by default; CK=$VAE/seqvae_v2_d32.pt selects d32
TPL=results/trace_pr_three_dataset_prepare/template_v060.json
DATASETS=${DATASETS:-"alfworld sciworld webshop"}
st() { $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$1/status.json" 2>/dev/null; }
say() { echo "$(date -Is) $*" >> $LOG; }
guard() { $PY -m tools.trace_v070_hash_guard > $Q/hash_guard_$1.json 2>&1 || { say "hash guard FAILED ($1)"; exit 2; }; }
mkdir -p $Q
say "TRACE v0.7.0 queue start ($DATASETS; checkpoint $CK)"
# The live launcher refuses to run while train_trace_seq_vae runs; wait for the VAE job to finish.
while ! grep -qx all $VAE/train_done.txt 2>/dev/null || pgrep -f "[t]rain_trace_seq_vae" > /dev/null; do sleep 60; done
[ -f $CK ] || { say "checkpoint $CK missing"; exit 3; }
guard start
for ds in $DATASETS; do
  INIT=results/trace_v070_${ds}_init_20261105
  ARM=results/trace_v070_latent_${ds}_20261105
  AFF=results/trace_v070_${ds}_20261105_affine.json
  READY=$Q/readiness_${ds}.json
  DB=trace_ablation_sci_48d26f8d18eba20f; [ $ds = webshop ] && DB=trace_ablation_web_22467f614f8be112
  if [ ! -f $INIT/prepared.json ]; then
    say "$ds prepare shared init"
    $PY -m tools.prepare_trace_v070_campaign shared-init --dataset $ds --template $TPL --run-dir $INIT \
      --memory-db $DB --seed 20261022 >> $Q/${ds}_prep.log 2>&1 || { say "$ds shared-init prepare FAILED"; exit 4; }
  fi
  if [ "$(st $INIT)" != "test_deferred" ]; then
    say "$ds init live (NoMemory + 5 Sobol + 5 LLM)"
    $PY -m tools.trace_v070_live --run-dir $INIT >> $Q/${ds}_init.log 2>&1
    s=$(st $INIT); say "$ds init ended with $s"
    [ "$s" = "test_deferred" ] || { say "$ds init unexpected status $s; stop"; exit 5; }
  fi
  if [ ! -f $READY ]; then
    say "$ds readiness R1-R5 ($CK)"
    nice -n 5 $PY -m tools.latent_readiness_v070 run --space-run $INIT --checkpoints $CK --out $READY --affine-cache $Q \
      >> $Q/${ds}_prep.log 2>&1 || { say "$ds readiness FAILED"; exit 6; }
  fi
  $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['checkpoints'];sys.exit(0 if all(v and v['passes'] for v in r.values()) else 1)" \
    $READY || { say "$ds readiness gates not passed ($READY); stop"; exit 7; }
  if [ ! -f $AFF ]; then
    say "$ds freeze affine on $INIT"
    nice -n 5 $PY -m tools.latent_readiness_v070 freeze-affine --checkpoint $CK --shared-init $INIT --out $AFF \
      >> $Q/${ds}_prep.log 2>&1 || { say "$ds freeze-affine FAILED"; exit 8; }
  fi
  say "$ds R6: $($PY -c "import json,sys;r=json.load(open(sys.argv[1]+'.json'));print(r['R6_init_roundtrip'],'/',r['n_init'])" $AFF)"
  if [ ! -f $ARM/prepared.json ]; then
    say "$ds prepare latent arm"
    $PY -m tools.prepare_trace_v070_campaign arm --backend latent-mobo-v1 --shared-init $INIT --dataset $ds \
      --template $TPL --run-dir $ARM --memory-db $DB --checkpoint $CK --affine $AFF --kernel matern \
      --structure independent --panel stratified >> $Q/${ds}_prep.log 2>&1 || { say "$ds arm prepare FAILED"; exit 9; }
  fi
  say "$ds arm live (TRAIN with staged Pareto + latency-cap pruning -> TEST)"
  $PY -m tools.trace_v070_live --run-dir $ARM >> $Q/${ds}_arm.log 2>&1
  s=$(st $ARM); say "$ds arm ended with $s"
  [ "$s" = "test_complete" ] || { say "$ds arm unexpected status $s; stop"; exit 10; }
  guard $ds
  say "$ds done"
done
say "TRACE v0.7.0 queue done"
