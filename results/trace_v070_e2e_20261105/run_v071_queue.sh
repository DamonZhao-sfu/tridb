#!/bin/bash
# TRACE v0.7.1 queue (user 2026-11-05). NOT STARTED by the implementer. run_v070_queue.sh stays as it is.
# Start only after the hold unit stopped trace-v070-e2e-20261105 ("queue HELD after ALF init" in queue.log).
#
# ALFWorld (mid-run switch): skip the completed init (results/trace_v070_alfworld_init_20261105, old SemFilter; status
#   test_deferred required) -> readiness R1-R5 -> freeze-affine -> prepare the arm with --old-protocol-init (recorded
#   exemption; SemFilter v2c from the first exploration round; stall rule) -> arm live (TRAIN -> TEST).
# ScienceWorld, WebShop: v0.7.1 from their init: prepare init (v2c) -> init live (NoMemory + 5 Sobol + 5 LLM) -> readiness
#   -> freeze-affine -> prepare arm (v2c, stall rule) -> arm live (TRAIN -> TEST).
# Exploration: q 4, 5,000 episodes, <= 80 configs, staged 20..100, latency cap in the barrier, stall rule patience 3.
# Episodes wait inside the launcher for the shared evaluation lease + idle GPUs. The hash guard fails closed.
set -u
cd /localhome/hza214/tridb
PY=.venv-optimizer/bin/python
export PYTHONPATH=/localhome/hza214/tridb
Q=results/trace_v070_e2e_20261105; LOG=$Q/queue_v071.log
CK=${CK:-results/trace_v070_seqvae_20261105/seqvae_v2_d16.pt}
TPL=results/trace_pr_three_dataset_prepare/template_v060.json
DATASETS=${DATASETS:-"alfworld sciworld webshop"}
st() { $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$1/status.json" 2>/dev/null; }
say() { echo "$(date -Is) $*" >> $LOG; }
guard() { $PY -m tools.trace_v071_hash_guard > $Q/hash_guard_v071_$1.json 2>&1 || { say "hash guard FAILED ($1)"; exit 2; }; }
say "TRACE v0.7.1 queue start ($DATASETS; checkpoint $CK)"
systemctl --user is-active -q trace-v070-e2e-20261105 && { say "the v0.7.0 queue is still active; stop"; exit 3; }
[ -f $CK ] || { say "checkpoint $CK missing"; exit 3; }
guard start
for ds in $DATASETS; do
  DB=trace_ablation_sci_48d26f8d18eba20f; [ $ds = webshop ] && DB=trace_ablation_web_22467f614f8be112
  SUF=; [ $ds = alfworld ] && [ "${ALF_RESTART:-0}" = 1 ] && SUF=_restart  # never reuses the mid-run arm's files
  ARM=results/trace_v071_latent_${ds}${SUF}_20261105
  AFF=results/trace_v071_${ds}${SUF}_20261105_affine.json
  READY=$Q/readiness_v071_${ds}${SUF}.json
  if [ $ds = alfworld ] && [ "${ALF_RESTART:-0}" != 1 ]; then  # ALF_RESTART=1: a fresh ALF init on v0.7.1 instead
    INIT=results/trace_v070_alfworld_init_20261105
    [ "$(st $INIT)" = "test_deferred" ] || { say "ALF init status $(st $INIT), expected test_deferred; stop"; exit 4; }
    EXEMPT=--old-protocol-init
    say "alfworld init complete (old SemFilter); skipped"
  else
    INIT=results/trace_v071_${ds}${SUF}_init_20261105
    EXEMPT=
    if [ ! -f $INIT/prepared.json ]; then
      say "$ds prepare shared init (v0.7.1)"
      $PY -m tools.prepare_trace_v071_campaign shared-init --dataset $ds --template $TPL --run-dir $INIT \
        --memory-db $DB --seed 20261022 >> $Q/${ds}_v071_prep.log 2>&1 || { say "$ds shared-init prepare FAILED"; exit 5; }
    fi
    if [ "$(st $INIT)" != "test_deferred" ]; then
      say "$ds init live (NoMemory + 5 Sobol + 5 LLM)"
      $PY -m tools.trace_v071_live --run-dir $INIT >> $Q/${ds}_v071_init.log 2>&1
      s=$(st $INIT); say "$ds init ended with $s"
      [ "$s" = "test_deferred" ] || { say "$ds init unexpected status $s; stop"; exit 6; }
    fi
  fi
  if [ ! -f $READY ]; then
    say "$ds readiness R1-R5 ($CK)"
    nice -n 5 $PY -m tools.latent_readiness_v070 run --space-run $INIT --checkpoints $CK --out $READY --affine-cache $Q \
      >> $Q/${ds}_v071_prep.log 2>&1 || { say "$ds readiness FAILED"; exit 7; }
  fi
  $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['checkpoints'];sys.exit(0 if all(v and v['passes'] for v in r.values()) else 1)" \
    $READY || { say "$ds readiness gates not passed ($READY); stop"; exit 8; }
  if [ ! -f $AFF ]; then
    say "$ds freeze affine on $INIT"
    nice -n 5 $PY -m tools.latent_readiness_v070 freeze-affine --checkpoint $CK --shared-init $INIT --out $AFF \
      >> $Q/${ds}_v071_prep.log 2>&1 || { say "$ds freeze-affine FAILED"; exit 9; }
  fi
  say "$ds R6: $($PY -c "import json,sys;r=json.load(open(sys.argv[1]+'.json'));print(r['R6_init_roundtrip'],'/',r['n_init'])" $AFF)"
  if [ ! -f $ARM/prepared.json ]; then
    say "$ds prepare latent arm (v0.7.1, q 4, stall rule ${EXEMPT:+, mid-run exemption})"
    $PY -m tools.prepare_trace_v071_campaign arm --backend latent-mobo-v1 --shared-init $INIT --dataset $ds \
      --template $TPL --run-dir $ARM --memory-db $DB --checkpoint $CK --affine $AFF --kernel matern \
      --structure independent --panel stratified $EXEMPT >> $Q/${ds}_v071_prep.log 2>&1 \
      || { say "$ds arm prepare FAILED"; exit 10; }
  fi
  say "$ds arm live (TRAIN -> TEST)"
  $PY -m tools.trace_v071_live --run-dir $ARM >> $Q/${ds}_v071_arm.log 2>&1
  s=$(st $ARM); say "$ds arm ended with $s"
  [ "$s" = "test_complete" ] || { say "$ds arm unexpected status $s; stop"; exit 11; }
  guard $ds
  say "$ds done"
done
say "TRACE v0.7.1 queue done"
