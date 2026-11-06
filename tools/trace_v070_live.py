"""Live launcher of one TRACE v0.7.0 run (shared init or latent arm; plan 2026-11-05 section 7).

Refuses any manifest that is not v0.7.0 (a v0.6.0 run stays on its own launcher). Sequence (as
``tools.trace_joint_live.main`` / ``tools.trace_latent_live``, which are imported read-only and never patched
here with ``backend_patches``):

1. validations (v0.7.0 blocks + the stock checks on the 100-task view), ``verify_artifacts``,
   ``service_preflight``, ``check_no_vae_training`` (not for ``--preflight-only``: it launches nothing and
   reports ``vae_training_running`` instead);
2. own ``queue.lock`` (exclusive) + the shared ``results/pool_evaluation_queue.lock`` lease + the idle-GPU /
   free-memory resource gate (three stable checks);
3. inside ``v070_patches``: the NoMemory TRAIN stage (missing episodes only; c16; same runtime as TRACE
   episodes) -> ``--baseline-only`` stops here; then the round loop (recover claims, ``run_round`` with the stock
   evaluator, ``samples.json``) until ``budget_complete``; then TEST (``trace_v070_holdout``: Pareto U NoMemory;
   the shared init returns ``test_deferred``).

    python -m tools.trace_v070_live --run-dir R [--preflight-only | --baseline-only]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from tools import trace_joint_live as live
from tools import trace_latent_live as tll
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_baseline import (
    baseline_entry,
    run_baseline,
    summarize,
)
from trimem.experiments.trace_v070_ledger import SampledLedger, write_samples_json
from trimem.experiments.trace_v070_runtime import is_v070, v070_patches


def _status(run: Path, status: str, **extra) -> None:
    driver.atomic_json(run / "status.json", {"status": status, "at": time.time(), **extra})


def _counts(run: Path, m: dict) -> dict:
    with SampledLedger(run / "ledger.sqlite", m) as led:
        return {"complete_configs": len(led.complete_configs()), "pending_episodes": len(led.pending()),
                "reserved_configs": led.db.execute("SELECT count(*) FROM configs").fetchone()[0]}


def vae_training_running() -> bool:
    return subprocess.run(["pgrep", "-f", "[t]rain_trace_seq_vae"], check=False, capture_output=True).returncode == 0


def load(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text())
    if not is_v070(m):
        raise SystemExit(f"{run}: not a TRACE v0.7.0 manifest; refusing (use the run's own launcher)")
    if not m.get("smoke"):
        from trimem.experiments.trace_v070_marker import check_prepared

        check_prepared(run)  # fix round 1: never launch an interrupted preparation
    return m


def preflight(run: Path, m: dict) -> dict:
    with v070_patches(m):
        driver.validate_manifest(m)
        live.verify_artifacts(m)
        live.service_preflight(m)
        counts = _counts(run, m)
    from trimem.experiments.trace_v070_baseline import load_baseline, parity_violations

    nm = baseline_entry(run, "TRAIN")
    bad = parity_violations(m, load_baseline(run, "TRAIN")) if nm is not None else ["NoMemory ledger missing"]
    if bad:
        raise SystemExit(f"{run}: NoMemory runtime parity violated: {bad}")
    return {"status": "passed", "nomemory_parity": "ok", "run_dir": str(run), "dataset": m["fixed"]["dataset"],
            "search_backend": m["search_backend"], "resources": live.resource_snapshot(),
            "vae_training_running": vae_training_running(), **counts,
            "nomemory_train": None if nm is None else {**summarize(nm), "tasks": len(nm["tasks"])}}


def baseline_stage(run: Path, m: dict) -> dict:
    from tools.trace_v070_nomemory import evaluate_nomemory, recover_nomemory_claims
    from trimem.experiments.trace_v070_baseline import load_baseline, parity_violations

    nm = baseline_entry(run, "TRAIN")
    if nm is not None:  # fix round 2: parity also when the ledger is already complete (or imported)
        bad = parity_violations(m, load_baseline(run, "TRAIN"))
        if bad:
            raise ValueError(f"NoMemory runtime differs from the TRACE episodes: {bad}")
    if nm is not None and nm["complete"]:
        return summarize(nm)
    if m["baselines"]["NoMemory"]["source"] == "imported":
        raise RuntimeError("imported NoMemory TRAIN ledger is incomplete; re-prepare the arm")
    _status(run, "nomemory_train", episodes_done=0 if nm is None else len(nm["receipts"]))
    live.verify_artifacts(m)
    live.verify_memory(m)
    live.service_preflight(m)
    return run_baseline(run, m, "TRAIN", evaluate_nomemory, recover=recover_nomemory_claims)


def backfill_pruning_log(run: Path, m: dict) -> int:
    """Append every stage decision committed in the ledger but missing from pruning_log.jsonl."""
    from trimem.experiments.trace_v070_pruning import backfill_from_ledger

    return backfill_from_ledger(run, m) if m.get("pruning") else 0


def round_loop(run: Path, m: dict) -> dict:
    from tools import trace_joint_protocol as proto

    while True:
        live.verify_artifacts(m)
        live.verify_memory(m)
        live.service_preflight(m)
        live.recover_claims(run, m)
        backfill_pruning_log(run, m)  # fix round 2: committed stage decisions reach the log before advancing
        result = driver.run_round(run, evaluator=live.evaluate)
        write_samples_json(run, m)
        counts = _counts(run, m)
        _status(run, result["status"], **counts)
        if result["status"] == "budget_complete":
            _status(run, "testing_train_frontier", **counts)
            test = proto.run_holdout(run, evaluator=live.evaluate)  # patched: run_pruned_holdout_v070
            _status(run, test["status"], train_configs=counts["complete_configs"],
                    test_configs=test.get("configurations", 0), test_episodes=test.get("episodes", 0),
                    baselines_tested=test.get("baselines_tested", []))
            return test
        if result["status"] == "pending":
            raise RuntimeError("Episodes still pending after bounded retries; inspect, then use the retry action")


def run(run_dir: Path, *, baseline_only: bool = False, sleep=time.sleep) -> dict:
    m = load(run_dir)
    tll.check_no_vae_training()
    with v070_patches(m):
        driver.validate_manifest(m)
    live.verify_artifacts(m)
    live.service_preflight(m)
    own = (run_dir / "queue.lock").open("a")
    try:  # fix round 1: a launcher that does not own the run never writes its status
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        own.close()
        raise SystemExit(f"{run_dir}: another launcher owns queue.lock; status left untouched")
    with own:  # fix round 2: the failure status is written while queue.lock is still held
        try:
            with (live.ROOT / "results/pool_evaluation_queue.lock").open("a") as lease:
                backfill_pruning_log(run_dir, m)  # fix round 2: startup back-fill of committed decisions
                tll._wait_for_resources(lease, [run_dir], sleep)
                with v070_patches(m):
                    _status(run_dir, "running", **_counts(run_dir, m))
                    nm = baseline_stage(run_dir, m)
                    write_samples_json(run_dir, m)
                    if baseline_only:
                        _status(run_dir, "nomemory_train_complete", nomemory=nm)
                        return {"status": "nomemory_train_complete", "nomemory": nm}
                    return round_loop(run_dir, m)
        except BaseException as exc:
            _status(run_dir, "stopped", error=repr(exc))
            raise


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args(argv)
    if a.preflight_only:
        print(json.dumps(preflight(a.run_dir, load(a.run_dir)), default=str))
        return
    print(json.dumps(run(a.run_dir, baseline_only=a.baseline_only), default=str))


if __name__ == "__main__":
    main()
