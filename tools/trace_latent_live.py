"""Live dispatcher and interleaved two-arm scheduler for the TRACE latent-MOBO campaign.

Patched at runtime (never edited on disk; the hashed files stay byte-identical):
  * ``trace_joint_plan_search._search_worker`` -> latent worker, only for ``latent-mobo-v1``
    runs, restored on exit. ``explicit-pr``, ``shared-init-v1`` and manifests without a
    ``search_backend`` key run the stock TRACE-PR path.
  * ``trace_joint_protocol.run_holdout`` -> ``run_pruned_holdout`` for a staged-pruning manifest
    with ``holdout.selection == "train_pareto"`` (user 2026-11-04). The stock selection requires
    ``budget`` complete configurations (with pruning, budget = 80 is a cap, not a count) and would
    rank pruned partial panels; the stratified TRACE-PR runs deferred TEST to trace_pr_test_now.
The lease/resource gate and the per-round sequence mirror ``tools.trace_joint_live.main``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Sequence

from tools import trace_joint_live as live
from tools.prepare_trace_latent_campaign import EXPLICIT_BACKEND, SHARED_BACKEND
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import digest

DEFAULT_ORDER_SEED = 20261104


@contextlib.contextmanager
def backend_patches(manifest: dict) -> Iterator[None]:
    backend = manifest.get("search_backend")
    with contextlib.ExitStack() as stack:
        if backend == trace_latent_search.BACKEND:
            stack.enter_context(trace_latent_search.install())
        elif not (backend is None or backend in (EXPLICIT_BACKEND, SHARED_BACKEND)):
            raise ValueError(f"unknown search_backend {backend!r}")
        if manifest.get("pruning") and (manifest.get("holdout") or {}).get("selection") == "train_pareto":
            stack.enter_context(_pruned_holdout_patch())
        yield


@contextlib.contextmanager
def _pruned_holdout_patch() -> Iterator[None]:
    from tools import trace_joint_protocol as proto

    original = proto.run_holdout
    proto.run_holdout = run_pruned_holdout
    try:
        yield
    finally:
        proto.run_holdout = original


def run_pruned_holdout(directory, *, evaluator):
    """TEST of a staged-pruning arm: ``trace_joint_protocol.run_holdout`` with two changes.

    1. TRAIN completion = the driver's own ``budget_complete`` rule under pruning: nothing pending
       and ``staged_batch_size == 0`` (no further configuration fits in full; budget = max configs).
    2. Candidates = configurations with their FULL TRAIN panel that were never pruned; the TRAIN
       sample-mean Pareto set is taken over them only. Pruned configurations (partial panels) are
       never TEST candidates (as trace_pr_test_now). Everything else (TEST ledger, manifest,
       evaluation, summary) is the stock code path's."""
    import fcntl as _fcntl

    from trimem.optimizer.plan_surrogate import aggregate_panel

    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    driver.validate_manifest(m)
    h = m["holdout"]
    if not m.get("pruning") or h.get("selection") != "train_pareto":
        raise ValueError("run_pruned_holdout needs a staged-pruning manifest with train_pareto TEST")
    if not m.get("smoke"):
        from tools.fixed_test_panel_guard import validate_ids

        validate_ids(m["fixed"]["dataset"], h["tasks"])
    space = driver.get_space(m)
    tasks = {str(t) for t in m["tasks"]}
    with SearchLedger(directory / "ledger.sqlite", m) as train:
        if train.pending() or driver.staged_batch_size(train, m) != 0:
            raise ValueError("Require complete TRAIN search before TEST selection")
        entries = train.complete_configs()
        reserved = train.db.execute("SELECT count(*) FROM configs").fetchone()[0]
    if len(entries) != reserved:
        raise ValueError("Require complete TRAIN search before TEST selection")
    full = [e for e in entries
            if not e["pruned"] and {str(r["task_id"]) for r in e["receipts"]} == tasks]
    obs = [aggregate_panel(space, e["config"], e["receipts"], task_ids=m["tasks"], protocol=m["protocol"],
                           allow_synthetic=m.get("smoke", False), allow_partial=False) for e in full]
    points = [(o.mean[0], -o.mean[1], -o.mean[2]) for o in obs]
    indices = [
        i for i, p in enumerate(points)
        if not any(all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p)) for q in points)
    ]
    selected = [obs[i].config for i in indices]
    selection = {
        "policy": h["selection"],
        "train_protocol": m["protocol"],
        "train_keys": [e["key"] for e in entries],
        "candidate_keys": [e["key"] for e in full],
        "excluded_pruned": [e["key"] for e in entries if e not in full],
        "candidate_rule": "never pruned and full TRAIN panel (user 2026-11-04)",
        "configs": selected,
        "train_means": [obs[i].mean for i in indices],
    }
    out = directory / "holdout"
    out.mkdir(exist_ok=True)
    test_manifest = {
        **m,
        "tasks": h["tasks"],
        "budget": len(selected),
        "protocol": m["protocol"] + "-test",
        "evaluation_split": "TEST",
        "selected_keys": [space.key(c) for c in selected],
    }
    with (out / "driver.lock").open("a") as lock:
        _fcntl.flock(lock, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            test.reserve(space, selected)
            pending = test.pending()
        driver.atomic_json(out / "selection.json", selection)
        driver.atomic_json(out / "manifest.json", test_manifest)
        if not m.get("smoke"):
            live.recover_claims(out, test_manifest)
            with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
                pending = test.pending()
        driver.evaluate_pending(out, test_manifest, space, pending, evaluator)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            if test.pending():
                raise RuntimeError("Uncertain TEST claims require evidence recovery")
            complete = test.complete_configs()
        n = len(h["tasks"])
        metrics = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")
        summary = {
            "status": "test_complete",
            "configurations": len(complete),
            "tasks_per_configuration": n,
            "episodes": n * len(complete),
            "results": [
                {"key": e["key"], "config": e["config"],
                 **{f: sum(r[f] for r in e["receipts"]) / n for f in metrics
                    if all(r.get(f) is not None for r in e["receipts"])}}
                for e in complete
            ],
        }
        driver.atomic_json(out / "summary.json", summary)
        return summary


def interleave_order(seed: int, step: int, arms: Sequence) -> list:
    """Deterministic shuffle: Random(first 16 hex digits of digest([seed, step]))."""
    rng = random.Random(int(digest([seed, step])[:16], 16))
    order = list(arms)
    rng.shuffle(order)
    return order


def check_no_vae_training() -> None:
    # "[t]rain..." keeps pgrep from matching its own command line / parent shell.
    found = subprocess.run(["pgrep", "-f", "[t]rain_trace_seq_vae"], capture_output=True)
    if found.returncode == 0:
        raise SystemExit("refusing to start: train_trace_seq_vae is running")


def _ledger_counts(directory: Path, manifest: dict) -> dict:
    with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
        return {
            "complete_configs": len(ledger.complete_configs()),
            "pending_episodes": len(ledger.pending()),
        }


def _append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _wait_for_resources(lease, status_dirs: Sequence[Path], sleep: Callable[[float], None]) -> None:
    stable = 0
    while True:
        resources = live.resource_snapshot()
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        ready = acquired and live.eligible_resources(
            resources["available_memory_bytes"], resources["gpu_utilization"]
        )
        stable = stable + 1 if ready else 0
        if stable >= 3:
            return
        if acquired:
            fcntl.flock(lease, fcntl.LOCK_UN)
        for d in status_dirs:
            driver.atomic_json(
                d / "status.json",
                {"status": "waiting_resources", "at": time.time(), **resources,
                 "evaluation_lease_available": acquired},
            )
        sleep(30)


def run_interleaved(
    run_dirs: Sequence[Path],
    order_seed: int = DEFAULT_ORDER_SEED,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    dirs = [Path(d) for d in run_dirs]
    manifests = {d: json.loads((d / "manifest.json").read_text()) for d in dirs}
    for d, m in manifests.items():
        driver.validate_manifest(m)
        live.verify_artifacts(m)
        live.service_preflight(m)
    first, *rest = manifests.values()
    for m in rest:
        if m["fixed"]["dataset"] != first["fixed"]["dataset"] or m["tasks"] != first["tasks"]:
            raise ValueError("interleaved arms must share fixed.dataset and tasks")
    with contextlib.ExitStack() as stack:
        for d in dirs:
            own = stack.enter_context((d / "queue.lock").open("a"))
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = stack.enter_context(
            (live.ROOT / "results/pool_evaluation_queue.lock").open("a")
        )
        _wait_for_resources(lease, dirs, sleep)
        active = list(dirs)
        step = 0
        while active:
            for d in interleave_order(order_seed, step, active):
                if _run_one(d, manifests[d], step) == "budget_complete":
                    active.remove(d)
            step += 1


def _run_one(d: Path, m: dict, step: int) -> str:
    log = d / "interleave_log.jsonl"
    _append(log, {"step": step, "arm": str(d), "phase": "before",
                  "resources": live.resource_snapshot(), "at": time.time()})
    try:
        live.verify_artifacts(m)
        live.verify_memory(m)
        live.service_preflight(m)
        live.recover_claims(d, m)
        with backend_patches(m):
            result = driver.run_round(d, evaluator=live.evaluate)
        _append(log, {"step": step, "arm": str(d), "phase": "after",
                      "status": result["status"], "at": time.time()})
        counts = _ledger_counts(d, m)
        status = {"status": result["status"], "at": time.time(), **counts}
        driver.atomic_json(d / "status.json", status)
        if result["status"] == "budget_complete" and m.get("holdout"):
            from tools.trace_joint_protocol import run_holdout

            driver.atomic_json(d / "status.json", {"status": "testing_train_frontier",
                                                   "at": time.time()})
            test = run_holdout(d, evaluator=live.evaluate)
            driver.atomic_json(
                d / "status.json",
                {"status": test["status"], "at": time.time(),
                 "train_configs": counts["complete_configs"],
                 "test_configs": test.get("configurations", 0),
                 "test_episodes": test.get("episodes", 0)},
            )
        if result["status"] == "pending":
            raise RuntimeError("Episodes still pending after bounded retries; inspect first")
    except BaseException as exc:
        driver.atomic_json(d / "status.json", {"status": "failed", "at": time.time(),
                                               "error": repr(exc)})
        raise
    return result["status"]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--interleave", nargs=2, type=Path, metavar=("R1", "R2"))
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    args = parser.parse_args(argv)
    if bool(args.run_dir) == bool(args.interleave):
        parser.error("give exactly one of --run-dir or --interleave")
    check_no_vae_training()
    if args.interleave:
        run_interleaved(args.interleave, args.order_seed)
        return
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    saved = sys.argv
    sys.argv = [saved[0], "--run-dir", str(args.run_dir)] + (
        ["--preflight-only"] if args.preflight_only else []
    )
    try:
        with backend_patches(manifest):
            live.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
