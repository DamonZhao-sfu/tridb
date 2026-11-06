"""Recovery helpers for a new, isolated shared initialization.

No dispatch occurs on import. The execution gate remains closed while the
live-service attestation and end-to-end recovery workflow are unfinished.
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from trimem.experiments import scienceworld_trace_ablation as ab


def scan_source_processes(markers, *, proc_root=Path("/proc")):
    """Find live or stopped same-user processes without exposing their environment."""
    if not markers or any(not isinstance(m, str) or len(m) < 8 for m in markers):
        raise ValueError("source process scan requires specific identity markers")
    needles = [m.encode() for m in markers]
    root = Path(proc_root)
    excluded = set()
    if root == Path("/proc"):
        pid = os.getpid()
        while pid > 1 and pid not in excluded:
            excluded.add(pid)
            try:
                stat = (root / str(pid) / "stat").read_text()
                pid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                break
    found = []
    for process in root.iterdir():
        if not process.name.isdigit() or int(process.name) in excluded:
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            state = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                continue
            command = (process / "cmdline").read_bytes()
            if any(marker in command for marker in needles):
                found.append(int(process.name))
                continue
            if (process / "comm").read_text().strip() in ("sshd", "(sd-pam)"):
                continue
            environment = (process / "environ").read_bytes().split(b"\0")
            # Guard helpers may inherit the guard's own marker list. It is not
            # evidence that they are workers of the historical experiment.
            searchable = [
                entry
                for entry in environment
                if not entry.startswith(b"TRIMEM_ABLATION_SOURCE_PROCESS_MARKERS=")
            ]
            if any(marker in entry for marker in needles for entry in searchable):
                found.append(int(process.name))
        except (OSError, IndexError) as exc:
            if not process.exists():
                continue
            raise ValueError("cannot complete source process scan") from exc
    return sorted(found)


def check_source_idle(markers, *, scan=scan_source_processes):
    try:
        if scan(markers):
            raise SystemExit(
                "historical source process active: pause and drain initialization"
            )
    except ValueError as exc:
        raise SystemExit(f"historical source process scan failed: {exc}") from exc
    return True


def resolve_stale_pending(
    plan, rows, *, source_experiment, observed_at_s, active_source_pids
):
    """Treat demonstrably stale unfinished rows as missing in a new experiment."""
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != ab._digest(payload):
        raise ValueError("pending recovery plan hash mismatch")
    if active_source_pids:
        raise ValueError("historical source has active or paused processes")
    expected = {(r["config_key"], r["query_id"]) for r in plan["blocked_pending"]}
    by_pair = {(r["config_key"], r["query_id"]): r for r in rows}
    if len(by_pair) != len(rows) or set(by_pair) != expected or not source_experiment:
        raise ValueError("pending source rows differ from the frozen plan")
    entries = []
    for pair, row in sorted(by_pair.items()):
        age = float(observed_at_s) - float(row["updated_at_s"])
        if (
            row.get("status") not in ("retrieving", "retrieved_pending_evaluation")
            or not row.get("retrieval_id")
            or row.get("generation") is not None
            or not math.isfinite(age)
            or not age >= 86400
        ):
            raise ValueError("pending row is not demonstrably stale and unfinished")
        entries.append(
            {
                "config_key": pair[0],
                "query_id": pair[1],
                "retrieval_id": str(row["retrieval_id"]),
                "status": row["status"],
                "updated_at_s": float(row["updated_at_s"]),
            }
        )
    missing = sorted(
        plan["missing"] + plan["blocked_pending"],
        key=lambda x: (x["config_key"], x["query_id"]),
    )
    resolved = {
        **payload,
        "missing": missing,
        "missing_count": len(missing),
        "blocked_pending": [],
        "blocked_pending_count": 0,
        "pending_disposition": {
            "source_experiment": source_experiment,
            "rows": entries,
            "observed_at_s": float(observed_at_s),
            "source_processes": 0,
            "action": "rerun_in_separate_experiment_without_modifying_source",
        },
    }
    return {**resolved, "sha256": ab._digest(resolved)}


def verify_pending_disposition(
    disposition,
    rows,
    *,
    source_experiment,
    markers,
    observed_at_s,
    scan=scan_source_processes,
):
    """Recheck stale source rows and live/paused processes immediately before use."""
    if (
        not isinstance(disposition, dict)
        or disposition.get("source_experiment") != source_experiment
        or disposition.get("action")
        != "rerun_in_separate_experiment_without_modifying_source"
        or disposition.get("source_processes") != 0
        or not math.isfinite(float(observed_at_s))
        or not math.isfinite(float(disposition["observed_at_s"]))
        or float(disposition["observed_at_s"]) > float(observed_at_s)
    ):
        raise ValueError("invalid pending disposition identity")
    if scan(markers):
        raise ValueError("historical source has active or paused processes")
    expected = {r["retrieval_id"]: r for r in disposition.get("rows", [])}
    actual = {}
    for row in rows:
        identity = str(row["retrieval_id"])
        if identity in actual or row.get("generation") is not None:
            raise ValueError("pending source status changed")
        actual[identity] = {
            "config_key": row["config_key"],
            "query_id": row["query_id"],
            "retrieval_id": identity,
            "status": row["status"],
            "updated_at_s": float(row["updated_at_s"]),
        }
        if (
            not math.isfinite(actual[identity]["updated_at_s"])
            or float(observed_at_s) - actual[identity]["updated_at_s"] < 86400
        ):
            raise ValueError("pending source row is not stale")
    if len(expected) != len(disposition.get("rows", [])) or expected != actual:
        raise ValueError("pending source rows changed since disposition")
    return True


def verify_semfilter_receipt(row, block):
    if row["protocol"]["space_config"]["plan"] not in ("P2", "P2+Rank", "P1+Rank"):
        return
    actual = (row.get("receipt") or {}).get("semfilter") or {}
    if actual.get("protocol") != "semfilter-v2" or any(
        actual.get(k) != block[k]
        for k in ("top_n", "batch", "max_tokens", "judge_input", "rubric")
    ):
        raise ValueError("SemFilter receipt differs from frozen v2 settings")


def checked_recovery_receipt(
    row,
    *,
    expected_pairs,
    sample_hashes,
    experiment,
    draft_hash,
    result_root,
    key_of,
    expected_semfilter=None,
):
    """Verify a newly evaluated TRAIN row before adding it to a seed snapshot."""
    protocol = row.get("protocol") or {}
    if expected_semfilter is not None:
        verify_semfilter_receipt(row, expected_semfilter)
    key = key_of(protocol["space_config"])
    if (
        (key, row.get("query_id")) not in expected_pairs
        or row.get("experiment_id") != experiment
        or row.get("split") != "train"
        or protocol.get("sample_hash") != sample_hashes.get(key)
        or protocol.get("ablation_phase") != "initialization"
        or protocol.get("ablation_draft_sha256") != draft_hash
        or not row.get("retrieval_id")
    ):
        raise ValueError("recovered initialization identity mismatch")
    path = Path(row["generation"]["result_file"])
    root = Path(result_root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("recovered result is outside the isolated initialization")
    raw = path.read_bytes()
    reward = ab.validate_historical_episode(row, json.loads(raw), {row["query_id"]})
    return {
        "config_key": key,
        "query_id": row["query_id"],
        "retrieval_id": str(row["retrieval_id"]),
        "reward": reward,
        "tokens": float(row["total_tokens"]),
        "latency_ms": float(row["e2e_ms"]),
        "source_experiment": experiment,
        "result_file": str(path.resolve()),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }


def reconcile_recovered_rows(rows, ledger, **options):
    """Validate the whole batch, then idempotently account actual completions."""
    receipts = {}
    for row in rows:
        receipt = checked_recovery_receipt(row, **options)
        pair = (receipt["config_key"], receipt["query_id"])
        if pair in receipts:
            raise ValueError("duplicate initialization measurement")
        receipts[pair] = receipt
    for (key, query), receipt in receipts.items():
        job = ab.physical_episode_job_id(
            key, query, arm="full", shared_initialization=True
        )
        ledger.complete(
            job,
            tokens=receipt["tokens"],
            elapsed_s=receipt["latency_ms"] / 1000,
            status="complete",
            metadata={
                "source_retrieval_id": receipt["retrieval_id"],
                "source_experiment": receipt["source_experiment"],
                "elapsed_kind": "episode_latency_not_wall_time",
            },
        )
    return receipts


def select_recovery_pairs(remaining, *, smoke):
    pairs = set(remaining)
    return set(sorted(pairs)[:1]) if smoke else pairs


def initialization_experiments(draft_hash):
    from trimem.experiments import memory_policy as mp

    base = "scienceworld-ablation-init-" + draft_hash[:16]
    return mp.tag(base), mp.tag(base + ":smoke")


def restore_smoke_result(row, *, result_root):
    """Restore one evaluated smoke result mistakenly quarantined as an orphan."""
    destination = Path(row["generation"]["result_file"])
    root = Path(result_root).resolve()
    if destination.is_symlink() or not destination.resolve().is_relative_to(root):
        raise ValueError("smoke result destination is outside the campaign")
    if destination.exists():
        return
    candidates = [
        p
        for p in (destination.parent / "_orphans").glob("*/" + destination.name)
        if p.is_file() and not p.is_symlink()
    ]
    if len(candidates) != 1:
        raise ValueError("smoke orphan result is missing or ambiguous")
    raw = candidates[0].read_bytes()
    ab.validate_historical_episode(row, json.loads(raw), {row["query_id"]})
    ab._install_verified_bytes(raw, destination, hashlib.sha256(raw).hexdigest())


def run_initialization_recovery(
    draft,
    plan,
    source_state_path,
    source_rows,
    target_identity,
    identity_evidence,
    resource_limits,
    system_limits,
    *,
    connection,
    campaign_root,
    test_ids=None,
    smoke=False,
):
    """Recover a seed into a distinct experiment, preserving all old artifacts.

    The launch gate is intentionally authoritative: live identity attestation
    and the complete campaign integration must be finished before dispatch.
    """
    from tools.scienceworld_ablation_clone_memory import verify_campaign_clone
    from tools.scienceworld_ablation_resource_preflight import collect
    from tools.scienceworld_ablation_runtime import verify_current_cgroup
    from tools.wait_sci_start import wait_for_admission
    from trimem.experiments import bo_pool_driver as driver
    from trimem.experiments import pool_branch

    driver.ensure_ablation_execution_ready(SimpleNamespace(ablation_arm="full"))
    ab.check_global_quota()
    ab.validate_ablation_environment(os.environ)
    ab.verify_manifest(draft)
    ab.validate_system_limits(system_limits)
    if draft.get("stage") != "code_only":
        raise ValueError("initialization requires the original draft manifest")
    root = Path(campaign_root).resolve()
    if not root.name.startswith("scienceworld_trace_ablation_"):
        raise ValueError("initialization requires an isolated campaign directory")
    raw_state = Path(source_state_path).read_bytes()
    if hashlib.sha256(raw_state).hexdigest() != draft["source_sha256"]:
        raise ValueError("historical initialization state changed")
    source = json.loads(raw_state)
    semfilter = (
        ab.verified_semfilter_v2(source, draft) if "semfilter" in draft else None
    )
    if (
        source.get("memory_policy") != driver.mp.SUCCESS_FAILURE
        or source.get("space_version") != driver.space.VERSION
    ):
        raise ValueError("historical source protocol is incompatible")
    configs = source["rounds"][0]["configs"]
    by_key = {driver.space.key(c["config"]): c for c in configs}
    if len(configs) != 10 or set(by_key) != set(draft["source_config_keys"]):
        raise ValueError("historical configuration identities changed")
    samples = [
        {"config_key": key, "sample_ids": c["sample"]["ids"]}
        for key, c in by_key.items()
    ]
    for config in configs:
        sample = config["sample"]
        ab.checked_initial_sample(
            sample["file"],
            sample["ids"],
            sample["sample_hash"],
            seed=sample["seed"],
            pool_hash=source["pool_hash"],
        )
    ab.initialization_recovery_jobs(samples, plan)
    initial_ids = {q for sample in samples for q in sample["sample_ids"]}
    if (
        test_ids is None
        or len(test_ids) != 200
        or len(set(test_ids)) != 200
        or set(test_ids) & (initial_ids | set(draft["validation_ids"]))
        or initial_ids & set(draft["validation_ids"])
    ):
        raise ValueError(
            "initialization TRAIN, validation and TEST identities overlap or are incomplete"
        )
    if len(target_identity) != len(ab.RUNTIME_IDENTITY_FIELDS):
        raise ValueError("complete target identity is required")
    ab.verify_identity_evidence(
        {"source_identity": target_identity, "target_identity": target_identity},
        identity_evidence,
    )
    memory_database = os.environ.get("TRIMEM_SCIWORLD_MEMORY_DB")
    target_path = identity_evidence["target"]["path"]
    ab.verify_postrun_identity(
        {"target_identity": target_identity}, target_path, target_path, memory_database
    )
    verify_campaign_clone(
        root,
        {"memory_database": memory_database, "identity_evidence": identity_evidence},
        {"target_identity": target_identity},
        os.environ,
    )
    cgroup = verify_current_cgroup(system_limits)["cgroup"]
    wait_for_admission(
        system_limits,
        check_quota=ab.check_global_quota,
        observe=collect,
        sleep=time.sleep,
        cgroup=cgroup,
    )
    if not driver.env_servers_healthy("sciworld"):
        raise ValueError("ScienceWorld services are unhealthy")
    if driver.mp.current() != driver.mp.SUCCESS_FAILURE:
        raise ValueError("initialization requires success+failure-v1 memory")

    # Validate every reused row against its original panel, stage and result.
    historical = {}
    reuse_pairs = {(r["config_key"], r["query_id"]) for r in plan["reusable"]}
    db_source_rows = connection.execute(
        f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train'",
        (source["experiment"],),
    ).fetchall()
    expected_source_pairs = {
        (sample["config_key"], query)
        for sample in samples
        for query in sample["sample_ids"]
    }

    def verify_source_pending(rows):
        live_pending = []
        for row in rows:
            if row.get("status") in ("retrieving", "retrieved_pending_evaluation"):
                key = driver.space.key(row["protocol"]["space_config"])
                if (key, row["query_id"]) in expected_source_pairs:
                    live_pending.append((key, row))
        disposition = plan.get("pending_disposition")
        if live_pending and not disposition:
            raise ValueError("historical pending initialization rows remain unresolved")
        if disposition:
            pending_records = [
                {
                    "config_key": key,
                    "query_id": row["query_id"],
                    "retrieval_id": row["retrieval_id"],
                    "status": row["status"],
                    "updated_at_s": row["updated_at"].timestamp(),
                    "generation": row.get("generation"),
                }
                for key, row in live_pending
            ]
            verify_pending_disposition(
                disposition,
                pending_records,
                source_experiment=source["experiment"],
                markers=[source["experiment"], str(Path(source_state_path).resolve())],
                observed_at_s=time.time(),
                scan=scan_source_processes,
            )

    verify_source_pending(db_source_rows)
    db_rows = {str(row["retrieval_id"]): row for row in db_source_rows}
    for row in source_rows:
        protocol = row.get("protocol") or {}
        key = driver.space.key(protocol["space_config"])
        pair = (key, row["query_id"])
        if pair not in reuse_pairs:
            continue
        if db_rows.get(str(row["retrieval_id"])) != row:
            raise ValueError("historical row differs from the database")
        if (
            pair in historical
            or row.get("experiment_id") != source["experiment"]
            or row.get("split") != "train"
        ):
            raise ValueError("duplicate or incompatible historical initialization row")
        cfg = by_key[key]
        stage = protocol.get("stage_n")
        if (
            stage not in (50, 100)
            or row["query_id"] not in cfg["staging"]["order"][:stage]
            or protocol.get("sample_hash") != cfg["sample"]["sample_hash"]
            or protocol.get("sample_seed") != cfg["sample"]["seed"]
            or protocol.get("pool_hash") != source["pool_hash"]
        ):
            raise ValueError("historical sample or stage identity differs")
        if (
            protocol.get("memory_policy") != source["memory_policy"]
            or protocol.get("space_version") != source["space_version"]
        ):
            raise ValueError("historical row protocol identity differs")
        path = Path(row["generation"]["result_file"])
        if path.is_symlink():
            raise ValueError("historical result is a symlink")
        result_bytes = path.read_bytes()
        reward = ab.validate_historical_episode(
            row, json.loads(result_bytes), set(cfg["sample"]["ids"])
        )
        historical[pair] = {
            "config_key": key,
            "query_id": row["query_id"],
            "retrieval_id": str(row["retrieval_id"]),
            "reward": reward,
            "tokens": float(row["total_tokens"]),
            "latency_ms": float(row["e2e_ms"]),
            "source_experiment": source["experiment"],
            "result_file": str(path.resolve()),
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    if set(historical) != reuse_pairs:
        raise ValueError("not all proposed historical reuse rows were verified")

    base = root / "initialization"
    ledger_path = root / "resource_ledger.sqlite"
    ledger = (
        ab.ResourceLedger.open(ledger_path, draft["manifest_sha256"], resource_limits)
        if ledger_path.exists()
        else ab.ResourceLedger.create(
            ledger_path, draft["manifest_sha256"], resource_limits
        )
    )
    ab.reserve_initialization_recovery(ledger, samples, plan)
    for (key, query), receipt in historical.items():
        ledger.complete(
            ab.physical_episode_job_id(
                key, query, arm="full", shared_initialization=True
            ),
            tokens=receipt["tokens"],
            elapsed_s=receipt["latency_ms"] / 1000,
            status="complete",
            metadata={
                "source_retrieval_id": receipt["retrieval_id"],
                "source_experiment": receipt["source_experiment"],
                "elapsed_kind": "episode_latency_not_wall_time",
            },
        )
    experiment, smoke_experiment = initialization_experiments(draft["manifest_sha256"])
    missing_pairs = {(r["config_key"], r["query_id"]) for r in plan["missing"]}
    sample_hashes = {key: c["sample"]["sample_hash"] for key, c in by_key.items()}
    opts = {
        "expected_pairs": missing_pairs,
        "sample_hashes": sample_hashes,
        "experiment": experiment,
        "draft_hash": draft["manifest_sha256"],
        "result_root": base,
        "key_of": driver.space.key,
    }
    if semfilter is not None:
        opts["expected_semfilter"] = semfilter

    def reconcile():
        rows = connection.execute(
            f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train' AND status='evaluated'",
            (experiment,),
        ).fetchall()
        return reconcile_recovered_rows(rows, ledger, **opts)

    recovered = reconcile()
    remaining = missing_pairs - set(recovered)
    if remaining:
        dispatch_pairs = select_recovery_pairs(remaining, smoke=smoke)
        run_experiment = smoke_experiment if smoke else experiment
        run_base = base / "smoke" if smoke else base
        smoke_job = "initialization-smoke:" + ab._digest(sorted(dispatch_pairs))
        if smoke:
            ledger.reserve("noise", smoke_job)
            prior = connection.execute(
                f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train' AND status='evaluated'",
                (smoke_experiment,),
            ).fetchall()
            if len(prior) > 1:
                raise ValueError("multiple evaluated smoke rows require reconciliation")
            if prior:
                restore_smoke_result(prior[0], result_root=run_base)
                receipt = checked_recovery_receipt(
                    prior[0],
                    **{
                        **opts,
                        "expected_pairs": dispatch_pairs,
                        "experiment": smoke_experiment,
                        "result_root": run_base,
                    },
                )
                ledger.complete(
                    smoke_job,
                    tokens=receipt["tokens"],
                    elapsed_s=receipt["latency_ms"] / 1000,
                    status="complete",
                    metadata={
                        "purpose": "low-concurrency smoke; excluded from initialization",
                        "source_retrieval_id": receipt["retrieval_id"],
                    },
                )
                return None
        grouped = defaultdict(list)
        for key, query in sorted(dispatch_pairs):
            grouped[key].append(query)
        arms = []
        for key, queries in grouped.items():
            cfg, sample = by_key[key]["config"], by_key[key]["sample"]
            dst = (
                run_base
                / "samples"
                / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            )
            ab.copy_initial_sample(
                sample["file"],
                dst,
                sample["ids"],
                sample["sample_hash"],
                seed=sample["seed"],
                pool_hash=source["pool_hash"],
            )
            copied_sample = {**sample, "file": str(dst)}
            session = driver.session_args(
                "sciworld",
                cfg,
                copied_sample,
                {"pool_hash": source["pool_hash"]},
                experiment=run_experiment,
                baseline_dir=run_base / "empty_baseline",
                label="ablation-initialization",
                sampling="per-config",
                reuse=False,
                extra={
                    "ablation_phase": "initialization",
                    "ablation_draft_sha256": draft["manifest_sha256"],
                },
            )
            arms.append(
                driver.Arm(
                    driver.cfg_name(cfg),
                    run_base / "configs" / driver.cfg_name(cfg),
                    queries,
                    session,
                )
            )
        jobs = (
            [smoke_job]
            if smoke
            else [
                ab.physical_episode_job_id(
                    key, query, arm="full", shared_initialization=True
                )
                for key, query in sorted(dispatch_pairs)
            ]
        )
        if smoke:
            ledger.reserve_many("noise", jobs)
        state = {
            "ablation_resource_ledger": str(ledger_path),
            "ablation_manifest_sha256": draft["manifest_sha256"],
            "ablation_resource_limits": resource_limits,
            "ablation_system_limits": system_limits,
        }
        changes = ab.worker_retry_environment(state)
        if semfilter is not None:
            changes.update(driver.semfilter_env(semfilter))
        changes[driver.RUNTIME_PROFILE_ENV] = driver.RUNTIME_PROFILE_C16
        if plan.get("pending_disposition"):
            changes["TRIMEM_ABLATION_SOURCE_PROCESS_MARKERS"] = json.dumps(
                [source["experiment"], str(Path(source_state_path).resolve())]
            )
        previous = {key: os.environ.get(key) for key in changes}
        old_concurrency = driver.CONCURRENCY["sciworld"]
        old_test_concurrency = driver.TEST_CONCURRENCY["sciworld"]
        try:
            os.environ.update(changes)
            driver.CONCURRENCY["sciworld"] = driver.TEST_CONCURRENCY["sciworld"] = 16
            driver.runtime_profile_want("sciworld")
            ab.check_global_quota()
            verify_source_pending(
                connection.execute(
                    f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train'",
                    (source["experiment"],),
                ).fetchall()
            )
            ledger.mark_dispatch_many(jobs)
            clock = ab.SearchClock(
                base / ("smoke_clock.json" if smoke else "initialization_clock.json"),
                draft["manifest_sha256"],
            )
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            clock.start(
                wall_s=time.time(), monotonic_ns=time.monotonic_ns(), boot_id=boot_id
            )
            try:
                driver.run_capped(
                    "sciworld", arms, 2 if smoke else 16, allow_restart=False
                )
            finally:
                clock.end(
                    wall_s=time.time(),
                    monotonic_ns=time.monotonic_ns(),
                    boot_id=boot_id,
                )
        finally:
            driver.CONCURRENCY["sciworld"] = old_concurrency
            driver.TEST_CONCURRENCY["sciworld"] = old_test_concurrency
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if smoke:
            smoke_rows = connection.execute(
                f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train' AND status='evaluated'",
                (run_experiment,),
            ).fetchall()
            if len(smoke_rows) != 1:
                raise SystemExit("initialization smoke incomplete; inspect its receipt")
            smoke_receipt = checked_recovery_receipt(
                smoke_rows[0],
                **{
                    **opts,
                    "expected_pairs": dispatch_pairs,
                    "experiment": run_experiment,
                    "result_root": run_base,
                },
            )
            ledger.complete(
                jobs[0],
                tokens=smoke_receipt["tokens"],
                elapsed_s=smoke_receipt["latency_ms"] / 1000,
                status="complete",
                metadata={
                    "purpose": "low-concurrency smoke; excluded from initialization",
                    "source_retrieval_id": smoke_receipt["retrieval_id"],
                },
            )
            return None
        recovered = reconcile()
    if set(recovered) != missing_pairs:
        raise SystemExit("initialization incomplete; resume after reviewing receipts")
    receipts = [historical[pair] for pair in sorted(historical)] + [
        recovered[pair] for pair in sorted(recovered)
    ]
    snapshot = ab.shared_initialization_snapshot(
        samples,
        receipts,
        test_ids,
        source_identity=target_identity,
        target_identity=target_identity,
    )
    raw = (
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    ab._install_verified_bytes(
        raw, base / "shared_initialization.json", hashlib.sha256(raw).hexdigest()
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "draft",
        "plan",
        "source-state",
        "target-identity",
        "identity-evidence",
        "resource-limits",
        "system-limits",
        "test-ids",
        "campaign-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate one initialization pair with at most two workers",
    )
    args = parser.parse_args()
    ab.check_global_quota()
    draft = json.loads(args.draft.read_text())
    plan_data = json.loads(args.plan.read_text())
    plan = plan_data.get("plan", plan_data)
    source_bytes = args.source_state.read_bytes()
    source = json.loads(source_bytes)
    ab.verify_manifest(draft)
    if hashlib.sha256(source_bytes).hexdigest() != draft["source_sha256"]:
        raise ValueError("source state differs from frozen draft")
    from trimem.experiments import bo_pool_driver as driver

    configs = [
        {"config_key": driver.space.key(c["config"]), "sample_ids": c["sample"]["ids"]}
        for c in source["rounds"][0]["configs"]
    ]
    jobs = ab.initialization_recovery_jobs(configs, plan)
    if not args.commit:
        print(
            json.dumps(
                {
                    "status": "validated_read_only",
                    "episodes": len(jobs),
                    "reused": plan["reusable_count"],
                    "to_evaluate": plan["missing_count"],
                    "execution_enabled": False,
                }
            )
        )
        return 0
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import pool_branch

    driver.ensure_ablation_execution_ready(SimpleNamespace(ablation_arm="full"))
    with psycopg.connect(
        driver.drv.OPT_DSN + " options='-c default_transaction_read_only=on'",
        row_factory=dict_row,
    ) as conn:
        source_rows = conn.execute(
            f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id=%s AND split='train'",
            (source["experiment"],),
        ).fetchall()
        snapshot = run_initialization_recovery(
            draft,
            plan,
            args.source_state,
            source_rows,
            json.loads(args.target_identity.read_text()),
            json.loads(args.identity_evidence.read_text()),
            json.loads(args.resource_limits.read_text()),
            json.loads(args.system_limits.read_text()),
            connection=conn,
            campaign_root=args.campaign_root,
            test_ids=json.loads(args.test_ids.read_text()),
            smoke=args.smoke,
        )
    if snapshot is None:
        print(json.dumps({"status": "smoke_complete", "episodes": 1}))
        return 0
    print(
        json.dumps(
            {
                "status": "complete",
                "snapshot_sha256": snapshot["sha256"],
                "episodes": snapshot["episode_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
