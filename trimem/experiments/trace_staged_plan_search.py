"""Versioned TRACE entry: Sobol 5 -> measured feedback -> LLM 5 -> BO q=4.

Existing source-frozen campaigns keep trace_joint_plan_search unchanged.
New campaigns explicitly use this entry; all results require a new run directory.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.joint_search_protocol import (
    SCHEDULE,
    EscapeSnapshot,
    normalized_hv,
    propose_with_escape,
    stagnation_status,
)
from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_schema import JointSpace, digest
from trimem.optimizer.plan_surrogate import (
    ObjectiveProtocol,
    aggregate_panel,
    fit_model,
)


def atomic_json(path, body):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    names = [
        "trimem/optimizer/" + n + ".py"
        for n in (
            "plan_schema",
            "plan_codec",
            "plan_completion",
            "plan_surrogate",
            "probabilistic_plan_search",
            "joint_search_controller",
            "joint_search_ledger",
            "joint_plan_analyzer",
            "joint_search_protocol",
            "joint_staged_analyzer",
        )
    ]
    names += [
        "trimem/experiments/" + n + ".py"
        for n in (
            "trace_joint_plan_search",
            "trace_staged_plan_search",
            "plan_execution_adapter",
            "session_retrieval",
        )
    ]
    import hashlib

    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}


def validate_manifest(manifest):
    tasks = manifest["tasks"]
    if len(set(map(str, tasks))) != len(tasks) or len(tasks) < 2:
        raise ValueError("Distinct TRAIN task ids required")
    initial = manifest.get("initial_configs", 10)
    budget = manifest["budget"]
    if (
        type(initial) is not int
        or type(budget) is not int
        or not 2 <= initial <= budget
    ):
        raise ValueError("Invalid initialization/total configuration budget")
    if manifest.get("new_configs", budget - initial) != budget - initial:
        raise ValueError("Initialization plus exploration must equal total budget")
    schedule = manifest.get("search_schedule", SCHEDULE)
    if (
        any(
            schedule.get(k) != SCHEDULE[k]
            for k in ("mode", "sobol_count", "agent_count", "forced_exploration")
        )
        or initial != 10
    ):
        raise ValueError("Staged protocol requires five Sobol plus five LLM configs")
    if (
        type(schedule.get("stagnation_patience")) is not int
        or schedule["stagnation_patience"] < 1
        or not 0 <= schedule.get("stagnation_epsilon", -1) < float("inf")
    ):
        raise ValueError("Invalid stagnation settings")
    if not manifest.get("smoke") and len(tasks) != 100:
        raise ValueError("Live protocol requires 100 TRAIN tasks")
    if (
        manifest.get("q", 4) not in (1, 2, 4)
        or not 1 <= manifest.get("task_concurrency", 16) <= 16
    ):
        raise ValueError("Invalid batch/task concurrency")
    if manifest.get("sources") != source_identity():
        raise ValueError("Execution source identity changed; create a new run manifest")


def get_space(manifest):
    m = manifest["space"]
    space = JointSpace(m["domains"], execution_fingerprint=m["execution_fingerprint"])
    if digest(space.manifest()) != digest(m):
        raise ValueError("Space manifest mismatch")
    return space


def observations(ledger, space, manifest):
    return [
        aggregate_panel(
            space,
            item["config"],
            item["receipts"],
            task_ids=manifest["tasks"],
            protocol=manifest["protocol"],
            allow_synthetic=manifest.get("smoke", False),
        )
        for item in ledger.complete_configs()
    ]


def prepare_run(directory, manifest):
    """Only reserve batch 1; batch 2 depends on all five completed panels."""
    manifest = {"initial_configs": 10, "q": 4, **manifest}
    manifest["search_schedule"] = {**SCHEDULE, **manifest.get("search_schedule", {})}
    manifest.setdefault("budget", 10 + manifest.get("new_configs", 30))
    manifest.setdefault("new_configs", manifest["budget"] - 10)
    validate_manifest(manifest)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Preparation requires a new empty run directory; use round to resume"
        )
    directory.mkdir(parents=True, exist_ok=True)
    space = get_space(manifest)
    initial = space.sample(5, seed=manifest.get("seed", 0))
    if len(initial) != 5:
        raise ValueError(
            "Space cannot supply five distinct Sobol initial configurations"
        )
    frozen = {
        **manifest,
        "objective": None,
        "initial_keys": [space.key(c) for c in initial],
    }
    with SearchLedger(directory / "ledger.sqlite", frozen) as ledger:
        ledger.reserve(space, initial)
    atomic_json(directory / "manifest.json", frozen)
    return frozen


def second_batch_record(directory, manifest, space, complete):
    """Load or validate an atomic stage-2 receipt against its measured evidence."""
    path = Path(directory) / "initialization_agent.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    first = [e for e in complete if e["key"] in manifest["initial_keys"]]
    if (
        record.get("manifest_hash") != digest(manifest)
        or record.get("archive_digest") != digest(first)
        or set(record.get("archive_keys", ())) != set(manifest["initial_keys"])
        or len(first) != 5
    ):
        raise ValueError("Stale initialization advice/evidence identity")
    configs = validate_second_configs(space, record.get("configs", []), manifest)
    if record.get("keys") != [space.key(c) for c in configs]:
        raise ValueError("Initialization advice config identity mismatch")
    return record


def validate_second_configs(space, configs, manifest):
    configs = [space.canonicalize(c) for c in configs]
    keys = {space.key(c) for c in configs}
    if len(configs) != 5 or len(keys) != 5 or keys & set(manifest["initial_keys"]):
        raise ValueError("Batch 2 requires five (5) distinct unmeasured configurations")
    return configs


def initial_keys(directory, manifest, space, complete):
    record = second_batch_record(directory, manifest, space, complete)
    return [*manifest["initial_keys"], *(record["keys"] if record else [])]


def freeze_objective(directory, manifest, obs, space):
    with SearchLedger(Path(directory) / "ledger.sqlite", manifest) as ledger:
        keys = initial_keys(directory, manifest, space, ledger.complete_configs())
    initial = [o for o in obs if space.key(o.config) in keys]
    if len(initial) != manifest.get("initial_configs", 10):
        raise ValueError(
            "Cannot freeze normalization before all initial panels complete"
        )
    objective = asdict(ObjectiveProtocol.from_initial(initial))
    path = Path(directory) / "objective.json"
    existing = (
        json.loads(path.read_text()) if path.exists() else manifest.get("objective")
    )
    if existing and digest(existing) != digest(objective):
        raise ValueError("Frozen initialization objective mismatch")
    if not path.exists():
        atomic_json(path, objective)
    return objective


def _search_worker(directory, manifest, entries, agent, seed, round_file):
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    directory = Path(directory)
    try:
        space = get_space(manifest)
        codec = JointCodec(space)
        obs = [
            aggregate_panel(
                space,
                e["config"],
                e["receipts"],
                task_ids=manifest["tasks"],
                protocol=manifest["protocol"],
                allow_synthetic=manifest.get("smoke", False),
            )
            for e in entries
        ]
        initializing = manifest.get("_initialize_agent", False)
        advice = {"status": "injected", "proposals": agent}
        settings = manifest.get("analyzer")
        if not agent and settings:
            from trimem.optimizer.analyzer_pool import QwenChat
            from trimem.optimizer.joint_staged_analyzer import advise

            if settings.get("backend") != "qwen" or not settings.get("model"):
                raise ValueError(
                    "Analyzer requires explicit qwen backend/model in manifest"
                )
            atomic_json(
                directory / "worker_phase.json",
                {"phase": "analyzer", "at": time.time()},
            )
            client = QwenChat(
                api_bases=[settings["api_base"]],
                model=settings["model"],
                max_tokens=min(int(settings.get("max_tokens", 8192)), 8192),
                timeout=90,
            )
            agent, advice = advise(
                space,
                obs,
                client,
                n_propose=5 if initializing else 4,
                purpose="initialization" if initializing else "ordinary",
            )
            atomic_json(directory / (round_file + ".analyzer.json"), advice)
        if initializing:
            atomic_json(
                directory / (round_file + ".json"),
                {
                    "configs": agent,
                    "advice": advice,
                    "archive_keys": [e["key"] for e in entries],
                    "archive_digest": digest(entries),
                    "manifest_hash": digest(
                        {k: v for k, v in manifest.items() if k != "_initialize_agent"}
                    ),
                },
            )
            return
        objective = ObjectiveProtocol(**manifest["objective"])
        atomic_json(directory / "worker_phase.json", {"phase": "gp", "at": time.time()})
        model = fit_model(
            codec, obs, objective, seed=seed, maxiter=manifest.get("gp_maxiter", 150)
        )
        atomic_json(
            directory / "worker_phase.json", {"phase": "search", "at": time.time()}
        )
        snapshot = EscapeSnapshot(codec, model, obs, objective, seed=seed)
        result = propose_with_escape(
            space,
            snapshot,
            [o.config for o in obs],
            agent=agent,
            seed=seed,
            q=manifest.get("q", 4),
            force_exploration=manifest["_search_progress"]["force_next"],
        )
        checkpoint = result["pr"].pop("checkpoint")
        torch.save(
            {
                "snapshot": snapshot.identity,
                "pr": checkpoint,
                "sources": manifest["sources"],
                "unique_acquisition_inputs": result["unique_acquisition_inputs"],
            },
            directory / (round_file + ".pt"),
        )
        result["archive_keys"] = [space.key(o.config) for o in obs]
        result["search_progress"] = manifest["_search_progress"]
        atomic_json(directory / (round_file + ".json"), result)
    except Exception as exc:
        atomic_json(directory / (round_file + ".error.json"), {"error": repr(exc)})
        raise


def guarded_search(
    directory, manifest, entries, *, agent=(), seed=0, round_file="round"
):
    """Monitor this exact spawned process: RSS 4 GiB, GP 125s, search 65s."""
    directory = Path(directory)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "4"
    phase_path = directory / "worker_phase.json"
    phase_path.unlink(missing_ok=True)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_search_worker,
        args=(str(directory), manifest, entries, list(agent), seed, round_file),
    )
    process.start()
    peak = 0
    began = time.monotonic()
    failure = None
    while process.is_alive():
        try:
            status = Path(f"/proc/{process.pid}/status").read_text()
            rss = (
                int(
                    next(
                        line.split()[1]
                        for line in status.splitlines()
                        if line.startswith("VmRSS:")
                    )
                )
                * 1024
            )
            peak = max(peak, rss)
        except (FileNotFoundError, StopIteration):
            pass
        phase = (
            json.loads(phase_path.read_text())
            if phase_path.exists()
            else {"phase": "startup", "at": time.time()}
        )
        limit = {"search": 65, "analyzer": 95}.get(phase["phase"], 125)
        if (
            peak > 4 * 1024**3
            or time.time() - phase["at"] > limit
            or time.monotonic() - began > 290
        ):
            failure = "worker RSS/deadline limit reached"
            process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
            break
        process.join(0.2)
    process.join()
    resource = {
        "peak_rss_bytes": peak,
        "wall_seconds": time.monotonic() - began,
        "exitcode": process.exitcode,
        "failure": failure,
        "cpu_threads": 4,
    }
    atomic_json(directory / (round_file + ".resources.json"), resource)
    if process.exitcode != 0 or failure:
        raise RuntimeError(
            f"Candidate generation failed: {resource}; see worker error receipt"
        )
    return json.loads((directory / (round_file + ".json")).read_text()), resource


def next_batch_size(manifest, complete):
    return min(manifest.get("q", 4), max(0, manifest["budget"] - complete))


def reserve_second_batch(directory, manifest, space, ledger, complete, agent):
    if len(complete) != 5 or {e["key"] for e in complete} != set(
        manifest["initial_keys"]
    ):
        raise ValueError(
            "All five first-batch panels must complete before LLM initialization"
        )
    record = second_batch_record(directory, manifest, space, complete)
    if record is None:
        if not agent and not manifest.get("analyzer"):
            return False
        if agent:
            result = {"configs": list(agent), "advice": {"status": "injected"}}
        else:
            raw_path = Path(directory) / "initialization_advice.json"
            if raw_path.exists():
                result = json.loads(raw_path.read_text())
            else:
                result, _ = guarded_search(
                    directory,
                    {**manifest, "_initialize_agent": True},
                    complete,
                    round_file="initialization_advice",
                )
            if result.get("archive_digest") != digest(complete) or result.get(
                "manifest_hash"
            ) != digest(manifest):
                raise ValueError("Stale raw initialization advice")
        configs = validate_second_configs(space, result["configs"], manifest)
        record = {
            "manifest_hash": digest(manifest),
            "archive_digest": digest(complete),
            "archive_keys": [e["key"] for e in complete],
            "configs": configs,
            "keys": [space.key(c) for c in configs],
            "advice": result["advice"],
        }
        # Persist validated advice before admission: resume cannot issue a second
        # paid model request or choose a different second batch after reservation.
        atomic_json(Path(directory) / "initialization_agent.json", record)
    ledger.reserve(space, record["configs"])
    return True


def search_progress(directory, manifest, space, obs, complete, objective):
    """Rebuild the counter from completed, ordered configurations, never polls."""
    by_key = {space.key(o.config): o for o in obs}
    known = set(initial_keys(directory, manifest, space, complete))
    if len(known) != 10 or not known <= by_key.keys():
        raise ValueError("Incomplete shared initialization")
    archive = [by_key[k] for k in sorted(known)]
    initial_nhv = normalized_hv(objective, archive)
    events = []
    for path in sorted(Path(directory).glob("round_*.json")):
        result = json.loads(path.read_text())
        if "selected" not in result:
            continue
        selected = result["selected"]
        keys = [r["key"] for r in selected]
        if not any(k in by_key for k in keys):
            continue  # Generated but not admitted yet: no feedback exists.
        if (
            not set(keys) <= by_key.keys()
            or set(keys) & known
            or len(set(keys)) != len(keys)
            or set(result["archive_keys"]) != known
        ):
            raise ValueError("Incomplete or inconsistent search history")
        for row in selected:
            known.add(row["key"])
            archive.append(by_key[row["key"]])
            events.append(
                {
                    "key": row["key"],
                    "nhv": normalized_hv(objective, archive),
                    "kind": "forced"
                    if row.get("selection_rule") == "forced_uncertainty_novelty"
                    else "ordinary",
                }
            )
    if known != by_key.keys():
        raise ValueError("Completed configuration lacks a durable selection receipt")
    schedule = manifest["search_schedule"]
    return {
        **stagnation_status(
            initial_nhv,
            events,
            patience=schedule["stagnation_patience"],
            epsilon=schedule["stagnation_epsilon"],
        ),
        "initial_nhv": initial_nhv,
        "events": events,
    }


def run_round(directory, *, evaluator=None, agent=()):
    with (Path(directory) / "driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another driver owns this run directory") from exc
        return _run_round(directory, evaluator=evaluator, agent=agent)


def _run_round(directory, *, evaluator=None, agent=()):
    """A pending batch is resumed before any new selection; uncertain claims stay pending."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    space = get_space(manifest)
    with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
        pending = ledger.pending()
        complete = ledger.complete_configs()
        second_batch_record(directory, manifest, space, complete)
        result = None
        if not pending:
            initial_count = manifest.get("initial_configs", 10)
            if len(complete) < initial_count:
                if not reserve_second_batch(
                    directory, manifest, space, ledger, complete, agent
                ):
                    return {
                        "status": "awaiting_initialization_advice",
                        "complete_configs": 5,
                    }
                pending = ledger.pending()
            else:
                obs = observations(ledger, space, manifest)
                objective = freeze_objective(directory, manifest, obs, space)
                progress = search_progress(
                    directory,
                    manifest,
                    space,
                    obs,
                    complete,
                    ObjectiveProtocol(**objective),
                )
                atomic_json(directory / "search_progress.json", progress)
                if len(complete) >= manifest["budget"]:
                    return {
                        "status": "budget_complete",
                        "complete_configs": len(complete),
                    }
                seed = manifest.get("seed", 0) + (
                    len(complete) - initial_count
                ) // manifest.get("q", 4)
                name = f"round_{len(complete):03d}"
                result_path = directory / (name + ".json")
                # A crash between generation and reservation reuses the frozen result.
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if set(result["archive_keys"]) != {
                        item["key"] for item in complete
                    }:
                        raise ValueError("Stale proposal archive; refusing replay")
                else:
                    result, _ = guarded_search(
                        directory,
                        {
                            **manifest,
                            "objective": objective,
                            "q": next_batch_size(manifest, len(complete)),
                            "_search_progress": progress,
                        },
                        complete,
                        agent=agent,
                        seed=seed,
                        round_file=name,
                    )
                remaining = next_batch_size(manifest, len(complete))
                selected = result["selected"]
                keys = [space.key(r["config"]) for r in selected]
                if (
                    len(keys) != remaining
                    or len(set(keys)) != remaining
                    or set(keys) & {e["key"] for e in complete}
                    or keys != [r["key"] for r in selected]
                ):
                    raise ValueError("Invalid or duplicate BO selection")
                ledger.reserve(space, [r["config"] for r in selected])
                pending = ledger.pending()
    if evaluator is None:
        return {
            "status": "reserved",
            "pending_episodes": len(pending),
            "proposal": result,
        }

    failed = threading.Event()

    def evaluate_one(item):
        with SearchLedger(directory / "ledger.sqlite", manifest) as writer:
            if not writer.claim(item["key"], item["task_id"]):
                return
            receipt = evaluator(item["config"], item["task_id"], manifest)
            # Individual errors remain claimed: inspect the durable service
            # receipt before a human authorizes any retry.
            if (
                receipt.get("split") != "TRAIN"
                or receipt.get("protocol") != manifest["protocol"]
            ):
                raise ValueError("Evaluator receipt protocol mismatch")
            if not manifest.get("smoke") and (
                receipt.get("config_key") != item["key"]
                or receipt.get("quality_source") != "environment_raw_score"
                or receipt.get("execution_fingerprint") != space.execution_fingerprint
            ):
                raise ValueError("Evaluator receipt identity mismatch")
            writer.finish(item["key"], item["task_id"], receipt)

    def evaluate(item):
        if failed.is_set():
            return
        try:
            return evaluate_one(item)
        except BaseException:
            failed.set()
            raise

    with ThreadPoolExecutor(max_workers=manifest.get("task_concurrency", 16)) as pool:
        list(pool.map(evaluate, [p for p in pending if p["state"] == "ready"]))
    with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
        obs = observations(ledger, space, manifest)
        if len(obs) >= manifest.get("initial_configs", 10):
            freeze_objective(directory, manifest, obs, space)
        return {
            "status": "pending" if ledger.pending() else "batch_complete",
            "complete_configs": len(obs),
            "pending_episodes": len(ledger.pending()),
            "proposal": result,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "round", "status"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", help="prepare JSON with a fresh staged-run manifest")
    parser.add_argument(
        "--evaluator", help="module:function returning one verified episode receipt"
    )
    parser.add_argument(
        "--agent-proposals",
        help="JSON: exactly 5 batch-2 proposals, or <=4 BO proposals",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        data = json.loads(Path(args.input).read_text())
        if "entries" in data:
            raise ValueError(
                "Historical imports cannot certify staged Sobol/LLM initialization"
            )
        result = prepare_run(args.run_dir, data["manifest"])
        print(json.dumps(result, indent=2))
    elif args.action == "round":
        evaluator = None
        if args.evaluator:
            module, name = args.evaluator.split(":", 1)
            evaluator = getattr(importlib.import_module(module), name)
        agent = (
            json.loads(Path(args.agent_proposals).read_text())
            if args.agent_proposals
            else ()
        )
        print(
            json.dumps(
                run_round(args.run_dir, evaluator=evaluator, agent=agent), indent=2
            )
        )
    else:
        manifest = json.loads((Path(args.run_dir) / "manifest.json").read_text())
        with SearchLedger(Path(args.run_dir) / "ledger.sqlite", manifest) as ledger:
            print(
                json.dumps(
                    {
                        "complete": len(ledger.complete_configs()),
                        "pending": ledger.pending(),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
