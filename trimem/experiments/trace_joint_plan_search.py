"""Recoverable v0.4.2 entry point. Historical imports require raw-score receipts.

CLI prepare/round use a JSON manifest and explicit evaluator module:function.
The bounded smoke uses the same worker/controller/ledger with fixture episodes.
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

from trimem.optimizer.joint_search_controller import propose_batch
from trimem.optimizer.joint_search_ledger import (
    EpisodeNotStarted,
    RetryableEpisodeError,
    SearchLedger,
)
from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_schema import JointSpace, digest
from trimem.optimizer.plan_surrogate import (
    AcquisitionSnapshot,
    ObjectiveProtocol,
    aggregate_panel,
    fit_model,
    validate_receipt,
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
        )
    ]
    names += [
        "trimem/experiments/" + n + ".py"
        for n in (
            "trace_joint_plan_search",
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
    holdout = manifest.get("holdout")
    if manifest.get("evaluation_split", "TRAIN") != "TRAIN":
        raise ValueError("TEST cannot enter the search driver")
    if holdout and (
        len(holdout["tasks"]) != 200
        or len(set(holdout["tasks"])) != 200
        or set(tasks) & set(holdout["tasks"])
        or holdout.get("selection") not in ("train_pareto", "all", "deferred")
    ):
        raise ValueError("Invalid or overlapping fixed TEST panel")
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
    initialization = manifest.get("initialization")
    if initialization and (
        initialization != {"method": "sobol_then_llm", "sobol": 5, "llm": 5}
        or initial != 10
    ):
        raise ValueError("Two-stage initialization requires five Sobol plus five LLM")
    if not manifest.get("smoke") and len(tasks) != 100:
        raise ValueError("Live protocol requires 100 TRAIN tasks")
    if (
        manifest.get("q", 4) not in (1, 2, 4)
        or not 1 <= manifest.get("task_concurrency", 16) <= 16
    ):
        raise ValueError("Invalid batch/task concurrency")
    if manifest.get("selection", "greedy") not in ("greedy", "channel-quota-v1"):
        raise ValueError("Unknown batch selection rule")
    pruning = manifest.get("pruning")
    if pruning is not None:
        from trimem.optimizer import staged_pruning as sp

        sizes = pruning.get("sizes") or []
        if (
            pruning.get("protocol") != sp.PROTOCOL
            or sorted(map(str, pruning.get("order", []))) != sorted(map(str, tasks))
            or not sizes or sizes[-1] != len(tasks)
            or any(type(x) is not int for x in sizes)
            or int(pruning.get("budget_episodes", 0)) < initial * len(tasks)
            or not 0 < float(pruning.get("z", 0)) <= 5
        ):
            raise ValueError("Invalid staged pruning settings")
        sp.stage_subsets([str(t) for t in pruning["order"]], sizes)  # strictly increasing, >= 2
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
            allow_partial=bool(manifest.get("pruning")),
            quality_scale=(manifest.get("pruning") or {}).get("quality_scale"),
        )
        for item in ledger.complete_configs()
    ]


def prepare_run(directory, manifest):
    """Reserve the frozen initialization count; actual episodes run via round."""
    from trimem.optimizer.joint_search_controller import initial_configs

    validate_manifest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    space = get_space(manifest)
    initial = (
        space.sample(5, seed=manifest.get("seed", 0))
        if manifest.get("initialization")
        else initial_configs(
            space,
            seed=manifest.get("seed", 0),
            count=manifest.get("initial_configs", 10),
        )
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


def initialization_keys(directory, manifest, space):
    keys = list(manifest["initial_keys"])
    if manifest.get("initialization"):
        path = Path(directory) / "initialization_llm.json"
        if path.exists():
            proposal = json.loads(path.read_text())
            if set(proposal["archive_keys"]) != set(keys):
                raise ValueError("Initialization proposal archive mismatch")
            keys += [space.key(r["config"]) for r in proposal["selected"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate initialization keys")
    return keys


def freeze_objective(directory, manifest, obs, space):
    keys = initialization_keys(directory, manifest, space)
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


def analyzer_client(settings):
    """The manifest's analyzer: local Qwen, or Codex CLI (user 2026-11-03:
    ChatGPT gpt-5.6-luna). Only TRAIN observations are ever sent."""
    if not settings or not settings.get("model"):
        raise ValueError("Analyzer requires an explicit backend and model in the manifest")
    if settings.get("backend") == "qwen":
        from trimem.optimizer.analyzer_pool import QwenChat

        return QwenChat(
            api_bases=[settings["api_base"]],
            model=settings["model"],
            max_tokens=min(int(settings.get("max_tokens", 8192)), 8192),
            timeout=90,
        )
    if settings.get("backend") == "codex":
        from trimem.optimizer.analyzer_session import CodexCLI

        return CodexCLI(
            model=settings["model"],
            effort=settings.get("effort", "medium"),
            timeout_s=float(settings.get("timeout_s", 900)),
        )
    raise ValueError(f"Unsupported analyzer backend {settings.get('backend')!r}")


def analyzer_deadline(settings):
    """Seconds the guarded worker allows the analyzer phase."""
    if settings and settings.get("backend") == "codex":
        return float(settings.get("timeout_s", 900)) + 60
    return 95


def initial_advice(space, obs, settings):
    from trimem.optimizer.joint_plan_analyzer import advise

    return advise(
        space, obs, analyzer_client(settings), max_proposals=5, initialization=True
    )


def reserve_llm_initialization(directory, manifest, ledger, space):
    """Persist phase B before reservation; replay never consumes another LLM call."""
    obs = observations(ledger, space, manifest)
    seen = {space.key(o.config) for o in obs}
    if len(obs) != 5 or seen != set(manifest["initial_keys"]):
        raise ValueError(
            "Five complete Sobol TRAIN panels required before LLM initialization"
        )
    path = Path(directory) / "initialization_llm.json"
    if path.exists():
        proposal = json.loads(path.read_text())
        if set(proposal["archive_keys"]) != seen:
            raise ValueError("Stale initialization proposal")
    else:
        configs, advice = initial_advice(space, obs, manifest.get("analyzer"))
        selected = []
        for config in configs:
            config = space.canonicalize(config)
            key = space.key(config)
            if key not in seen and len(selected) < 5:
                selected.append({"config": config, "source": "llm"})
                seen.add(key)
        missing = 5 - len(selected)
        if missing:
            selected += [
                {"config": c, "source": "sobol_fallback"}
                for c in space.sample(
                    missing, seed=manifest.get("seed", 0) + 777, avoid=seen
                )
            ]
        proposal = {
            "archive_keys": list(manifest["initial_keys"]),
            "selected": selected,
            "analyzer": advice,
        }
        atomic_json(path, proposal)
    keys = [space.key(r["config"]) for r in proposal["selected"]]
    if (
        len(keys) != 5
        or len(set(keys)) != 5
        or set(keys) & set(manifest["initial_keys"])
    ):
        raise ValueError(
            "LLM initialization must reserve five unique novel configurations"
        )
    ledger.reserve(space, [r["config"] for r in proposal["selected"]])
    return proposal


def import_initial(directory, manifest, entries):
    """Validate all ten panels before admitting anything; never relabel scores."""
    validate_manifest(manifest)
    if manifest.get("initialization"):
        raise ValueError(
            "Two-stage initialization requires fresh Sobol measurements before LLM"
        )
    space = get_space(manifest)
    count = manifest.get("initial_configs", 10)
    if len(entries) != count or len({space.key(e["config"]) for e in entries}) != count:
        raise ValueError(
            f"Exactly {count} unique complete initialization panels required"
        )
    if not manifest.get("smoke"):
        for entry in entries:
            for receipt in entry["receipts"]:
                if (
                    receipt.get("config_key") != space.key(entry["config"])
                    or receipt.get("execution_fingerprint")
                    != space.execution_fingerprint
                    or receipt.get("quality_source") != "environment_raw_score"
                ):
                    raise ValueError(
                        "Unverified historical execution/config/raw-score identity"
                    )
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
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    objective = asdict(ObjectiveProtocol.from_initial(obs))
    frozen = {
        **manifest,
        "objective": objective,
        "initial_keys": [space.key(o.config) for o in obs],
    }
    with SearchLedger(directory / "ledger.sqlite", frozen) as ledger:
        ledger.reserve(space, [o.config for o in obs])
        for e in entries:
            key = space.key(e["config"])
            for receipt in e["receipts"]:
                ledger.claim(key, receipt["task_id"])
                ledger.finish(key, receipt["task_id"], receipt)
    atomic_json(directory / "manifest.json", frozen)
    return frozen


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
                allow_partial=bool(manifest.get("pruning")),
                quality_scale=(manifest.get("pruning") or {}).get("quality_scale"),
            )
            for e in entries
        ]
        objective = ObjectiveProtocol(**manifest["objective"])
        settings = manifest.get("analyzer")
        if not agent and settings:
            from trimem.optimizer.joint_plan_analyzer import advise

            client = analyzer_client(settings)
            atomic_json(
                directory / "worker_phase.json",
                {"phase": "analyzer", "at": time.time()},
            )
            agent, advice = advise(space, obs, client)
            atomic_json(directory / (round_file + ".analyzer.json"), advice)
        atomic_json(directory / "worker_phase.json", {"phase": "gp", "at": time.time()})
        model = fit_model(
            codec, obs, objective, seed=seed, maxiter=manifest.get("gp_maxiter", 150)
        )
        atomic_json(
            directory / "worker_phase.json", {"phase": "search", "at": time.time()}
        )
        snapshot = AcquisitionSnapshot(codec, model, obs, objective, seed=seed)
        result = propose_batch(
            space,
            snapshot,
            [o.config for o in obs],
            agent=agent,
            seed=seed,
            q=manifest.get("q", 4),
            selection=manifest.get("selection", "greedy"),
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
        atomic_json(directory / (round_file + ".json"), result)
    except Exception as exc:
        atomic_json(directory / (round_file + ".error.json"), {"error": repr(exc)})
        raise


def guarded_search(
    directory, manifest, entries, *, agent=(), seed=0, round_file="round"
):
    """Monitor this exact spawned process: RSS 4 GiB, GP/startup 125 s, analyzer 95 s.

    The candidate search phase (PR, greedy selection, diagnostic) has no time
    limit (user 2026-11-03); it is bounded by its step and unique-input budgets.
    """
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
        limit = {
            "search": None,
            "analyzer": analyzer_deadline(manifest.get("analyzer")),
        }.get(phase["phase"], 125)
        if peak > 4 * 1024**3 or (
            limit is not None and time.time() - phase["at"] > limit
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
        result = None
        if not pending and manifest.get("initialization") and len(complete) == 5:
            result = reserve_llm_initialization(directory, manifest, ledger, space)
            pending = ledger.pending()
        if not pending:
            objective = freeze_objective(
                directory, manifest, observations(ledger, space, manifest), space
            )
            if manifest.get("pruning"):
                batch_size = staged_batch_size(ledger, manifest)
                if len(complete) >= manifest.get("initial_configs", 10) and batch_size == 0:
                    return {"status": "budget_complete", "complete_configs": len(complete)}
            elif len(complete) >= manifest["budget"]:
                return {"status": "budget_complete", "complete_configs": len(complete)}
            else:
                batch_size = next_batch_size(manifest, len(complete))
            initial_count = manifest.get("initial_configs", 10)
            if len(complete) < initial_count:
                raise ValueError("Initialization is incomplete")
            seed = manifest.get("seed", 0) + (
                len(complete) - initial_count
            ) // manifest.get("q", 4)
            name = f"round_{len(complete):03d}"
            result_path = directory / (name + ".json")
            # A crash between generation and reservation reuses the frozen result.
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if set(result["archive_keys"]) != {item["key"] for item in complete}:
                    raise ValueError("Stale proposal archive; refusing replay")
            else:
                result, _ = guarded_search(
                    directory,
                    {
                        **manifest,
                        "objective": objective,
                        "q": batch_size,
                    },
                    complete,
                    agent=agent,
                    seed=seed,
                    round_file=name,
                )
            remaining = batch_size
            ledger.reserve(space, [r["config"] for r in result["selected"][:remaining]])
            pending = ledger.pending()
    if evaluator is None:
        return {
            "status": "reserved",
            "pending_episodes": len(pending),
            "proposal": result,
        }

    if manifest.get("pruning") and (directory / "objective.json").exists():
        staged_evaluate(directory, manifest, space, pending, evaluator)
    else:
        evaluate_pending(directory, manifest, space, pending, evaluator)
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


def _stage_obj(receipts, tasks, quality_scale=None, z=1.96):
    """staged_pruning.Obj of one configuration on the task subset ``tasks`` (all must be done).

    ``quality_scale`` (user 2026-11-04): floor the quality SE at the Agresti-Coull SE of the
    mean reward rescaled to [0, 1] by ``quality_scale``. Without it a prefix where every task
    succeeds has SE 0, and a cheaper configuration that also succeeds everywhere weakly
    dominates the optimistic corner at the ceiling."""
    import math
    import statistics

    from trimem.optimizer import staged_pruning as sp

    by = {str(r["task_id"]): r for r in receipts}
    rows = [by[t] for t in tasks]
    n = len(rows)
    cols = [[float(r[f]) for r in rows] for f in ("raw_score", "total_tokens", "latency_seconds")]
    se = [statistics.stdev(c) / math.sqrt(n) if n > 1 else 0.0 for c in cols]
    if quality_scale:
        p = min(1.0, max(0.0, statistics.mean(cols[0]) / quality_scale))
        p_ac = (p * n + z * z / 2) / (n + z * z)
        se[0] = max(se[0], quality_scale * math.sqrt(p_ac * (1 - p_ac) / (n + z * z)))
    return sp.Obj(*(statistics.mean(c) for c in cols), *se, n=n)


def pruning_budget(manifest):
    from trimem.optimizer import staged_pruning as sp

    p = manifest["pruning"]
    return sp.Budget(int(p["budget_episodes"]), len(manifest["tasks"]), int(manifest["budget"]))


def staged_batch_size(ledger, manifest):
    """staged_pruning.next_batch_size: episodes saved by pruning buy more configurations, every new
    configuration must still fit in full, and at most ``budget`` (max_configs) configurations."""
    from trimem.optimizer import staged_pruning as sp

    used = ledger.db.execute("SELECT count(*) FROM episodes WHERE state='done'").fetchone()[0]
    n_configs = ledger.db.execute("SELECT count(*) FROM configs").fetchone()[0]
    return sp.next_batch_size(pruning_budget(manifest), used, n_configs, manifest.get("q", 4))


def staged_evaluate(directory, manifest, space, pending, evaluator):
    """Staged profiling with 3-D Pareto pruning, reusing ``trimem.optimizer.staged_pruning``
    (protocol staged-pareto-prune-v2, user 2026-10-19/20; reused for TRACE-PR 2026-11-04).

    A BO batch runs nested prefixes of ``pruning.order`` (one stratified order of the frozen
    TRAIN panel, ``staged_pruning.stage_order``) at ``pruning.sizes``. At every checkpoint
    before the last, ``staged_pruning.decide`` prunes a configuration whose optimistic corner is
    weakly dominated by (a) the point estimate of a fully evaluated configuration or (b) the
    pessimistic corner of another configuration on the same stage subset; ``guard_plans``
    keeps one configuration per plan order without a fully evaluated configuration.
    Initialization configurations are protected. Quality SEs are floored at the Agresti-Coull SE
    (``pruning.quality_scale``; see ``_stage_obj``). Decisions go to ``pruning_log.jsonl``.
    """
    from trimem.optimizer import staged_pruning as sp

    directory = Path(directory)
    p = manifest["pruning"]
    z, sizes, order = float(p["z"]), [int(x) for x in p["sizes"]], [str(t) for t in p["order"]]
    qs = p.get("quality_scale")
    subsets = sp.stage_subsets(order, sizes)
    batch = sorted({x["key"] for x in pending})
    protected = set(initialization_keys(directory, manifest, space))
    for i, subset in enumerate(subsets):
        members = set(subset)
        with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
            current = [
                x for x in ledger.pending()
                if x["key"] in batch and str(x["task_id"]) in members
            ]
        if current:
            evaluate_pending(directory, manifest, space, current, evaluator)
        with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
            live = [k for k in batch if ledger.db.execute(
                "SELECT count(*) FROM episodes WHERE config=? AND state='pruned'", (k,)).fetchone()[0] == 0]
            done = {k: {str(r["task_id"]) for r in ledger.receipts(k)} for k in live}
            # Stage barrier: every live configuration must have its whole prefix done (an
            # unresolved claim or failure stops the run, as in the unstaged path).
            missing = {k: len(members - d) for k, d in done.items() if not members <= d}
            if missing:
                raise RuntimeError(f"Stage {len(subset)} incomplete for {missing}; recover claims first")
        if i == len(subsets) - 1:
            return
        with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
            receipts = {
                k: ledger.receipts(k)
                for (k,) in ledger.db.execute("SELECT key FROM configs").fetchall()
            }
            bodies = dict(ledger.db.execute("SELECT key, body FROM configs").fetchall())
            pruned = {k for (k,) in ledger.db.execute(
                "SELECT DISTINCT config FROM episodes WHERE state='pruned'")}
            full = {
                k: _stage_obj(r, order, qs, z) for k, r in receipts.items()
                if k not in pruned and {str(x["task_id"]) for x in r} == set(order)
            }
            stage_ok = {k for k, r in receipts.items() if members <= {str(x["task_id"]) for x in r}}
            # Decide a checkpoint once: on resume, a configuration that already has work beyond
            # this prefix passed this checkpoint before (its decision is in pruning_log.jsonl).
            undecided = [k for k in live if done[k] == members]
            decisions, objs = {}, {}
            for key in undecided:
                objs[key] = _stage_obj(receipts[key], subset, qs, z)
                partial = {k: _stage_obj(receipts[k], subset, qs, z) for k in stage_ok if k != key}
                decisions[key] = sp.decide(
                    objs[key], {k: o for k, o in full.items() if k != key},
                    stage=len(subset), last=False, protected=key in protected, z=z, partial=partial,
                )
            plan_of = {k: json.loads(bodies[k])["filter_order"] for k in decisions}
            covered = {json.loads(bodies[k])["filter_order"] for k in full}
            decisions = sp.guard_plans(decisions, objs, plan_of, covered, z)
            with (directory / "pruning_log.jsonl").open("a") as log:
                for key, d in decisions.items():
                    log.write(json.dumps({"at": time.time(), "key": key, "stage": len(subset),
                                          **d.as_dict()}, default=list) + "\n")
            ledger.prune_many({k: json.dumps(d.as_dict(), default=list)
                               for k, d in decisions.items() if d.prune})


def check_receipt(receipt, item, manifest, space):
    """Validate before finish(): a done receipt can never be rewritten."""
    split = manifest.get("evaluation_split", "TRAIN")
    validate_receipt(
        receipt,
        split=split,
        protocol=manifest["protocol"],
        allow_synthetic=manifest.get("smoke", False),
    )
    if str(receipt.get("task_id")) != str(item["task_id"]):
        raise ValueError("Receipt task identity mismatch")
    if not manifest.get("smoke"):
        if (
            receipt.get("config_key") != item["key"]
            or receipt.get("quality_source") != "environment_raw_score"
            or receipt.get("execution_fingerprint") != space.execution_fingerprint
        ):
            raise ValueError("Evaluator receipt identity mismatch")
        runtime = manifest.get("fixed", {}).get("runtime")
        if runtime is not None and receipt.get("runtime") != runtime:
            raise ValueError("Evaluator runtime identity mismatch")


def evaluate_pending(directory, manifest, space, pending, evaluator):
    """Dispatch a frozen panel with explicit failure classes.

    - EpisodeNotStarted: no worker launched; the claim returns to ready.
    - RetryableEpisodeError: worker confirmed terminated, no result; retried
      automatically while attempts < ``max_attempts`` (default 2).
    - Anything else (including an invalid receipt): uncertain or completed
      evidence. Marked failed, never retried automatically.
    After ``failure_breaker`` consecutive failures (default 3) no new episode is
    admitted; admitted ones finish. Remaining failures raise, so the service
    stops and the operator decides with the ``retry`` action.
    """
    directory = Path(directory)
    max_attempts = int(manifest.get("max_attempts", 2))
    breaker = int(manifest.get("failure_breaker", 3))
    lock = threading.Lock()
    state = {"consecutive": 0, "tripped": False, "errors": []}

    def note_failure(exc):
        with lock:
            state["consecutive"] += 1
            state["errors"].append(repr(exc))
            if state["consecutive"] >= breaker:
                state["tripped"] = True

    def evaluate_one(item):
        with SearchLedger(directory / "ledger.sqlite", manifest) as writer:
            with lock:  # admission check and claim are one step
                if state["tripped"] or not writer.claim(item["key"], item["task_id"]):
                    return
            try:
                receipt = evaluator(item["config"], item["task_id"], manifest)
            except EpisodeNotStarted as exc:
                writer.release(item["key"], item["task_id"])
                note_failure(exc)
                return
            except RetryableEpisodeError as exc:
                writer.fail(item["key"], item["task_id"], repr(exc), retryable=True)
                note_failure(exc)
                return
            except Exception as exc:
                writer.fail(item["key"], item["task_id"], repr(exc))
                note_failure(exc)
                return
            try:
                check_receipt(receipt, item, manifest, space)
            except Exception as exc:  # completed but invalid: never auto-retried
                writer.fail(item["key"], item["task_id"], repr(exc))
                note_failure(exc)
                return
            writer.finish(item["key"], item["task_id"], receipt)
            with lock:
                state["consecutive"] = 0

    items = {(p["key"], p["task_id"]): p for p in pending}
    for _ in range(max_attempts):
        if state["tripped"]:
            break
        with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
            ledger.retry(max_attempts=max_attempts)
            batch = [
                items[(p["key"], p["task_id"])]
                for p in ledger.pending()
                if p["state"] == "ready" and (p["key"], p["task_id"]) in items
            ]
        if not batch:
            break
        with ThreadPoolExecutor(
            max_workers=manifest.get("task_concurrency", 16)
        ) as pool:
            list(pool.map(evaluate_one, batch))
    with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
        failed = ledger.episodes("failed")
    if failed or state["tripped"]:
        raise RuntimeError(
            f"{len(failed)} episodes failed (breaker={state['tripped']}); "
            f"last errors: {state['errors'][-3:]}; inspect, then run the retry action"
        )


def owned(directory):
    """Exclusive campaign ownership for CLI actions that change ledger state.

    The live service holds the same queue.lock for its whole life (TRAIN rounds,
    recovery and TEST), so a manual round or retry cannot race it.
    """
    handle = (Path(directory) / "queue.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("A live service owns this run directory") from exc
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "round", "status", "retry"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--input", help="prepare JSON: manifest and entries with ten raw panels"
    )
    parser.add_argument(
        "--evaluator", help="module:function returning one verified episode receipt"
    )
    parser.add_argument(
        "--agent-proposals", help="JSON array of <=4 schema-valid analyzer proposals"
    )
    args = parser.parse_args()
    if args.action == "prepare":
        data = json.loads(Path(args.input).read_text())
        result = (
            import_initial(args.run_dir, data["manifest"], data["entries"])
            if "entries" in data
            else prepare_run(args.run_dir, data["manifest"])
        )
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
        with owned(args.run_dir):
            print(
                json.dumps(
                    run_round(args.run_dir, evaluator=evaluator, agent=agent), indent=2
                )
            )
    elif args.action == "retry":
        # Explicit operator decision: every failed episode becomes ready again,
        # including uncertain ones; inspect their evidence first.
        manifest = json.loads((Path(args.run_dir) / "manifest.json").read_text())
        with owned(args.run_dir), SearchLedger(
            Path(args.run_dir) / "ledger.sqlite", manifest
        ) as ledger:
            failed = ledger.episodes("failed")
            print(
                json.dumps(
                    {"retried": ledger.retry(include_uncertain=True), "failed": failed},
                    indent=2,
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
