"""Concrete Math/ALE factory: frozen inputs, private native graph, metered LLM.

Use through ``python -m trimem.optimizer run --factory
trimem.optimizer.runtime:math_ale_factory ...``. Merely importing this module
does not load a model, start Postgres, evaluate code, or send a request.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .evaluator import CohortEvaluator, EvaluationProtocol, EvalUnit, IsolatedBranch
from .schema import stable_hash
from .snapshot import FrozenSnapshot, file_sha256

FROZEN_MODE = "first_decision_then_frozen_policy"
ROUTED_MODE = "per_round_routed_policy"


def validate_inputs(config):
    """Read/hash inputs only: no tokenizer import, DB, subprocess or HTTP request."""
    from .math_ale import CheckpointSpec

    if config.get("application_mode") not in (FROZEN_MODE, ROUTED_MODE):
        raise ValueError(
            "application_mode must be the frozen continuation policy or "
            "the per-round routed policy"
        )
    if config.get("graph_policy") != "frozen_no_ingest":
        raise ValueError("first edition runtime requires frozen_no_ingest")
    snapshots = {}
    for task, value in config["snapshots"].items():
        if not value.get("sha256"):
            raise ValueError("every snapshot needs an explicit frozen sha256")
        snapshots[task] = FrozenSnapshot.read(value["path"], value["sha256"])
    checkpoints = {}
    for value in config["checkpoints"]:
        if any(
            not value.get(name + "_sha256") for name in ("code", "spec", "evaluator")
        ):
            raise ValueError("checkpoint code/spec/evaluator digests must be frozen")
        checkpoint = CheckpointSpec(**value)
        checkpoint.load()
        if checkpoint.task_id in checkpoints:
            raise ValueError("runtime v1 allows one checkpoint per task in a cohort")
        if checkpoint.task_id not in snapshots:
            raise ValueError("checkpoint has no frozen graph snapshot")
        query = snapshots[checkpoint.task_id].query(checkpoint.checkpoint_id)
        if query.task != checkpoint.task_id:
            raise ValueError("retrieval query task differs from the checkpoint task")
        checkpoints[checkpoint.task_id] = checkpoint
    if not checkpoints or set(checkpoints) != set(snapshots):
        raise ValueError("checkpoints and snapshots must cover exactly the same tasks")
    for item in config.get("dependency_files", []):
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("a frozen evaluator dependency changed")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a nonempty unique list of integers")
    return checkpoints, snapshots


@contextmanager
def math_ale_factory(config):
    """Yield a paired evaluator; starts a new PG cluster only during evaluation.

    Snapshot reconstruction is setup telemetry outside the Q/T observation. Its
    original construction cost belongs to an independently reported corpus cost.
    Each branch starts with an empty semantic cache. The retrieved context is
    reused by the fixed continuation; this does not implement online graph growth.

    In ``per_round_routed_policy`` mode the candidate is a ``RoutedPolicy``: each
    round it picks an action from the current state, then retrieves on the
    branch's single private cluster. Memory is cached per action hash (the query
    is the frozen snapshot query), so a repeated action costs no new retrieval
    unless the cached context no longer fits the current prompt window.
    """
    from .adapters import NativeRetrievalAdapter
    from .clients import OpenAICompatibleTransport
    from .executor import NativeExecutor
    from .math_ale import MathAleContinuation, SubprocessScorer
    from .private_pg import PrivatePostgres
    from .semantic_usage import (
        ExclusiveClusterLease,
        ExclusiveSemanticUsageCollector,
        MeteredSemanticJudge,
    )
    from .tokenization import LocalChatTokenizer

    config = json.loads(json.dumps(config, allow_nan=False))
    checkpoints, snapshots = validate_inputs(config)
    routed = config["application_mode"] == ROUTED_MODE
    tokenizer = LocalChatTokenizer(config["tokenizer_path"])
    if config["protocol"]["tokenizer_version"] != tokenizer.tokenizer_id:
        raise ValueError(
            "configured tokenizer digest differs from local tokenizer files"
        )
    # Bind endpoint/model/decoding, scorer, graph, and checkpoint inputs to the
    # protocol as well as the user's descriptive prompt version. Secrets are
    # referenced only by environment variable name, never persisted here.
    config_hash = stable_hash(config)
    protocol = EvaluationProtocol.from_dict(
        {
            **config["protocol"],
            "baseline_policy_version": "frozen-no-ingest-v1:" + config_hash,
        }
    )
    units = [
        EvalUnit(
            task,
            checkpoint.checkpoint_id,
            snapshots[task].snapshot_id,
            seed,
            checkpoint.initial_fitness,
        )
        for task, checkpoint in checkpoints.items()
        for seed in config["seeds"]
    ]
    trace_directory = Path(config["trace_directory"])
    trace_directory.mkdir(parents=True, exist_ok=True)

    def client_options(options):
        options = dict(options)
        api_key_env = options.pop("api_key_env", None)
        if "api_key" in options:
            raise ValueError(
                "reference API credentials by environment variable, not config"
            )
        return {
            **options,
            "api_key": os.environ.get(api_key_env) if api_key_env else None,
            "prompt_token_count": tokenizer.count_messages,
            "tokenizer_id": tokenizer.tokenizer_id,
        }

    with ExitStack() as resources:
        generation = OpenAICompatibleTransport(**client_options(config["generation"]))
        resources.callback(generation.close)
        semantic = OpenAICompatibleTransport(**client_options(config["semantic"]))
        resources.callback(semantic.close)
        if generation.model_version != protocol.model_version:
            raise ValueError("generation model version differs from the protocol")
        if semantic.model_version != generation.model_version:
            raise ValueError(
                "runtime v1 requires the same deployed tokenizer/model version for semantic and generation calls"
            )
        scorer_options = dict(config["scorer"])
        scorer_options["environment"] = {
            **scorer_options.get("environment", {}),
            "CUDA_VISIBLE_DEVICES": "",
        }
        scorer = SubprocessScorer(**scorer_options)

        @contextmanager
        def branch_factory(unit, action_hash):
            validate_inputs(config)
            branch_id = uuid4().hex
            evidence = json.dumps(
                {
                    "boundary": "isolated_state_and_private_cluster_if_retrieving",
                    "action_hash": action_hash,
                    "snapshot": unit.graph_snapshot,
                    "semantic_cache": "empty",
                    "graph_policy": "frozen_no_ingest",
                    "config_sha256": config_hash,
                },
                sort_keys=True,
            )
            # Lazy creation means the no-memory branch never starts Postgres.
            with ExitStack() as branch_resources:
                resource = {"stack": branch_resources, "pg": None, "retrieval": None}
                if routed:
                    resource.update(executor=None, memory_cache={}, retrievals=[])
                yield IsolatedBranch(branch_id, unit.graph_snapshot, resource, evidence)

        def executor(context):
            resource = context.isolated_branch.resource
            if routed and resource["executor"] is not None:
                # One private cluster per branch, reused by every round.
                return resource["executor"]
            if resource["pg"] is not None:
                raise ValueError(
                    "first-decision protocol must retrieve only once per branch"
                )
            pg = resource["stack"].enter_context(PrivatePostgres(**config["postgres"]))
            resource["pg"] = pg
            snapshots[context.unit.task_id].load_native(pg.connection)
            pg.connection.execute(
                "SELECT set_config('trimem.model',%s,false)", (semantic.model,)
            )
            lease = ExclusiveClusterLease(
                pg.system_identifier,
                "postgres",
                pg.backend_pid,
                context.isolated_branch.isolation_evidence,
                True,
                True,
            )
            collector = ExclusiveSemanticUsageCollector(
                lease=lease,
                connection_factory=pg.connect_worker,
                judge_factory=lambda ledger: MeteredSemanticJudge(
                    semantic,
                    ledger,
                    max_output_tokens=semantic.max_output_tokens,
                ),
            )
            executor = NativeExecutor(pg.connection, collector)
            if routed:
                resource["executor"] = executor
            return executor

        def trace_sink(context, state):
            target = trace_directory / (context.isolated_branch.branch_id + ".json")
            target.write_text(
                json.dumps(
                    {
                        "action_protocol": protocol.protocol_id,
                        "unit_id": context.unit.unit_id,
                        "branch_id": context.isolated_branch.branch_id,
                        "checkpoint_digests": state.checkpoint.digests,
                        "best_code": state.best_code,
                        "best_fitness": state.best_fitness,
                        "rounds": state.round,
                        "history": state.history,
                        "ledger": context.ledger.to_dict(),
                        "retrieval": context.isolated_branch.resource["retrieval"],
                        **routed_telemetry(context),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )

        continuation = MathAleContinuation(
            checkpoints,
            client=generation,
            scorer=scorer,
            render_chat=tokenizer.render,
            trace_sink=trace_sink,
        )
        native = NativeRetrievalAdapter(
            executor=executor,
            query_context=lambda context, state: snapshots[context.unit.task_id].query(
                context.unit.checkpoint_id,
            ),
            tokenizer=tokenizer,
            tokenizer_id=tokenizer.tokenizer_id,
            render_prompt=continuation.render_prompt,
            model_context_window=generation.model_context_window,
            max_output_tokens=generation.max_output_tokens,
        )

        def retrieve(action, context, state):
            result = native(action, context, state)
            context.isolated_branch.resource["retrieval"] = result.to_dict()
            return result

        direction = 1 if protocol.quality_direction == "maximize" else -1
        corpus_best = {}
        if routed:
            from .router import corpus_best_fitness

            corpus_best = {
                task: corpus_best_fitness(snapshot.document, task, direction)
                for task, snapshot in snapshots.items()
            }

        def fits_window(context, state, text):
            prompt = continuation.render_prompt(context, state, text)
            used = len(tokenizer.encode(prompt, add_special_tokens=False))
            return (
                used + generation.max_output_tokens <= generation.model_context_window
            )

        def routed_retrieve(policy, context, state):
            from .router import RoutedMemory, RoutedPolicy, online_features

            if not isinstance(policy, RoutedPolicy):
                raise TypeError("per-round routed mode requires a RoutedPolicy")
            resource = context.isolated_branch.resource
            cache = resource["memory_cache"]
            features = online_features(
                state.history,
                initial_fitness=context.unit.initial_fitness,
                best_before=state.best_fitness,
                corpus_best=corpus_best[context.unit.task_id],
                tokens_so_far=context.ledger.total_tokens,
                direction=direction,
            )
            choice = policy.choose(features, cached=cache)
            action = choice.action
            cached = cache.get(action.action_hash)
            hit = cached is not None and fits_window(context, state, cached.text)
            before = {event.request_id for event in context.ledger.events}
            if hit:
                result = cached
            else:
                result = native(action, context, state)
                cache[action.action_hash] = result
                resource["retrievals"].append(
                    {"round": state.round + 1, **result.to_dict()}
                )
            new = [e for e in context.ledger.events if e.request_id not in before]
            routing = {
                "action": action.to_dict(),
                "action_hash": action.action_hash,
                "features": features,
                "scores": choice.scores,
                "cache_hit": hit,
                # A cached action whose context no longer fits is re-retrieved.
                "cache_refit": cached is not None and not hit,
                "execution_id": result.receipt.execution_id,
                "packed_ids": list(result.packed.retained_ids),
                "receipt": None if hit else result.to_dict()["receipt"],
                "retrieval_request_ids": [e.request_id for e in new],
                "retrieval_tokens": sum(e.total_tokens for e in new),
            }
            return RoutedMemory(result.text, routing)

        def routed_telemetry(context):
            resource = context.isolated_branch.resource
            if not routed:
                return {}
            return {"retrievals": resource["retrievals"]}

        base_runner = (
            continuation.runner(routed_retrieve, retrieve_each_step=True)
            if routed
            else continuation.runner(retrieve)
        )

        def runner(action, context):
            result = base_runner(action, context)
            return replace(
                result,
                telemetry={
                    **result.telemetry,
                    "retrieval": context.isolated_branch.resource["retrieval"],
                    "application_mode": config["application_mode"],
                    "graph_policy": config["graph_policy"],
                    "runtime_config_sha256": config_hash,
                    **(
                        {**routed_telemetry(context), "policy": action.to_dict()}
                        if routed
                        else {}
                    ),
                },
            )

        yield CohortEvaluator(protocol, units, runner, branch_factory)
