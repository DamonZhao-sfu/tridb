"""Locked 1k/10k/100k/1M retrieval-only systems-scale experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from bench.agent_memory.evomembench.dataset import PINNED_REVISION, load_crossep_know
from bench.agent_memory.evomembench.injection import fit_injection
from bench.agent_memory.evomembench.modeling import experience_query
from bench.agent_memory.evomembench.multi_system import (
    LiveMultiSystemExperienceStore,
    MultiSystemConfig,
    MultiSystemQuery,
)
from bench.agent_memory.evomembench.run_pilot import _append_jsonl, _load_selected
from bench.agent_memory.evomembench.run_systems import _footprint
from bench.agent_memory.evomembench.scale import (
    BatchExperienceStrategy,
    ScaleRecord,
    iter_scaled_experiences,
)
from bench.agent_memory.evomembench.scale_snapshot import export_gem_scale_snapshot
from bench.agent_memory.evomembench.system_protocol import (
    balanced_arm_order,
    FrozenTokenizerCounter,
    SystemTrace,
    compare_parity,
    embedding_sha256,
    require_formal_authorization,
    require_parity,
    sha256_text,
)
from bench.agent_memory.evomembench.task_signature import knowledge_task_signature
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import InteractionEvent, RetrievalMode
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)


SCALE_SCHEMA_VERSION = "evomembench_gem_system_scale_v0.1.0"


def _batches(values: Iterable[ScaleRecord], size: int) -> Iterable[list[ScaleRecord]]:
    batch: list[ScaleRecord] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_batch(
    memory: TriDBGovernedMemory,
    records: Sequence[ScaleRecord],
    *,
    valid_from: str,
) -> dict[str, int | float]:
    grouped: dict[str, list[ScaleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.unit.scope_id].append(record)
    observed: dict[str, int | float] = {
        "experience_records": len(records),
        "units_created": 0,
        "edges_created": 0,
        "seconds": 0.0,
        "llm_calls": 0,
        "embedding_calls": 0,
        "extraction_input_tokens": 0,
        "extraction_output_tokens": 0,
        "embedding_input_tokens": 0,
    }
    for scope_id, scoped in grouped.items():
        events = [
            InteractionEvent(
                scope_id=scope_id,
                external_id=record.unit.source_external_ids[0],
                content=record.unit.memory_payload,
                kind="systems_decoy",
                event_time=valid_from,
                event_order=record.index,
            )
            for record in scoped
        ]
        result = memory.ingest(
            events,
            strategy=BatchExperienceStrategy(
                [record.unit for record in scoped], valid_from=valid_from
            ),
            scope_id=scope_id,
        )
        if not result.committed:
            raise RuntimeError(result.aborted_reason)
        observed["units_created"] += result.delta.units_created
        observed["edges_created"] += result.delta.edges_created
        observed["seconds"] += result.cost.seconds
        observed["llm_calls"] += result.cost.llm_calls
        observed["embedding_calls"] += result.cost.embed_calls
        observed["extraction_input_tokens"] += result.cost.prompt_tokens
        observed["extraction_output_tokens"] += result.cost.completion_tokens
        observed["embedding_input_tokens"] += result.cost.embed_input_tokens
    return observed


def _queries(source_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "evomembench_scale_queries_v0.1.0":
        raise ValueError("unsupported scale query manifest")
    if manifest.get("source_revision") != PINNED_REVISION:
        raise ValueError("scale query manifest source revision is not pinned")
    relative = Path(str(manifest["dataset_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe scale query dataset path")
    source = (source_root / relative).resolve()
    if source_root.resolve() not in source.parents:
        raise ValueError("scale query dataset escapes source root")
    corpus = load_crossep_know(source)
    by_uid = {
        episode.episode_uid: episode
        for context in corpus.contexts
        for episode in context.episodes
    }
    ids = [str(value) for value in manifest["episode_uids"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("scale query manifest must pin 100 unique IDs")
    encoded_ids = json.dumps(
        ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded_ids).hexdigest() != manifest.get("episode_uids_sha256"):
        raise ValueError("scale query manifest ID digest mismatch")
    missing = sorted(set(ids) - set(by_uid))
    if missing:
        raise ValueError(f"scale query manifest IDs are absent: {missing[:3]}")
    return [
        {
            "query_id": uid,
            "task_signature": knowledge_task_signature(by_uid[uid]),
        }
        for uid in ids
    ]


def _namespace(run_id: str, history_size: int, digest: str) -> str:
    token = hashlib.sha256(f"{run_id}:{history_size}:{digest}".encode()).hexdigest()[
        :16
    ]
    return f"scale_{history_size}_{token}"


def _gem_once(
    memory: TriDBGovernedMemory,
    *,
    scope_id: str,
    raw: dict[str, Any],
    history_size: int,
    args: argparse.Namespace,
    count_tokens: FrozenTokenizerCounter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = experience_query(
        scope_id=scope_id,
        task_signature=str(raw["task_signature"]),
        cutoff_ordinal=history_size + 1,
        mode=RetrievalMode.FUSED,
        top_k=args.k,
        hops=args.hops,
        m_seeds=args.m_seeds,
        term_cond=args.term_cond,
        reinforce=False,
        validity_states=("active",),
    )
    query = replace(
        query,
        text=None,
        embedding=tuple(raw["embedding"]),
        m_seeds=args.m_seeds,
        term_cond=args.term_cond,
    )
    began = time.perf_counter()
    result = memory.retrieve(query)
    database_ms = (time.perf_counter() - began) * 1000
    if not result.committed:
        raise RuntimeError(result.aborted_reason)
    probes = dict(result.probes)
    probe_read_ms = float(probes.get("instrumentation_probe_read_ms") or 0.0)
    probes["raw_database_retrieval_including_instrumentation_ms"] = database_ms
    database_ms = max(0.0, database_ms - probe_read_ms)
    unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
    assembly_started = time.perf_counter()
    items = _load_selected(memory, unit_ids)
    accepted, injection, injection_tokens = fit_injection(
        items,
        max_items=args.k,
        token_budget=args.injection_token_budget,
        count_tokens=count_tokens,
    )
    assembly_ms = (time.perf_counter() - assembly_started) * 1000
    selected_ids = tuple(str(item["episode_uid"]) for item in accepted)
    return (
        {
            "selected_ids": selected_ids,
            "selected_ordinals": tuple(int(item["ordinal"]) for item in accepted),
            "injection": injection,
            "injection_tokens": injection_tokens,
        },
        {
            "database_ms": database_ms,
            "assembly_ms": assembly_ms,
            "total_ms": database_ms + assembly_ms,
            "probes": probes,
            "cost": asdict(result.cost),
        },
    )


def _stable_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(result["selected_ids"]), sha256_text(str(result["injection"])))


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"median": 0.0, "p95": 0.0, "p99": 0.0}

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {"median": median, "p95": percentile(0.95), "p99": percentile(0.99)}


def _assemble_long_context(
    payloads: Sequence[str], task_signature: str
) -> tuple[str, str, float]:
    began = time.perf_counter()
    history = "\n\n".join(payloads)
    prompt = (
        "Use the complete, time-ordered cross-session memory below to solve the "
        "current task. Do not omit or truncate prior episodes.\n\n"
        f"Prior cross-session memory:\n{history}\n\n"
        f"Current task:\n{task_signature}"
    )
    return history, prompt, (time.perf_counter() - began) * 1000


def _long_context_point(
    memory: TriDBGovernedMemory,
    *,
    scope_id: str,
    history_size: int,
    count_tokens: FrozenTokenizerCounter,
    args: argparse.Namespace,
    query_rows: Sequence[dict[str, Any]],
    output: Path,
    prior_overflow: bool,
) -> dict[str, Any]:
    if prior_overflow:
        return {
            "history_size": history_size,
            "status": "context_overflow_inherited",
            "truncation": False,
            "assembly_repetitions": 0,
        }
    fetch_started = time.perf_counter()
    rows = memory.store.conn.execute(
        "SELECT fv.value FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id=u.id"
        " WHERE u.scope_id=%s AND u.state='active'"
        " AND u.metadata->>'node_kind'='experience'"
        " AND u.metadata->>'validity_state'='active'"
        " AND fv.field='memory_payload' AND fv.valid_to IS NULL ORDER BY u.id",
        (scope_id,),
    ).fetchall()
    source_fetch_ms = (time.perf_counter() - fetch_started) * 1000
    payloads = tuple(str(row[0]) for row in rows)
    preview_history, preview_prompt, preview_assembly_ms = _assemble_long_context(
        payloads, str(query_rows[0]["task_signature"])
    )
    token_started = time.perf_counter()
    preview_tokens = count_tokens(preview_prompt)
    preview_tokenization_ms = (time.perf_counter() - token_started) * 1000
    projected = preview_tokens + args.answer_max_tokens + args.context_safety_tokens
    overflow = projected > args.context_window_tokens
    common = {
        "history_size": history_size,
        "eligible_history_episodes": len(rows),
        "history_bytes": len(preview_history.encode()),
        "history_sha256": sha256_text(preview_history),
        "preview_prompt_tokens": preview_tokens,
        "projected_context_tokens": projected,
        "context_window_tokens": args.context_window_tokens,
        "source_fetch_ms_excluded": source_fetch_ms,
        "preview_assembly_ms": preview_assembly_ms,
        "preview_tokenization_ms": preview_tokenization_ms,
        "truncation": False,
        "token_source": "frozen_answer_tokenizer_json_exact_rendered_scale_prompt",
    }
    if overflow:
        return {
            **common,
            "status": "context_overflow",
            "assembly_repetitions": 1,
            "measured_queries": 0,
        }

    assembly_values: list[float] = []
    tokenization_values: list[float] = []
    total_values: list[float] = []
    for raw in query_rows:
        stable_tokens: int | None = None
        for _ in range(args.warmups):
            _history, prompt, _assembly_ms = _assemble_long_context(
                payloads, str(raw["task_signature"])
            )
            token_started = time.perf_counter()
            observed_tokens = count_tokens(prompt)
            _tokenization_ms = (time.perf_counter() - token_started) * 1000
            stable_tokens = stable_tokens or observed_tokens
            if observed_tokens != stable_tokens:
                raise RuntimeError("Long Context token count is not deterministic")
        for repetition in range(args.repetitions):
            history, prompt, assembly_ms = _assemble_long_context(
                payloads, str(raw["task_signature"])
            )
            token_started = time.perf_counter()
            observed_tokens = count_tokens(prompt)
            tokenization_ms = (time.perf_counter() - token_started) * 1000
            stable_tokens = stable_tokens or observed_tokens
            if observed_tokens != stable_tokens:
                raise RuntimeError("Long Context token count is not deterministic")
            total_ms = assembly_ms + tokenization_ms
            assembly_values.append(assembly_ms)
            tokenization_values.append(tokenization_ms)
            total_values.append(total_ms)
            _append_jsonl(
                output / "traces" / f"long_context_{history_size}.jsonl",
                SystemTrace(
                    run_id=args.run_id,
                    track="Systems-Scale",
                    arm="long_context",
                    target_id=f"{raw['query_id']}:{repetition}",
                    scope_id=scope_id,
                    history_size=history_size,
                    status="complete",
                    latency_ms={
                        "prompt_assembly": assembly_ms,
                        "tokenization": tokenization_ms,
                        "total": total_ms,
                    },
                    tokens={
                        "answer_input": observed_tokens,
                        "projected_context": observed_tokens
                        + args.answer_max_tokens
                        + args.context_safety_tokens,
                    },
                    intermediate={
                        "eligible_history_episodes": len(payloads),
                        "history_bytes": len(history.encode()),
                        "peak_application_materialized_ids": len(payloads),
                    },
                    selected_ids=(),
                    injection_sha256=sha256_text(history),
                    probes={
                        "repetition": repetition,
                        "truncation": False,
                        "source_fetch_ms_excluded": source_fetch_ms,
                    },
                ).as_dict(),
            )
    return {
        **common,
        "status": "complete",
        "assembly_repetitions": len(query_rows) * (args.warmups + args.repetitions),
        "measured_queries": len(query_rows),
        "measured_rows": len(total_values),
        "latency_ms": {
            "prompt_assembly": _latency_summary(assembly_values),
            "tokenization": _latency_summary(tokenization_values),
            "total": _latency_summary(total_values),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_authorization(
        formal=bool(args.formal), authorization=str(args.authorization)
    )
    if args.graph_work_budget < 128:
        raise ValueError("graph-work budget must be at least 128")
    if args.warmups < 1 or args.repetitions < 1:
        raise ValueError("scale warmups and repetitions must be positive")
    if args.history_sizes != sorted(set(args.history_sizes)):
        raise ValueError("history sizes must be unique and increasing")
    if args.formal and args.history_sizes != [1000, 10000, 100000, 1000000]:
        raise ValueError("formal scale points are locked to 1k/10k/100k/1M")
    if args.formal and (args.warmups != 10 or args.repetitions != 30):
        raise ValueError(
            "formal scale repetitions are locked to 10 warmup + 30 measured"
        )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing scale output: {output}")
    output.mkdir(parents=True)
    source_root = Path(args.source_root)
    query_rows = _queries(source_root, Path(args.query_manifest))
    count_tokens = FrozenTokenizerCounter(args.tokenizer_json)
    ledger = CallLedger()
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.timeout,
        ledger=ledger,
    )
    embedder = PhasedEmbedder(embedding_client)
    with embedder.phase("query"):
        vectors = embedder.encode([str(row["task_signature"]) for row in query_rows])
    for row, vector in zip(query_rows, vectors, strict=True):
        row["embedding"] = [float(value) for value in vector]
        row["embedding_sha256"] = embedding_sha256(vector)
    (output / "frozen_queries.json").write_text(
        json.dumps(query_rows, ensure_ascii=False, indent=2) + "\n"
    )

    memory = TriDBGovernedMemory.connect(
        args.dsn, dim=args.embedding_dim, embedder=embedder
    )
    memory.init_schema()
    existing_units = int(
        memory.store.conn.execute("SELECT count(*) FROM gem_unit").fetchone()[0]
    )
    if existing_units:
        memory.close()
        raise ValueError(
            "systems-scale DSN must have an empty GEM store to avoid physical-index interference"
        )
    settings = memory.store.conn.execute(
        "SELECT set_config('tjs.graph_scoring', 'membership', false),"
        " set_config('tjs.graph_work_budget', %s, false)",
        (str(args.graph_work_budget),),
    ).fetchone()
    if tuple(settings) != ("membership", str(args.graph_work_budget)):
        raise RuntimeError(f"failed to lock TJS scale settings: {settings!r}")
    valid_from = datetime.now(timezone.utc).isoformat()
    loaded = 0
    points: list[dict[str, Any]] = []
    long_overflow = False
    began = time.time()
    try:
        for history_size in args.history_sizes:
            construction = {
                "experience_records": 0,
                "units_created": 0,
                "edges_created": 0,
                "seconds": 0.0,
                "llm_calls": 0,
                "embedding_calls": 0,
                "extraction_input_tokens": 0,
                "extraction_output_tokens": 0,
                "embedding_input_tokens": 0,
            }
            generated = iter_scaled_experiences(
                count=history_size - loaded,
                primary_scope=args.scope_id,
                seed=args.seed,
                start_index=loaded,
            )
            for batch_index, batch in enumerate(
                _batches(generated, args.ingest_batch_size)
            ):
                with embedder.phase("construction"):
                    observed = _load_batch(memory, batch, valid_from=valid_from)
                for key in construction:
                    construction[key] += observed[key]
                _append_jsonl(
                    output / "per_update.jsonl",
                    {
                        "schema_version": "evomembench_scale_update_v0.1.0",
                        "run_id": args.run_id,
                        "history_size_target": history_size,
                        "batch_index": batch_index,
                        "first_global_index": batch[0].index,
                        "last_global_index": batch[-1].index,
                        "kind_counts": {
                            kind: sum(record.kind == kind for record in batch)
                            for kind in sorted({record.kind for record in batch})
                        },
                        **observed,
                    },
                )
            loaded = history_size
            snapshot = export_gem_scale_snapshot(
                memory,
                scope_root=args.scope_id,
                output_dir=output / "snapshots" / str(history_size),
                fetch_size=args.snapshot_batch_size,
            )

            config = MultiSystemConfig(
                namespace=_namespace(args.run_id, history_size, snapshot.digest),
                ann_overfetch=args.ann_overfetch,
                milvus_hnsw_m=args.milvus_hnsw_m,
                milvus_ef_construction=args.milvus_ef_construction,
                milvus_ef=args.milvus_ef,
            )
            parity_passed = 0
            arm_order_schedule: dict[str, Any] = {}
            with LiveMultiSystemExperienceStore(config) as store:
                load_metrics = store.load_snapshot(snapshot)
                for query_index, raw in enumerate(query_rows):
                    reference: dict[str, Any] | None = None
                    multi_reference: Any | None = None
                    full_stable: tuple[Any, ...] | None = None
                    multi_stable: tuple[Any, ...] | None = None
                    query = MultiSystemQuery(
                        query_id=str(raw["query_id"]),
                        scope_id=args.scope_id,
                        embedding=tuple(raw["embedding"]),
                        cutoff_ordinal=history_size + 1,
                        k=args.k,
                        m_seeds=args.m_seeds,
                        hops=args.hops,
                        graph_work_budget=args.graph_work_budget,
                        token_budget=args.injection_token_budget,
                        validity_states=("active",),
                    )
                    warmup_orders = []
                    for warmup in range(args.warmups):
                        order = balanced_arm_order(
                            ("full_gem", "multi_system"),
                            block_index=query_index * args.warmups + warmup,
                            seed=args.seed + history_size,
                        )
                        warmup_orders.append(list(order))
                        for arm in order:
                            if arm == "full_gem":
                                observed_full, _timing = _gem_once(
                                    memory,
                                    scope_id=args.scope_id,
                                    raw=raw,
                                    history_size=history_size,
                                    args=args,
                                    count_tokens=count_tokens,
                                )
                                full_stable = full_stable or _stable_key(observed_full)
                                if _stable_key(observed_full) != full_stable:
                                    raise RuntimeError(
                                        "Full GEM warmup output is not deterministic"
                                    )
                                reference = observed_full
                            else:
                                observed_multi = store.query(
                                    query, count_tokens=count_tokens
                                )
                                key = (
                                    observed_multi.selected_ids,
                                    sha256_text(observed_multi.injection),
                                )
                                multi_stable = multi_stable or key
                                if key != multi_stable:
                                    raise RuntimeError(
                                        "multi-system warmup output is not deterministic"
                                    )
                                multi_reference = observed_multi
                    if reference is None or multi_reference is None:
                        raise RuntimeError(
                            "scale warmup did not execute both system arms"
                        )
                    parity = compare_parity(
                        expected_ids=reference["selected_ids"],
                        observed_ids=multi_reference.selected_ids,
                        expected_injection=reference["injection"],
                        observed_injection=multi_reference.injection,
                    )
                    _append_jsonl(
                        output / "parity" / f"scale_{history_size}.jsonl",
                        {"query_id": raw["query_id"], **parity.as_dict()},
                    )
                    require_parity(parity)
                    parity_passed += 1
                    measured_orders = []
                    for repetition in range(args.repetitions):
                        order = balanced_arm_order(
                            ("full_gem", "multi_system"),
                            block_index=query_index * args.repetitions + repetition,
                            seed=args.seed + history_size,
                        )
                        measured_orders.append(list(order))
                        for arm in order:
                            if arm == "full_gem":
                                observed_full, timing = _gem_once(
                                    memory,
                                    scope_id=args.scope_id,
                                    raw=raw,
                                    history_size=history_size,
                                    args=args,
                                    count_tokens=count_tokens,
                                )
                                if _stable_key(observed_full) != full_stable:
                                    raise RuntimeError(
                                        "Full GEM measured output is not deterministic"
                                    )
                                probes = timing["probes"]
                                _append_jsonl(
                                    output
                                    / "traces"
                                    / f"full_gem_{history_size}.jsonl",
                                    SystemTrace(
                                        run_id=args.run_id,
                                        track="Systems-Scale",
                                        arm="full_gem",
                                        target_id=f"{raw['query_id']}:{repetition}",
                                        scope_id=args.scope_id,
                                        history_size=history_size,
                                        status="complete",
                                        latency_ms={
                                            "database_retrieval": timing["database_ms"],
                                            "prompt_assembly": timing["assembly_ms"],
                                            "total": timing["total_ms"],
                                        },
                                        tokens={
                                            "memory_injection": observed_full[
                                                "injection_tokens"
                                            ]
                                        },
                                        intermediate={
                                            "vector_candidates_examined": probes.get(
                                                "candidates_examined"
                                            ),
                                            "graph_edges_examined": probes.get(
                                                "graph_examined"
                                            ),
                                            "graph_reached": probes.get(
                                                "graph_reached"
                                            ),
                                            "relational_candidates_examined": probes.get(
                                                "relational_candidates_examined"
                                            ),
                                            "relational_candidates_survived": probes.get(
                                                "relational_candidates_passed"
                                            ),
                                            "peak_application_materialized_ids": len(
                                                observed_full["selected_ids"]
                                            ),
                                            "final_results": len(
                                                observed_full["selected_ids"]
                                            ),
                                            "cross_process_transfer_bytes": 0,
                                        },
                                        selected_ids=tuple(
                                            observed_full["selected_ids"]
                                        ),
                                        injection_sha256=sha256_text(
                                            observed_full["injection"]
                                        ),
                                        probes={
                                            **probes,
                                            "repetition": repetition,
                                            "snapshot_digest": snapshot.digest,
                                            "arm_order_position": list(order).index(
                                                arm
                                            ),
                                        },
                                    ).as_dict(),
                                )
                            else:
                                observed_multi = store.query(
                                    query, count_tokens=count_tokens
                                )
                                key = (
                                    observed_multi.selected_ids,
                                    sha256_text(observed_multi.injection),
                                )
                                if key != multi_stable:
                                    raise RuntimeError(
                                        "multi-system measured output is not deterministic"
                                    )
                                _append_jsonl(
                                    output
                                    / "traces"
                                    / f"multi_system_{history_size}.jsonl",
                                    SystemTrace(
                                        run_id=args.run_id,
                                        track="Systems-Scale",
                                        arm="multi_system",
                                        target_id=f"{raw['query_id']}:{repetition}",
                                        scope_id=args.scope_id,
                                        history_size=history_size,
                                        status="complete",
                                        latency_ms=observed_multi.latency_ms,
                                        tokens={
                                            "memory_injection": observed_multi.injection_tokens
                                        },
                                        intermediate=observed_multi.intermediate,
                                        selected_ids=observed_multi.selected_ids,
                                        injection_sha256=sha256_text(
                                            observed_multi.injection
                                        ),
                                        probes={
                                            **observed_multi.probes,
                                            "repetition": repetition,
                                            "snapshot_digest": snapshot.digest,
                                            "parity_passed": True,
                                            "arm_order_position": list(order).index(
                                                arm
                                            ),
                                        },
                                    ).as_dict(),
                                )
                    arm_order_schedule[str(raw["query_id"])] = {
                        "warmup": warmup_orders,
                        "measured": measured_orders,
                    }
                load_metrics["cleanup"] = store.cleanup_owned_namespace()

            order_path = output / "arm_order" / f"scale_{history_size}.json"
            order_path.parent.mkdir(parents=True, exist_ok=True)
            order_path.write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_balanced_arm_schedule_v0.1.0",
                        "unit": "query_repetition",
                        "seed": args.seed + history_size,
                        "schedule": arm_order_schedule,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )

            long_point = _long_context_point(
                memory,
                scope_id=args.scope_id,
                history_size=history_size,
                count_tokens=count_tokens,
                args=args,
                query_rows=query_rows,
                output=output,
                prior_overflow=long_overflow,
            )
            long_overflow = long_overflow or str(long_point["status"]).startswith(
                "context_overflow"
            )
            points.append(
                {
                    "history_size": history_size,
                    "construction": construction,
                    "snapshot": {
                        "digest": snapshot.digest,
                        "units": snapshot.unit_count,
                        "edges": snapshot.edge_count,
                        "database_sha256": snapshot.database_sha256,
                    },
                    "footprint": _footprint(memory),
                    "multi_system_load": load_metrics,
                    "parity_queries": parity_passed,
                    "arm_order_schedule": str(order_path.relative_to(output)),
                    "long_context": long_point,
                }
            )
    finally:
        memory.close()
    receipt = {
        "schema_version": SCALE_SCHEMA_VERSION,
        "status": "complete",
        "formal": bool(args.formal),
        "run_id": args.run_id,
        "history_sizes": args.history_sizes,
        "queries": len(query_rows),
        "warmups": args.warmups,
        "measured_repetitions": args.repetitions,
        "operating_point": {
            "k": args.k,
            "m_seeds": args.m_seeds,
            "hops": args.hops,
            "term_cond": args.term_cond,
            "graph_scoring": "membership",
            "graph_work_budget": args.graph_work_budget,
            "injection_token_budget": args.injection_token_budget,
        },
        "embedding_calls": ledger.summary(),
        "embedding_tokens": ledger.tokens(),
        "tokenizer": {"path": count_tokens.path, "sha256": count_tokens.sha256},
        "points": points,
        "agent_outcome_measured": False,
        "elapsed_seconds": time.time() - began,
    }
    (output / "run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--history-sizes", nargs="+", type=int, default=[1000, 10000, 100000, 1000000]
    )
    parser.add_argument("--ingest-batch-size", type=int, default=256)
    parser.add_argument("--snapshot-batch-size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--term-cond", type=int, default=32)
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--injection-token-budget", type=int, default=4096)
    parser.add_argument("--ann-overfetch", type=int, default=8)
    parser.add_argument("--milvus-hnsw-m", type=int, default=16)
    parser.add_argument("--milvus-ef-construction", type=int, default=200)
    parser.add_argument("--milvus-ef", type=int, default=128)
    parser.add_argument("--long-context-base-tokens", type=int, default=1024)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    parser.add_argument("--context-window-tokens", type=int, default=262144)
    parser.add_argument("--context-safety-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--authorization", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    preexisted = output.exists()
    try:
        run(args)
    except BaseException as exc:
        if not preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCALE_SCHEMA_VERSION,
                        "status": "failed",
                        "formal": bool(args.formal),
                        "run_id": args.run_id,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    main()
