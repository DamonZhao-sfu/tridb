"""Formal-ready CrossEp-Know quality/system runner (does not auto-launch).

The live multi-system arm is measured by replaying the frozen Full-GEM snapshot
through :mod:`multi_system` and must pass injection parity before its latency is
admitted.  This runner creates the no-memory, strict long-context, and Full-GEM
agent outcomes needed by that replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

from bench.agent_memory.evomembench.dataset import EvoEpisode, load_crossep_know
from bench.agent_memory.evomembench.injection import fit_injection
from bench.agent_memory.evomembench.long_context import knowledge_history_item
from bench.agent_memory.evomembench.manifest import (
    load_manifest,
    manifest_sha256,
    verify_assets,
)
from bench.agent_memory.evomembench.modeling import (
    PreparedExperienceStrategy,
    knowledge_experience,
    sanitize_postgres_text,
)
from bench.agent_memory.evomembench.run_pilot import (
    VLLMTokenCounter,
    _append_jsonl,
    _retrieve,
    select_pilot_contexts,
)
from bench.agent_memory.evomembench.run_systems import _footprint
from bench.agent_memory.evomembench.system_protocol import (
    balanced_arm_order,
    HistoryItem,
    SystemTrace,
    append_history_to_messages,
    embedding_sha256,
    materialize_long_context,
    require_formal_authorization,
    sha256_text,
)
from bench.agent_memory.evomembench.system_snapshot import export_gem_snapshot
from bench.agent_memory.evomembench.vllm_metrics import (
    collect as collect_vllm_metrics,
    collect_attributable_delta,
)
from bench.agent_memory.evomembench.task_signature import knowledge_task_signature
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import InteractionEvent
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)


ARMS = ("memory_off", "long_context", "full_gem")


def _shard_contexts(
    contexts: Sequence[Any], *, shard_count: int, shard_index: int
) -> tuple[tuple[int, Any], ...]:
    if shard_count < 1:
        raise ValueError("context shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("context shard index must be within the shard count")
    selected = tuple(
        (index, context)
        for index, context in enumerate(contexts)
        if index % shard_count == shard_index
    )
    if not selected:
        raise ValueError("context shard is empty")
    return selected


def _base_prompt_tokens(episode: EvoEpisode, count_tokens: Any) -> int:
    return count_tokens.count_chat(
        [asdict(item) for item in episode.messages],
        chat_template_kwargs={"enable_thinking": False},
    )


def _event_time(ordinal: int) -> str:
    return (
        datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ordinal)
    ).isoformat()


def _gem_intermediate(probes: dict[str, Any], returned: int) -> dict[str, Any]:
    candidates = probes.get("candidates_examined")
    graph_examined = probes.get("graph_examined")
    operator_items_proxy = max(
        [
            returned,
            *([int(candidates)] if isinstance(candidates, (int, float)) else []),
            *(
                [int(graph_examined)]
                if isinstance(graph_examined, (int, float))
                else []
            ),
        ]
    )
    return {
        "vector_candidates_examined": candidates,
        "graph_edges_examined": graph_examined,
        "graph_reached": probes.get("graph_reached"),
        "relational_candidates_examined": probes.get("relational_candidates_examined"),
        "relational_candidates_survived": probes.get("relational_candidates_passed"),
        "peak_application_materialized_ids": returned,
        "peak_operator_items_proxy": operator_items_proxy,
        "peak_operator_items_proxy_is_exact": False,
        "final_results": returned,
        "cross_process_transfer_bytes": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_authorization(
        formal=bool(getattr(args, "formal", False)),
        authorization=str(getattr(args, "authorization", "")),
    )
    if args.graph_work_budget < 128:
        raise ValueError("tjs.graph_work_budget must be at least 128")
    context_shard_count = int(getattr(args, "context_shard_count", 1))
    context_shard_index = int(getattr(args, "context_shard_index", 0))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    manifest = load_manifest(args.manifest)
    verified = verify_assets(args.source_root, manifest)
    manifest_hash = manifest_sha256(manifest)
    source_path = Path(args.source_root) / manifest["assets"][0]["path"]
    corpus = load_crossep_know(source_path)
    selected_contexts = select_pilot_contexts(
        corpus.contexts, count=args.contexts, seed=args.seed
    )
    indexed_contexts = _shard_contexts(
        selected_contexts,
        shard_count=context_shard_count,
        shard_index=context_shard_index,
    )
    contexts = tuple(context for _index, context in indexed_contexts)
    global_index_by_uid = {
        context.context_uid: index for index, context in indexed_contexts
    }
    count_tokens = VLLMTokenCounter(
        args.tokenizer_base_url or args.answer_base_url.removesuffix("/v1"),
        args.answer_model,
        timeout=args.timeout,
    )
    ledger = CallLedger()
    embedder = PhasedEmbedder(
        OpenAIEmbeddingClient(
            args.embedding_base_url,
            args.api_key,
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            timeout=args.timeout,
            ledger=ledger,
        )
    )
    chat = OpenAIChatClient(
        args.answer_base_url, args.api_key, timeout=args.timeout, ledger=ledger
    )
    memory = TriDBGovernedMemory.connect(
        args.dsn, dim=args.embedding_dim, embedder=embedder
    )
    memory.init_schema()
    footprint_before = _footprint(memory)
    operating_point = memory.store.conn.execute(
        "SELECT set_config('tjs.graph_scoring', %s, false),"
        " set_config('tjs.graph_work_budget', %s, false)",
        (args.graph_scoring, str(args.graph_work_budget)),
    ).fetchone()
    if tuple(operating_point) != (
        args.graph_scoring,
        str(args.graph_work_budget),
    ):
        raise RuntimeError(f"failed to lock TJS operating point: {operating_point!r}")
    predictions = 0
    arm_order_schedule = {
        context.source_context_id: list(
            balanced_arm_order(
                args.arms,
                block_index=global_index_by_uid[context.context_uid],
                seed=args.seed,
            )
        )
        for context in contexts
    }
    (output / "arm_order_schedule.json").write_text(
        json.dumps(
            {
                "schema_version": "evomembench_balanced_arm_schedule_v0.1.0",
                "unit": "context",
                "seed": args.seed,
                "context_shard_count": context_shard_count,
                "context_shard_index": context_shard_index,
                "global_context_indices": {
                    context.source_context_id: global_index_by_uid[context.context_uid]
                    for context in contexts
                },
                "schedule": arm_order_schedule,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    began = time.time()
    try:
        for context_index, context in enumerate(contexts, 1):
            for arm in arm_order_schedule[context.source_context_id]:
                scope_id = f"{args.run_id}:{arm}:{context.context_uid}"
                history: list[HistoryItem] = []
                for episode in context.episodes:
                    query_started = time.perf_counter()
                    raw_signature = knowledge_task_signature(episode)
                    signature = sanitize_postgres_text(raw_signature)
                    task_signature_nul_replacements = raw_signature.count("\x00")
                    base_tokens = _base_prompt_tokens(episode, count_tokens)
                    selected_ids: list[str] = []
                    selected_ordinals: list[int] = []
                    probes: dict[str, Any]
                    retrieval_ms = 0.0
                    query_embedding_ms = 0.0
                    database_retrieval_ms = 0.0
                    query_embedding_tokens = 0
                    query_embedding: list[float] | None = None
                    if arm == "memory_off":
                        injection = ""
                        injection_tokens = 0
                        probes = {
                            "mode": "off",
                            "termination_reason": "memory_disabled",
                        }
                    elif arm == "long_context":
                        materialized = materialize_long_context(
                            history,
                            count_tokens=count_tokens,
                            base_prompt_tokens=base_tokens,
                            context_window_tokens=args.context_window_tokens,
                            reserved_generation_tokens=args.answer_max_tokens,
                            safety_tokens=args.context_safety_tokens,
                        )
                        injection = materialized.text
                        injection_tokens = materialized.history_tokens
                        selected_ids = list(materialized.episode_uids)
                        selected_ordinals = list(materialized.ordinals)
                        retrieval_ms = materialized.assembly_ms
                        probes = {
                            "mode": "complete_eligible_raw_history",
                            "history_episodes": len(selected_ids),
                            "history_bytes": materialized.history_bytes,
                            "truncation": False,
                            "context_overflow_provisional": materialized.overflow,
                        }
                    else:
                        if episode.ordinal > 0:
                            embedding_started = time.perf_counter()
                            with embedder.phase("query"):
                                query_embedding = embedder.encode([signature])[0]
                            query_embedding_ms = (
                                time.perf_counter() - embedding_started
                            ) * 1000
                            usage = (
                                ledger.records[-1].get("detail", {}).get("usage", {})
                            )
                            query_embedding_tokens = int(usage.get("prompt_tokens", 0))
                        items, probes, database_retrieval_ms = _retrieve(
                            memory,
                            arm="gem_fused",
                            scope_id=scope_id,
                            episode=episode,
                            top_k=args.top_k,
                            hops=args.hops,
                            m_seeds=args.m_seeds,
                            term_cond=args.term_cond,
                            query_embedding=query_embedding,
                        )
                        retrieval_ms = query_embedding_ms + database_retrieval_ms
                        probes = {
                            **probes,
                            "graph_scoring": args.graph_scoring,
                            "graph_work_budget": args.graph_work_budget,
                        }
                        accepted, injection, injection_tokens = fit_injection(
                            items,
                            max_items=args.top_k,
                            token_budget=args.injection_token_budget,
                            count_tokens=count_tokens,
                        )
                        if episode.ordinal > 0:
                            if not items:
                                raise RuntimeError(
                                    "full_gem retrieved no eligible history for "
                                    f"{episode.episode_uid} at ordinal {episode.ordinal}"
                                )
                            if not accepted or injection_tokens <= 0:
                                raise RuntimeError(
                                    "full_gem dropped all eligible history while "
                                    f"formatting {episode.episode_uid} at ordinal "
                                    f"{episode.ordinal}"
                                )
                        probes = {
                            **probes,
                            "experience_overflow_policy": "truncate_head_tail",
                            "truncated_experiences": sum(
                                bool(item.get("injection_truncated"))
                                for item in accepted
                            ),
                        }
                        selected_ids = [str(item["episode_uid"]) for item in accepted]
                        selected_ordinals = [int(item["ordinal"]) for item in accepted]

                    database_instrumentation_ms = float(
                        probes.get("instrumentation_probe_read_ms") or 0.0
                    )

                    serialization_started = time.perf_counter()
                    messages = append_history_to_messages(
                        [asdict(item) for item in episode.messages], injection
                    )
                    request_payload = {
                        "model": args.answer_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": args.answer_max_tokens,
                        "seed": args.seed,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    client_request_bytes = len(
                        json.dumps(request_payload, allow_nan=False).encode()
                    )
                    prompt_serialization_ms = (
                        time.perf_counter() - serialization_started
                    ) * 1000
                    tokenization_started = time.perf_counter()
                    answer_prompt_tokens = count_tokens.count_chat(
                        messages,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    prompt_tokenization_ms = (
                        time.perf_counter() - tokenization_started
                    ) * 1000
                    overflow = (
                        answer_prompt_tokens
                        + args.answer_max_tokens
                        + args.context_safety_tokens
                        > args.context_window_tokens
                    )
                    probes["context_overflow"] = overflow
                    if overflow:
                        generated_text = ""
                        generation_usage: dict[str, int] = {}
                        generation_seconds = 0.0
                        ttft_seconds = 0.0
                        status = "context_overflow"
                        model_metrics: dict[str, Any] = {
                            "attributable": False,
                            "reason": "context_overflow_no_model_request",
                            "telemetry_overhead_ms": 0.0,
                        }
                    else:
                        metrics_before = collect_vllm_metrics(
                            args.answer_base_url.removesuffix("/v1"),
                            model=args.answer_model,
                            timeout=args.timeout,
                        )
                        generated = chat.stream_chat(
                            model=args.answer_model,
                            messages=messages,
                            max_tokens=args.answer_max_tokens,
                            temperature=0.0,
                            seed=args.seed,
                        )
                        generated_text = generated.text
                        generation_usage = generated.usage
                        generation_seconds = (
                            generated.completed_at - generated.request_started
                        )
                        ttft_seconds = (
                            generated.first_token_at - generated.request_started
                        )
                        observed_prompt_tokens = int(
                            generation_usage.get("prompt_tokens", -1)
                        )
                        if observed_prompt_tokens != answer_prompt_tokens:
                            raise RuntimeError(
                                "live tokenizer/generation prompt token mismatch: "
                                f"{answer_prompt_tokens} != {observed_prompt_tokens}"
                            )
                        model_metrics = collect_attributable_delta(
                            args.answer_base_url.removesuffix("/v1"),
                            metrics_before,
                            model=args.answer_model,
                            expected_requests=1,
                            timeout=min(2.0, args.timeout),
                        )
                        if args.formal and model_metrics["attributable"] is not True:
                            raise RuntimeError(
                                "vLLM per-request timing attribution failed: "
                                f"{model_metrics}"
                            )
                        status = "complete"
                    e2e_ms_raw = (time.perf_counter() - query_started) * 1000
                    model_telemetry_overhead_ms = float(
                        model_metrics.get("telemetry_overhead_ms", 0.0)
                    )
                    telemetry_overhead_ms = (
                        model_telemetry_overhead_ms + database_instrumentation_ms
                    )
                    e2e_ms = max(0.0, e2e_ms_raw - telemetry_overhead_ms)
                    prompt_details = generation_usage.get("prompt_tokens_details") or {}
                    prefix_cache_hit_tokens = prompt_details.get("cached_tokens")
                    response_for_history = (
                        generated_text
                        if generated_text
                        else "[context_overflow: no model response]"
                    )
                    unit = knowledge_experience(
                        episode,
                        response=response_for_history,
                        scope_id=scope_id,
                    )
                    experience_nul_replacements = int(
                        unit.metadata.get("postgres_nul_replacements", 0)
                    )
                    probes["task_signature_nul_replacements"] = (
                        task_signature_nul_replacements
                    )
                    probes["experience_database_nul_replacements"] = (
                        experience_nul_replacements
                    )

                    prediction = {
                        "idx": episode.source_task_id,
                        "messages": [asdict(item) for item in episode.messages],
                        "model_output": generated_text,
                        "rubrics": list(episode.rubrics),
                        "metadata": {
                            "task_id": episode.source_task_id,
                            "context_id": context.source_context_id,
                            "context_category": episode.category,
                            "sub_category": episode.subcategory,
                            "ordinal": episode.ordinal,
                            "status": status,
                        },
                        "memory_type": arm,
                        "memory_retrieved": injection,
                        "stats": {
                            "retrieval_ms": retrieval_ms,
                            "injection_tokens": injection_tokens,
                            "generation_usage": generation_usage,
                            "generation_seconds": generation_seconds,
                            "model_ttft_seconds": ttft_seconds,
                            "answer_prompt_tokens": answer_prompt_tokens,
                            "answer_prompt_token_source": "live_vllm_chat_template",
                            "prompt_serialization_ms": prompt_serialization_ms,
                            "prompt_tokenization_ms": prompt_tokenization_ms,
                            "client_request_bytes": client_request_bytes,
                            "database_instrumentation_ms": database_instrumentation_ms,
                            "model_telemetry_overhead_ms": model_telemetry_overhead_ms,
                            "telemetry_overhead_ms": telemetry_overhead_ms,
                            "prefix_cache_hit_tokens": prefix_cache_hit_tokens,
                            "prefix_cache_hit_tokens_available": (
                                prefix_cache_hit_tokens is not None
                            ),
                            "context_overflow": overflow,
                            "task_signature_nul_replacements": (
                                task_signature_nul_replacements
                            ),
                            "experience_database_nul_replacements": (
                                experience_nul_replacements
                            ),
                            "experience_overflow_policy": (
                                "not_applicable"
                                if arm in {"memory_off", "long_context"}
                                else "truncate_head_tail"
                            ),
                            "truncated_experiences": int(
                                probes.get("truncated_experiences", 0)
                            ),
                            "injection_policy": (
                                "none"
                                if arm == "memory_off"
                                else (
                                    "complete_history_append_no_truncation"
                                    if arm == "long_context"
                                    else "bounded_memory_append"
                                )
                            ),
                        },
                    }
                    _append_jsonl(output / "predictions" / f"{arm}.jsonl", prediction)
                    intermediate = (
                        {
                            "database_candidates": None,
                            "history_episodes_materialized": len(selected_ids),
                            "history_prompt_bytes": len(injection.encode()),
                        }
                        if arm == "long_context"
                        else (
                            _gem_intermediate(probes, len(selected_ids))
                            if arm == "full_gem"
                            else {
                                "database_candidates": None,
                                "history_episodes_materialized": 0,
                            }
                        )
                    )
                    trace = SystemTrace(
                        run_id=args.run_id,
                        track="CrossEp-Know",
                        arm=arm,
                        target_id=episode.episode_uid,
                        scope_id=scope_id,
                        history_size=episode.ordinal,
                        status=status,
                        latency_ms={
                            "memory_or_prompt_assembly": retrieval_ms,
                            "query_embedding": query_embedding_ms,
                            "database_retrieval": database_retrieval_ms,
                            "prompt_serialization": prompt_serialization_ms,
                            "prompt_tokenization": prompt_tokenization_ms,
                            "model_ttft": ttft_seconds * 1000,
                            "model_prefill": float(
                                model_metrics.get("request_prefill_time_ms", 0.0)
                            ),
                            "model_decode": float(
                                model_metrics.get("request_decode_time_ms", 0.0)
                            ),
                            "model_server_end_to_end": float(
                                model_metrics.get("e2e_request_latency_ms", 0.0)
                            ),
                            "generation": generation_seconds * 1000,
                            "end_to_end": e2e_ms,
                        },
                        tokens={
                            "answer_prompt": int(
                                generation_usage.get(
                                    "prompt_tokens", answer_prompt_tokens
                                )
                            ),
                            "answer_completion": int(
                                generation_usage.get("completion_tokens", 0)
                            ),
                            "model_prefill": max(
                                0,
                                int(
                                    generation_usage.get(
                                        "prompt_tokens", answer_prompt_tokens
                                    )
                                )
                                - int(prefix_cache_hit_tokens or 0),
                            ),
                            "memory_injection": injection_tokens,
                            "query_embedding_input": query_embedding_tokens,
                            **(
                                {"prefix_cache_hit": int(prefix_cache_hit_tokens)}
                                if prefix_cache_hit_tokens is not None
                                else {}
                            ),
                        },
                        intermediate={
                            **intermediate,
                            "client_to_model_request_bytes": client_request_bytes,
                        },
                        selected_ids=tuple(selected_ids),
                        injection_sha256=sha256_text(injection),
                        probes={
                            **probes,
                            "context_id": context.source_context_id,
                            "context_category": episode.category,
                            "ordinal": episode.ordinal,
                            "vllm_timing": model_metrics,
                            "raw_end_to_end_including_telemetry_ms": e2e_ms_raw,
                        },
                    )
                    _append_jsonl(output / "traces" / f"{arm}.jsonl", trace.as_dict())
                    if arm == "full_gem" and episode.ordinal > 0:
                        if query_embedding is None:
                            raise RuntimeError("missing canonical task embedding")
                        _append_jsonl(
                            output / "queries" / "full_gem.jsonl",
                            {
                                "query_id": episode.episode_uid,
                                "scope_id": scope_id,
                                "cutoff_ordinal": episode.ordinal,
                                "task_signature": signature,
                                "task_signature_sha256": sha256_text(signature),
                                "task_embedding": query_embedding,
                                "task_embedding_sha256": embedding_sha256(
                                    query_embedding
                                ),
                                "k": args.top_k,
                                "m_seeds": args.m_seeds,
                                "hops": args.hops,
                                "term_cond": args.term_cond,
                                "graph_scoring": args.graph_scoring,
                                "graph_work_budget": args.graph_work_budget,
                                "token_budget": args.injection_token_budget,
                                "expected_ids": selected_ids,
                                "expected_ordinals": selected_ordinals,
                                "expected_injection": injection,
                                "expected_injection_sha256": sha256_text(injection),
                            },
                        )
                    predictions += 1

                    if arm == "long_context":
                        update_started = time.perf_counter()
                        raw_history = knowledge_history_item(
                            episode, response_for_history
                        )
                        history.append(raw_history)
                        _append_jsonl(
                            output / "updates" / f"{arm}.jsonl",
                            {
                                "episode_uid": episode.episode_uid,
                                "construction_seconds": time.perf_counter()
                                - update_started,
                                "construction_tokens": 0,
                                "embedding_tokens": 0,
                                "wal_bytes": 0,
                                "history_append_bytes": len(raw_history.text.encode()),
                            },
                        )
                    elif arm == "full_gem":
                        event_time = _event_time(episode.ordinal)
                        wal_before = memory.store.conn.execute(
                            "SELECT pg_current_wal_lsn()"
                        ).fetchone()[0]
                        with embedder.phase("construction"):
                            update = memory.ingest(
                                [
                                    InteractionEvent(
                                        scope_id=scope_id,
                                        external_id=episode.source_task_id,
                                        content=unit.memory_payload,
                                        session_id=episode.episode_uid,
                                        role="agent_trajectory",
                                        kind="experience",
                                        event_time=event_time,
                                        event_order=episode.ordinal,
                                    )
                                ],
                                strategy=PreparedExperienceStrategy(
                                    unit, valid_from=event_time
                                ),
                                scope_id=scope_id,
                            )
                        if not update.committed:
                            raise RuntimeError(update.aborted_reason)
                        wal_after = memory.store.conn.execute(
                            "SELECT pg_current_wal_lsn()"
                        ).fetchone()[0]
                        wal_bytes = int(
                            memory.store.conn.execute(
                                "SELECT pg_wal_lsn_diff(%s,%s)",
                                (wal_after, wal_before),
                            ).fetchone()[0]
                        )
                        _append_jsonl(
                            output / "updates" / f"{arm}.jsonl",
                            {
                                "episode_uid": episode.episode_uid,
                                "delta": asdict(update.delta),
                                "cost": asdict(update.cost),
                                "wal_bytes": wal_bytes,
                            },
                        )
                if arm == "full_gem":
                    snapshot_path = (
                        output
                        / "snapshots"
                        / "full_gem"
                        / f"{context.context_uid.rsplit(':', 1)[-1]}.json"
                    )
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    export_gem_snapshot(memory, scope_id=scope_id).write(snapshot_path)
                print(
                    f"[evomembench-system/{arm}] context "
                    f"{context_index}/{len(contexts)} {context.source_context_id}",
                    flush=True,
                )
        footprint_after = _footprint(memory)
    finally:
        memory.close()
    return {
        "schema_version": "evomembench_know_system_run_v0.1.0",
        "status": "complete",
        "formal": bool(args.formal),
        "run_id": args.run_id,
        "source_revision": corpus.source_revision,
        "manifest_sha256": manifest_hash,
        "verified_assets": verified,
        "contexts": [item.source_context_id for item in contexts],
        "selected_contexts_before_sharding": len(selected_contexts),
        "context_shard_count": context_shard_count,
        "context_shard_index": context_shard_index,
        "global_context_indices": {
            context.source_context_id: global_index_by_uid[context.context_uid]
            for context in contexts
        },
        "arms": list(args.arms),
        "arm_order_policy": "seeded_latin_rotation_by_context",
        "arm_order_schedule": arm_order_schedule,
        "predictions": predictions,
        "cross_episode_decisions_per_arm": sum(
            len(context.episodes) - 1 for context in contexts
        ),
        "prompt_policies": {
            "memory_off": "none",
            "long_context": "complete_history_append_no_truncation",
            "full_gem": "bounded_memory_append_with_head_tail_experience_overflow",
        },
        "experience_overflow_policy": "truncate_head_tail",
        "context_window_tokens": args.context_window_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "operating_point": {
            "top_k": args.top_k,
            "m_seeds": args.m_seeds,
            "hops": args.hops,
            "term_cond": args.term_cond,
            "graph_scoring": args.graph_scoring,
            "graph_work_budget": args.graph_work_budget,
            "injection_token_budget": args.injection_token_budget,
        },
        "model_calls": ledger.summary(),
        "database_footprint_before": footprint_before,
        "database_footprint_after": footprint_after,
        "elapsed_seconds": time.time() - began,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("protocol_manifest.json")),
    )
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="qwen3.8")
    parser.add_argument("--tokenizer-base-url", default=None)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--contexts", type=int, default=120)
    parser.add_argument("--context-shard-count", type=int, default=1)
    parser.add_argument("--context-shard-index", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--term-cond", type=int, default=32)
    parser.add_argument(
        "--graph-scoring", choices=("membership",), default="membership"
    )
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--injection-token-budget", type=int, default=4096)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    parser.add_argument("--context-window-tokens", type=int, default=262144)
    parser.add_argument("--context-safety-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="mark an explicitly authorized formal run; launchers gate this flag",
    )
    parser.add_argument("--authorization", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output_preexisted = output.exists()
    try:
        receipt = run(args)
    except BaseException as exc:
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_know_system_run_v0.1.0",
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
    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
