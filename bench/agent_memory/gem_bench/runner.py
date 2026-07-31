"""Execute one operating point over LongMemEval_S*: construct, serve, grade.

Reproduces [AM] §4.1 (per-query serving latency vs accuracy, construction
excluded), §4.2 (the construction / retrieval / generation phase split and its
energy), and §4.8 (effective TTFT structure and tail width) for the TriDB/GEM
arms defined in :mod:`.points`.

The loop shape is the paper's: inject once per history, then query many.

    for each of five ~360 K-token histories
        construct   ingest (+ revise)        -> phase "construction"
        serve       60 x (retrieve, generate) -> phases "retrieval" / "generation"
        maintain    forget tick               -> phase "maintenance"

**Timing follows the embedRAG arm exactly**, down to the field names in
``timing``, because [AM] Fig. 2 and Fig. 10 compare arms and would otherwise be
comparing harnesses. Two consequences worth stating:

*Query embedding is issued here, not inside* ``retrieve``. Passing
``Query(embedding=...)`` splits pre-answer wait into embedding / retrieval /
assembly / queue+prefill the way Fig. 10 does. The embed call is still counted —
by the ledger rather than by the operator's meter — and the summary cross-checks
the two counters against each other rather than trusting either.

*A forget tick is neither construction nor serving.* [AM] Table 3 prices
"Construct + 300 QA"; a maintenance tick is a third thing, so it is timed,
reported, and kept out of both totals.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.energy import GpuEnergySampler
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.types import (
    Hit,
    InteractionEvent,
    Query,
)
from bench.agent_memory.gem_bench import points as pointsmod
from bench.agent_memory.serving import (
    CallLedger,
    LedgerChatExtractor,
    LongMemEvalWorkload,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
    _write_jsonl,
    build_judge_prompt,
    extract_retrieval_query,
    fit_answer_prompt,
    parse_judge_yes_no,
)

#: The deterministic strategy stores a chunk as a single ``content`` field. When
#: that is all a unit holds, the memory handed to the generator is the raw chunk
#: — byte-identical to what the embedRAG pipeline sends — so the G2 comparison
#: measures retrieval rather than prompt formatting.
RAW_CONTENT_FIELD = "content"


def hit_texts(hits: Sequence[Hit], limit: int) -> list[str]:
    """Render ranked hits into one memory string per unit, rank order preserved.

    ``retrieve`` returns one hit per (unit, field), so a unit carrying four
    fields would otherwise consume four of the generator's memory slots.
    """
    order: list[int] = []
    grouped: dict[int, dict[str, Any]] = {}
    for hit in hits:
        entry = grouped.get(hit.unit_id)
        if entry is None:
            entry = {"title": hit.title, "fields": []}
            grouped[hit.unit_id] = entry
            order.append(hit.unit_id)
        if hit.field_name and hit.value is not None:
            entry["fields"].append((hit.field_name, hit.value))

    texts: list[str] = []
    for unit_id in order[:limit]:
        entry = grouped[unit_id]
        fields = entry["fields"]
        if len(fields) == 1 and fields[0][0] == RAW_CONTENT_FIELD:
            texts.append(str(fields[0][1]))
            continue
        body = "\n".join(f"{name}: {value}" for name, value in fields)
        texts.append(f"{entry['title']}\n{body}" if body else str(entry["title"]))
    return texts


def _cost_dict(cost: Any) -> dict[str, Any]:
    return asdict(cost)


def _scope_for(point: pointsmod.OperatingPoint, workload: LongMemEvalWorkload) -> str:
    return f"{point.key}__{workload.scope_id}"


def connect_memory(
    args: argparse.Namespace,
    point: pointsmod.OperatingPoint,
    embedder: PhasedEmbedder,
) -> TriDBGovernedMemory:
    """One governed store per operating point.

    ``P_t`` is part of the configuration, not a constant: the paradigm proxies
    run with the engine's base policies so they stay faithful to systems that
    have no policy layer at all, while the GEM-conformant arm carries the seeded
    ``propagate-on-change`` / ``reinforce-on-read`` rules that make C3 and C6
    enforceable. Reporting one number across both without saying so would hide
    the cost of governance, which is the comparison that row exists for.
    """
    engine = (
        PolicyEngine(extra=seed_policies())
        if point.key == "gem_conformant"
        else PolicyEngine()
    )
    memory = TriDBGovernedMemory.connect(
        args.dsn,
        dim=args.embedding_dim,
        embedder=embedder,
        policy_engine=engine,
        max_rejection_rate=args.max_rejection_rate,
    )
    memory.init_schema()
    return memory


def construct_history(
    *,
    memory: TriDBGovernedMemory,
    point: pointsmod.OperatingPoint,
    workload: LongMemEvalWorkload,
    scope_id: str,
    history_index: int,
    args: argparse.Namespace,
    embedder: PhasedEmbedder,
    extractor: Any,
    sampler: GpuEnergySampler,
    lifecycle_started: float,
) -> dict[str, Any]:
    """Ingest one history (and revise it, when the point says so)."""
    from bench.agent_memory.demo.scenario import reset_scope

    # The native graph has no vertex delete; reset_scope tombstones active edges
    # before the relational rows cascade. Reimplementing that here would be a
    # second copy of a subtle invariant.
    reset_evidence = reset_scope(memory, scope_id)

    strategy = pointsmod.build_strategy(
        point,
        extractor=extractor,
        embedder=embedder,
        construction_model=args.answer_model,
        chunk_tokens=args.chunk_size,
        embedding_batch_size=args.embedding_batch_size,
        max_rounds=args.agentic_max_rounds,
        max_tool_calls=args.agentic_max_tool_calls,
    )

    event = InteractionEvent(
        scope_id=scope_id,
        external_id=f"{workload.scope_id}_history",
        content=workload.context,
        session_id=workload.scope_id,
        kind="history",
        metadata={
            "source": workload.source,
            "history_index": history_index,
            "operating_point": point.key,
        },
    )

    sampler.mark()  # bracket the phase exactly; see GpuEnergySampler.mark
    started = time.perf_counter()
    with embedder.phase("construction"):
        ingest_result = memory.ingest([event], strategy=strategy, scope_id=scope_id)
    ingest_ended = time.perf_counter()
    if not ingest_result.committed:
        raise RuntimeError(
            f"{point.key}/{scope_id} ingest aborted: {ingest_result.aborted_reason}"
        )

    revise_payload: dict[str, Any] | None = None
    if point.revise:
        evidence_ids = list(ingest_result.units)
        if args.revise_max_evidence:
            evidence_ids = evidence_ids[: args.revise_max_evidence]
        with embedder.phase("construction"):
            revision = memory.revise(
                scope_id,
                evidence=[{"unit_id": unit_id} for unit_id in evidence_ids],
                max_hops=args.revise_max_hops,
            )
        if not revision.committed:
            raise RuntimeError(
                f"{point.key}/{scope_id} revise aborted: {revision.aborted_reason}"
            )
        revise_payload = {
            "evidence_units": len(evidence_ids),
            "evidence_capped": bool(
                args.revise_max_evidence
                and len(ingest_result.units) > args.revise_max_evidence
            ),
            "delta": asdict(revision.delta),
            "cost": _cost_dict(revision.cost),
            "repairs": len(revision.repairs),
        }
    ended = time.perf_counter()
    sampler.mark()

    return {
        "history_index": history_index,
        "scope_id": scope_id,
        "source_scope": workload.scope_id,
        "questions": len(workload.questions),
        "reset": reset_evidence,
        "strategy": strategy.name,
        "strategy_variant": getattr(strategy, "mode", None),
        "construction_start_offset_seconds": started - lifecycle_started,
        "construction_end_offset_seconds": ended - lifecycle_started,
        "ingest_seconds": ingest_ended - started,
        "revise_seconds": ended - ingest_ended if point.revise else 0.0,
        "construction_seconds": ended - started,
        "units_created": ingest_result.delta.units_created,
        "fields_appended": ingest_result.delta.fields_appended,
        "edges_created": ingest_result.delta.edges_created,
        "ingest_cost": _cost_dict(ingest_result.cost),
        "revise": revise_payload,
        # [AM] Recommendation 10 / §4.4: a capped run and a run that hit the
        # capability floor are RECORDED operating points, never silent.
        "capped": bool(ingest_result.capped),
        "rejected": len(ingest_result.rejected),
        "rejected_examples": [dict(item) for item in ingest_result.rejected[:5]],
        "energy": sampler.window(started, ended).to_dict(),
    }


def serve_question(
    *,
    memory: TriDBGovernedMemory,
    point: pointsmod.OperatingPoint,
    question: Any,
    history_index: int,
    question_index: int,
    scope_id: str,
    args: argparse.Namespace,
    embedder: PhasedEmbedder,
    answer_client: OpenAIChatClient,
    token_counter: Any,
    sampler: GpuEnergySampler,
    lifecycle_started: float,
) -> dict[str, Any]:
    """One query: embed, retrieve, assemble, stream. Nothing is batched."""
    sampler.mark()  # bracket the phase exactly; see GpuEnergySampler.mark
    admitted = time.perf_counter()
    retrieval_query = extract_retrieval_query(question.question)

    query_embedding_started = time.perf_counter()
    with embedder.phase("query"):
        query_vector = embedder.encode([retrieval_query])[0]
    query_embedding_ended = time.perf_counter()
    if len(query_vector) != args.embedding_dim:
        raise RuntimeError(
            f"query embedding has {len(query_vector)} dimensions; "
            f"expected {args.embedding_dim}"
        )

    retrieval_started = time.perf_counter()
    result = memory.retrieve(
        Query(
            scope_id=scope_id,
            text=retrieval_query,
            embedding=query_vector,
            k=args.top_k,
            mode=point.mode,
            route=point.route,
            reinforce=point.reinforce,
            term_cond=args.term_cond,
        )
    )
    retrieval_ended = time.perf_counter()
    if not result.committed:
        raise RuntimeError(
            f"{point.key}/{question.question_id} retrieve aborted: "
            f"{result.aborted_reason}"
        )

    prompt_started = time.perf_counter()
    messages, prompt_hit_count, estimated_prompt_tokens = fit_answer_prompt(
        question.question,
        hit_texts(result.hits, args.max_prompt_memories),
        token_counter=token_counter,
        token_budget=args.prompt_token_budget,
    )
    prompt_ended = time.perf_counter()

    generation = answer_client.stream_chat(
        model=args.answer_model,
        messages=messages,
        max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    sampler.mark()

    timing = {
        "query_admitted_offset_seconds": admitted - lifecycle_started,
        "query_embedding_seconds": query_embedding_ended - query_embedding_started,
        "gem_retrieval_seconds": retrieval_ended - retrieval_started,
        # Named for the embedRAG arm's field so both summaries key alike.
        "tridb_retrieval_seconds": retrieval_ended - retrieval_started,
        "prompt_assembly_seconds": prompt_ended - prompt_started,
        "vllm_queue_prefill_seconds": (
            generation.first_token_at - generation.request_started
        ),
        "effective_ttft_seconds": generation.first_token_at - admitted,
        "decode_seconds": generation.completed_at - generation.first_token_at,
        "generation_seconds": generation.completed_at - generation.request_started,
        "retrieval_phase_seconds": (
            (query_embedding_ended - query_embedding_started)
            + (retrieval_ended - retrieval_started)
            + (prompt_ended - prompt_started)
        ),
        "total_seconds": generation.completed_at - admitted,
        "generation_first_token_offset_seconds": (
            generation.first_token_at - lifecycle_started
        ),
        "generation_last_token_offset_seconds": (
            generation.completed_at - lifecycle_started
        ),
    }

    unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
    return {
        "history_index": history_index,
        "question_index": question_index,
        "operating_point": point.key,
        "scope_id": scope_id,
        "question_id": question.question_id,
        "qa_pair_id": question.qa_pair_id,
        "question_type": question.question_type,
        "question": question.question,
        "answer": question.answer,
        "prediction": generation.text,
        "retrieval_query": retrieval_query,
        "retrieved_units": len(unit_ids),
        "retrieved_hits": len(result.hits),
        "retrieved": [
            {
                "rank": rank,
                "unit_id": hit.unit_id,
                "title": hit.title,
                "field": hit.field_name,
                "score": hit.score,
                "via": hit.via,
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
        "prompt_memory_count": prompt_hit_count,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "usage": generation.usage,
        "retrieval_cost": _cost_dict(result.cost),
        # Engine honesty counters, copied without renaming: a censored or
        # right-censored run is a different operating point, not a faster one.
        "probes": dict(result.probes),
        "timing": timing,
        "energy": sampler.window(admitted, generation.completed_at).to_dict(),
    }


def maintenance_tick(
    *,
    memory: TriDBGovernedMemory,
    scope_id: str,
    history_index: int,
    sampler: GpuEnergySampler,
) -> dict[str, Any]:
    sampler.mark()  # bracket the phase exactly; see GpuEnergySampler.mark
    started = time.perf_counter()
    result = memory.forget(scope_id)
    ended = time.perf_counter()
    sampler.mark()
    if not result.committed:
        raise RuntimeError(f"{scope_id} forget aborted: {result.aborted_reason}")
    return {
        "history_index": history_index,
        "scope_id": scope_id,
        "seconds": ended - started,
        "delta": asdict(result.delta),
        "cost": _cost_dict(result.cost),
        "demoted": len(result.demoted),
        "energy": sampler.window(started, ended).to_dict(),
    }


def grade(
    *,
    predictions: Sequence[dict[str, Any]],
    judge_client: OpenAIChatClient | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if judge_client is None:
        return []
    results: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        prompt = build_judge_prompt(
            prediction["question_type"],
            prediction["question"],
            prediction["answer"],
            prediction["prediction"],
            abstention="_abs" in prediction["question_id"],
        )
        raw, usage, judge_seconds = judge_client.chat(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
            seed=args.seed,
            ledger_kind="judge",
        )
        results.append(
            {
                "question_id": prediction["question_id"],
                "question_type": prediction["question_type"],
                "judge_model": args.judge_model,
                "correct": parse_judge_yes_no(raw),
                "raw": raw,
                "usage": usage,
                "judge_seconds": judge_seconds,
            }
        )
        _write_jsonl(output_dir / "judge_results.jsonl", results)
        print(
            f"[gem-judge] {index}/{len(predictions)} "
            f"{prediction['question_id']} correct={results[-1]['correct']}",
            file=sys.stderr,
            flush=True,
        )
    return results


def run_point(
    point: pointsmod.OperatingPoint,
    *,
    workloads: Sequence[LongMemEvalWorkload],
    args: argparse.Namespace,
    sampler: GpuEnergySampler,
    output_dir: Path,
) -> dict[str, Any]:
    """Construct, serve and grade every history for one operating point."""
    from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker

    ledger = CallLedger()
    answer_client = OpenAIChatClient(
        args.answer_base_url,
        args.answer_api_key,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.embedding_api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    embedder = PhasedEmbedder(embedding_client)
    extractor = LedgerChatExtractor(
        answer_client,
        model=args.answer_model,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.construction_max_tokens,
    )
    judge_client = (
        None
        if args.skip_judge
        else OpenAIChatClient(
            args.judge_base_url,
            args.judge_api_key,
            timeout=args.request_timeout,
            ledger=ledger,
        )
    )
    chunker = TiktokenSentenceChunker(
        chunk_size=args.chunk_size,
        tokenizer_model=args.chunk_tokenizer,
    )

    memory = connect_memory(args, point, embedder)
    predictions: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    maintenance_records: list[dict[str, Any]] = []
    lifecycle_started = time.perf_counter()
    try:
        for history_index, workload in enumerate(workloads):
            scope_id = _scope_for(point, workload)
            record = construct_history(
                memory=memory,
                point=point,
                workload=workload,
                scope_id=scope_id,
                history_index=history_index,
                args=args,
                embedder=embedder,
                extractor=extractor,
                sampler=sampler,
                lifecycle_started=lifecycle_started,
            )
            construction_records.append(record)
            _write_jsonl(output_dir / "construction.jsonl", construction_records)
            print(
                f"[gem/{point.key}] constructed {scope_id} "
                f"units={record['units_created']} "
                f"{record['construction_seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )

            for question_index, question in enumerate(workload.questions):
                prediction = serve_question(
                    memory=memory,
                    point=point,
                    question=question,
                    history_index=history_index,
                    question_index=question_index,
                    scope_id=scope_id,
                    args=args,
                    embedder=embedder,
                    answer_client=answer_client,
                    token_counter=chunker.count,
                    sampler=sampler,
                    lifecycle_started=lifecycle_started,
                )
                predictions.append(prediction)
                _write_jsonl(output_dir / "predictions.jsonl", predictions)
                timing = prediction["timing"]
                print(
                    f"[gem/{point.key}] {len(predictions)}/"
                    f"{sum(len(w.questions) for w in workloads)} "
                    f"{question.question_id} "
                    f"ttft={timing['effective_ttft_seconds']:.3f}s "
                    f"total={timing['total_seconds']:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

            if point.forget:
                maintenance_records.append(
                    maintenance_tick(
                        memory=memory,
                        scope_id=scope_id,
                        history_index=history_index,
                        sampler=sampler,
                    )
                )
                _write_jsonl(output_dir / "maintenance.jsonl", maintenance_records)
    finally:
        memory.close()

    lifecycle_seconds = time.perf_counter() - lifecycle_started
    judge_results = grade(
        predictions=predictions,
        judge_client=judge_client,
        args=args,
        output_dir=output_dir,
    )
    _write_jsonl(output_dir / "call_ledger.jsonl", ledger.records)

    from bench.agent_memory.gem_bench.summarize import summarize_point

    return summarize_point(
        point=point,
        args=args,
        predictions=predictions,
        judge_results=judge_results,
        ledger=ledger,
        construction_records=construction_records,
        maintenance_records=maintenance_records,
        lifecycle_seconds=lifecycle_seconds,
        sampler=sampler,
    )
