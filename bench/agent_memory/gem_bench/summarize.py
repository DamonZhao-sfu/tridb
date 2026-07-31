"""Turn one operating point's raw records into [AM] §4.1 / §4.2 / §4.8 numbers.

Nothing here measures anything; it only aggregates what :mod:`.runner` recorded.
Kept separate so a completed run can be re-summarised from ``predictions.jsonl``
without re-serving 300 queries.

Three sections, each pinned to the figure it feeds:

``section_4_1``  Fig. 2 — accuracy against mean QA wallclock per query.
                 Construction is EXCLUDED, per the paper's own caption.
``section_4_2``  Fig. 3 / Fig. 4 / Table 3 — the construction vs retrieval vs
                 generation split, lifecycle calls and tokens, joules, and
                 joules per correct answer.
``section_4_8``  Fig. 10 / Fig. 11 — effective TTFT against total user wait, and
                 tail width as p95/p50.

Every energy figure is ``None`` when NVML was unavailable, and ``partial`` when
some window went unsampled. A summary never silently substitutes zero for an
unmeasured joule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Sequence

from bench.agent_memory.energy import GpuEnergySampler
from bench.agent_memory.gem_bench.points import OperatingPoint
from bench.agent_memory.serving import (
    CallLedger,
    _wilson_interval,
    judge_protocol_label,
    latency_summary,
)

SCHEMA_VERSION = "tridb_gem_longmemeval_v0.1.0"


def _energy_total(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum the joules of a set of phase windows, reporting what was missed."""
    joules = 0.0
    measured = 0
    missing = 0
    partial_coverage = 0
    for record in records:
        window = record.get("energy") or {}
        value = window.get("joules")
        if value is None:
            missing += 1
            continue
        joules += float(value)
        measured += 1
        if float(window.get("coverage") or 0.0) < 0.999:
            partial_coverage += 1
    return {
        "joules": None if measured == 0 else joules,
        "windows_measured": measured,
        "windows_missing": missing,
        "windows_partial_coverage": partial_coverage,
        "complete": missing == 0 and partial_coverage == 0 and measured > 0,
    }


def _sum_costs(records: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    """Fold GEM ``PhaseCost`` dicts, so the operator's own meter is reported too."""
    fields = (
        "llm_calls",
        "embed_calls",
        "embed_sequences",
        "prompt_tokens",
        "completion_tokens",
        "embed_input_tokens",
        "db_statements",
    )
    totals = dict.fromkeys(fields, 0)
    for record in records:
        cost = record.get(key) or {}
        for name in fields:
            totals[name] += int(cost.get(name, 0) or 0)
    return totals


def summarize_point(
    *,
    point: OperatingPoint,
    args: argparse.Namespace,
    predictions: Sequence[dict[str, Any]],
    judge_results: Sequence[dict[str, Any]],
    ledger: CallLedger,
    construction_records: Sequence[dict[str, Any]],
    maintenance_records: Sequence[dict[str, Any]],
    lifecycle_seconds: float,
    sampler: GpuEnergySampler,
) -> dict[str, Any]:
    timings = [prediction["timing"] for prediction in predictions]
    queries = len(predictions)

    judged = [result for result in judge_results if "correct" in result]
    correct = sum(bool(result["correct"]) for result in judged)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for result in judged:
        by_type[str(result["question_type"])].append(bool(result["correct"]))

    construction_seconds = sum(
        float(record["construction_seconds"]) for record in construction_records
    )
    maintenance_seconds = sum(
        float(record["seconds"]) for record in maintenance_records
    )
    qa_seconds = sum(float(timing["total_seconds"]) for timing in timings)
    retrieval_seconds = sum(
        float(timing["retrieval_phase_seconds"]) for timing in timings
    )
    generation_seconds = sum(float(timing["generation_seconds"]) for timing in timings)

    construction_energy = _energy_total(construction_records)
    qa_energy = _energy_total(predictions)
    maintenance_energy = _energy_total(maintenance_records)
    lifecycle_joules = None
    if construction_energy["joules"] is not None and qa_energy["joules"] is not None:
        lifecycle_joules = construction_energy["joules"] + qa_energy["joules"]
        if maintenance_energy["joules"] is not None:
            lifecycle_joules += maintenance_energy["joules"]

    calls = ledger.summary()
    tokens = ledger.tokens()
    accuracy = None if not judged else correct / len(judged)

    # The operator's own meter and the HTTP ledger count the same embed calls by
    # two independent routes. Disagreement means one of them is wrong, and a
    # silent disagreement would corrupt Table 3's Calls column — so it is
    # reported rather than reconciled.
    operator_construction_cost = _sum_costs(construction_records, "ingest_cost")
    operator_retrieval_cost = _sum_costs(predictions, "retrieval_cost")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operating_point": point.to_dict(),
        "workload": {
            "histories": len(construction_records),
            "questions": queries,
            "top_k": args.top_k,
            "max_prompt_memories": args.max_prompt_memories,
            "prompt_token_budget": args.prompt_token_budget,
            "chunk_size_tokens": args.chunk_size,
            "term_cond": args.term_cond,
            "serial_execution": True,
        },
        "accuracy": {
            "judge_model": None if args.skip_judge else args.judge_model,
            "judge_protocol": judge_protocol_label(
                enabled=not args.skip_judge,
                model=args.judge_model,
                base_url=args.judge_base_url,
            ),
            "judged": len(judged),
            "correct": correct,
            "accuracy": accuracy,
            "wilson_95": _wilson_interval(correct, len(judged)),
            "by_question_type": {
                question_type: {
                    "count": len(values),
                    "correct": sum(values),
                    "accuracy": sum(values) / len(values),
                }
                for question_type, values in sorted(by_type.items())
            },
        },
        "walltime": {
            "lifecycle_seconds": lifecycle_seconds,
            "construction_seconds": construction_seconds,
            "qa_seconds": qa_seconds,
            "maintenance_seconds": maintenance_seconds,
            "construction_plus_qa_seconds": construction_seconds + qa_seconds,
            "judge_seconds_excluded": sum(
                float(result.get("judge_seconds", 0)) for result in judge_results
            ),
            "by_history": list(construction_records),
        },
        "latency": {
            name: latency_summary(timing[field] for timing in timings)
            for name, field in (
                ("effective_ttft_seconds", "effective_ttft_seconds"),
                ("total_time_seconds", "total_seconds"),
                ("query_embedding_seconds", "query_embedding_seconds"),
                ("retrieval_seconds", "gem_retrieval_seconds"),
                ("retrieval_phase_seconds", "retrieval_phase_seconds"),
                ("prompt_assembly_seconds", "prompt_assembly_seconds"),
                ("vllm_queue_prefill_seconds", "vllm_queue_prefill_seconds"),
                ("generation_seconds", "generation_seconds"),
                ("decode_seconds", "decode_seconds"),
            )
        },
        "calls": calls,
        "tokens": tokens,
        "operator_meter": {
            "construction": operator_construction_cost,
            "retrieval": operator_retrieval_cost,
            "note": (
                "GEM PhaseCost counters, independent of the HTTP ledger. Query "
                "embedding is issued by the runner for the Fig. 10 breakdown, so "
                "retrieval.embed_calls is 0 by construction and the ledger's "
                "query_embedding count is authoritative for that call."
            ),
        },
        "energy": {
            "sampler": sampler.describe(),
            "construction": construction_energy,
            "qa": qa_energy,
            "maintenance": maintenance_energy,
            "lifecycle_joules": lifecycle_joules,
            "joules_per_correct": (
                None
                if lifecycle_joules is None or not correct
                else lifecycle_joules / correct
            ),
        },
        "construction_health": {
            "capped_histories": sum(
                1 for record in construction_records if record.get("capped")
            ),
            "rejected_units": sum(
                int(record.get("rejected", 0)) for record in construction_records
            ),
            "max_rejection_rate": args.max_rejection_rate,
            "note": (
                "[AM] §4.4: a construction model below an algorithm's capability "
                "floor corrupts the store rather than merely lowering accuracy. A "
                "run over max_rejection_rate is a FAILED CONFIGURATION, not a "
                "datapoint."
            ),
        },
    }

    summary["section_4_1"] = {
        "figure": "Fig. 2 — serving latency vs accuracy (construction excluded)",
        "accuracy": accuracy,
        "mean_qa_wallclock_per_query_seconds": (
            None if not queries else qa_seconds / queries
        ),
        "mean_retrieval_per_query_seconds": (
            None if not queries else retrieval_seconds / queries
        ),
        "mean_generation_per_query_seconds": (
            None if not queries else generation_seconds / queries
        ),
        "queries": queries,
    }
    summary["section_4_2"] = {
        "figure": "Fig. 3 / Fig. 4 / Table 3 — phase cost and lifecycle energy",
        "construction_wallclock_seconds": construction_seconds,
        "retrieval_per_query_seconds": (
            None if not queries else retrieval_seconds / queries
        ),
        "generation_per_query_seconds": (
            None if not queries else generation_seconds / queries
        ),
        "lifecycle_wallclock_seconds": construction_seconds + qa_seconds,
        "model_calls": calls["paper_model_calls"],
        "construction_calls": calls["construction_calls"],
        "qa_calls": calls["qa_calls"],
        "construction_tokens": (
            tokens["construction_prompt_tokens"]
            + tokens["construction_completion_tokens"]
        ),
        "qa_tokens": tokens["qa_prompt_tokens"] + tokens["qa_completion_tokens"],
        "construction_kilojoules": (
            None
            if construction_energy["joules"] is None
            else construction_energy["joules"] / 1000.0
        ),
        "qa_kilojoules": (
            None if qa_energy["joules"] is None else qa_energy["joules"] / 1000.0
        ),
        "total_kilojoules": (
            None if lifecycle_joules is None else lifecycle_joules / 1000.0
        ),
        "joules_per_correct": summary["energy"]["joules_per_correct"],
        "correct": correct,
    }
    ttft = summary["latency"]["effective_ttft_seconds"]
    total = summary["latency"]["total_time_seconds"]
    summary["section_4_8"] = {
        "figure": "Fig. 10 / Fig. 11 — effective TTFT and QA tail width",
        "ttft_p50_seconds": ttft.get("p50"),
        "total_p50_seconds": total.get("p50"),
        "post_first_token_streaming_p50_seconds": (
            None
            if ttft.get("p50") is None or total.get("p50") is None
            else total["p50"] - ttft["p50"]
        ),
        "qa_p50_seconds": total.get("p50"),
        "qa_p95_seconds": total.get("p95"),
        "qa_p95_over_p50": total.get("p95_over_p50"),
        "ttft_p95_over_p50": ttft.get("p95_over_p50"),
        # [AM] Insight 8's mechanism: an LLM-bounded phase widens the tail
        # because the model, not the corpus, decides when it is done.
        "bound_regime": (
            "llm_bounded" if point.uses_llm_construction else "algorithm_bounded"
        ),
        "iteration_caps": {
            "agentic_max_rounds": args.agentic_max_rounds,
            "agentic_max_tool_calls": args.agentic_max_tool_calls,
            "retrieval_term_cond": args.term_cond,
        },
    }
    return summary


def paper_sections(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cross-arm view: the three sections, one row per operating point."""

    def rows(section: str) -> list[dict[str, Any]]:
        return [
            {
                "operating_point": summary["operating_point"]["key"],
                "paradigm": summary["operating_point"]["paradigm"],
                "label": summary["operating_point"]["label"],
                **{
                    key: value
                    for key, value in summary[section].items()
                    if key != "figure"
                },
            }
            for summary in summaries
        ]

    def spread(section: str, field: str) -> dict[str, Any]:
        values = [
            float(row[field])
            for row in rows(section)
            if row.get(field) is not None and float(row[field]) > 0
        ]
        if len(values) < 2:
            return {"min": None, "max": None, "ratio": None, "arms": len(values)}
        return {
            "min": min(values),
            "max": max(values),
            "ratio": max(values) / min(values),
            "arms": len(values),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "TriDB/GEM operating points only. Each arm is a GEM SETTING chosen to "
            "reproduce an [AM] paradigm's cost shape, never the named system. "
            "These rows belong beside [AM]'s paradigm claims (Insight 1, 2, 8), "
            "not beside its per-system bars."
        ),
        "section_4_1": {
            "figure": "Fig. 2 — serving latency vs accuracy (construction excluded)",
            "long_context_baseline": (
                "not run — this reproduction measures serving latency only"
            ),
            "rows": rows("section_4_1"),
            "serving_latency_spread": spread(
                "section_4_1", "mean_qa_wallclock_per_query_seconds"
            ),
        },
        "section_4_2": {
            "figure": "Fig. 3 / Fig. 4 / Table 3 — phase cost and lifecycle energy",
            "rows": rows("section_4_2"),
            "construction_wallclock_spread": spread(
                "section_4_2", "construction_wallclock_seconds"
            ),
            "lifecycle_energy_spread": spread("section_4_2", "total_kilojoules"),
            "energy_per_correct_spread": spread("section_4_2", "joules_per_correct"),
        },
        "section_4_8": {
            "figure": "Fig. 10 / Fig. 11 — effective TTFT and QA tail width",
            "rows": rows("section_4_8"),
            "ttft_spread": spread("section_4_8", "ttft_p50_seconds"),
        },
        "section_4_7": (
            "not reproduced — per-user footprint scaling from 64K to 1M tokens is "
            "out of scope for this run"
        ),
    }
