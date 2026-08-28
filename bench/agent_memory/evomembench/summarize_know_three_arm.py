"""Aggregate the v0.2 CrossEp-Know No-Memory/GEM/Polyglot experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from bench.agent_memory.evomembench.metrics import clustered_paired_effect
from bench.agent_memory.evomembench.system_protocol import verify_trace


AGENT_ARMS = ("memory_off", "full_gem")
REPORT_ARMS = ("no_memory", "gem", "polyglot")


def _rows(path: Path, *, traces: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if traces and not all(verify_trace(row) for row in rows):
        raise ValueError(f"trace digest verification failed: {path}")
    return rows


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values]
    return {
        "n": len(observed),
        "mean": mean(observed) if observed else None,
        "p50": median(observed) if observed else None,
        "p95": _quantile(observed, 0.95),
        "p99": _quantile(observed, 0.99),
    }


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    metadata = row["metadata"]
    return str(metadata["context_id"]), str(metadata["task_id"])


def _validate_shards(
    shard_dirs: Sequence[Path], polyglot_dirs: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(shard_dirs) != 2 or len(polyglot_dirs) != 2:
        raise ValueError("v0.2 protocol requires exactly two answer/context shards")
    receipts = [
        json.loads((path / "run_receipt.json").read_text()) for path in shard_dirs
    ]
    poly_receipts = [
        json.loads((path / "run_receipt.json").read_text()) for path in polyglot_dirs
    ]
    expected_indices = {0, 1}
    observed_indices = {int(row.get("context_shard_index", -1)) for row in receipts}
    if observed_indices != expected_indices:
        raise ValueError(f"context shard indices differ: {observed_indices}")
    context_ids: set[str] = set()
    for receipt in receipts:
        if receipt.get("status") != "complete":
            raise ValueError("agent shard is incomplete")
        if set(receipt.get("arms", [])) != set(AGENT_ARMS):
            raise ValueError("agent shard must contain only memory_off and full_gem")
        if int(receipt.get("context_shard_count", -1)) != 2:
            raise ValueError("agent shard does not declare two-way sharding")
        if int(receipt.get("selected_contexts_before_sharding", -1)) != 120:
            raise ValueError("agent shard was not selected from all 120 contexts")
        current = {str(value) for value in receipt.get("contexts", [])}
        if context_ids & current:
            raise ValueError("context appears in both answer shards")
        context_ids.update(current)
    if len(context_ids) != 120:
        raise ValueError(f"expected 120 unique contexts, got {len(context_ids)}")
    for receipt in poly_receipts:
        if receipt.get("status") != "complete" or not receipt.get("all_parity_passed"):
            raise ValueError("Polyglot replay is incomplete or failed parity")
        if float(receipt.get("parity_fraction", 0.0)) != 1.0:
            raise ValueError("Polyglot replay parity is not exactly 100%")
    return receipts, poly_receipts


def _arm_quality(
    by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    cross_keys: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    all_scores = [int(row["score"]) for row in by_key.values()]
    cross_scores = [int(by_key[key]["score"]) for key in cross_keys]
    contexts: dict[str, list[int]] = {}
    categories: dict[str, list[int]] = {}
    ordinals: dict[int, list[int]] = {}
    for key in cross_keys:
        row = by_key[key]
        metadata = row["metadata"]
        score = int(row["score"])
        contexts.setdefault(str(metadata["context_id"]), []).append(score)
        categories.setdefault(
            str(metadata.get("context_category", "unknown")), []
        ).append(score)
        ordinals.setdefault(int(metadata["ordinal"]), []).append(score)
    return {
        "episodes": len(all_scores),
        "reuse_decisions": len(cross_scores),
        "strict_rubric_accuracy": mean(all_scores),
        "cross_episode_accuracy": mean(cross_scores),
        "macro_context_accuracy": mean(mean(values) for values in contexts.values()),
        "accuracy_by_context_category": {
            key: {"n": len(values), "accuracy": mean(values)}
            for key, values in sorted(categories.items())
        },
        "accuracy_by_ordinal": {
            str(key): {"n": len(values), "accuracy": mean(values)}
            for key, values in sorted(ordinals.items())
        },
    }


def _trace_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {str(row["target_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("duplicate target ID in system traces")
    return mapped


def _token_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "answer_prompt",
        "answer_completion",
        "model_prefill",
        "memory_injection",
        "query_embedding_input",
        "prefix_cache_hit",
    )
    materialized = list(rows)
    return {
        key: sum(int(row.get("tokens", {}).get(key, 0)) for row in materialized)
        for key in keys
    }


def _paired_ratio(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired latency vectors differ in length")
    pairs = [(float(a), float(b)) for a, b in zip(left, right, strict=True) if a > 0]
    ratios = [b / a for a, b in pairs]
    return {
        "n": len(pairs),
        "ratio_right_over_left": _distribution(ratios),
        "absolute_difference_right_minus_left_ms": _distribution(
            b - a for a, b in pairs
        ),
        "left_faster_fraction": (
            sum(a < b for a, b in pairs) / len(pairs) if pairs else None
        ),
    }


def _cluster_log_ratio(
    target_ids: Sequence[str],
    left: Sequence[float],
    right: Sequence[float],
    context_by_target: Mapping[str, str],
    *,
    repetitions: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    ratios: list[float] = []
    for target, a, b in zip(target_ids, left, right, strict=True):
        if a <= 0 or b <= 0:
            continue
        ratio = b / a
        ratios.append(ratio)
        grouped.setdefault(context_by_target[target], []).append(math.log(ratio))
    if not ratios:
        return {"n": 0, "cluster_bootstrap_95_ci": None}
    rng = random.Random(42)
    clusters = sorted(grouped)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(clusters) for _ in clusters]
        draws.append(
            math.exp(median(value for cluster in sampled for value in grouped[cluster]))
        )
    draws.sort()
    return {
        "n": len(ratios),
        "clusters": len(clusters),
        "median_ratio_right_over_left": median(ratios),
        "cluster_bootstrap_95_ci": [
            draws[int(0.025 * (repetitions - 1))],
            draws[int(0.975 * (repetitions - 1))],
        ],
        "bootstrap_seed": 42,
        "bootstrap_repetitions": repetitions,
    }


def summarize(
    shard_dirs: Sequence[Path],
    polyglot_dirs: Sequence[Path],
    *,
    bootstrap_repetitions: int = 10_000,
) -> dict[str, Any]:
    shard_receipts, poly_receipts = _validate_shards(shard_dirs, polyglot_dirs)
    graded: dict[str, list[dict[str, Any]]] = {arm: [] for arm in AGENT_ARMS}
    traces: dict[str, list[dict[str, Any]]] = {arm: [] for arm in AGENT_ARMS}
    poly_traces: list[dict[str, Any]] = []
    for shard in shard_dirs:
        for arm in AGENT_ARMS:
            graded[arm].extend(_rows(shard / "graded" / f"{arm}.jsonl"))
            traces[arm].extend(_rows(shard / "traces" / f"{arm}.jsonl", traces=True))
    for polyglot in polyglot_dirs:
        poly_traces.extend(
            _rows(polyglot / "traces" / "multi_system.jsonl", traces=True)
        )

    by_arm = {arm: {_key(row): row for row in rows} for arm, rows in graded.items()}
    if any(len(rows) != 884 for rows in by_arm.values()):
        raise ValueError("each judged agent arm must contain all 884 episodes")
    if set(by_arm["memory_off"]) != set(by_arm["full_gem"]):
        raise ValueError("No Memory and GEM judged target sets differ")
    keys = sorted(by_arm["memory_off"])
    cross_keys = [
        key for key in keys if int(by_arm["memory_off"][key]["metadata"]["ordinal"]) > 0
    ]
    if len(cross_keys) != 764:
        raise ValueError(f"expected 764 reuse decisions, got {len(cross_keys)}")

    def no_score(key: tuple[str, str]) -> int:
        return int(by_arm["memory_off"][key]["score"])

    def gem_score(key: tuple[str, str]) -> int:
        return int(by_arm["full_gem"][key]["score"])

    reuse = clustered_paired_effect(
        ((key[0], no_score(key), gem_score(key)) for key in cross_keys),
        repetitions=bootstrap_repetitions,
    )
    beneficial = sum(no_score(key) == 0 and gem_score(key) == 1 for key in cross_keys)
    harmful = sum(no_score(key) == 1 and gem_score(key) == 0 for key in cross_keys)
    reuse.update(
        {
            "beneficial_flips": beneficial,
            "harmful_flips": harmful,
            "beneficial_flip_rate": beneficial / len(cross_keys),
            "harmful_flip_rate": harmful / len(cross_keys),
        }
    )

    no_traces = _trace_map(
        [row for row in traces["memory_off"] if int(row["history_size"]) > 0]
    )
    gem_traces = _trace_map(
        [row for row in traces["full_gem"] if int(row["history_size"]) > 0]
    )
    poly_map = _trace_map(poly_traces)
    if not (set(no_traces) == set(gem_traces) == set(poly_map)):
        raise ValueError("three-arm system target sets differ")
    target_ids = sorted(gem_traces)
    if len(target_ids) != 764:
        raise ValueError("three-arm system traces do not cover 764 reuse decisions")
    # Trace target_id is the canonical episode UID, while judged rows key by
    # source task ID. Build the mapping from the trace's explicit probe.
    context_by_target = {
        target: str(gem_traces[target]["probes"]["context_id"]) for target in target_ids
    }

    no_e2e = [
        float(no_traces[target]["latency_ms"]["end_to_end"]) for target in target_ids
    ]
    gem_e2e = [
        float(gem_traces[target]["latency_ms"]["end_to_end"]) for target in target_ids
    ]
    gem_query_embedding = [
        float(gem_traces[target]["latency_ms"]["query_embedding"])
        for target in target_ids
    ]
    gem_retrieval = [
        float(gem_traces[target]["latency_ms"]["database_retrieval"])
        for target in target_ids
    ]
    poly_retrieval = [
        float(poly_map[target]["latency_ms"]["total"]) for target in target_ids
    ]
    poly_e2e = [
        max(0.0, gem_total - gem_memory) + poly_memory
        for gem_total, gem_memory, poly_memory in zip(
            gem_e2e, gem_retrieval, poly_retrieval, strict=True
        )
    ]
    retrieval_ratio = _paired_ratio(gem_retrieval, poly_retrieval)
    retrieval_ratio["context_clustered_log_ratio"] = _cluster_log_ratio(
        target_ids,
        gem_retrieval,
        poly_retrieval,
        context_by_target,
        repetitions=bootstrap_repetitions,
    )

    gem_intermediate = [gem_traces[target]["intermediate"] for target in target_ids]
    poly_intermediate = [poly_map[target]["intermediate"] for target in target_ids]
    reduction = [
        float(poly["peak_materialized_ids"])
        / max(1.0, float(gem["peak_application_materialized_ids"]))
        for gem, poly in zip(gem_intermediate, poly_intermediate, strict=True)
    ]
    critical_generation_seconds = max(
        float(receipt["elapsed_seconds"]) for receipt in shard_receipts
    )
    sequential_generation_seconds = sum(
        float(receipt["elapsed_seconds"]) for receipt in shard_receipts
    )
    ci_lower = float(reuse["clustered_bootstrap_95_ci"][0])
    latency_ci = retrieval_ratio["context_clustered_log_ratio"][
        "cluster_bootstrap_95_ci"
    ]
    return {
        "schema_version": "evomembench_know_three_arm_summary_v0.2.0",
        "status": "complete",
        "track": "CrossEp-Know",
        "protocol_arms": list(REPORT_ARMS),
        "dataset": {"contexts": 120, "episodes": 884, "reuse_decisions": 764},
        "quality": {
            "no_memory": _arm_quality(by_arm["memory_off"], cross_keys),
            "gem": _arm_quality(by_arm["full_gem"], cross_keys),
            "polyglot": {
                **_arm_quality(by_arm["full_gem"], cross_keys),
                "provenance": (
                    "inherited from GEM only after 100% ordered-ID and injection-SHA "
                    "parity; current task, deterministic answer model, and prompt are shared"
                ),
                "independent_answer_generation": False,
            },
            "gem_vs_no_memory": reuse,
            "polyglot_vs_no_memory": {
                **reuse,
                "provenance": "identical to GEM after fail-closed prompt parity",
            },
            "gem_vs_polyglot": {
                "paired_mean_delta": 0.0,
                "output_identity": "derived from exact prompt parity",
            },
        },
        "latency_ms": {
            "no_memory_end_to_end_measured": _distribution(no_e2e),
            "gem_end_to_end_measured": _distribution(gem_e2e),
            "shared_task_embedding_measured": _distribution(gem_query_embedding),
            "gem_retrieval_measured": _distribution(gem_retrieval),
            "polyglot_retrieval_measured": _distribution(poly_retrieval),
            "polyglot_end_to_end_parity_reconstructed": _distribution(poly_e2e),
            "polyglot_over_gem_retrieval": retrieval_ratio,
            "polyglot_over_gem_reconstructed_end_to_end": _paired_ratio(
                gem_e2e, poly_e2e
            ),
            "no_memory_note": (
                "No Memory has no retrieval path; its measured E2E is a quality/control "
                "reference, not a database-speed victory"
            ),
            "embedding_fairness_note": (
                "GEM-vs-Polyglot retrieval compares physical database work only; the "
                "same measured task-embedding time is retained in both E2E paths"
            ),
        },
        "tokens": {
            "no_memory": _token_totals(no_traces.values()),
            "gem": _token_totals(gem_traces.values()),
            "polyglot": {
                **_token_totals(gem_traces.values()),
                "provenance": "inherited from GEM after exact injection parity",
            },
        },
        "intermediate": {
            "gem_peak_application_materialized_ids": _distribution(
                float(row["peak_application_materialized_ids"])
                for row in gem_intermediate
            ),
            "polyglot_peak_application_materialized_ids": _distribution(
                float(row["peak_materialized_ids"]) for row in poly_intermediate
            ),
            "polyglot_over_gem_reduction_factor": _distribution(reduction),
        },
        "parity": {
            "queries": sum(int(row["queries"]) for row in poly_receipts),
            "passed": sum(int(row["parity_passed"]) for row in poly_receipts),
            "fraction": 1.0,
            "returned_set": 1.0,
            "returned_order": 1.0,
            "injection_sha256": 1.0,
        },
        "throughput": {
            "answer_replica_count": 2,
            "context_shards": 2,
            "agent_predictions": sum(int(row["predictions"]) for row in shard_receipts),
            "critical_path_generation_seconds": critical_generation_seconds,
            "sum_of_shard_generation_seconds": sequential_generation_seconds,
            "predictions_per_critical_path_second": (
                sum(int(row["predictions"]) for row in shard_receipts)
                / critical_generation_seconds
            ),
            "execution": "two independent single-GPU Qwen3.8-27B-FP8 replicas",
        },
        "claim_gates": {
            "memory_utility": {
                "passed": float(reuse["paired_mean_delta"]) >= 0.03 and ci_lower > 0,
                "threshold": "gain >= 0.03 and context-cluster bootstrap CI lower > 0",
            },
            "gem_polyglot_parity_100pct": True,
            "gem_faster_than_polyglot": {
                "passed": bool(
                    retrieval_ratio["left_faster_fraction"] is not None
                    and float(retrieval_ratio["left_faster_fraction"]) >= 0.8
                    and latency_ci is not None
                    and float(latency_ci[0]) > 1.0
                ),
                "threshold": (
                    "GEM faster on >=80% paired queries and Polyglot/GEM latency "
                    "ratio CI lower > 1"
                ),
            },
            "peak_materialized_id_reduction_at_least_5x": (
                float(median(reduction)) >= 5.0
            ),
        },
        "judge": {
            "implementation": "pinned upstream EvoMemBench eval.py via lossless adapter",
            "model": "local Qwen3.8-27B-FP8",
            "paper_equivalent_closed_judge": False,
        },
        "embedding": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "execution": "two GPU BF16 pooling replicas, one per context shard",
            "exact_vector_reuse_for_polyglot": True,
        },
        "hardware_claim": (
            "x86_64 dual-GPU off-target execution; not GX10/ARM64 PG13.4 sign-off"
        ),
        "unavailable_retrieval_quality_metrics": {
            name: "N/A: EvoMemBench has no independent source-experience qrels"
            for name in ("recall_at_k", "ndcg_at_k", "mrr", "graph_path_recall")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--polyglot-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing summary: {args.output}")
    report = summarize(
        args.shard_run_dir,
        args.polyglot_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
