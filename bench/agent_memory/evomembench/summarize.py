"""Aggregate EvoMemBench outcome, systems, and honest N/A metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from bench.agent_memory.evomembench.metrics import clustered_paired_effect

MIN_ANSWER_MAX_TOKENS = 4096


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _p95(values: Iterable[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _average(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def summarize(root: Path, *, allow_invalid_outcome: bool = False) -> dict[str, Any]:
    run = json.loads((root / "run_receipt.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("cannot summarize an incomplete run")
    invalid_reasons: list[str] = []
    if run.get("injection_policy") != "replace_equal_total_input_tokens":
        invalid_reasons.append(
            "memory injection did not replace an equal-sized token slot"
        )
    if int(run.get("answer_max_tokens") or 0) < MIN_ANSWER_MAX_TOKENS:
        invalid_reasons.append(
            "answer_max_tokens was absent or below the pinned 4096-token protocol"
        )
    outcome_valid = not invalid_reasons
    if not outcome_valid and not allow_invalid_outcome:
        raise ValueError("invalid outcome run: " + "; ".join(invalid_reasons))
    arms = run["arms"]
    graded: dict[str, list[dict[str, Any]]] = {
        arm: _rows(root / "graded" / f"{arm}.jsonl") for arm in arms
    }
    baseline = {
        (row["metadata"]["context_id"], row["metadata"]["task_id"]): int(row["score"])
        for row in graded["memory_off"]
    }
    metrics: dict[str, Any] = {}
    for arm in arms:
        predictions = _rows(root / "predictions" / f"{arm}.jsonl")
        receipts = _rows(root / "receipts" / f"{arm}.jsonl")
        scores = graded[arm]
        if not (len(predictions) == len(receipts) == len(scores)):
            raise ValueError(f"{arm}: prediction/receipt/grade count mismatch")
        score_by_key = {
            (row["metadata"]["context_id"], row["metadata"]["task_id"]): int(
                row["score"]
            )
            for row in scores
        }
        if set(score_by_key) != set(baseline):
            raise ValueError(f"{arm}: scored task set differs from memory_off")
        cross_rows = [row for row in scores if int(row["metadata"]["ordinal"]) > 0]
        cross_scores = [int(row["score"]) for row in cross_rows]
        baseline_cross = [
            baseline[(row["metadata"]["context_id"], row["metadata"]["task_id"])]
            for row in cross_rows
        ]
        paired_effect = clustered_paired_effect(
            [
                (
                    row["metadata"]["context_id"],
                    baseline[
                        (row["metadata"]["context_id"], row["metadata"]["task_id"])
                    ],
                    int(row["score"]),
                )
                for row in cross_rows
            ]
        )
        ordinal_scores: dict[int, list[int]] = {}
        for row in scores:
            ordinal_scores.setdefault(int(row["metadata"]["ordinal"]), []).append(
                int(row["score"])
            )
        probes = [row.get("probes", {}) for row in receipts]
        updates = (
            [] if arm == "memory_off" else _rows(root / "updates" / f"{arm}.jsonl")
        )
        retrieval_ms = [float(row["stats"]["retrieval_ms"]) for row in predictions]
        generation_seconds = [
            float(row["stats"]["generation_seconds"]) for row in predictions
        ]
        model_ttft_seconds = [
            float(row["stats"]["model_ttft_seconds"]) for row in predictions
        ]
        generation_usage = [
            row["stats"].get("generation_usage", {}) for row in predictions
        ]
        completion_cap = run.get("answer_max_tokens")
        observed_completion_max = max(
            int(row.get("completion_tokens", 0)) for row in generation_usage
        )
        if any(
            row["stats"].get("system_tokens_original")
            != row["stats"].get("system_tokens_final")
            for row in predictions
            if int(row["stats"].get("system_tokens_replaced", 0)) > 0
        ):
            raise ValueError(f"{arm}: token replacement parity violation")
        metrics[arm] = {
            "episodes": len(scores),
            "cross_episode_decisions": len(cross_scores),
            "strict_rubric_accuracy": (
                mean(int(row["score"]) for row in scores) if outcome_valid else None
            ),
            "cross_episode_strict_accuracy": (
                mean(cross_scores) if outcome_valid else None
            ),
            "cross_episode_delta_vs_memory_off": (
                mean(cross_scores) - mean(baseline_cross) if outcome_valid else None
            ),
            "cross_episode_paired_effect": paired_effect if outcome_valid else None,
            "accuracy_by_ordinal": (
                {
                    str(ordinal): {"n": len(values), "accuracy": mean(values)}
                    for ordinal, values in sorted(ordinal_scores.items())
                }
                if outcome_valid
                else None
            ),
            "retrieval_latency_ms_p50": median(retrieval_ms),
            "retrieval_latency_ms_p95": _p95(retrieval_ms),
            "generation_seconds_p50": median(generation_seconds),
            "generation_seconds_p95": _p95(generation_seconds),
            "model_ttft_seconds_p50": median(model_ttft_seconds),
            "model_ttft_seconds_p95": _p95(model_ttft_seconds),
            "generation_prompt_tokens_mean": _average(
                row.get("prompt_tokens") for row in generation_usage
            ),
            "generation_completion_tokens_mean": _average(
                row.get("completion_tokens") for row in generation_usage
            ),
            "generation_total_tokens_mean": _average(
                row.get("total_tokens") for row in generation_usage
            ),
            "generation_completion_cap_tokens": completion_cap,
            "generation_completion_cap_fraction": (
                sum(
                    row.get("completion_tokens") == completion_cap
                    for row in generation_usage
                )
                / len(generation_usage)
                if completion_cap is not None
                else None
            ),
            "generation_observed_completion_token_max": observed_completion_max,
            "generation_fraction_at_observed_completion_max": sum(
                int(row.get("completion_tokens", 0)) == observed_completion_max
                for row in generation_usage
            )
            / len(generation_usage),
            "injection_tokens_mean": mean(
                int(row["stats"]["injection_tokens"]) for row in predictions
            ),
            "candidates_examined_mean": _average(
                row.get("candidates_examined") for row in probes
            ),
            "graph_edges_examined_mean": _average(
                row.get("graph_examined") for row in probes
            ),
            "graph_reached_mean": _average(row.get("graph_reached") for row in probes),
            "graph_reached_available": any(
                bool(row.get("graph_reached_available")) for row in probes
            ),
            "construction_seconds_mean": _average(
                row.get("cost", {}).get("seconds") for row in updates
            ),
            "wal_bytes_per_update_mean": _average(
                row.get("wal_bytes_interval") for row in updates
            ),
            "edges_created_total": sum(
                int(row.get("delta", {}).get("edges_created", 0)) for row in updates
            ),
            "operator_usage": predictions[0]["stats"]["operations"],
        }
    return {
        "schema_version": "evomembench_gem_summary_v0.1.0",
        "run_id": run["run_id"],
        "source_revision": run["source_revision"],
        "outcome_valid": outcome_valid,
        "outcome_invalid_reason": None if outcome_valid else "; ".join(invalid_reasons),
        "outcome_invalid_reasons": invalid_reasons,
        "metrics": metrics,
        "unavailable_metrics": {
            name: "N/A: EvoMemBench does not release independent source relevance qrels"
            for name in run["unavailable_metrics"]
        },
        "interpretation_gate": (
            "A positive GEM delta supports cross-session outcome transfer at this "
            "pilot operating point; it does not by itself establish retrieval recall "
            "or large-corpus database scaling."
            if outcome_valid
            else "Outcome comparison is prohibited; only wiring and systems telemetry may be used."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-invalid-outcome", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)
    summary = summarize(root, allow_invalid_outcome=args.allow_invalid_outcome)
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
