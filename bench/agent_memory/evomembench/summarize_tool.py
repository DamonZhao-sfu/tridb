"""Aggregate official CrossEp-Tool outcomes and the 12-cell transfer matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from bench.agent_memory.evomembench.metrics import clustered_paired_effect
from bench.agent_memory.evomembench.protocol import ordered_transfer_pairs


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _completed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if not row.get("error", "").strip()]


def _contains_error(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            ("error" in str(key).casefold() and bool(item)) or _contains_error(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_error(item) for item in value)
    return False


def _official_cost_metrics(
    rows: list[dict[str, str]], *, sample_dir: Path | None = None
) -> dict[str, float | int]:
    completed = _completed(rows)
    successful = [row for row in completed if float(row.get("success", 0)) == 1.0]
    row_ids = {row["id"] for row in rows if row.get("id")}
    detail_paths = (
        {
            path.stem: path
            for path in sample_dir.glob("**/*.json")
            if path.stem in row_ids
        }
        if sample_dir is not None
        else {}
    )
    details = []
    for row in rows:
        path = detail_paths.get(row.get("id", ""))
        details.append(
            json.loads(path.read_text()) if path is not None and path.is_file() else {}
        )
    force_terminated = sum(
        bool(detail.get("force_quit"))
        or (
            detail.get("checker_result", {}).get("error", {}).get("error_type")
            == "multi_turn:force_terminated"
        )
        or (
            detail.get("checker_result", {}).get("error_type")
            == "multi_turn:force_terminated"
        )
        for detail in details
    )
    tool_or_api_errors = sum(
        bool(row.get("error", "").strip())
        or _contains_error(detail.get("inference_log", []))
        for row, detail in zip(rows, details, strict=True)
    )

    def average(key: str) -> float:
        return _mean(completed, key) if completed else 0.0

    return {
        "n_failed": len(rows) - len(completed),
        "avg_step_count": average("step_count"),
        "successful_tasks": len(successful),
        "avg_steps_to_success": (
            mean(float(row["step_count"]) for row in successful) if successful else 0.0
        ),
        "force_termination_count": force_terminated,
        "force_termination_rate": force_terminated / len(rows) if rows else 0.0,
        "tool_or_api_error_count": tool_or_api_errors,
        "tool_or_api_error_rate": tool_or_api_errors / len(rows) if rows else 0.0,
        **{
            f"avg_{key}": average(key)
            for key in (
                "latency_s_inference",
                "input_tokens_inference",
                "output_tokens_inference",
                "total_tokens_inference",
                "latency_s_memory",
                "input_tokens_memory",
                "output_tokens_memory",
                "total_tokens_memory",
                "embedding_tokens",
                "latency_s_total",
                "input_tokens_total",
                "output_tokens_total",
                "total_tokens_total",
            )
        },
    }


def summarize(root: Path, *, bootstrap_repetitions: int = 10_000) -> dict[str, Any]:
    receipt = json.loads((root / "run_receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("cannot summarize an incomplete CrossEp-Tool run")
    arms = receipt["arms"]
    expected_per_environment = int(receipt["episodes_per_environment"])
    if receipt.get("query_mode") == "gem_task_seed":
        policies = receipt.get("token_slot_policy")
        if not isinstance(policies, dict):
            # Compatibility with calibration receipts written before the formal
            # systems protocol added a strict long-context arm.
            policies = {arm: policies for arm in arms}
        for arm in arms:
            expected_policy = (
                "complete_history_append_no_truncation"
                if arm == "long_context"
                else (None if arm == "memory_off" else "fixed_neutral_replacement")
            )
            if expected_policy is not None and policies.get(arm) != expected_policy:
                raise ValueError(
                    f"invalid Tool outcome run: {arm} policy is {policies.get(arm)!r}"
                )
        expected_slot = int(receipt["token_slot_tokens"])
        for arm in arms:
            if arm in {"memory_off", "long_context"}:
                continue
            receipt_paths = [
                *sorted((root / "phase1" / arm).glob("*/retrieval_receipts.jsonl")),
                *sorted((root / "phase2" / arm).glob("*/retrieval_receipts.jsonl")),
            ]
            if not receipt_paths:
                raise ValueError(f"{arm}: missing retrieval receipts")
            for path in receipt_paths:
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    observed = int(json.loads(line).get("slot_tokens", -1))
                    if observed != expected_slot:
                        raise ValueError(
                            f"{arm}: token slot mismatch {observed} != {expected_slot}"
                        )
    in_environment_baseline: dict[str, dict[str, float]] = {}
    for environment in ("gorilla_fs", "vehicle_control", "trading_bot", "travel_api"):
        rows = _csv(root / "phase1" / "memory_off" / environment / "per_sample.csv")
        if len(rows) != expected_per_environment:
            raise ValueError(
                f"memory_off/{environment}: {len(rows)} rows, expected "
                f"{expected_per_environment}"
            )
        in_environment_baseline.update(
            {
                row["id"]: {
                    "success": float(row["success"]),
                    "progress": float(row["progress"]),
                }
                for row in rows
            }
        )

    baseline: dict[tuple[str, str], dict[str, float]] = {}
    baseline_mode = "600_executed_target_evaluations"
    for source, target in ordered_transfer_pairs():
        cell = f"{source}__to__{target}"
        path = root / "phase2" / "memory_off" / cell / "per_sample.csv"
        if not path.is_file():
            if receipt.get("formal"):
                raise ValueError(
                    f"formal No Memory target evaluation is missing: {cell}"
                )
            baseline_mode = "legacy_200_in_environment_scores_reused"
            for sample_id, value in in_environment_baseline.items():
                baseline[(cell, sample_id)] = value
            continue
        rows = _csv(path)
        if len(rows) != expected_per_environment:
            raise ValueError(
                f"memory_off/{cell}: {len(rows)} rows, expected "
                f"{expected_per_environment}"
            )
        for row in rows:
            baseline[(cell, row["id"])] = {
                "success": float(row["success"]),
                "progress": float(row["progress"]),
            }
    if receipt.get("formal") and len(baseline) != 600:
        raise ValueError(
            f"formal No Memory expected 600 target rows, got {len(baseline)}"
        )

    in_env: dict[str, Any] = {}
    transfer: dict[str, Any] = {}
    transfer_values: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for arm in arms:
        env_rows: list[dict[str, str]] = []
        for environment in (
            "gorilla_fs",
            "vehicle_control",
            "trading_bot",
            "travel_api",
        ):
            env_rows.extend(
                _csv(root / "phase1" / arm / environment / "per_sample.csv")
            )
        if len(env_rows) != 4 * expected_per_environment:
            raise ValueError(f"{arm}: incomplete in-environment evaluation")
        in_env[arm] = {
            "n": len(env_rows),
            "success_rate": _mean(env_rows, "success"),
            "progress_score": _mean(env_rows, "progress"),
            **_official_cost_metrics(env_rows, sample_dir=root / "phase1" / arm),
        }
        if arm == "memory_off":
            continue
        cells: dict[str, Any] = {}
        arm_values: dict[tuple[str, str], dict[str, float]] = {}
        paired_success: list[tuple[str, float, float]] = []
        paired_progress: list[tuple[str, float, float]] = []
        for source, target in ordered_transfer_pairs():
            cell = f"{source}__to__{target}"
            rows = _csv(root / "phase2" / arm / cell / "per_sample.csv")
            if len(rows) != expected_per_environment:
                raise ValueError(
                    f"{arm}/{cell}: {len(rows)} rows, expected "
                    f"{expected_per_environment}"
                )
            cells[cell] = {
                "n": len(rows),
                "success_rate": _mean(rows, "success"),
                "progress_score": _mean(rows, "progress"),
                **_official_cost_metrics(rows, sample_dir=root / "phase2" / arm / cell),
                "success_gain_vs_no_memory": _mean(rows, "success")
                - mean(baseline[(cell, row["id"])]["success"] for row in rows),
                "progress_gain_vs_no_memory": _mean(rows, "progress")
                - mean(baseline[(cell, row["id"])]["progress"] for row in rows),
            }
            for row in rows:
                arm_values[(cell, row["id"])] = {
                    "success": float(row["success"]),
                    "progress": float(row["progress"]),
                }
                paired_success.append(
                    (
                        cell,
                        baseline[(cell, row["id"])]["success"],
                        float(row["success"]),
                    )
                )
                paired_progress.append(
                    (
                        cell,
                        baseline[(cell, row["id"])]["progress"],
                        float(row["progress"]),
                    )
                )
        transfer[arm] = {
            "cells": cells,
            "worst_pair_by_success_gain": min(
                (
                    {
                        "pair": pair,
                        "gain": metrics["success_gain_vs_no_memory"],
                        "success_rate": metrics["success_rate"],
                    }
                    for pair, metrics in cells.items()
                ),
                key=lambda row: (row["gain"], row["pair"]),
            ),
            "worst_pair_by_progress_gain": min(
                (
                    {
                        "pair": pair,
                        "gain": metrics["progress_gain_vs_no_memory"],
                        "progress_score": metrics["progress_score"],
                    }
                    for pair, metrics in cells.items()
                ),
                key=lambda row: (row["gain"], row["pair"]),
            ),
            "success_effect_vs_no_memory": clustered_paired_effect(
                paired_success, repetitions=bootstrap_repetitions
            ),
            "progress_effect_vs_no_memory": clustered_paired_effect(
                paired_progress, repetitions=bootstrap_repetitions
            ),
        }
        transfer_values[arm] = arm_values
    claim_gates: dict[str, Any] | None = None
    if "gem_fused" in transfer:
        success = transfer["gem_fused"]["success_effect_vs_no_memory"]
        progress = transfer["gem_fused"]["progress_effect_vs_no_memory"]
        success_pass = (
            float(success["paired_mean_delta"]) >= 0.05
            and float(success["clustered_bootstrap_95_ci"][0]) > 0
        )
        progress_pass = (
            float(progress["paired_mean_delta"]) >= 0.05
            and float(progress["clustered_bootstrap_95_ci"][0]) > 0
        )
        claim_gates = {
            "memory_utility_pass": success_pass or progress_pass,
            "memory_utility_threshold": (
                "exact-success gain >= 0.05 or progress gain >= 0.05, "
                "with paired CI lower > 0"
            ),
        }
        if "long_context" in transfer_values:
            full = transfer_values["gem_fused"]
            long_context = transfer_values["long_context"]
            if set(full) != set(long_context):
                raise ValueError("gem_fused and long_context transfer tasks differ")
            noninferiority = clustered_paired_effect(
                (
                    (
                        key[0],
                        long_context[key]["success"],
                        full[key]["success"],
                    )
                    for key in sorted(full)
                ),
                repetitions=bootstrap_repetitions,
            )
            claim_gates.update(
                {
                    "full_gem_vs_long_context": noninferiority,
                    "long_context_noninferiority_pass": (
                        float(noninferiority["paired_mean_delta"]) >= -0.02
                    ),
                    "noninferiority_margin": -0.02,
                }
            )
    return {
        "schema_version": "evomembench_gem_tool_summary_v0.1.0",
        "run_id": receipt["run_id"],
        "source_revision": receipt["source_revision"],
        "outcome_valid": receipt.get("query_mode") == "gem_task_seed",
        "outcome_validity_gate": (
            "fixed_neutral_replacement"
            if receipt.get("query_mode") == "gem_task_seed"
            else "upstream_parity_not_a_token_matched_causal_protocol"
        ),
        "actual_release_rows": 200,
        "no_memory_target_evaluation_mode": baseline_mode,
        "no_memory_target_evaluations": len(baseline),
        "in_environment": in_env,
        "cross_environment_transfer": transfer,
        "claim_gates": claim_gates,
        "unavailable_retrieval_metrics": {
            name: "N/A: no independent source-experience relevance qrels"
            for name in (
                "evidence_recall_at_k",
                "dependency_recall_at_k",
                "ndcg_at_k",
                "mrr",
                "graph_path_recall",
                "oracle_gap",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    summary = summarize(args.run_dir, bootstrap_repetitions=args.bootstrap_repetitions)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
