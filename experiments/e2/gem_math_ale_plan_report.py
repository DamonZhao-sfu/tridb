"""Validate, aggregate, and plot the approved Math/ALE physical-plan matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import psycopg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.e2.gem_math_ale_plan_agent_matrix import (
    ALE_TASKS,
    INJECTION_FREQUENCY,
    INJECTION_RATE,
    INJECTION_SLOTS,
    ITERATIONS,
    MATH_TASKS,
    PLANS,
    SEED,
    TASKS,
)

DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
STAGES = (
    "embed_ms",
    "ann_ms",
    "graph_ms",
    "predicate_ms",
    "dedup_rank_ms",
    "hydrate_ms",
    "executor_overhead_ms",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fields(rows: list[dict[str, Any]], fallback: Iterable[str]) -> list[str]:
    ordered = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                ordered.append(field)
    return ordered or list(fallback)


def _references(dsn: str) -> dict[str, dict[str, float]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT b.task_uid,r.seed_fit,b.corpus_best FROM"
            " (SELECT task_uid,max(fitness) corpus_best FROM gem_eg_node"
            "  WHERE is_valid AND fitness IS NOT NULL GROUP BY 1) b"
            " JOIN (SELECT task_uid,max(fitness) seed_fit FROM gem_eg_node"
            "       WHERE parent_node_uid IS NULL AND is_valid"
            "       AND fitness IS NOT NULL GROUP BY 1) r USING(task_uid)"
            " WHERE b.task_uid=ANY(%s)",
            (list(TASKS),),
        ).fetchall()
    return {
        str(task): {
            "seed": float(seed),
            "corpus_best": float(best),
            "gap": float(best) - float(seed),
        }
        for task, seed, best in rows
    }


def _quality(
    cell: Path, receipt: dict[str, Any], ref: dict[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = _read_jsonl(cell / "evolution_trace.jsonl")
    by_iteration: dict[int, list[float]] = defaultdict(list)
    distinct: set[str] = set()
    zero = scored = copied = 0
    for row in trace:
        iteration = int(row.get("iteration", 0))
        score = (row.get("child_metrics") or {}).get("combined_score")
        if score is not None and math.isfinite(float(score)):
            value = float(score)
            by_iteration[iteration].append(value)
            scored += 1
            zero += value == 0.0
        code = str(row.get("child_code") or "").strip()
        if code:
            distinct.add(hashlib.sha256(code.encode()).hexdigest())
            prompt = row.get("prompt") or {}
            prompt_text = (
                f"{prompt.get('system', '')}\n{prompt.get('user', '')}"
                if isinstance(prompt, dict)
                else str(prompt)
            )
            copied += int(len(code) > 200 and code in prompt_text)

    seed = float(receipt["live_seed_fitness"])
    gap = float(ref["corpus_best"]) - seed
    best = seed
    trajectories: list[dict[str, Any]] = []
    reached = {25: None, 50: None, 75: None, 90: None}
    for iteration in range(1, ITERATIONS + 1):
        candidates = by_iteration.get(iteration, ())
        if candidates:
            best = max(best, max(candidates))
        normalized = (best - seed) / gap if gap > 0 else None
        clipped = None if normalized is None else max(0.0, min(1.0, normalized))
        trajectories.append(
            {
                "domain": receipt["task_uid"].split(":", 1)[0],
                "task_uid": receipt["task_uid"],
                "task": receipt["task_uid"].split(":", 1)[1],
                "physical_plan": receipt["physical_plan"],
                "seed": SEED,
                "iteration": iteration,
                "raw_best_so_far": best,
                "normalized_gain": normalized,
                "normalized_gain_clipped": clipped,
                "evaluation_observed": bool(candidates),
            }
        )
        if normalized is not None:
            for threshold in reached:
                if reached[threshold] is None and normalized >= threshold / 100:
                    reached[threshold] = iteration

    normalized_final = (best - seed) / gap if gap > 0 else None
    auc_values = [row["normalized_gain_clipped"] for row in trajectories]
    quality = {
        "domain": receipt["task_uid"].split(":", 1)[0],
        "task_uid": receipt["task_uid"],
        "task": receipt["task_uid"].split(":", 1)[1],
        "physical_plan": receipt["physical_plan"],
        "seed": SEED,
        "seed_score": seed,
        "stored_seed_score": ref["seed"],
        "corpus_best": ref["corpus_best"],
        "raw_best_score": best,
        "best_score": best if receipt["task_uid"].startswith("math:") else None,
        "best_combined_score": best if receipt["task_uid"].startswith("ale:") else None,
        "raw_gain": best - seed,
        "normalized_gain": normalized_final,
        "normalized_auc": statistics.fmean(auc_values) if gap > 0 else None,
        "t25": reached[25],
        "t50": reached[50],
        "t75": reached[75],
        "t90": reached[90],
        "iterations_expected": ITERATIONS,
        "iterations_traced": receipt.get("iterations_traced"),
        "trace_rows": len(trace),
        "successful_evaluations": scored,
        "successful_iteration_fraction": len(by_iteration) / ITERATIONS,
        "failed_iteration_fraction": 1.0 - len(by_iteration) / ITERATIONS,
        "zero_score_fraction": zero / scored if scored else None,
        "distinct_programs": len(distinct),
        "copy_count": copied,
        "copy_rate": copied / len(trace) if trace else None,
    }
    return quality, trajectories


def _latency(
    cell: Path, receipt: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    telemetry = _read_jsonl(cell / "retrieval_telemetry.jsonl")
    failures = []
    raw = []
    for row in telemetry:
        total = float(row.get("retriever_total_ms", 0.0))
        stage_sum = sum(float(row.get(stage, 0.0)) for stage in STAGES)
        discrepancy = abs(total - stage_sum) / max(total, 1e-9)
        if discrepancy > 0.05:
            failures.append(
                {
                    "cell": cell.name,
                    "iteration": row.get("iteration"),
                    "reason": "stage_reconciliation",
                    "fraction": discrepancy,
                }
            )
        raw.append(
            {
                "domain": receipt["task_uid"].split(":", 1)[0],
                "task_uid": receipt["task_uid"],
                "physical_plan": receipt["physical_plan"],
                "seed": SEED,
                **row,
                "stage_sum_ms": stage_sum,
                "stage_discrepancy_fraction": discrepancy,
            }
        )
    totals = [float(row["retriever_total_ms"]) for row in telemetry]
    summary: dict[str, Any] = {
        "domain": receipt["task_uid"].split(":", 1)[0],
        "task_uid": receipt["task_uid"],
        "task": receipt["task_uid"].split(":", 1)[1],
        "physical_plan": receipt["physical_plan"],
        "seed": SEED,
        "retrieval_calls": len(telemetry),
        "retrieval_p50_ms": _percentile(totals, 0.50),
        "retrieval_p95_ms": _percentile(totals, 0.95),
        "retrieval_mean_ms": statistics.fmean(totals) if totals else None,
    }
    for stage in STAGES:
        values = [float(row.get(stage, 0.0)) for row in telemetry]
        summary[f"stage_{stage.removesuffix('_ms')}_mean_ms"] = (
            statistics.fmean(values) if values else None
        )
    work_fields = (
        "seeds_consumed",
        "graph_examined",
        "predicate_probes",
        "raw_reached",
        "distinct_reached",
        "dedup_hits",
        "returned",
        "missing_changes",
    )
    for field in work_fields:
        values = [float(row[field]) for row in telemetry if row.get(field) is not None]
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
    return summary, raw, failures


def _injection(cell: Path) -> dict[str, float | None]:
    rows = _read_jsonl(cell / "injection_trace.jsonl")
    if not rows:
        return {
            "injection_gate_open_fraction": None,
            "injection_render_rate": None,
            "injection_error_fraction": None,
        }
    open_rows = [row for row in rows if row.get("gate_open")]
    rendered = sum(int(row.get("rendered") or 0) for row in open_rows)
    budget = sum(int(row.get("budget") or 0) for row in open_rows)
    return {
        "injection_gate_open_fraction": len(open_rows) / len(rows),
        "injection_render_rate": rendered / budget if budget else None,
        "injection_error_fraction": sum(bool(row.get("error")) for row in rows)
        / len(rows),
    }


def _figure_rows(
    qualities: list[dict[str, Any]], latencies: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    rows = []
    quality_metrics = {
        "raw_best_score": "score",
        "best_score": "score",
        "best_combined_score": "score",
        "normalized_gain": "fraction",
        "normalized_auc": "fraction",
        "zero_score_fraction": "fraction",
        "successful_evaluation_fraction": "fraction",
        "successful_iteration_fraction": "fraction",
        "failed_iteration_fraction": "fraction",
        "copy_rate": "fraction",
        "injection_gate_open_fraction": "fraction",
        "injection_render_rate": "fraction",
        "injection_error_fraction": "fraction",
        "final_private_performance": "performance_points",
        "private_performance_delta_from_seed": "performance_points",
        "private_performance_delta_vs_nocontext": "performance_points",
    }
    latency_metrics = {
        "retrieval_p50_ms": "ms",
        "retrieval_p95_ms": "ms",
        "retrieval_mean_ms": "ms",
        "retrieval_calls": "count",
        **{f"stage_{stage.removesuffix('_ms')}_mean_ms": "ms" for stage in STAGES},
    }
    for figure, inputs, metrics in (
        (1, qualities, quality_metrics),
        (3, latencies, latency_metrics),
    ):
        for item in inputs:
            for metric, unit in metrics.items():
                value = item.get(metric)
                if value is None:
                    continue
                rows.append(
                    {
                        "figure": figure,
                        "domain": item["domain"],
                        "task": item["task"],
                        "arm": "gem",
                        "physical_plan": item["physical_plan"],
                        "seed": SEED,
                        "injection_frequency": INJECTION_FREQUENCY,
                        "injection_rate": INJECTION_RATE,
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                        "source": source,
                    }
                )
    return rows


def _save(fig: Any, root: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plots(
    qualities: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    latencies: list[dict[str, Any]],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    colors = {"vfwd": "#4C78A8", "rrev": "#F58518", "aivg": "#54A24B"}
    for domain, domain_tasks in (("math", MATH_TASKS), ("ale", ALE_TASKS)):
        rows = [row for row in qualities if row["domain"] == domain]
        fig, axes = plt.subplots(
            len(domain_tasks), 1, figsize=(9, 1.8 * len(domain_tasks))
        )
        for axis, task_uid in zip(axes, domain_tasks, strict=True):
            task_rows = {
                row["physical_plan"]: row for row in rows if row["task_uid"] == task_uid
            }
            axis.bar(
                PLANS,
                [task_rows[p]["normalized_gain"] for p in PLANS],
                color=[colors[p] for p in PLANS],
            )
            axis.axhline(0, color="black", linewidth=0.5)
            axis.set_ylabel(
                task_uid.split(":", 1)[1], rotation=0, ha="right", va="center"
            )
        axes[0].set_title(
            f"{domain.upper()} normalized final quality (one seed, descriptive)"
        )
        _save(fig, root, f"{domain}_quality_by_task")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, domain in zip(axes, ("math", "ale"), strict=True):
        for plan in PLANS:
            points = [
                row
                for row in trajectories
                if row["domain"] == domain and row["physical_plan"] == plan
            ]
            means = []
            for iteration in range(1, ITERATIONS + 1):
                values = [
                    row["normalized_gain_clipped"]
                    for row in points
                    if row["iteration"] == iteration
                ]
                means.append(statistics.fmean(values))
            axis.plot(
                range(1, ITERATIONS + 1), means, label=plan.upper(), color=colors[plan]
            )
        axis.set(
            title=f"{domain.upper()} normalized trajectory",
            xlabel="iteration",
            ylabel="mean clipped gain",
        )
        axis.legend()
    _save(fig, root, "quality_trajectories")

    tasks = list(TASKS)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 1.65 * len(tasks)))
    for axis, task_uid in zip(axes, tasks, strict=True):
        task_rows = {
            row["physical_plan"]: row
            for row in latencies
            if row["task_uid"] == task_uid
        }
        axis.bar(
            PLANS,
            [task_rows[p]["retrieval_p50_ms"] for p in PLANS],
            color=[colors[p] for p in PLANS],
        )
        axis.scatter(
            PLANS,
            [task_rows[p]["retrieval_p95_ms"] for p in PLANS],
            marker="_",
            color="black",
        )
        axis.set_yscale("log")
        axis.set_ylabel(task_uid.replace(":", "/"), rotation=0, ha="right", va="center")
    axes[0].set_title("Retrieval latency by task: p50 bars, p95 markers (log ms)")
    _save(fig, root, "latency_by_task")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, domain in zip(axes, ("math", "ale"), strict=True):
        bottom = [0.0] * len(PLANS)
        for stage in STAGES:
            key = f"stage_{stage.removesuffix('_ms')}_mean_ms"
            values = [
                statistics.fmean(
                    row[key]
                    for row in latencies
                    if row["domain"] == domain and row["physical_plan"] == plan
                )
                for plan in PLANS
            ]
            axis.bar(PLANS, values, bottom=bottom, label=stage.removesuffix("_ms"))
            bottom = [left + right for left, right in zip(bottom, values, strict=True)]
        axis.set(title=f"{domain.upper()} mean stage breakdown", ylabel="ms")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, root, "latency_stage_breakdown")

    private_rows = [
        row for row in qualities if row.get("final_private_performance") is not None
    ]
    if private_rows:
        fig, axes = plt.subplots(len(ALE_TASKS), 1, figsize=(9, 1.8 * len(ALE_TASKS)))
        for axis, task_uid in zip(axes, ALE_TASKS, strict=True):
            task_rows = {
                row["physical_plan"]: row
                for row in private_rows
                if row["task_uid"] == task_uid
            }
            axis.bar(
                PLANS,
                [task_rows[plan]["final_private_performance"] for plan in PLANS],
                color=[colors[plan] for plan in PLANS],
            )
            axis.set_ylabel(
                task_uid.split(":", 1)[1], rotation=0, ha="right", va="center"
            )
        axes[0].set_title("ALE held-out private performance by task")
        _save(fig, root, "ale_private_quality_by_task")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, domain in zip(axes, ("math", "ale"), strict=True):
        for plan in PLANS:
            quality_values = [
                row["normalized_auc"]
                for row in qualities
                if row["domain"] == domain
                and row["physical_plan"] == plan
                and row["normalized_auc"] is not None
            ]
            latency_values = [
                row["retrieval_mean_ms"]
                for row in latencies
                if row["domain"] == domain
                and row["physical_plan"] == plan
                and row["retrieval_mean_ms"] is not None
            ]
            axis.scatter(
                statistics.fmean(latency_values),
                statistics.fmean(quality_values),
                s=80,
                label=plan.upper(),
                color=colors[plan],
            )
        axis.set(
            title=f"{domain.upper()} quality–latency",
            xlabel="mean retrieval latency (ms)",
            ylabel="mean normalized AUC",
        )
        axis.set_xscale("log")
        axis.legend()
    _save(fig, root, "quality_latency_pareto")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [
        f"{row['domain']}/{row['task']}/{row['physical_plan']}" for row in qualities
    ]
    axes[0].bar(
        range(len(qualities)),
        [row["failed_iteration_fraction"] for row in qualities],
        color=[colors[row["physical_plan"]] for row in qualities],
    )
    axes[0].set(title="Failed/discarded iteration fraction", ylabel="fraction")
    axes[1].bar(
        range(len(qualities)),
        [row.get("injection_render_rate") or 0.0 for row in qualities],
        color=[colors[row["physical_plan"]] for row in qualities],
    )
    axes[1].set(title="Injection render yield", ylabel="rendered / open-gate budget")
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    _save(fig, root, "failure_injection_diagnostics")


def _merge_private(
    qualities: list[dict[str, Any]], path: Path | None
) -> tuple[int, list[str]]:
    if path is None:
        return 0, []
    rows = json.loads(path.read_text())
    index = {
        (f"ale:{row['problem']}", row.get("physical_plan")): row
        for row in rows
        if row.get("physical_plan") in PLANS
    }
    missing = []
    merged = 0
    for quality in qualities:
        if quality["domain"] != "ale":
            continue
        row = index.get((quality["task_uid"], quality["physical_plan"]))
        if row is None:
            missing.append(f"{quality['task_uid']}/{quality['physical_plan']}")
            continue
        for field in (
            "seed_private_performance",
            "final_private_performance",
            "private_performance_delta_from_seed",
            "private_performance_delta_vs_nocontext",
            "generalization_label",
            "private_case_count",
        ):
            quality[field] = row.get(field)
        merged += 1
    return merged, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--private-json", type=Path)
    parser.add_argument("--require-private", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.run / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    references = _references(args.dsn)
    expected = {(task, plan) for task in TASKS for plan in PLANS}
    expected_from_manifest = {
        (row["task_uid"], row["physical_plan"]) for row in manifest["cells"]
    }
    exclusions: list[dict[str, Any]] = []
    qualities: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    raw_telemetry: list[dict[str, Any]] = []
    reconciliation_failures: list[dict[str, Any]] = []
    seen = set()
    for row in manifest["cells"]:
        cell = Path(row["path"])
        receipt_path = cell / "run_receipt.json"
        if not receipt_path.is_file():
            exclusions.append({"cell": cell.name, "reason": "missing_receipt"})
            continue
        receipt = json.loads(receipt_path.read_text())
        invariant = all(
            (
                receipt.get("status") == "complete",
                receipt.get("task_uid") == row["task_uid"],
                receipt.get("physical_plan") == row["physical_plan"],
                receipt.get("resolved_seed") == SEED,
                receipt.get("iterations_expected") == ITERATIONS,
                receipt.get("iterations_traced") == ITERATIONS,
                receipt.get("injection_frequency") == INJECTION_FREQUENCY,
                receipt.get("injection_rate_requested") == INJECTION_RATE,
                receipt.get("injection_slots") == INJECTION_SLOTS,
                receipt.get("max_injected") == 5,
                receipt.get("inject_as") == "changes",
                receipt.get("injection_render_contract")
                == "external_program_code_is_changes_description",
                receipt.get("language")
                == ("cpp" if row["domain"] == "ale" else "python"),
                receipt.get("live_seed_fitness") is not None,
            )
        )
        if not invariant:
            exclusions.append({"cell": cell.name, "reason": "receipt_invariant"})
            continue
        key = (row["task_uid"], row["physical_plan"])
        seen.add(key)
        quality, curve = _quality(cell, receipt, references[row["task_uid"]])
        quality.update(_injection(cell))
        latency, telemetry, failures = _latency(cell, receipt)
        if not telemetry:
            exclusions.append({"cell": cell.name, "reason": "missing_telemetry"})
            continue
        qualities.append(quality)
        trajectories.extend(curve)
        latencies.append(latency)
        raw_telemetry.extend(telemetry)
        reconciliation_failures.extend(failures)

    private_cells, private_missing = _merge_private(qualities, args.private_json)
    complete = (
        expected_from_manifest == expected
        and seen == expected
        and len(qualities) == 51
        and len(latencies) == 51
        and not exclusions
        and not reconciliation_failures
        and (not args.require_private or (private_cells == 30 and not private_missing))
    )
    if (args.require_formal or args.require_private) and not complete:
        status = "failed"
    else:
        status = "passed" if not reconciliation_failures else "failed"

    quality_fields = _fields(qualities, ("domain", "task_uid", "physical_plan"))
    latency_fields = _fields(latencies, ("domain", "task_uid", "physical_plan"))
    trajectory_fields = _fields(
        trajectories, ("domain", "task_uid", "physical_plan", "iteration")
    )
    _write_csv(args.run / "quality_by_task.csv", qualities, quality_fields)
    _write_csv(args.run / "quality_trajectories.csv", trajectories, trajectory_fields)
    _write_csv(args.run / "latency_by_task.csv", latencies, latency_fields)
    _write_csv(args.run / "excluded_cells.csv", exclusions, ("cell", "reason"))
    with (args.run / "raw_retrieval_telemetry.jsonl").open("w") as handle:
        for row in raw_telemetry:
            handle.write(json.dumps(row) + "\n")
    figure_rows = _figure_rows(qualities, latencies, str(args.run))
    _write_csv(
        args.run / "figure_data.csv",
        figure_rows,
        (
            "figure",
            "domain",
            "task",
            "arm",
            "physical_plan",
            "seed",
            "injection_frequency",
            "injection_rate",
            "metric",
            "value",
            "unit",
            "source",
        ),
    )
    if qualities and latencies:
        _plots(qualities, trajectories, latencies, args.run / "figures")

    correctness = manifest.get("preflight", {}).get("correctness", {}).get("receipt")
    if correctness:
        (args.run / "correctness_gate.json").write_text(
            json.dumps(correctness, indent=2)
        )

    report = (
        "# EvoTrace Math/ALE physical-plan experiment\n\n"
        f"Validation status: **{status}**. Formal cells: {len(qualities)}/51; "
        f"telemetry calls: {len(raw_telemetry)}; held-out ALE cells: "
        f"{private_cells}/30.\n\n"
        "Quality is the task-local evaluator score. Oracle parity is only the "
        "correctness gate. These one-seed results are descriptive and are not a "
        "GX10 sign-off.\n"
    )
    (args.run / "report.md").write_text(report)

    receipt = {
        "schema_version": "gem_math_ale_physical_plan_validation_v1",
        "status": status,
        "passed": status == "passed",
        "formal_complete": complete,
        "expected_cells": 51,
        "quality_cells": len(qualities),
        "latency_cells": len(latencies),
        "trajectory_rows": len(trajectories),
        "telemetry_rows": len(raw_telemetry),
        "private_cells": private_cells,
        "private_missing": private_missing,
        "excluded_cells": exclusions,
        "reconciliation_failures": reconciliation_failures,
        "hardware_claim": "stock PostgreSQL x86_64; not a GX10 sign-off",
        "artifacts": {},
    }
    for name in (
        "matrix_manifest.json",
        "quality_by_task.csv",
        "quality_trajectories.csv",
        "latency_by_task.csv",
        "figure_data.csv",
        "excluded_cells.csv",
        "raw_retrieval_telemetry.jsonl",
        "report.md",
        "correctness_gate.json",
    ):
        path = args.run / name
        if path.is_file():
            receipt["artifacts"][name] = _sha256(path)
    if args.private_json is not None:
        receipt["artifacts"]["private_json"] = _sha256(args.private_json)
    figures = args.run / "figures"
    if figures.is_dir():
        for path in sorted(figures.iterdir()):
            if path.is_file():
                receipt["artifacts"][str(path.relative_to(args.run))] = _sha256(path)
    (args.run / "validation_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
