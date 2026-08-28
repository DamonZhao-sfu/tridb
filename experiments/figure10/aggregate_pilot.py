"""Aggregate the seven-system Figure-10 same-host PILOT.

The command is deliberately fail-closed: every configured system must have a
complete 300-query receipt and the same ordered question IDs before any table
or figure is written.  This keeps partial queue state out of headline output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SYSTEMS = (
    ("tridb_gem", "TriDB/GEM fused", "tridb_gem/pilot_b1"),
    ("mem0", "Mem0", "mem0/pilot_b1"),
    ("cognee", "Cognee full-PG", "cognee/pilot_b4"),
    ("memos", "MemOS", "memos/pilot_b2"),
    ("mandol", "Mandol", "mandol/pilot_b1"),
    ("graphiti", "Graphiti (Zep OSS proxy)", "graphiti/pilot_b1"),
    (
        "evermemos",
        "EverMemOS (shared embed)",
        "evermemos/shared_embed_pilot_b1",
    ),
)

EXPECTED_QUESTIONS = 300
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_824


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * quantile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows) if rows else None,
        "p50": percentile(rows, 0.50),
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
        "max": max(rows) if rows else None,
    }


def cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float],
    statistic: Callable[[Sequence[float]], float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile CI resampling the five history clusters, not questions."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if "history_index" not in row:
            raise ValueError("bootstrap row lacks history_index")
        grouped.setdefault(int(row["history_index"]), []).append(row)
    clusters = sorted(grouped)
    if not clusters:
        raise ValueError("bootstrap needs at least one history cluster")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled: list[float] = []
        for cluster in generator.choices(clusters, k=len(clusters)):
            sampled.extend(value(row) for row in grouped[cluster])
        estimates.append(float(statistic(sampled)))
    low = percentile(estimates, 0.025)
    high = percentile(estimates, 0.975)
    assert low is not None and high is not None
    return low, high


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = run_dir / "run_receipt.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not receipt_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"incomplete Figure-10 run: {run_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    serving = receipt.get("serving") or {}
    if not (
        receipt.get("status") == "complete"
        and serving.get("questions") == EXPECTED_QUESTIONS
        and serving.get("successes") == EXPECTED_QUESTIONS
        and serving.get("failures") == 0
    ):
        raise RuntimeError(
            f"run did not pass 300/300 zero-failure gate: {run_dir}: "
            f"status={receipt.get('status')!r} serving={serving!r}"
        )
    rows = read_jsonl(predictions_path)
    if len(rows) != EXPECTED_QUESTIONS or not all(row.get("success") for row in rows):
        raise RuntimeError(f"prediction evidence is not 300 successful rows: {run_dir}")
    return receipt, rows


def timing_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row["timing"][field]) for row in rows]


def headline_row(
    key: str,
    label: str,
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ttft = distribution(timing_values(rows, "effective_ttft_seconds"))
    total = distribution(timing_values(rows, "total_seconds"))
    context = distribution(timing_values(rows, "context_ready_seconds"))
    mean_ttft_ci = cluster_bootstrap_ci(
        rows,
        lambda row: float(row["timing"]["effective_ttft_seconds"]),
        statistics.fmean,
    )
    p50_ttft_ci = cluster_bootstrap_ci(
        rows,
        lambda row: float(row["timing"]["effective_ttft_seconds"]),
        statistics.median,
    )
    mean_total_ci = cluster_bootstrap_ci(
        rows,
        lambda row: float(row["timing"]["total_seconds"]),
        statistics.fmean,
    )
    p50_total_ci = cluster_bootstrap_ci(
        rows,
        lambda row: float(row["timing"]["total_seconds"]),
        statistics.median,
    )
    return {
        "system": key,
        "display_label": label,
        "identity": "reproduction" if key == "mem0" else "protocol_extension",
        "claim": receipt.get("claim"),
        "questions": len(rows),
        "failures": 0,
        "mean_ttft_s": ttft["mean"],
        "mean_ttft_ci95_low_s": mean_ttft_ci[0],
        "mean_ttft_ci95_high_s": mean_ttft_ci[1],
        "mean_total_s": total["mean"],
        "mean_total_ci95_low_s": mean_total_ci[0],
        "mean_total_ci95_high_s": mean_total_ci[1],
        "p50_ttft_s": ttft["p50"],
        "p50_ttft_ci95_low_s": p50_ttft_ci[0],
        "p50_ttft_ci95_high_s": p50_ttft_ci[1],
        "p50_total_s": total["p50"],
        "p50_total_ci95_low_s": p50_total_ci[0],
        "p50_total_ci95_high_s": p50_total_ci[1],
        "ttft_p90_s": ttft["p90"],
        "ttft_p95_s": ttft["p95"],
        "ttft_p99_s": ttft["p99"],
        "ttft_max_s": ttft["max"],
        "total_p90_s": total["p90"],
        "total_p95_s": total["p95"],
        "total_p99_s": total["p99"],
        "total_max_s": total["max"],
        "context_ready_mean_s": context["mean"],
        "cold_first_ttft_s": rows[0]["timing"]["effective_ttft_seconds"],
        "cold_first_total_s": rows[0]["timing"]["total_seconds"],
        "overflow_top5_count": sum(
            bool(row.get("overflow_fallback_to_top5")) for row in rows
        ),
    }


def decomposition_row(
    key: str, label: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fields = {
        "embedding_s": "query_embedding_seconds",
        "post_embedding_retrieval_s": "post_embedding_retrieval_seconds",
        "context_ready_s": "context_ready_seconds",
        "prompt_assembly_s": "prompt_assembly_seconds",
        "answer_queue_prefill_s": "answer_queue_prefill_seconds",
        "decode_s": "decode_seconds",
        "ttft_s": "effective_ttft_seconds",
        "total_s": "total_seconds",
    }
    result: dict[str, Any] = {"system": key, "display_label": label}
    for output, source in fields.items():
        result[f"mean_{output}"] = statistics.fmean(timing_values(rows, source))
        result[f"p50_{output}"] = statistics.median(timing_values(rows, source))
    observations = [str(row["timing"].get("t1_observation")) for row in rows]
    result["observed_embedding_span_fraction"] = sum(
        value == "observed_embedding_span" for value in observations
    ) / len(observations)
    result["opaque_t1_fraction"] = sum(
        value == "opaque_upper_bound_at_context_ready" for value in observations
    ) / len(observations)
    return result


def workload_row(
    key: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rounds = [
        float((row.get("search") or {}).get("retrieval_rounds") or 1) for row in rows
    ]
    context_counts = [float(row.get("prompt_context_count") or 0) for row in rows]
    prompt_tokens = [float(row.get("estimated_prompt_tokens") or 0) for row in rows]
    memory_llm_calls = sum(row.get("phase") == "memory_retrieval" for row in ledger)
    return {
        "system": key,
        "display_label": label,
        "query_memory_llm_calls": memory_llm_calls,
        "retrieval_rounds_mean": statistics.fmean(rounds),
        "retrieval_rounds_p95": percentile(rounds, 0.95),
        "context_items_mean": statistics.fmean(context_counts),
        "estimated_final_prompt_tokens_mean": statistics.fmean(prompt_tokens),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    figure_dir: Path,
    headline: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    decomposition: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["display_label"]) for row in headline]
    for estimator in ("mean", "p50"):
        ttft = [float(row[f"{estimator}_ttft_s"]) for row in headline]
        total = [float(row[f"{estimator}_total_s"]) for row in headline]
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.bar(labels, ttft, label="query → first token")
        ax.bar(
            labels,
            [end - first for first, end in zip(ttft, total, strict=True)],
            bottom=ttft,
            label="first token → final token",
        )
        ax.set_ylabel("seconds")
        ax.set_title(f"Figure-10 protocol: {estimator} TTFT and Total")
        ax.tick_params(axis="x", rotation=28)
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"figure10_{estimator}_stacked.{suffix}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for key, label, _ in SYSTEMS:
        values = sorted(timing_values(predictions[key], "effective_ttft_seconds"))
        y = [(index + 1) / len(values) for index in range(len(values))]
        ax.step(values, y, where="post", label=label)
    ax.set_xlabel("TTFT (seconds)")
    ax.set_ylabel("ECDF")
    ax.set_title("TTFT distribution (300 serial queries per system)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"ttft_ecdf.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = list(range(len(labels)))
    width = 0.25
    for offset, metric, name in (
        (-width, "p50_ttft_s", "P50"),
        (0.0, "ttft_p95_s", "P95"),
        (width, "ttft_p99_s", "P99"),
    ):
        ax.bar(
            [value + offset for value in x],
            [row[metric] for row in headline],
            width,
            label=name,
        )
    ax.set_xticks(x, labels, rotation=28)
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("TTFT tail latency")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"ttft_tail.{suffix}", dpi=180)
    plt.close(fig)

    components = (
        ("mean_context_ready_s", "memory context ready"),
        ("mean_prompt_assembly_s", "prompt assembly"),
        ("mean_answer_queue_prefill_s", "answer queue + prefill"),
        ("mean_decode_s", "decode"),
    )
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = [0.0] * len(decomposition)
    for field, label in components:
        values = [float(row[field]) for row in decomposition]
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    ax.set_ylabel("mean seconds")
    ax.set_title("End-to-end latency decomposition")
    ax.tick_params(axis="x", rotation=28)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"latency_decomposition.{suffix}", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def run(root: Path) -> None:
    aggregate_dir = root / "aggregates"
    figure_dir = root / "figures"
    if aggregate_dir.exists() or figure_dir.exists():
        raise FileExistsError("refusing existing aggregate or figure directory")

    receipts: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    question_order: list[str] | None = None
    headline: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    workload: list[dict[str, Any]] = []
    for key, label, relative in SYSTEMS:
        run_dir = root / "runs" / relative
        receipt, rows = validate_run(run_dir)
        current_order = [str(row["question_id"]) for row in rows]
        if question_order is None:
            question_order = current_order
        elif current_order != question_order:
            raise RuntimeError(f"question order mismatch for {key}")
        ledger = read_jsonl(run_dir / "call_ledger.jsonl")
        receipts[key] = receipt
        predictions[key] = rows
        headline.append(headline_row(key, label, receipt, rows))
        decomposition.append(decomposition_row(key, label, rows))
        workload.append(workload_row(key, label, rows, ledger))

    aggregate_dir.mkdir()
    figure_dir.mkdir()
    write_csv(aggregate_dir / "figure10_headline.csv", headline)
    write_csv(aggregate_dir / "latency_decomposition.csv", decomposition)
    write_csv(aggregate_dir / "retrieval_workload.csv", workload)
    plot_results(figure_dir, headline, predictions, decomposition)

    report = [
        "# Figure 10 seven-system same-host PILOT\n",
        "## Material Passport\n",
        "- Origin: academic-research-suite / experiment-agent",
        "- Verification Status: ANALYZED (one fresh build; not H100 numerical replication)",
        "- Workload: 5 histories, 451 chunks, 300 serial queries per system",
        "- Answer model: Qwen3.8-27B-FP8, TP=2 on GPUs 0/1, thinking disabled",
        "- Embedding model: Qwen3-Embedding-0.6B",
        "- Bootstrap: 2,000 percentile resamples clustered by history",
        "\n## TTFT : Total\n",
        "| System | mean TTFT : Total (s) | p50 TTFT : Total (s) | failures |",
        "|---|---:|---:|---:|",
    ]
    for row in headline:
        report.append(
            f"| {row['display_label']} | {row['mean_ttft_s']:.6f} : "
            f"{row['mean_total_s']:.6f} | {row['p50_ttft_s']:.6f} : "
            f"{row['p50_total_s']:.6f} | {row['failures']} |"
        )
    report.extend(
        [
            "\n## Interpretation boundary\n",
            "These rows are a same-host controlled PILOT with one fresh build per system. "
            "They are not the plan's three-build H100 headline and must not be presented as "
            "an exact numerical reproduction of the paper. Mem0 is the only paper-overlap "
            "row; all other systems are protocol extensions.",
            "\nQuality/judge results are intentionally absent until the separately frozen "
            "MemoryAgentBench judge phase is executed. Latency alone does not establish a "
            "matched-quality speedup.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
