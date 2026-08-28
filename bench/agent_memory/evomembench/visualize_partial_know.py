"""Create a clearly provisional dashboard from an in-progress Know run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable


EXPECTED_EPISODES = 884
EXPECTED_REUSE_DECISIONS = 764


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values]
    return {
        "n": len(observed),
        "p50": median(observed) if observed else None,
        "p95": _percentile(observed, 0.95),
    }


def _numeric(rows: Iterable[dict[str, Any]], *path: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _read_receipt(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def build_snapshot(run_root: Path) -> dict[str, Any]:
    traces: dict[str, list[dict[str, Any]]] = {"memory_off": [], "full_gem": []}
    receipts: dict[str, dict[str, Any] | None] = {}
    graded = 0
    for shard in (0, 1):
        shard_root = run_root / f"shard_{shard}"
        receipts[str(shard)] = _read_receipt(shard_root / "run_receipt.json")
        for arm in traces:
            traces[arm].extend(_rows(shard_root / "traces" / f"{arm}.jsonl"))
            graded += len(_rows(shard_root / "graded" / f"{arm}.jsonl"))

    complete = {
        arm: [row for row in rows if row.get("status") == "complete"]
        for arm, rows in traces.items()
    }
    reusable = {
        arm: [row for row in rows if int(row.get("history_size", 0)) > 0]
        for arm, rows in complete.items()
    }
    maps = {
        arm: {str(row["target_id"]): row for row in rows}
        for arm, rows in reusable.items()
    }
    paired_ids = sorted(set(maps["memory_off"]) & set(maps["full_gem"]))
    paired = {arm: [maps[arm][target_id] for target_id in paired_ids] for arm in maps}

    polyglot_rows = sum(
        len(_rows(run_root / f"polyglot_{shard}" / "traces" / "multi_system.jsonl"))
        for shard in (0, 1)
    )
    root_receipt = _read_receipt(run_root / "run_receipt.json")
    failed_receipts = [
        receipt
        for receipt in [root_receipt, *receipts.values()]
        if receipt is not None and receipt.get("status") == "failed"
    ]
    warnings = [
        "The run is incomplete; no final claim may use this snapshot.",
        "Judge quality is unavailable until agent generation finishes.",
        "Polyglot metrics are unavailable until replay finishes.",
    ]
    if failed_receipts:
        warnings.append("A run or shard receipt reports failure; partial artifacts are diagnostic only.")
    else:
        warnings.append("No failure receipt is present at capture time; the run remains in progress.")
    snapshot: dict[str, Any] = {
        "schema_version": "evomembench_partial_visualization_snapshot_v0.1.0",
        "status": "provisional_invalid_for_claims",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_root": str(run_root.resolve()),
        "warnings": warnings,
        "progress": {
            "no_memory": {
                "observed": len(traces["memory_off"]),
                "expected": EXPECTED_EPISODES,
            },
            "gem": {"observed": len(traces["full_gem"]), "expected": EXPECTED_EPISODES},
            "judge": {"observed": graded, "expected": EXPECTED_EPISODES * 2},
            "polyglot": {
                "observed": polyglot_rows,
                "expected": EXPECTED_REUSE_DECISIONS,
            },
        },
        "shard_receipts": receipts,
        "paired_reuse_decisions": len(paired_ids),
        "paired_metrics": {},
        "gem_retrieval": {},
        "quality": "N/A: judge has not started",
    }
    for arm, rows in paired.items():
        snapshot["paired_metrics"][arm] = {
            "end_to_end_ms": _distribution(_numeric(rows, "latency_ms", "end_to_end")),
            "ttft_ms": _distribution(_numeric(rows, "latency_ms", "model_ttft")),
            "answer_prompt_tokens": _distribution(
                _numeric(rows, "tokens", "answer_prompt")
            ),
            "answer_completion_tokens": _distribution(
                _numeric(rows, "tokens", "answer_completion")
            ),
            "memory_injection_tokens": _distribution(
                _numeric(rows, "tokens", "memory_injection")
            ),
        }

    gem_rows = reusable["full_gem"]
    snapshot["gem_retrieval"] = {
        "reuse_rows": len(gem_rows),
        "latency_ms": {
            "query_embedding": _distribution(
                _numeric(gem_rows, "latency_ms", "query_embedding")
            ),
            "database_retrieval": _distribution(
                _numeric(gem_rows, "latency_ms", "database_retrieval")
            ),
            "memory_or_prompt_assembly": _distribution(
                _numeric(gem_rows, "latency_ms", "memory_or_prompt_assembly")
            ),
        },
        "operator_work": {
            name: _distribution(_numeric(gem_rows, "intermediate", name))
            for name in (
                "vector_candidates_examined",
                "graph_edges_examined",
                "graph_reached",
                "relational_candidates_examined",
                "relational_candidates_survived",
                "final_results",
            )
        },
    }
    return snapshot


def _metric(snapshot: dict[str, Any], arm: str, name: str, percentile: str) -> float:
    value = snapshot["paired_metrics"][arm][name][percentile]
    return float(value or 0.0)


def render(snapshot: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "EvoMemBench CrossEp-Know — PROVISIONAL / INVALID FOR CLAIMS",
        fontsize=16,
        fontweight="bold",
        color="#9b2226",
    )

    progress = snapshot["progress"]
    labels = ["No Memory", "GEM", "Judge", "Polyglot"]
    keys = ["no_memory", "gem", "judge", "polyglot"]
    fractions = [progress[key]["observed"] / progress[key]["expected"] for key in keys]
    bars = axes[0, 0].barh(
        labels, fractions, color=["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"]
    )
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_title("Pipeline progress")
    for bar, key in zip(bars, keys, strict=True):
        item = progress[key]
        axes[0, 0].text(
            min(0.98, bar.get_width() + 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{item['observed']}/{item['expected']}",
            va="center",
            fontsize=9,
        )

    arm_labels = ["No Memory", "GEM"]
    arm_keys = ["memory_off", "full_gem"]
    colors = ["#457b9d", "#2a9d8f"]
    for axis, metric, title, divisor in (
        (axes[0, 1], "end_to_end_ms", "Paired reuse E2E latency", 1000.0),
        (axes[1, 0], "ttft_ms", "Paired reuse TTFT", 1000.0),
    ):
        p50 = [_metric(snapshot, arm, metric, "p50") / divisor for arm in arm_keys]
        p95 = [_metric(snapshot, arm, metric, "p95") / divisor for arm in arm_keys]
        x = range(2)
        axis.bar(
            [value - 0.18 for value in x], p50, width=0.36, label="p50", color=colors
        )
        axis.bar(
            [value + 0.18 for value in x],
            p95,
            width=0.36,
            label="p95",
            color=colors,
            alpha=0.45,
            hatch="//",
        )
        axis.set_xticks(list(x), arm_labels)
        axis.set_ylabel("seconds")
        axis.set_title(title)
        axis.legend()

    prompt_p50 = [
        _metric(snapshot, arm, "answer_prompt_tokens", "p50") for arm in arm_keys
    ]
    injection_p50 = [
        _metric(snapshot, arm, "memory_injection_tokens", "p50") for arm in arm_keys
    ]
    x = range(2)
    axes[1, 1].bar(
        [value - 0.18 for value in x],
        prompt_p50,
        width=0.36,
        color=colors,
        label="answer prompt p50 (includes injection)",
    )
    axes[1, 1].bar(
        [value + 0.18 for value in x],
        injection_p50,
        width=0.36,
        color=["#a8dadc", "#94d2bd"],
        label="memory injection p50",
    )
    axes[1, 1].set_xticks(list(x), arm_labels)
    axes[1, 1].set_ylabel("tokens")
    axes[1, 1].set_title("Paired reuse token footprint")
    axes[1, 1].legend(fontsize=8)
    fig.text(
        0.5,
        0.005,
        f"paired reuse decisions n={snapshot['paired_reuse_decisions']} · quality=N/A · run incomplete",
        ha="center",
        color="#9b2226",
    )
    fig.savefig(output_dir / "provisional_dashboard.png", dpi=180)
    fig.savefig(output_dir / "provisional_dashboard.svg")
    plt.close(fig)

    retrieval = snapshot["gem_retrieval"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        "Partial GEM retrieval profile — PROVISIONAL",
        fontsize=15,
        fontweight="bold",
        color="#9b2226",
    )
    stage_keys = ["query_embedding", "database_retrieval", "memory_or_prompt_assembly"]
    stage_labels = ["Embedding", "DB retrieval", "Memory total"]
    p50 = [float(retrieval["latency_ms"][key]["p50"] or 0) for key in stage_keys]
    p95 = [float(retrieval["latency_ms"][key]["p95"] or 0) for key in stage_keys]
    x = range(len(stage_keys))
    axes[0].bar([value - 0.18 for value in x], p50, 0.36, label="p50", color="#2a9d8f")
    axes[0].bar([value + 0.18 for value in x], p95, 0.36, label="p95", color="#e9c46a")
    axes[0].set_xticks(list(x), stage_labels, rotation=12)
    axes[0].set_ylabel("milliseconds")
    axes[0].set_title(f"Latency (reuse rows n={retrieval['reuse_rows']})")
    axes[0].legend()

    work = retrieval["operator_work"]
    work_keys = [
        "vector_candidates_examined",
        "graph_edges_examined",
        "relational_candidates_examined",
        "final_results",
    ]
    label_by_key = {
        "vector_candidates_examined": "Vector candidates",
        "graph_edges_examined": "Graph edges",
        "relational_candidates_examined": "Relational candidates",
        "final_results": "Final results",
    }
    work_keys = [key for key in work_keys if int(work[key]["n"]) > 0]
    work_labels = [label_by_key[key] for key in work_keys]
    medians = [float(work[key]["p50"] or 0) for key in work_keys]
    axes[1].barh(work_labels, medians, color=["#457b9d", "#e76f51", "#2a9d8f"])
    axes[1].set_xlabel("items, median")
    axes[1].set_title("Tri-modal operator work")
    axes[1].text(
        0.99,
        0.02,
        "Relational candidate probe: N/A",
        transform=axes[1].transAxes,
        ha="right",
        color="#6c757d",
        fontsize=9,
    )
    fig.savefig(output_dir / "provisional_gem_retrieval.png", dpi=180)
    fig.savefig(output_dir / "provisional_gem_retrieval.svg")
    plt.close(fig)

    readme = f"""# Provisional EvoMemBench visualization

Status: **INVALID FOR CLAIMS**. Captured at `{snapshot["captured_at"]}`.

- No Memory: {progress["no_memory"]["observed"]}/{progress["no_memory"]["expected"]}
- GEM: {progress["gem"]["observed"]}/{progress["gem"]["expected"]}
- Paired reuse decisions in latency plots: {snapshot["paired_reuse_decisions"]}
- Quality: N/A (judge not started)
- Polyglot: not available until parity replay
- Runtime status: {snapshot["warnings"][-1]}
- Relational candidate probe: N/A in current instrumentation; it is not zero.

`metrics_snapshot.json` is the machine-readable source for both figures.
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(build_snapshot(args.run_root), args.output_dir)


if __name__ == "__main__":
    main()
