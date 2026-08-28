"""Generate the final Track C Markdown report from verified local artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .protocol import _write_json, benchmark_code_sha256
from .quality import verify_quality_tree
from .verify import verify_tree

PAPER_REFERENCE = {
    "mem0": {
        "search_mean": 1089.0,
        "search_p90": 1397.0,
        "search_p99": 4637.0,
        "add_mean": 888.0,
        "add_p90": 1650.0,
        "add_p99": 2841.0,
    },
    "memos": {
        "search_mean": 440.5,
        "search_p90": 528.4,
        "search_p99": 777.1,
        "add_mean": 191.9,
        "add_p90": 211.6,
        "add_p99": 376.4,
    },
    "zep": {
        "search_mean": 571.7,
        "search_p90": 614.8,
        "search_p99": 5348.7,
        "add_mean": 239.0,
        "add_p90": 254.5,
        "add_p99": 375.1,
    },
}

SYSTEM_ORDER = (
    "tridb_gem",
    "mem0",
    "memos",
    "cognee",
    "graphiti_zep_oss_proxy",
)
SYSTEM_LABELS = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0",
    "memos": "MemOS",
    "cognee": "Cognee",
    "graphiti_zep_oss_proxy": "Graphiti (Zep OSS proxy)",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _number(value: Any, digits: int = 3) -> str:
    if value in (None, "", "None"):
        return "—"
    return f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    if value in (None, "", "None"):
        return "—"
    return f"{int(float(value)):,}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rendered.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(rendered)


def _pooled(metrics: Sequence[dict[str, str]], phase: str) -> list[dict[str, str]]:
    by_system = {
        row["system"]: row
        for row in metrics
        if row["phase"] == phase and row["build_id"] == "pooled"
    }
    return [by_system[system] for system in SYSTEM_ORDER if system in by_system]


def _latency_table(metrics: Sequence[dict[str, str]], phase: str) -> str:
    def estimate(row: dict[str, str], metric: str) -> str:
        key = f"service_{metric}_ms"
        low = f"service_{metric}_ci_low_ms"
        high = f"service_{metric}_ci_high_ms"
        return f"{_number(row[key])} [{_number(row[low])}, {_number(row[high])}]"

    rows = []
    for row in _pooled(metrics, phase):
        rows.append(
            (
                SYSTEM_LABELS[row["system"]],
                estimate(row, "mean"),
                estimate(row, "p90"),
                estimate(row, "p99"),
                _number(100 * float(row["success_rate"]), 2),
                _number(row["scheduled_qps"]),
                _number(row["admission_qps"]),
                _number(row["actual_qps"]),
                _number(row["admission_lag_p99_ms"]),
            )
        )
    return _table(
        (
            "System",
            "mean [95% CI] (ms)",
            "P90 [95% CI] (ms)",
            "P99 [95% CI] (ms)",
            "success (%)",
            "scheduled QPS",
            "admission QPS",
            "completion QPS",
            "admission-lag P99 (ms)",
        ),
        rows,
    )


def _quality_table(rows: Sequence[dict[str, str]]) -> str:
    pooled = {
        row["system"]: row
        for row in rows
        if row["build_id"] == "pooled" and row["category"] == "overall"
    }
    result = []
    for system in SYSTEM_ORDER:
        if system not in pooled:
            continue
        row = pooled[system]
        result.append(
            (
                SYSTEM_LABELS[system],
                _integer(row["total"]),
                _integer(row["evaluated"]),
                _number(100 * float(row["score"]), 2),
                _number(100 * float(row["conditional_score"]), 2),
                _number(100 * float(row["coverage"]), 2),
            )
        )
    return _table(
        (
            "System",
            "questions",
            "judged",
            "score (%)",
            "conditional score (%)",
            "coverage (%)",
        ),
        result,
    )


def _failure_table(metrics: Sequence[dict[str, str]]) -> str:
    rows = []
    for phase in ("search", "add:native", "add:source_to_searchable"):
        for row in _pooled(metrics, phase):
            rows.append(
                (
                    SYSTEM_LABELS[row["system"]],
                    phase,
                    _integer(row["formal_requests"]),
                    _integer(row["failed"]),
                    row["error_classes"],
                )
            )
    return _table(("System", "Phase", "Admissions", "Failed", "Error classes"), rows)


def _paper_anchor_table(metrics: Sequence[dict[str, str]]) -> str:
    search = {row["system"]: row for row in _pooled(metrics, "search")}
    native = {row["system"]: row for row in _pooled(metrics, "add:native")}
    rows = []
    for system in ("mem0", "memos"):
        paper = PAPER_REFERENCE[system]
        for phase, local, paper_prefix in (
            ("Search", search[system], "search"),
            ("Native-Add", native[system], "add"),
        ):
            for metric in ("mean", "p90", "p99"):
                paper_value = float(paper[f"{paper_prefix}_{metric}"])
                local_value = float(local[f"service_{metric}_ms"])
                rows.append(
                    (
                        SYSTEM_LABELS[system],
                        phase,
                        metric.upper(),
                        _number(paper_value),
                        _number(local_value),
                        _number(local_value / paper_value),
                    )
                )
    for phase, prefix in (("Search", "search"), ("Native-Add", "add")):
        for metric in ("mean", "p90", "p99"):
            rows.append(
                (
                    "Zep",
                    phase,
                    metric.upper(),
                    _number(PAPER_REFERENCE["zep"][f"{prefix}_{metric}"]),
                    "BLOCKED",
                    "—",
                )
            )
    return _table(
        (
            "System",
            "Phase",
            "Metric",
            "Paper (ms)",
            "Local Track C (ms)",
            "Local / paper",
        ),
        rows,
    )


def generate_report(artifact_root: str | Path) -> str:
    root = Path(artifact_root).resolve()
    schedule = yaml.safe_load((root / "formal_schedule.yaml").read_text())
    expected_hash = str(schedule["benchmark_code_sha256"])
    launcher_hash = str(schedule["run_one_script_sha256"])
    protocol_hash = str(schedule["protocol_receipt_sha256"])
    schedule_hash = hashlib.sha256(
        (root / "formal_schedule.yaml").read_bytes()
    ).hexdigest()
    if benchmark_code_sha256() != expected_hash:
        raise RuntimeError("report code does not match the frozen schedule hash")

    verification = verify_tree(
        root / "runs",
        expected_code_hash=expected_hash,
        expected_launcher_hash=launcher_hash,
        expected_schedule_hash=schedule_hash,
        expected_protocol_hash=protocol_hash,
    )
    if not (
        verification["all_found_runs_pass"]
        and verification["formal_tree_exact"]
        and verification["coverage_complete_for_five_systems"]
        and verification["frozen_benchmark_code_hash_matches"]
        and verification["frozen_launcher_hash_matches"]
        and verification["frozen_schedule_hash_matches"]
        and verification["frozen_protocol_hash_matches"]
    ):
        raise RuntimeError("formal run verification is incomplete")
    expected_quality = {
        (str(run["system"]), str(run["build_id"]))
        for run in verification["runs"]
        if run["phase"] == "search" and run["passed"]
    }
    quality_verification = verify_quality_tree(
        root / "quality",
        expected_runs=expected_quality,
        expected_code_hash=expected_hash,
    )
    if not (
        quality_verification["coverage_exact"]
        and quality_verification["all_found_runs_pass"]
    ):
        raise RuntimeError("quality verification is incomplete")

    aggregate_receipt = json.loads(
        (root / "aggregates" / "aggregate_receipt.json").read_text()
    )
    if not (
        aggregate_receipt.get("schema_version") == "table5_track_c_aggregate_v0.2.0"
        and aggregate_receipt.get("benchmark_code_sha256") == expected_hash
        and aggregate_receipt.get("verification_passed") is True
        and aggregate_receipt.get("quality_verification_passed") is True
    ):
        raise RuntimeError("aggregate receipt is not tied to verified inputs")
    expected_aggregate_files = {
        "request_metrics.csv",
        "quality_gate.csv",
        "latency_decomposition.csv",
        "resource_and_build_cost.csv",
        "answer_quality.csv",
        "quality_noninferiority.csv",
    }
    aggregate_hashes = aggregate_receipt.get("output_sha256", {})
    if set(aggregate_hashes) != expected_aggregate_files:
        raise RuntimeError("aggregate receipt does not cover every final CSV")
    for filename, digest in aggregate_hashes.items():
        path = root / "aggregates" / filename
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(f"aggregate hash mismatch: {filename}")

    expected_figures = (
        "search_latency_ecdf",
        "add_latency_ecdf",
        "search_latency_box",
        "search_p99_bar",
        "add_p99_bar",
        "formal_success_rate",
        "latency_quality_frontier",
    )
    missing_figures = [
        f"{stem}.{suffix}"
        for stem in expected_figures
        for suffix in ("pdf", "png")
        if not (root / "figures" / f"{stem}.{suffix}").is_file()
    ]
    if missing_figures:
        raise RuntimeError(f"missing final figures: {missing_figures}")
    figure_receipt = json.loads(
        (root / "figures" / "figure_receipt.json").read_text(encoding="utf-8")
    )
    if not (
        figure_receipt.get("schema_version") == "table5_track_c_figure_receipt_v1"
        and figure_receipt.get("benchmark_code_sha256") == expected_hash
    ):
        raise RuntimeError("figure receipt is not tied to the frozen code hash")
    expected_figure_files = {
        f"{stem}.{suffix}" for stem in expected_figures for suffix in ("pdf", "png")
    }
    figure_hashes = figure_receipt.get("figures", {})
    if set(figure_hashes) != expected_figure_files:
        raise RuntimeError("figure receipt does not cover every final figure")
    for filename, digest in figure_hashes.items():
        path = root / "figures" / filename
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(f"figure hash mismatch: {filename}")

    protocol = yaml.safe_load((root / "protocol_receipt.yaml").read_text())
    metrics = _csv(root / "aggregates" / "request_metrics.csv")
    answer_quality = _csv(root / "aggregates" / "answer_quality.csv")
    noninferiority = _csv(root / "aggregates" / "quality_noninferiority.csv")
    decomposition = _csv(root / "aggregates" / "latency_decomposition.csv")
    resources = _csv(root / "aggregates" / "resource_and_build_cost.csv")

    gates = []
    for row in noninferiority:
        gates.append(
            (
                SYSTEM_LABELS.get(row["reference"], row["reference"]),
                _number(100 * float(row["candidate_score"]), 2),
                _number(100 * float(row["reference_score"]), 2),
                _number(100 * float(row["delta"]), 2),
                _number(100 * float(row["one_sided_lower_bound"]), 2),
                "PASS" if row["passes_noninferiority"] == "True" else "FAIL",
            )
        )

    build_rows = []
    for row in resources:
        if row["phase"] != "search":
            continue
        build_rows.append(
            (
                SYSTEM_LABELS[row["system"]],
                row["build_id"],
                _number(row["build_wall_seconds"]),
                _integer(row["database_or_store_bytes"]),
                _integer(row["build_peak_runner_process_rss_bytes"]),
            )
        )

    fused_rows = []
    for row in decomposition:
        if row["system"] == "tridb_gem" and row["build_id"] == "pooled":
            fused_rows.append(
                (
                    row["phase"],
                    _number(row["candidates_examined_mean"]),
                    _number(row["graph_examined_mean"]),
                    _number(row["bridges_injected_mean"]),
                    row["termination_reasons"],
                )
            )

    deviations = "\n".join(
        f"- {value}" for value in protocol.get("known_deviations", [])
    )
    return f"""# Table 5 Track C controlled-model comparison

状态：完成五个系统的同机 controlled comparison；Graphiti 始终标为 `Graphiti (Zep OSS proxy)`，不声称复刻 production Zep。

## Material Passport

- Dataset: LoCoMo, SHA-256 `{protocol["dataset"]["sha256"]}`
- Formal workload: 每个 system/workload 使用 3 个独立 fresh states；每个 Search state 1,787 requests，每个 Add state 2,000 admissions
- Models: Qwen3-32B FP8（thinking off）+ Qwen3-Embedding-0.6B（1024-d）
- Frozen benchmark code SHA-256: `{expected_hash}`
- Verification: {verification["runs_found"]} run receipts passed; {quality_verification["runs_found"]} quality receipts passed
- Environment: same-host RTX PRO 6000 controlled comparison；不是论文 H800 数值复刻，也不是 GX10 sign-off

## Search（不含最终答案生成）

{_latency_table(metrics, "search")}

## Native-Add

{_latency_table(metrics, "add:native")}

## Source-to-searchable Add

{_latency_table(metrics, "add:source_to_searchable")}

## LoCoMo answer quality

{_quality_table(answer_quality)}

质量分数把 retrieval/answer/judge failure 保留在总分母；conditional score 只描述成功完成 judge 的子集。

## Failure denominator

{_failure_table(metrics)}

### TriDB/GEM 的配对 conversation-cluster 2-point non-inferiority gate

{_table(("Reference", "TriDB score (%)", "Reference score (%)", "Delta (pp)", "One-sided 95% lower (pp)", "Gate"), gates)}

## 与论文 Table 5 的锚点对照

{_paper_anchor_table(metrics)}

论文锚点只帮助比较数量级和排序。本地列使用统一 Qwen 模型与不同硬件，不能解释为论文绝对数值的逐点复刻。

## Search construction cost

{_table(("System", "Build", "wall (s)", "store bytes", "runner peak RSS"), build_rows)}

## TriDB/GEM fused telemetry

{_table(("Phase", "candidates examined", "graph examined", "bridges", "termination reasons"), fused_rows)}

完整的 per-build CI、scheduler drift、queue/user-visible latency、错误类别、分解和资源数据分别见：

- `aggregates/request_metrics.csv`
- `aggregates/latency_decomposition.csv`
- `aggregates/resource_and_build_cost.csv`
- `aggregates/quality_gate.csv`
- `aggregates/answer_quality.csv`
- `aggregates/quality_noninferiority.csv`
- `runs/verification.json`

## Zep boundary

Production Zep Context Graph 在可获得发行形态中是 managed/proprietary，本机没有凭证，且不能把 managed Zep 指向本地冻结模型。经用户批准，本报告只加入开源 Graphiti，并始终使用 `Graphiti (Zep OSS proxy)` 标签；它不是论文 Table 5 的 production Zep 数值复刻。

## 已知偏差

{deviations}

此外，本轮没有对共享 PostgreSQL/model-server 进程实施整机 CPU affinity 或 RAM hard cap；公平性依靠同一主机、同一组常驻模型服务、固定 GPU 绑定和一次只运行一个 measured system。该限制禁止把小幅差异过度解释为系统性优势。

## 图表

- `figures/search_latency_ecdf.pdf`
- `figures/search_p99_bar.pdf`
- `figures/add_latency_ecdf.pdf`
- `figures/add_p99_bar.pdf`
- `figures/search_latency_box.pdf`
- `figures/formal_success_rate.pdf`
- `figures/latency_quality_frontier.pdf`
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.artifact_root).resolve()
    output = Path(args.output).resolve() if args.output else root / "REPORT.md"
    report = generate_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    receipt = {
        "schema_version": "table5_track_c_report_receipt_v1",
        "report": str(output),
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "benchmark_code_sha256": benchmark_code_sha256(),
        "aggregate_receipt_sha256": hashlib.sha256(
            (root / "aggregates" / "aggregate_receipt.json").read_bytes()
        ).hexdigest(),
        "figure_receipt_sha256": hashlib.sha256(
            (root / "figures" / "figure_receipt.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
