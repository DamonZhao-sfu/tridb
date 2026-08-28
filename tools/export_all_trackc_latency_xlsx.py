#!/usr/bin/env python3
"""Export every collected Track C Search/Add latency receipt to one XLSX.

The workbook separates the selected Qwen3-32B QPS sweep from exploratory
Qwen3.8 results and preserves every duplicate, preliminary, failed, and partial
receipt in an audit sheet.  Incomplete receipt statistics are derived from the
already-written formal.jsonl only and are labelled provisional.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "bench/out"
DEFAULT_OUTPUT = REPO / "haikaidocs/all_system_qps_search_add_summary_2026-08-23.xlsx"
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
ANSWER_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

SYSTEMS = {
    "tridb_gem": ("TriDB/GEM", "TriDB branch 82f963f"),
    "mem0": ("Mem0", "2.0.18"),
    "memos": ("MemOS", "2.0.30"),
    "cognee": ("Cognee", "1.5.0"),
    "mandol": ("Mandol", "0.1.0 / commit 407247b"),
    "graphiti_zep_oss_proxy": (
        "Graphiti (Zep OSS proxy; not production Zep)",
        "graphiti-core 0.29.3",
    ),
    "evermemos": ("EverMemOS (EverOS OSS)", "commit 48fc908"),
    "evermemos_paper_proxy": (
        "EverMemOS (paper-era official-network fork)",
        "commit 806ad055",
    ),
}

QWEN32_ROOT = "table5_track_c_qps_sweep_snapshot_2026_08_20_v2"
PREPARED_ADD_ROOT = "expc_prepared_item_add_2026_08_23_v10"
EVERMEMOS_QWEN32_SEARCH_ROOT = "expc_evermemos_qwen32_tp2_gpu01_2026_08_23_v9"
COGNEE_TIMEOUT300_ROOT = "table5_reproduction_2026_08_19/cognee_gpu0_search_timeout300_v1"
SEARCH_RETRY_ROOT_PREFIX = "expc_search_timeout300_retries_2026_08_23_v14_"
MANDOL_SEARCH_ROOT_PREFIX = "table5_track_c_mandol_search_resume_2026_08_23_v"
GRAPHITI_QWEN32_ROOT_PREFIX = "expc_graphiti_llm_free_2026_08_23_v14m"
GRAPHITI_QWEN38_ROOT_PREFIX = "expc_graphiti_qwen38_search_add_2026_08_23_v16"
GRAPHITI_LLM_FREE_ROOT_PREFIXES = (
    GRAPHITI_QWEN32_ROOT_PREFIX,
    GRAPHITI_QWEN38_ROOT_PREFIX,
)
QWEN32_LEGACY_ROOTS = {
    "table5_reproduction_2026_08_19",
    "table5_track_c_qps_sweep_2026_08_20_v1",
    QWEN32_ROOT,
}

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
GOOD_FILL = PatternFill("solid", fgColor="E2F0D9")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="F4CCCC")
INFO_FILL = PatternFill("solid", fgColor="D9EAF7")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def argv_value(receipt: dict[str, Any], flag: str) -> str | None:
    argv = nested(receipt, "execution", "argv") or []
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return str(argv[index + 1]) if index + 1 < len(argv) else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean": sum(rows) / len(rows),
        "p50": percentile(rows, 0.50),
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
        "max": max(rows),
    }


def derive_partial(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "formal.jsonl"
    if not path.is_file():
        return None
    records: list[dict[str, Any]] = []
    parse_errors = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    if not records:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": None,
            "errors": {},
            "service_latency_ms": {"count": 0},
            "user_visible_latency_ms": {"count": 0},
            "queue_latency_ms": {"count": 0},
            "parse_errors": parse_errors,
        }
    successful = [row for row in records if row.get("success") is True]
    errors = Counter(
        str(row.get("error") or "unspecified failure")
        for row in records
        if row.get("success") is not True
    )
    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "success_rate": len(successful) / len(records),
        "errors": dict(errors),
        "service_latency_ms": distribution(
            row["service_latency_ms"]
            for row in successful
            if isinstance(row.get("service_latency_ms"), (int, float))
        ),
        "user_visible_latency_ms": distribution(
            row["user_visible_latency_ms"]
            for row in records
            if isinstance(row.get("user_visible_latency_ms"), (int, float))
        ),
        "queue_latency_ms": distribution(
            row["queue_latency_ms"]
            for row in records
            if isinstance(row.get("queue_latency_ms"), (int, float))
        ),
        "parse_errors": parse_errors,
    }


def timeout_count(summary: dict[str, Any]) -> int:
    return sum(
        int(count)
        for error, count in (summary.get("errors") or {}).items()
        if "timeout" in str(error).lower()
    )


def result_root_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root)
    return relative.parts[0]


def selected_canonical(path: Path, receipt: dict[str, Any], source_root: Path) -> bool:
    relative = path.relative_to(source_root).as_posix()
    root = result_root_name(path, source_root)
    system = receipt.get("system")
    phase = receipt.get("phase")
    if relative.startswith(f"{QWEN32_ROOT}/runs/"):
        # The prepared-item runs supersede the earlier opaque/native Mem0 and
        # MemOS Add points.  Cognee Search@10 was rerun with a 300 s timeout so
        # that every admitted query receives a terminal outcome.
        if system in {"mem0", "memos"} and phase == "add:native":
            return False
        if system == "cognee" and phase == "search":
            qps = nested(receipt, "protocol", "qps")
            if qps == 10.0:
                # QPS=10 is superseded by the complete timeout-300 rerun.
                return False
            if qps == 5.0 and any(
                source_root.glob(
                    f"{SEARCH_RETRY_ROOT_PREFIX}*/**/run_receipt.json"
                )
            ):
                return False
        if system == "memos" and phase == "search" and nested(
            receipt, "protocol", "qps"
        ) == 10.0 and any(
            source_root.glob(f"{SEARCH_RETRY_ROOT_PREFIX}*/**/run_receipt.json")
        ):
            return False
        return True
    if relative.startswith(f"{PREPARED_ADD_ROOT}/"):
        return system in {"mem0", "memos", "evermemos_paper_proxy"} and phase == "add:native"
    if relative.startswith(f"{EVERMEMOS_QWEN32_SEARCH_ROOT}/"):
        return system == "evermemos_paper_proxy" and phase == "search"
    if relative.startswith(f"{COGNEE_TIMEOUT300_ROOT}/"):
        return system == "cognee" and phase == "search"
    if root.startswith(SEARCH_RETRY_ROOT_PREFIX):
        return (
            phase == "search"
            and (
                system == "cognee"
                and nested(receipt, "protocol", "qps") == 5.0
                or system == "memos"
                and nested(receipt, "protocol", "qps") == 10.0
            )
        )
    if root.startswith(MANDOL_SEARCH_ROOT_PREFIX):
        return system == "mandol" and phase == "search"
    if root.startswith(GRAPHITI_LLM_FREE_ROOT_PREFIXES):
        return system == "graphiti_zep_oss_proxy" and phase in {
            "search",
            "add:native",
        }
    if system == "graphiti_zep_oss_proxy":
        if any(
            any(source_root.glob(f"{prefix}*/graphiti/**/run_receipt.json"))
            for prefix in GRAPHITI_LLM_FREE_ROOT_PREFIXES
        ):
            return False
        if root.startswith("expc_graphiti_qwen32_gpu1_"):
            return phase in {"search", "add:native"}
        if phase == "search":
            return "/search_v1_no_timestamps/" in f"/{relative}"
        if phase == "add:native":
            return "/add_native/" in f"/{relative}"
    if system == "evermemos_paper_proxy" and root.startswith(
        "expc_evermemos_qwen32_gpu1_"
    ):
        return phase in {"search", "add:native"}
    if relative.startswith("expc_evermemos_locomo_2026_08_21/search/"):
        return not any(
            source_root.glob(
                f"{EVERMEMOS_QWEN32_SEARCH_ROOT}/**/run_receipt.json"
            )
        )
    return False


def model_identity(
    path: Path, receipt: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    root = result_root_name(path, source_root)
    answer_model = argv_value(receipt, "--answer-model")
    embedding_model = argv_value(receipt, "--embedding-model")
    answer_endpoint = argv_value(receipt, "--answer-base-url")
    embedding_endpoint = argv_value(receipt, "--embedding-base-url")
    if (
        root in QWEN32_LEGACY_ROOTS
        or root.startswith(SEARCH_RETRY_ROOT_PREFIX)
        or root.startswith(MANDOL_SEARCH_ROOT_PREFIX)
        or root.startswith(GRAPHITI_QWEN32_ROOT_PREFIX)
        or answer_model == "Qwen/Qwen3-32B"
    ):
        evidence = (
            "per-run argv plus frozen root model receipts/protocol"
            if root == "table5_reproduction_2026_08_19"
            else "per-run endpoints plus shared Qwen32 launch logs/protocol; revision not repeated in receipt"
        )
        return {
            "cohort": "Qwen3-32B-FP8 controlled",
            "answer_artifact": "Qwen/Qwen3-32B-FP8",
            "answer_served": answer_model or "Qwen/Qwen3-32B",
            "answer_revision": ANSWER_REVISION,
            "answer_endpoint": answer_endpoint or "http://127.0.0.1:8000/v1",
            "embedding_artifact": embedding_model or "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": EMBEDDING_REVISION,
            "embedding_endpoint": embedding_endpoint or "http://127.0.0.1:8001/v1",
            "evidence": evidence,
        }

    if answer_model == "qwen3.8" or root.startswith("expc_"):
        return {
            "cohort": "Qwen3.8 exploratory",
            "answer_artifact": "Qwen/Qwen3.8-27B-FP8",
            "answer_served": answer_model or "qwen3.8",
            "answer_revision": "not recorded",
            "answer_endpoint": answer_endpoint or "not recorded",
            "embedding_artifact": embedding_model or "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "not recorded",
            "embedding_endpoint": embedding_endpoint or "not recorded",
            "evidence": "explicit per-run argv; artifact from preserved launcher; revisions absent",
        }

    return {
        "cohort": "unknown/unbound",
        "answer_artifact": answer_model or "not recorded",
        "answer_served": answer_model or "not recorded",
        "answer_revision": "not recorded",
        "answer_endpoint": answer_endpoint or "not recorded",
        "embedding_artifact": embedding_model or "not recorded",
        "embedding_revision": "not recorded",
        "embedding_endpoint": embedding_endpoint or "not recorded",
        "evidence": "insufficient receipt/model provenance",
    }


def suitability(path: Path, receipt: dict[str, Any], selected: bool) -> tuple[str, str]:
    status = str(receipt.get("status") or "unknown")
    summary = receipt.get("formal_summary") or {}
    if status != "complete":
        return "NO", f"receipt status={status}; partial statistics are provisional"
    if "/failures/" in f"/{path.as_posix()}":
        return "NO", "preserved failed-attempt directory"
    if (
        "/preliminary/" in f"/{path.as_posix()}"
        or "/reduced_v1/" in f"/{path.as_posix()}"
    ):
        return "NO", "preliminary/reduced superseded run"
    if int(summary.get("successful") or 0) == 0:
        return "NO", "zero successful requests; service percentiles unavailable"
    total = int(summary.get("total") or 0)
    failed = int(summary.get("failed") or 0)
    if failed:
        return (
            "DEGRADED",
            f"{failed}/{total} requests failed; service latency is success-only and total latency includes failures",
        )
    target_qps = nested(receipt, "protocol", "qps")
    completion_qps = summary.get("completion_throughput_qps") or summary.get(
        "actual_qps"
    )
    if (
        isinstance(target_qps, (int, float))
        and isinstance(completion_qps, (int, float))
        and completion_qps < 0.9 * target_qps
    ):
        return (
            "OVERLOADED",
            f"all requests terminated, but completion throughput {completion_qps:.3f} QPS is below 90% of target {target_qps:g} QPS",
        )
    if receipt.get("system") == "graphiti_zep_oss_proxy":
        conformance = receipt.get("llm_conformance") or {}
        if (
            selected
            and receipt.get("phase") == "add:native"
            and conformance.get("passed") is True
            and conformance.get("llm_call_count") == 0
        ):
            return (
                "EXPLORATORY",
                "selected Graphiti prepared-item Add; generation LLM forbidden fail-closed; compare only within the recorded Qwen3.8 cohort",
            )
        if argv_value(receipt, "--answer-model") == "Qwen/Qwen3-32B" and selected:
            return (
                "YES",
                "selected Qwen32 Graphiti OSS proxy point with formal per-group FIFO guard",
            )
        if receipt.get("phase") == "add:native":
            return "NO", "exploratory Graphiti Add predates formal per-group FIFO guard"
        return (
            "EXPLORATORY",
            "Qwen3.8 and not production Zep; compare only within cohort",
        )
    if receipt.get("system") == "evermemos":
        return "EXPLORATORY", "Qwen3.8 EverOS exploratory point; no Qwen32 counterpart"
    if selected:
        return "YES", "selected Qwen32 QPS-sweep point"
    return "NO", "duplicate or superseded receipt retained for audit"


def stage_role(receipt: dict[str, Any]) -> str:
    conformance = receipt.get("llm_conformance") or {}
    if conformance.get("passed") is True and conformance.get("llm_call_count") == 0:
        return "prepared-item insertion; LLM forbidden fail-closed; observed LLM calls=0"
    stages = nested(receipt, "time_breakdown", "stages") or {}
    if "llm" in stages:
        return "LLM span included in measured request"
    if receipt.get("phase") == "search":
        return "answer LLM excluded from timed retrieval; model may affect construction"
    return (
        "no explicit LLM span; native/opaque implementation may include internal work"
    )


def receipt_row(path: Path, source_root: Path) -> dict[str, Any]:
    receipt = load_json(path)
    run_dir = path.parent
    summary = receipt.get("formal_summary")
    metric_source = "receipt formal_summary"
    if not isinstance(summary, dict) or not isinstance(summary.get("total"), int):
        summary = derive_partial(run_dir)
        metric_source = (
            "provisional: derived from existing formal.jsonl"
            if summary is not None
            else "no collected formal latency"
        )
    summary = summary or {}
    model = model_identity(path, receipt, source_root)
    selected = selected_canonical(path, receipt, source_root)
    headline, reason = suitability(path, receipt, selected)
    label, version = SYSTEMS.get(
        str(receipt.get("system")),
        (str(receipt.get("system") or "unknown"), "not recorded"),
    )
    return {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(REPO).as_posix(),
        "root": result_root_name(path, source_root),
        "receipt": receipt,
        "summary": summary,
        "metric_source": metric_source,
        "selected": selected,
        "headline": headline,
        "headline_reason": reason,
        "system": receipt.get("system"),
        "system_label": label,
        "system_version": version,
        "phase": receipt.get("phase"),
        "workload": (
            "Native Add" if receipt.get("phase") == "add:native" else "Search"
        ),
        "qps": nested(receipt, "protocol", "qps"),
        "timeout_seconds": nested(receipt, "protocol", "timeout_seconds"),
        "status": receipt.get("status"),
        "dataset_sha256": nested(receipt, "dataset", "sha256"),
        "dataset_match": nested(receipt, "dataset", "sha256") == DATASET_SHA256,
        "model": model,
        "llm_role": stage_role(receipt),
    }


def ms_to_s(value: Any) -> float | None:
    return None if not isinstance(value, (int, float)) else float(value) / 1000.0


def stat(row: dict[str, Any], family: str, key: str) -> Any:
    return nested(row, "summary", family, key)


SUMMARY_HEADERS = [
    "System",
    "Version/source",
    "Workload",
    "Target QPS",
    "Receipt status",
    "Headline suitability",
    "Reason / limitation",
    "Metric source",
    "Formal collected",
    "Successful",
    "Failed",
    "Timeouts",
    "Success rate",
    "Admission QPS",
    "Completion QPS",
    "Service mean (s; success-only)",
    "Service P50 (s; success-only)",
    "Service P90 (s; success-only)",
    "Service P95 (s; success-only)",
    "Service P99 (s; success-only)",
    "Service max (s; success-only)",
    "Total/user-visible mean (s; all)",
    "Total/user-visible P50 (s; all)",
    "Total/user-visible P90 (s; all)",
    "Total/user-visible P95 (s; all)",
    "Total/user-visible P99 (s; all)",
    "Total/user-visible max (s; all)",
    "Queue mean (s)",
    "Queue P99 (s)",
    "TTFT (s)",
    "TTFT status",
    "Evidence Recall@20",
    "Evidence any-hit@20",
    "Model cohort",
    "Answer artifact",
    "Answer served name",
    "Answer revision",
    "Answer endpoint",
    "Embedding artifact",
    "Embedding revision",
    "Embedding endpoint",
    "LLM role in measured request",
    "LLM conformance",
    "Observed LLM calls",
    "Semantic boundary",
    "Model provenance",
    "Build ID",
    "Dataset SHA256",
    "Started at",
    "Completed at",
    "Receipt path",
]


def summary_values(row: dict[str, Any]) -> list[Any]:
    summary = row["summary"]
    receipt = row["receipt"]
    return [
        row["system_label"],
        row["system_version"],
        row["workload"],
        row["qps"],
        row["status"],
        row["headline"],
        row["headline_reason"],
        row["metric_source"],
        summary.get("total"),
        summary.get("successful"),
        summary.get("failed"),
        timeout_count(summary),
        summary.get("success_rate"),
        summary.get("admission_qps") or nested(receipt, "protocol", "qps"),
        summary.get("completion_throughput_qps") or summary.get("actual_qps"),
        ms_to_s(stat(row, "service_latency_ms", "mean")),
        ms_to_s(stat(row, "service_latency_ms", "p50")),
        ms_to_s(stat(row, "service_latency_ms", "p90")),
        ms_to_s(stat(row, "service_latency_ms", "p95")),
        ms_to_s(stat(row, "service_latency_ms", "p99")),
        ms_to_s(stat(row, "service_latency_ms", "max")),
        ms_to_s(stat(row, "user_visible_latency_ms", "mean")),
        ms_to_s(stat(row, "user_visible_latency_ms", "p50")),
        ms_to_s(stat(row, "user_visible_latency_ms", "p90")),
        ms_to_s(stat(row, "user_visible_latency_ms", "p95")),
        ms_to_s(stat(row, "user_visible_latency_ms", "p99")),
        ms_to_s(stat(row, "user_visible_latency_ms", "max")),
        ms_to_s(stat(row, "queue_latency_ms", "mean")),
        ms_to_s(stat(row, "queue_latency_ms", "p99")),
        None,
        "not instrumented; Search measures retrieval only",
        nested(receipt, "quality", "evidence_recall_at_20"),
        nested(receipt, "quality", "evidence_any_hit_at_20"),
        row["model"]["cohort"],
        row["model"]["answer_artifact"],
        row["model"]["answer_served"],
        row["model"]["answer_revision"],
        row["model"]["answer_endpoint"],
        row["model"]["embedding_artifact"],
        row["model"]["embedding_revision"],
        row["model"]["embedding_endpoint"],
        row["llm_role"],
        nested(receipt, "llm_conformance", "passed"),
        nested(receipt, "llm_conformance", "llm_call_count"),
        receipt.get("semantic_boundary"),
        row["model"]["evidence"],
        receipt.get("build_id"),
        row["dataset_sha256"],
        receipt.get("started_at"),
        receipt.get("completed_at") or receipt.get("failed_at"),
        row["path"],
    ]


def configure_sheet(sheet: Any, *, freeze: str = "A2") -> None:
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for column in range(1, sheet.max_column + 1):
        width = max(
            len(str(sheet.cell(row=row, column=column).value or ""))
            for row in range(1, min(sheet.max_row, 200) + 1)
        )
        sheet.column_dimensions[get_column_letter(column)].width = min(
            55, max(10, width + 2)
        )


def style_summary(sheet: Any) -> None:
    configure_sheet(sheet)
    for row in range(2, sheet.max_row + 1):
        suitability_cell = sheet.cell(row, 6)
        suitability_cell.fill = (
            GOOD_FILL
            if suitability_cell.value == "YES"
            else INFO_FILL
            if suitability_cell.value == "EXPLORATORY"
            else WARN_FILL
            if suitability_cell.value in {"DEGRADED", "OVERLOADED"}
            else BAD_FILL
        )
        sheet.cell(row, 13).number_format = "0.00%"
        for column in range(16, 30):
            sheet.cell(row, column).number_format = "0.000"
        for column in (32, 33):
            sheet.cell(row, column).number_format = "0.00%"


def write_summary_sheet(
    workbook: Workbook, name: str, rows: list[dict[str, Any]]
) -> Any:
    sheet = workbook.create_sheet(name)
    sheet.append(SUMMARY_HEADERS)
    for row in rows:
        sheet.append(summary_values(row))
    style_summary(sheet)
    return sheet


COMPARISON_HEADERS = [
    "System",
    "Version/source",
    "Target QPS",
    "Model cohort",
    "Receipt status",
    "Usability",
    "Reason / limitation",
    "Formal collected",
    "Successful",
    "Failed",
    "Timeouts",
    "Success rate",
    "Admission QPS",
    "Completion QPS",
    "Service mean (ms; success-only)",
    "Service P50 (ms; success-only)",
    "Service P95 (ms; success-only)",
    "Service P99 (ms; success-only)",
    "Total mean (ms; all)",
    "Total P99 (ms; all)",
    "Queue P99 (ms)",
    "LLM role in measured request",
    "LLM conformance",
    "Observed LLM calls",
    "Semantic boundary",
    "Receipt path",
]


HEADLINE_SYSTEMS = [
    "tridb_gem",
    "mem0",
    "memos",
    "cognee",
    "mandol",
    "graphiti_zep_oss_proxy",
    "evermemos_paper_proxy",
]


def comparison_values(row: dict[str, Any]) -> list[Any]:
    summary = row["summary"]
    receipt = row["receipt"]
    return [
        row["system_label"],
        row["system_version"],
        row["qps"],
        row["model"]["cohort"],
        row["status"],
        row["headline"],
        row["headline_reason"],
        summary.get("total"),
        summary.get("successful"),
        summary.get("failed"),
        timeout_count(summary),
        summary.get("success_rate"),
        summary.get("admission_qps") or nested(receipt, "protocol", "qps"),
        summary.get("completion_throughput_qps") or summary.get("actual_qps"),
        stat(row, "service_latency_ms", "mean"),
        stat(row, "service_latency_ms", "p50"),
        stat(row, "service_latency_ms", "p95"),
        stat(row, "service_latency_ms", "p99"),
        stat(row, "user_visible_latency_ms", "mean"),
        stat(row, "user_visible_latency_ms", "p99"),
        stat(row, "queue_latency_ms", "p99"),
        row["llm_role"],
        nested(receipt, "llm_conformance", "passed"),
        nested(receipt, "llm_conformance", "llm_call_count"),
        receipt.get("semantic_boundary"),
        row["path"],
    ]


def write_comparison_sheet(
    workbook: Workbook,
    name: str,
    rows: list[dict[str, Any]],
    workload: str,
) -> Any:
    sheet = workbook.create_sheet(name)
    sheet.append(COMPARISON_HEADERS)
    selected = {
        (row["system"], int(row["qps"])): row
        for row in rows
        if row["selected"]
        and row["workload"] == workload
        and row["qps"] is not None
    }
    for system in HEADLINE_SYSTEMS:
        label, version = SYSTEMS[system]
        for qps in (1, 5, 10):
            row = selected.get((system, qps))
            if row is None:
                sheet.append(
                    [
                        label,
                        version,
                        qps,
                        None,
                        "MISSING",
                        "NO",
                        "No selected formal receipt collected",
                    ]
                )
            else:
                sheet.append(comparison_values(row))
    configure_sheet(sheet)
    for row_index in range(2, sheet.max_row + 1):
        usability = sheet.cell(row_index, 6)
        usability.fill = (
            GOOD_FILL
            if usability.value == "YES"
            else INFO_FILL
            if usability.value == "EXPLORATORY"
            else WARN_FILL
            if usability.value in {"DEGRADED", "OVERLOADED"}
            else BAD_FILL
        )
        sheet.cell(row_index, 12).number_format = "0.00%"
        for column in range(13, 22):
            sheet.cell(row_index, column).number_format = "0.000"
    return sheet


def write_readme(
    workbook: Workbook, rows: list[dict[str, Any]], generated_at: str
) -> None:
    sheet = workbook.create_sheet("README")
    selected = [row for row in rows if row["selected"]]
    qwen32 = [row for row in selected if row["model"]["cohort"].startswith("Qwen3-32B")]
    other = [row for row in selected if row not in qwen32]
    partial = [row for row in rows if row["status"] != "complete"]
    content = [
        ("Workbook", "All collected Track C Search/Native-Add latency inventory"),
        ("Generated UTC", generated_at),
        ("Source root", str(DEFAULT_SOURCE.resolve())),
        ("Run receipts discovered", len(rows)),
        ("Selected QPS points", len(selected)),
        ("Selected Qwen32 points", len(qwen32)),
        ("Selected other-model exploratory points", len(other)),
        ("Incomplete/failed receipts retained", len(partial)),
        (
            "Dataset",
            "LoCoMo: 10 conversations, 5,882 events, 1,787 formal Search queries",
        ),
        (
            "Latency headline",
            "Service latency is success-only; Total/user-visible latency includes failures and client timeout",
        ),
        (
            "TTFT",
            "Not measured by these retrieval/Add workloads; cells intentionally blank",
        ),
        (
            "Search model role",
            "Answer generation is outside timed Search; construction may still use the listed LLM",
        ),
        (
            "Native Add model role",
            "Mem0, MemOS, and EverMemOS selected Add points use prepared-memory-item insertion with fail-closed zero-LLM conformance; other systems retain their recorded native boundary",
        ),
        (
            "Qwen32 comparison",
            "Use 'Qwen32 Selected' only; it excludes Graphiti/EverMemOS Qwen3.8 rows",
        ),
        (
            "Graphiti boundary",
            "Graphiti is an OSS proxy and must not be labelled production Zep",
        ),
        (
            "Graphiti Add warning",
            "Selected Graphiti Add uses prepared-item insertion with fail-closed zero-LLM conformance; old failed high-level rows remain audit-only",
        ),
        (
            "EverMemOS warning",
            "Paper-era EverMemOS Qwen32 Search and prepared-item Add are selected; the older EverOS OSS point remains audit-only",
        ),
        (
            "Graphiti model cohort",
            "Current v16 Graphiti uses the recorded Qwen3.8 canonical graph and is kept outside the Qwen32 headline cohort; measured Search/prepared Add forbid generation LLM calls",
        ),
        (
            "Mandol Search",
            "No formal Search QPS=1/5/10 receipts are currently available; cells remain MISSING",
        ),
        (
            "Cognee Search@10",
            "Uses the complete 300-second-timeout rerun; the earlier 60-second partial run remains in All Receipts",
        ),
        (
            "Hardware claim",
            "Same-host workstation measurements; not paper H800 and not GX10 sign-off",
        ),
        (
            "Partial rows",
            "Derived only from existing formal.jsonl, marked provisional and excluded from headline",
        ),
    ]
    sheet.append(["Field", "Value"])
    for key, value in content:
        sheet.append([key, value])
    configure_sheet(sheet)
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 110


def write_coverage(workbook: Workbook, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("Coverage Matrix")
    headers = ["System"] + [
        f"{workload}@{qps}"
        for workload in ("Search", "Native Add")
        for qps in (1, 5, 10)
    ]
    sheet.append(headers)
    selected = {
        (row["system"], row["workload"], int(row["qps"])): row
        for row in rows
        if row["selected"] and row["qps"] is not None
    }
    for system in HEADLINE_SYSTEMS:
        label = SYSTEMS[system][0]
        values: list[Any] = [label]
        for workload in ("Search", "Native Add"):
            for qps in (1, 5, 10):
                row = selected.get((system, workload, qps))
                if row is None:
                    values.append("MISSING")
                else:
                    successful = row["summary"].get("successful")
                    total = row["summary"].get("total")
                    values.append(f"{row['status']}: {successful}/{total}")
        sheet.append(values)
    configure_sheet(sheet)
    for row_index, system in enumerate(HEADLINE_SYSTEMS, start=2):
        for offset, workload in enumerate(("Search", "Native Add")):
            for qps_index, qps in enumerate((1, 5, 10)):
                cell = sheet.cell(row_index, 2 + offset * 3 + qps_index)
                selected_row = selected.get((system, workload, qps))
                if selected_row is None or selected_row["headline"] == "NO":
                    cell.fill = BAD_FILL
                elif selected_row["headline"] in {"DEGRADED", "OVERLOADED"}:
                    cell.fill = WARN_FILL
                elif selected_row["headline"] == "EXPLORATORY":
                    cell.fill = INFO_FILL
                else:
                    cell.fill = GOOD_FILL


def write_stage_breakdown(workbook: Workbook, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("Stage Breakdown")
    headers = [
        "Selected",
        "System",
        "Workload",
        "QPS",
        "Model cohort",
        "Receipt status",
        "Stage",
        "Requests with stage",
        "Spans",
        "Stage mean (s)",
        "Stage P50 (s)",
        "Stage P90 (s)",
        "Stage P99 (s)",
        "Stage work (s; may overlap)",
        "Breakdown schema",
        "Receipt path",
    ]
    sheet.append(headers)
    for row in rows:
        breakdown = row["receipt"].get("time_breakdown") or {}
        for stage, payload in sorted((breakdown.get("stages") or {}).items()):
            union = payload.get("per_request_union_ms") or {}
            sheet.append(
                [
                    row["selected"],
                    row["system_label"],
                    row["workload"],
                    row["qps"],
                    row["model"]["cohort"],
                    row["status"],
                    stage,
                    payload.get("request_count"),
                    payload.get("span_count"),
                    ms_to_s(union.get("mean")),
                    ms_to_s(union.get("p50")),
                    ms_to_s(union.get("p90")),
                    ms_to_s(union.get("p99")),
                    ms_to_s(payload.get("stage_work_ms")),
                    breakdown.get("schema_version"),
                    row["path"],
                ]
            )
    configure_sheet(sheet)
    for row in range(2, sheet.max_row + 1):
        for column in range(10, 15):
            sheet.cell(row, column).number_format = "0.000"


def write_models(workbook: Workbook, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("Model Provenance")
    headers = [
        "Model cohort",
        "Answer artifact",
        "Answer served name",
        "Answer revision",
        "Embedding artifact",
        "Embedding revision",
        "Evidence",
        "Receipt count",
        "Selected point count",
        "Interpretation",
    ]
    sheet.append(headers)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        model = row["model"]
        key = (
            model["cohort"],
            model["answer_artifact"],
            model["answer_served"],
            model["answer_revision"],
            model["embedding_artifact"],
            model["embedding_revision"],
            model["evidence"],
        )
        grouped.setdefault(key, []).append(row)
    for key, cohort_rows in sorted(grouped.items()):
        interpretation = (
            "controlled Qwen32 cohort"
            if key[0].startswith("Qwen3-32B")
            else "exploratory only; do not mix with Qwen32"
        )
        sheet.append(
            [
                *key,
                len(cohort_rows),
                sum(row["selected"] for row in cohort_rows),
                interpretation,
            ]
        )
    configure_sheet(sheet)


def write_all_receipts(workbook: Workbook, rows: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("All Receipts")
    headers = [
        "Selected point",
        "Result root",
        *SUMMARY_HEADERS,
        "Raw phase",
        "Timeout seconds",
        "Dataset match",
        "Formal JSONL parse errors",
        "Errors JSON",
        "Benchmark code SHA256",
        "Git commit",
        "Relative receipt path",
    ]
    sheet.append(headers)
    for row in rows:
        summary = row["summary"]
        sheet.append(
            [
                row["selected"],
                row["root"],
                *summary_values(row),
                row["phase"],
                row["timeout_seconds"],
                row["dataset_match"],
                summary.get("parse_errors"),
                json.dumps(
                    summary.get("errors") or {}, ensure_ascii=False, sort_keys=True
                ),
                nested(row, "receipt", "execution", "benchmark_code_sha256"),
                nested(row, "receipt", "execution", "git_commit"),
                row["relative_path"],
            ]
        )
    configure_sheet(sheet)


def add_latency_chart(sheet: Any, title: str) -> None:
    if sheet.max_row < 2:
        return
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = title
    chart.y_axis.title = "Service mean latency (s)"
    chart.x_axis.title = "System / workload / QPS"
    chart.height = 8
    chart.width = 18
    service_mean_column = SUMMARY_HEADERS.index("Service mean (s; success-only)") + 1
    chart.add_data(
        Reference(
            sheet,
            min_col=service_mean_column,
            max_col=service_mean_column,
            min_row=1,
            max_row=sheet.max_row,
        ),
        titles_from_data=True,
    )
    chart.set_categories(Reference(sheet, min_col=1, min_row=2, max_row=sheet.max_row))
    sheet.add_chart(chart, "A40")


def build_workbook(source_root: Path, output: Path) -> dict[str, Any]:
    receipt_paths = sorted(source_root.glob("**/run_receipt.json"))
    rows = [receipt_row(path, source_root) for path in receipt_paths]
    rows.sort(
        key=lambda row: (
            row["system_label"],
            row["workload"],
            float(row["qps"] or -1),
            row["relative_path"],
        )
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = "All Track C system/QPS latency results"
    workbook.properties.creator = "TriDB Track C audit exporter"
    workbook.properties.description = (
        "All collected Search/Native-Add latency receipts with model provenance"
    )
    write_readme(workbook, rows, generated_at)
    selected = [row for row in rows if row["selected"]]
    qwen32 = [
        row for row in selected if row["model"]["cohort"] == "Qwen3-32B-FP8 controlled"
    ]
    other = [row for row in selected if row not in qwen32]
    qwen_sheet = write_summary_sheet(workbook, "Qwen32 Selected", qwen32)
    write_summary_sheet(workbook, "Other Model Selected", other)
    write_comparison_sheet(workbook, "Search Comparison", rows, "Search")
    write_comparison_sheet(workbook, "Add Comparison", rows, "Native Add")
    write_coverage(workbook, rows)
    write_stage_breakdown(workbook, rows)
    write_models(workbook, rows)
    write_all_receipts(workbook, rows)
    add_latency_chart(qwen_sheet, "Selected Qwen32 service mean latency")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    workbook.save(temporary)
    os.replace(temporary, output)

    reopened = load_workbook(output, read_only=True, data_only=True)
    validation = {
        "output": str(output.resolve()),
        "generated_at": generated_at,
        "receipt_count": len(rows),
        "complete_receipts": sum(row["status"] == "complete" for row in rows),
        "incomplete_receipts": sum(row["status"] != "complete" for row in rows),
        "selected_points": len(selected),
        "selected_qwen32_points": len(qwen32),
        "selected_other_model_points": len(other),
        "sheets": reopened.sheetnames,
        "sheet_rows": {sheet: reopened[sheet].max_row for sheet in reopened.sheetnames},
        "bytes": output.stat().st_size,
    }
    reopened.close()
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing existing workbook: {output}")
    result = build_workbook(source_root, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
