"""Validate and report a GEM Math/ALE Track-C physical-plan replay artifact."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.e2.gem_math_plan_replay import STAGES, _summary

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    args = parser.parse_args(argv)
    raw_path = args.run / "raw.jsonl"
    rows = [json.loads(line) for line in raw_path.open() if line.strip()]
    measured = [row for row in rows if not row["warmup"]]
    import psycopg

    with psycopg.connect(args.dsn) as conn:
        environment = {
            "postgres": conn.execute("SELECT version()").fetchone()[0],
            "extensions": dict(
                conn.execute(
                    "SELECT extname,extversion FROM pg_extension"
                    " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
                ).fetchall()
            ),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        }
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_repetitions = int(manifest["measured_repetitions"])
    domains = sorted({task.split(":", 1)[0] for task in manifest["tasks"]})
    manifest["environment"] = environment
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = _summary(measured)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))

    repetitions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    hashes: dict[str, set[str]] = defaultdict(set)
    reconciliation_failures = []
    for row in measured:
        repetitions[(row["query_uid"], row["physical_plan"])][
            str(row["repetition"])
        ] += 1
        hashes[row["query_uid"]].add(row["logical_spec_hash"])
        classified = sum(float(row[name]) for name in STAGES[:-1])
        total = float(row["retriever_total_ms"])
        discrepancy = abs(total - classified) / max(total, 1e-9)
        if discrepancy > 0.05:
            reconciliation_failures.append(
                {
                    "query_uid": row["query_uid"],
                    "plan": row["physical_plan"],
                    "fraction": discrepancy,
                }
            )
    bad_repetitions = [
        {"query_uid": key[0], "plan": key[1], "counts": dict(counts)}
        for key, counts in repetitions.items()
        if len(counts) != expected_repetitions
        or any(value != 1 for value in counts.values())
    ]
    hash_failures = [uid for uid, values in hashes.items() if len(values) != 1]
    exact_plans = ("vfwd", "rrev")
    exact_rows = [
        row
        for row in summary
        if not row["task_uid"].startswith("__") and row["physical_plan"] in exact_plans
    ]
    aivg_rows = [
        row
        for row in summary
        if not row["task_uid"].startswith("__") and row["physical_plan"] == "aivg"
    ]
    gates = {
        "all_queries_have_expected_repetitions": not bad_repetitions,
        "one_logical_hash_per_query": not hash_failures,
        "stage_reconciliation_within_5pct": not reconciliation_failures,
        "vfwd_rrev_exact_order": bool(exact_rows)
        and all(row["exact_order_rate"] == 1.0 for row in exact_rows),
        "aivg_recall_at_10_ge_0_95": bool(aivg_rows)
        and all(row["recall_at_10"] >= 0.95 for row in aivg_rows),
    }
    gate_receipt: dict[str, Any] = {
        "schema_version": "gem_math_ale_plan_report_v2",
        "domains": domains,
        "expected_repetitions": expected_repetitions,
        "gates": gates,
        "bad_repetitions": bad_repetitions,
        "hash_failures": hash_failures,
        "reconciliation_failures": reconciliation_failures[:20],
        "passed": all(gates.values()),
        "hardware_claim": "stock PostgreSQL 16 x86_64; not a GX10 sign-off",
        "environment": environment,
    }
    (args.run / "gate_receipt.json").write_text(json.dumps(gate_receipt, indent=2))

    lines = [
        "# GEM+ EvoTrace Math/ALE physical-plan Track-C report",
        "",
        "Stock PostgreSQL 16/x86_64 validation only; this is not a GX10 benchmark sign-off.",
        "",
        "| plan | exact order | recall@10 | p50 total ms | p95 total ms | p50 ANN | p50 graph | p50 predicate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in domains:
        global_rows = {
            row["physical_plan"]: row
            for row in summary
            if row["task_uid"] == f"__all_{domain}__"
        }
        for plan in ("vfwd", "rrev", "aivg"):
            row = global_rows[plan]
            lines.append(
                f"| {domain}/{plan.upper()} | {row['exact_order_rate']:.3f} | "
                f"{row['recall_at_10']:.3f} | {row['retriever_total_ms_p50']:.3f} "
                f"| {row['retriever_total_ms_p95']:.3f} | {row['ann_ms_p50']:.3f} "
                f"| {row['graph_ms_p50']:.3f} | {row['predicate_ms_p50']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The query track replays stored vectors, so `embed_ms=0`; online embedding is measured in agent telemetry.",
            "Agent quality is measured separately by the local task evaluators; oracle parity is only this correctness gate.",
            "",
            f"All gates passed: **{all(gates.values())}**.",
        ]
    )
    (args.run / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate_receipt, indent=2))
    return 0 if gate_receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
