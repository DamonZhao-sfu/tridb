"""Replay CrossEp-Tool frozen source banks on the live three-system baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from bench.agent_memory.evomembench.multi_system import (
    LiveMultiSystemExperienceStore,
    MultiSystemConfig,
    MultiSystemQuery,
)
from bench.agent_memory.evomembench.run_pilot import _append_jsonl
from bench.agent_memory.evomembench.system_protocol import (
    FrozenTokenizerCounter,
    SystemTrace,
    compare_parity,
    embedding_sha256,
    require_formal_authorization,
    require_parity,
    sha256_text,
)
from bench.agent_memory.evomembench.system_snapshot import ExperienceSnapshot


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _namespace(run_id: str, environment: str, digest: str) -> str:
    token = hashlib.sha256(f"{run_id}:{environment}:{digest}".encode()).hexdigest()[:16]
    return f"tool_{environment[:12]}_{token}".lower()


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_authorization(
        formal=bool(args.formal), authorization=str(args.authorization)
    )
    source = Path(args.tool_run_dir)
    receipt = json.loads((source / "run_receipt.json").read_text())
    if receipt.get("status") != "complete" or "gem_fused" not in receipt.get(
        "arms", []
    ):
        raise ValueError("Tool source run lacks a complete gem_fused arm")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing Tool replay output: {output}")
    output.mkdir(parents=True)
    snapshot_paths = sorted((source / "system_snapshots" / "full_gem").glob("*.json"))
    snapshots = {path.stem: ExperienceSnapshot.read(path) for path in snapshot_paths}
    if set(snapshots) != {
        "gorilla_fs",
        "vehicle_control",
        "trading_bot",
        "travel_api",
    }:
        raise ValueError("Tool replay requires all four frozen source banks")

    query_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in snapshots}
    for path in sorted(
        (source / "phase2" / "gem_fused").glob("*/retrieval_receipts.jsonl")
    ):
        for row in _rows(path):
            query_rows[str(row["source_environment"])].append(row)
    expected = 12 * int(receipt["episodes_per_environment"])
    if sum(len(rows) for rows in query_rows.values()) != expected:
        raise ValueError("Tool replay query count does not match the 12-cell protocol")
    flat_queries = [row for rows in query_rows.values() for row in rows]
    graph_budgets = {int(row["graph_work_budget"]) for row in flat_queries}
    graph_scorings = {str(row["graph_scoring"]) for row in flat_queries}
    if graph_budgets != {args.graph_work_budget}:
        raise ValueError(
            f"replay graph-work budget differs from Full GEM: {graph_budgets}"
        )
    if graph_scorings != {"membership"}:
        raise ValueError(
            f"live multi-system baseline requires membership scoring: {graph_scorings}"
        )

    count_tokens = FrozenTokenizerCounter(args.tokenizer_json)
    total = 0
    passed = 0
    loads: list[dict[str, Any]] = []
    began = time.time()
    for source_environment in sorted(snapshots):
        snapshot = snapshots[source_environment]
        config = MultiSystemConfig(
            namespace=_namespace(args.run_id, source_environment, snapshot.digest),
            ann_overfetch=args.ann_overfetch,
            milvus_hnsw_m=args.milvus_hnsw_m,
            milvus_ef_construction=args.milvus_ef_construction,
            milvus_ef=args.milvus_ef,
        )
        rows = query_rows[source_environment]
        with LiveMultiSystemExperienceStore(config) as store:
            load_metrics = store.load_snapshot(snapshot)
            loads.append(load_metrics)
            for raw in rows:
                vector = [float(value) for value in raw["task_embedding"]]
                if embedding_sha256(vector) != raw["task_embedding_sha256"]:
                    raise ValueError(
                        f"canonical task embedding digest failed: {raw['sample_id']}"
                    )
                result = store.query(
                    MultiSystemQuery(
                        query_id=str(raw["sample_id"]),
                        scope_id=snapshot.scope_id,
                        embedding=tuple(float(value) for value in vector),
                        cutoff_ordinal=int(raw["cutoff_ordinal"]),
                        k=int(raw["top_k"]),
                        m_seeds=int(raw["m_seeds"]),
                        hops=int(raw["hops"]),
                        graph_work_budget=int(raw["graph_work_budget"]),
                        token_budget=int(raw["token_budget"]),
                        source_phases=("in_env",),
                    ),
                    count_tokens=count_tokens,
                )
                parity = compare_parity(
                    expected_ids=raw["selected_episode_uids"],
                    observed_ids=result.selected_ids,
                    expected_injection=str(raw["injection_text"]),
                    observed_injection=result.injection,
                )
                _append_jsonl(
                    output / "parity.jsonl",
                    {
                        "sample_id": raw["sample_id"],
                        "source_environment": source_environment,
                        "target_environment": raw["target_environment"],
                        **parity.as_dict(),
                    },
                )
                total += 1
                passed += int(parity.passed)
                if not parity.passed:
                    require_parity(parity)
                trace = SystemTrace(
                    run_id=args.run_id,
                    track="CrossEp-Tool",
                    arm="multi_system",
                    target_id=(
                        f"{source_environment}__to__{raw['target_environment']}:"
                        f"{raw['sample_id']}"
                    ),
                    scope_id=snapshot.scope_id,
                    history_size=sum(
                        unit.node_kind == "experience" for unit in snapshot.units
                    ),
                    status="complete",
                    latency_ms=result.latency_ms,
                    tokens={
                        "memory_injection": result.injection_tokens,
                        "answer_prompt": 0,
                        "answer_completion": 0,
                    },
                    intermediate=result.intermediate,
                    selected_ids=result.selected_ids,
                    injection_sha256=sha256_text(result.injection),
                    probes={
                        **result.probes,
                        "source_environment": source_environment,
                        "target_environment": raw["target_environment"],
                        "phase": "transfer",
                        "parity_passed": True,
                    },
                )
                _append_jsonl(output / "traces" / "multi_system.jsonl", trace.as_dict())
            load_metrics["cleanup"] = store.cleanup_owned_namespace()
    return {
        "schema_version": "evomembench_tool_multi_system_replay_v0.1.0",
        "status": "complete",
        "formal": bool(args.formal),
        "run_id": args.run_id,
        "source_run": str(source),
        "queries": total,
        "parity_passed": passed,
        "parity_fraction": passed / total if total else 0.0,
        "all_parity_passed": total == passed,
        "load_metrics": loads,
        "canonical_task_embeddings": {
            "source": "frozen_exact_vectors_from_full_gem_queries",
            "queries": total,
        },
        "tokenizer": {
            "path": count_tokens.path,
            "sha256": count_tokens.sha256,
            "implementation": count_tokens.implementation,
            "execution": "cpu_only_no_answer_gpu_request",
        },
        "elapsed_seconds": time.time() - began,
        "token_accounting_note": (
            "answer-model tokens are inherited from the Full-GEM Tool run only "
            "after ordered-ID and injection-SHA parity"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--ann-overfetch", type=int, default=8)
    parser.add_argument("--milvus-hnsw-m", type=int, default=16)
    parser.add_argument("--milvus-ef-construction", type=int, default=200)
    parser.add_argument("--milvus-ef", type=int, default=128)
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--authorization", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    preexisted = output.exists()
    try:
        result = run(args)
    except BaseException as exc:
        if not preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_tool_multi_system_replay_v0.1.0",
                        "status": "failed",
                        "formal": bool(args.formal),
                        "run_id": args.run_id,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                )
                + "\n"
            )
        raise
    (output / "run_receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
