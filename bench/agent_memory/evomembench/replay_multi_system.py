"""Replay frozen Full-GEM queries on the live three-system baseline.

This command performs measured work and is never invoked by preflight.  A
formal run requires the explicit protocol authorization token in addition to
``--formal``; pilot invocations remain visibly labelled as non-formal.
"""

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


def _namespace(run_id: str, index: int, digest: str) -> str:
    token = hashlib.sha256(f"{run_id}:{index}:{digest}".encode()).hexdigest()[:16]
    return f"evo_{index:03d}_{token}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_authorization(
        formal=bool(args.formal), authorization=str(args.authorization)
    )
    source = Path(args.know_run_dir)
    if (
        json.loads((source / "run_receipt.json").read_text()).get("status")
        != "complete"
    ):
        raise ValueError("Full-GEM source run is incomplete")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing replay output: {output}")
    output.mkdir(parents=True)

    snapshots = [
        ExperienceSnapshot.read(path)
        for path in sorted((source / "snapshots" / "full_gem").glob("*.json"))
    ]
    if not snapshots:
        raise FileNotFoundError("no Full-GEM snapshots found")
    snapshot_by_scope = {snapshot.scope_id: snapshot for snapshot in snapshots}
    if len(snapshot_by_scope) != len(snapshots):
        raise ValueError("duplicate scope in Full-GEM snapshots")
    queries = [
        row
        for row in _rows(source / "queries" / "full_gem.jsonl")
        if int(row["cutoff_ordinal"]) > 0
    ]
    graph_budgets = {int(row["graph_work_budget"]) for row in queries}
    graph_scorings = {str(row["graph_scoring"]) for row in queries}
    if graph_budgets != {args.graph_work_budget}:
        raise ValueError(
            f"replay graph-work budget differs from Full GEM: {graph_budgets}"
        )
    if graph_scorings != {"membership"}:
        raise ValueError(
            f"live multi-system baseline requires membership scoring: {graph_scorings}"
        )
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        by_scope.setdefault(str(query["scope_id"]), []).append(query)
    if set(by_scope) != set(snapshot_by_scope):
        raise ValueError("query scopes do not match frozen snapshots")

    count_tokens = FrozenTokenizerCounter(args.tokenizer_json)
    total_queries = 0
    parity_passed = 0
    began = time.time()
    loads: list[dict[str, Any]] = []
    for index, scope_id in enumerate(sorted(by_scope)):
        snapshot = snapshot_by_scope[scope_id]
        config = MultiSystemConfig(
            namespace=_namespace(args.run_id, index, snapshot.digest),
            ann_overfetch=args.ann_overfetch,
            milvus_hnsw_m=args.milvus_hnsw_m,
            milvus_ef_construction=args.milvus_ef_construction,
            milvus_ef=args.milvus_ef,
        )
        with LiveMultiSystemExperienceStore(config) as store:
            load_metrics = store.load_snapshot(snapshot)
            loads.append(load_metrics)
            scope_queries = sorted(
                by_scope[scope_id], key=lambda item: int(item["cutoff_ordinal"])
            )
            for raw in scope_queries:
                vector = [float(value) for value in raw["task_embedding"]]
                if embedding_sha256(vector) != raw["task_embedding_sha256"]:
                    raise ValueError(
                        f"canonical task embedding digest failed: {raw['query_id']}"
                    )
                result = store.query(
                    MultiSystemQuery(
                        query_id=str(raw["query_id"]),
                        scope_id=scope_id,
                        embedding=tuple(float(value) for value in vector),
                        cutoff_ordinal=int(raw["cutoff_ordinal"]),
                        k=int(raw["k"]),
                        m_seeds=int(raw["m_seeds"]),
                        hops=int(raw["hops"]),
                        graph_work_budget=int(raw["graph_work_budget"]),
                        token_budget=int(raw["token_budget"]),
                    ),
                    count_tokens=count_tokens,
                )
                parity = compare_parity(
                    expected_ids=raw["expected_ids"],
                    observed_ids=result.selected_ids,
                    expected_injection=str(raw["expected_injection"]),
                    observed_injection=result.injection,
                )
                _append_jsonl(
                    output / "parity.jsonl",
                    {
                        "query_id": raw["query_id"],
                        "scope_id": scope_id,
                        **parity.as_dict(),
                    },
                )
                total_queries += 1
                parity_passed += int(parity.passed)
                if not parity.passed:
                    require_parity(parity)
                trace = SystemTrace(
                    run_id=args.run_id,
                    track="CrossEp-Know",
                    arm="multi_system",
                    target_id=str(raw["query_id"]),
                    scope_id=scope_id,
                    history_size=int(raw["cutoff_ordinal"]),
                    status="complete",
                    latency_ms=result.latency_ms,
                    tokens={
                        "memory_injection": result.injection_tokens,
                        # Generation is shared with Full GEM only after exact
                        # injection parity and is not duplicated in retrieval replay.
                        "answer_prompt": 0,
                        "answer_completion": 0,
                    },
                    intermediate=result.intermediate,
                    selected_ids=result.selected_ids,
                    injection_sha256=sha256_text(result.injection),
                    probes={**result.probes, "parity_passed": True},
                )
                _append_jsonl(output / "traces" / "multi_system.jsonl", trace.as_dict())
            load_metrics["cleanup"] = store.cleanup_owned_namespace()
    return {
        "schema_version": "evomembench_multi_system_replay_v0.1.0",
        "status": "complete",
        "formal": bool(args.formal),
        "run_id": args.run_id,
        "source_run": str(source),
        "snapshots": len(snapshots),
        "queries": total_queries,
        "parity_passed": parity_passed,
        "parity_fraction": parity_passed / total_queries if total_queries else 0.0,
        "all_parity_passed": parity_passed == total_queries,
        "load_metrics": loads,
        "canonical_task_embeddings": {
            "source": "frozen_exact_vectors_from_full_gem_queries",
            "queries": total_queries,
        },
        "tokenizer": {
            "path": count_tokens.path,
            "sha256": count_tokens.sha256,
            "implementation": count_tokens.implementation,
            "execution": "cpu_only_no_answer_gpu_request",
        },
        "elapsed_seconds": time.time() - began,
        "token_accounting_note": (
            "multi-system answer tokens are inherited from Full GEM only after exact "
            "ordered-ID and injection-SHA parity; this replay does not call the answer model"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--know-run-dir", required=True)
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
        receipt = run(args)
    except BaseException as exc:
        if not preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_multi_system_replay_v0.1.0",
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
    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
