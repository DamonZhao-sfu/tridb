"""Checkpointed E0 plan-space runner."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tools.e0.common import artifact_record, environment_record, sha256_file, write_json

from .backend import build_backend
from .config import enumerate_plans, load_config
from .model import QuerySpec, canonical_json


def observation_key(
    dataset: str, query_id: str, plan_id: str, repetition: int
) -> tuple[str, str, str, int]:
    return dataset, query_id, plan_id, repetition


def completed_keys(path: Path, config_sha256: str) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    result = set()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"{path}:{lineno}: config hash differs; use a new output directory"
            )
        result.add(
            observation_key(
                row["dataset"], row["query_id"], row["plan_id"], int(row["repetition"])
            )
        )
    return result


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(canonical_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _query_record(query: QuerySpec) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query_hop_limit": query.hop_limit,
        "template": query.template,
        "annotation_status": query.annotation_status,
        "answer_count": len(query.answer_ids),
    }


def limit_plans_balanced(plans: list[Any], limit: int) -> list[Any]:
    """Take a deterministic round-robin sample across shapes, then append default."""
    defaults = [plan for plan in plans if plan.is_default]
    groups: dict[str, list[Any]] = {}
    for plan in plans:
        if not plan.is_default:
            groups.setdefault(plan.shape, []).append(plan)
    selected: list[Any] = []
    offsets = {shape: 0 for shape in groups}
    while len(selected) < limit:
        added = False
        for shape in sorted(groups):
            offset = offsets[shape]
            if offset < len(groups[shape]) and len(selected) < limit:
                selected.append(groups[shape][offset])
                offsets[shape] += 1
                added = True
        if not added:
            break
    selected.extend(plan for plan in defaults if plan not in selected)
    return selected


def run(
    config_path: Path,
    output_dir: Path,
    *,
    backend: str,
    datasets: list[str] | None,
    query_limit: int | None,
    plan_limit: int | None,
    repetitions: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    selected = datasets or list(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "observations.jsonl"
    done = completed_keys(raw_path, config_sha256)
    counts = {"written": 0, "resumed": len(done), "errors": 0}
    started = time.time()
    repetitions = repetitions or int(config["repetitions"])
    top_n = int(config["top_n"])
    timeout_ms = float(config["timeout_seconds"]) * 1000.0
    input_artifacts: list[dict[str, Any]] = [artifact_record(config_path)]

    with raw_path.open("a", encoding="utf-8") as handle:
        for dataset_name in selected:
            spec = config["datasets"][dataset_name]
            for field in (
                "nodes",
                "edges",
                "embeddings",
                "query_embeddings",
                "queries",
            ):
                input_artifacts.append(artifact_record(Path(spec[field])))
            dataset = build_backend(backend, dataset_name, spec)
            queries = dataset.load_queries(Path(spec["queries"]))
            if query_limit is not None:
                queries = queries[:query_limit]
            plans = enumerate_plans(config, dataset_name)
            if plan_limit is not None:
                plans = limit_plans_balanced(plans, plan_limit)
            hops = {plan.hops for plan in plans}

            for query in queries:
                dataset.prepare_query(query, hops)
                for _ in range(int(config["warmups"])):
                    default = next(
                        (plan for plan in plans if plan.is_default), plans[0]
                    )
                    dataset.execute(query, default, top_n=top_n)
                for plan in plans:
                    for repetition in range(repetitions):
                        key = observation_key(
                            dataset_name, query.query_id, plan.plan_id, repetition
                        )
                        if key in done:
                            continue
                        try:
                            result = dataset.execute(query, plan, top_n=top_n)
                            if result["latency_ms"] > timeout_ms:
                                result["status"] = "timeout"
                        except Exception as exc:
                            result = {
                                "status": "error",
                                "backend": backend,
                                "valid_for_system_latency_claims": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            counts["errors"] += 1
                        row = {
                            "schema_version": "e0-plan-observation-v0.2.0",
                            "config_sha256": config_sha256,
                            "dataset": dataset_name,
                            **_query_record(query),
                            **plan.as_dict(),
                            "repetition": repetition,
                            **result,
                        }
                        _append(handle, row)
                        counts["written"] += 1
                        done.add(key)
                        if result["status"] == "error":
                            raise RuntimeError(
                                f"{dataset_name}/{query.query_id}/{plan.plan_id}: "
                                f"{result['error']}"
                            )
            close = getattr(dataset, "close", None)
            if close is not None:
                close()

    manifest = {
        "schema_version": "e0-plan-run-v0.2.0",
        "environment": environment_record(),
        "backend": backend,
        "valid_for_system_latency_claims": backend in {"polyglot_live", "tridb_live"},
        "config": artifact_record(config_path),
        "inputs": input_artifacts,
        "output": artifact_record(raw_path),
        "datasets": selected,
        "query_limit": query_limit,
        "plan_limit": plan_limit,
        "repetitions": repetitions,
        "counts": counts,
        "seconds": round(time.time() - started, 3),
        "complete": counts["errors"] == 0,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e0/plan_space_v0.2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["parquet_reference", "polyglot_live", "tridb_live"],
        default="parquet_reference",
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--plan-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    manifest = run(
        args.config,
        args.output_dir,
        backend=args.backend,
        datasets=args.datasets,
        query_limit=args.query_limit,
        plan_limit=args.plan_limit,
        repetitions=args.repetitions,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
