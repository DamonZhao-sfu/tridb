"""Paired E1 grid runner with separate pilot and full-grid claim gates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.e0.plan_spread.backend import build_backend
from experiments.e0.plan_spread.model import QuerySpec
from experiments.e0.plan_spread.reference_backend import ReferenceDataset
from tools.e0.common import artifact_record, environment_record, sha256_file, write_json

from .config import load_config
from .model import (
    OperatingPoint,
    enumerate_points,
    limit_points_balanced,
    normalize_observation,
)


def system_order(repetition: int) -> list[str]:
    """Alternating AB/BA yields an A-B-B-A request sequence across two blocks."""
    systems = ["tridb_live", "polyglot_tuned"]
    return systems if repetition % 2 == 0 else list(reversed(systems))


def select_systems(
    config: dict[str, Any], requested: list[str] | None
) -> list[str]:
    """Validate an optional resumable system-isolated execution phase."""
    headline = list(config["systems"]["headline"])
    if requested is None:
        return headline
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested systems must be non-empty and unique")
    unknown = sorted(set(requested) - set(headline))
    if unknown:
        raise ValueError(f"requested systems are not headline systems: {unknown}")
    return [system for system in headline if system in requested]


def observation_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        row["dataset"],
        row["query_id"],
        row["point_id"],
        row["system"],
        int(row["repetition"]),
    )


def completed_keys(
    path: Path, config_sha256: str, run_kind: str
) -> set[tuple[str, str, str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"{path}:{lineno}: config hash differs; use a new output directory"
            )
        if row.get("run_kind") != run_kind:
            raise RuntimeError(
                f"{path}:{lineno}: run kind differs; use a new output directory"
            )
        keys.add(observation_key(row))
    return keys


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_queries(path: Path) -> list[QuerySpec]:
    return [
        QuerySpec.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def design_summary(
    config: dict[str, Any],
    *,
    datasets: list[str] | None,
    query_limit: int | None,
    point_limit: int | None,
    repetitions: int,
    run_kind: str = "pilot",
) -> dict[str, Any]:
    selected = datasets or list(config["datasets"])
    summary: dict[str, Any] = {"run_kind": f"{run_kind}_dry_run", "datasets": {}}
    total = 0
    for dataset in selected:
        spec = config["datasets"][dataset]
        queries = _load_queries(Path(spec["queries"]))
        if query_limit is not None:
            queries = queries[:query_limit]
        points = (
            []
            if not queries
            else limit_points_balanced(
                enumerate_points(config, queries[0]), point_limit
            )
        )
        observations = len(queries) * len(points) * 2 * repetitions
        total += observations
        summary["datasets"][dataset] = {
            "queries": len(queries),
            "points_per_query": len(points),
            "systems": 2,
            "repetitions": repetitions,
            "observations": observations,
            "valid_for_performance_claims": spec["valid_for_performance_claims"],
        }
    summary["total_observations"] = total
    summary["valid_for_headline_claims"] = False
    summary["would_be_headline_eligible_on_success"] = bool(
        run_kind == "full_grid"
        and all(
            config["datasets"][name]["valid_for_performance_claims"]
            for name in selected
        )
    )
    return summary


def _set_effort(backend: Any, point: OperatingPoint) -> None:
    setter = getattr(backend, "set_ann_effort", None)
    if setter is None:
        raise RuntimeError(f"{backend.backend_name} cannot set ANN effort")
    setter(point.ann_effort)


def run_pilot(
    config_path: Path,
    output_dir: Path,
    *,
    datasets: list[str] | None,
    query_limit: int | None,
    point_limit: int | None,
    repetitions: int,
    warmups: int,
    run_kind: str = "pilot",
) -> dict[str, Any]:
    if run_kind not in {"pilot", "full_grid"}:
        raise ValueError("run_kind must be pilot or full_grid")
    full_grid = run_kind == "full_grid"
    if full_grid and (query_limit is not None or point_limit is not None):
        raise ValueError("full_grid forbids query and point limits")
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    selected = datasets or list(config["datasets"])
    unknown = sorted(set(selected) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "observations.jsonl"
    done = completed_keys(raw_path, config_sha256, run_kind)
    counts = {"written": 0, "resumed": len(done), "errors": 0, "censored": 0}
    started = time.time()
    failure: str | None = None
    timeout_ms = float(config["measurement"]["soft_timeout_seconds"]) * 1000.0
    inputs = [artifact_record(config_path)]

    try:
        with raw_path.open("a", encoding="utf-8") as handle:
            for dataset_name in selected:
                spec = config["datasets"][dataset_name]
                inputs.extend(
                    artifact_record(Path(spec[field]))
                    for field in (
                        "nodes",
                        "edges",
                        "embeddings",
                        "query_embeddings",
                        "queries",
                    )
                )
                reference = ReferenceDataset(dataset_name, spec)
                queries = reference.load_queries(Path(spec["queries"]))
                if query_limit is not None:
                    queries = queries[:query_limit]
                backend_name = str(spec["backend_dataset_name"])
                backends: dict[str, Any] = {}
                try:
                    for system in config["systems"]["headline"]:
                        backends[system] = build_backend(
                            config["systems"]["backend_factory"][system],
                            backend_name,
                            spec,
                        )
                    for query in queries:
                        points = limit_points_balanced(
                            enumerate_points(config, query), point_limit
                        )
                        reference.prepare_query(query, {query.hop_limit})
                        valid_ids = reference.valid_result_ids(query, query.hop_limit)
                        for backend in backends.values():
                            backend.prepare_query(query, {query.hop_limit})
                        for point in points:
                            for system, backend in backends.items():
                                _set_effort(backend, point)
                                for _ in range(warmups):
                                    backend.execute(
                                        query,
                                        point.plan,
                                        top_n=int(config["semantics"]["output_top_n"]),
                                    )
                            for repetition in range(repetitions):
                                order = system_order(repetition)
                                for order_index, system in enumerate(order):
                                    key = (
                                        dataset_name,
                                        query.query_id,
                                        point.point_id,
                                        system,
                                        repetition,
                                    )
                                    if key in done:
                                        continue
                                    backend = backends[system]
                                    _set_effort(backend, point)
                                    try:
                                        result = backend.execute(
                                            query,
                                            point.plan,
                                            top_n=int(
                                                config["semantics"]["output_top_n"]
                                            ),
                                        )
                                        if float(result["latency_ms"]) > timeout_ms:
                                            result["status"] = "soft_timeout"
                                            result["error_or_censor_reason"] = (
                                                "latency_exceeded_soft_timeout"
                                            )
                                            counts["censored"] += 1
                                    except Exception as exc:
                                        result = {
                                            "status": "error",
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                        counts["errors"] += 1
                                    row = normalize_observation(
                                        config_sha256=config_sha256,
                                        dataset=dataset_name,
                                        dataset_valid_for_performance=bool(
                                            spec["valid_for_performance_claims"]
                                        ),
                                        system=system,
                                        query=query,
                                        point=point,
                                        repetition=repetition,
                                        randomized_block=repetition,
                                        order_in_block=order_index,
                                        result=result,
                                        valid_result_ids=valid_ids,
                                        run_kind=run_kind,
                                        allow_headline_claim=full_grid,
                                    )
                                    _append(handle, row)
                                    counts["written"] += 1
                                    done.add(key)
                                    if row["status"] != "ok":
                                        raise RuntimeError(
                                            f"{dataset_name}/{query.query_id}/"
                                            f"{point.point_id}/{system}: "
                                            f"{row['error_or_censor_reason']}"
                                        )
                                    if not row["result_validity"]:
                                        counts["errors"] += 1
                                        raise RuntimeError(
                                            f"{dataset_name}/{query.query_id}/"
                                            f"{point.point_id}/{system}: "
                                            "result failed exact constraint validity"
                                        )
                finally:
                    for backend in backends.values():
                        backend.close()
    except Exception as exc:  # failure is recorded once; the pilot is never retried
        failure = f"{type(exc).__name__}: {exc}"
        if counts["errors"] == 0 and counts["censored"] == 0:
            counts["errors"] += 1

    manifest = {
        "schema_version": f"e1-composition-{run_kind}-run-v0.1.0",
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "UNVERIFIED",
        },
        "environment": environment_record(),
        "config": artifact_record(config_path),
        "inputs": inputs,
        "output": artifact_record(raw_path),
        "datasets": selected,
        "query_limit": query_limit,
        "point_limit": point_limit,
        "repetitions": repetitions,
        "warmups": warmups,
        "counts": counts,
        "seconds": round(time.time() - started, 3),
        "complete": failure is None
        and counts["errors"] == 0
        and counts["censored"] == 0,
        "failure": failure,
        "valid_for_headline_claims": bool(
            full_grid
            and failure is None
            and counts["errors"] == 0
            and counts["censored"] == 0
            and all(
                config["datasets"][name]["valid_for_performance_claims"]
                for name in selected
            )
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/e1/composition_v0.1.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/e1/composition/pilot_v0.1")
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--query-limit", type=int, default=1)
    parser.add_argument("--point-limit", type=int, default=6)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--run-kind", choices=["pilot", "full_grid"], default="pilot")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="connect to live stores; without this flag only print the pilot design",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    repetitions = args.repetitions or int(
        config["measurement"][
            "repetitions" if args.run_kind == "full_grid" else "pilot_repetitions"
        ]
    )
    warmups = (
        args.warmups
        if args.warmups is not None
        else int(config["measurement"]["warmups"])
    )
    query_limit = None if args.run_kind == "full_grid" else args.query_limit
    point_limit = None if args.run_kind == "full_grid" else args.point_limit
    if not args.execute:
        print(
            json.dumps(
                design_summary(
                    config,
                    datasets=args.datasets,
                    query_limit=query_limit,
                    point_limit=point_limit,
                    repetitions=repetitions,
                    run_kind=args.run_kind,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = run_pilot(
        args.config,
        args.output_dir,
        datasets=args.datasets,
        query_limit=query_limit,
        point_limit=point_limit,
        repetitions=repetitions,
        warmups=warmups,
        run_kind=args.run_kind,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
