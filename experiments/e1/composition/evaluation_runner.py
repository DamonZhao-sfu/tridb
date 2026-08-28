"""Measure frozen E1 points on held-out evaluation queries."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.e0.plan_spread.backend import build_backend
from experiments.e0.plan_spread.reference_backend import ReferenceDataset
from tools.e0.common import artifact_record, sha256_file, write_json

from .config import load_config
from .model import OperatingPoint, normalize_observation
from .runner import _set_effort, select_systems, system_order
from .staging import (
    load_query_specs,
    load_staged_config,
    staged_dataset,
    stratified_query_split,
)


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_frozen(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "e1-operating-points-v0.2.0":
        raise ValueError("unexpected operating-point schema")
    if not data.get("selection_uses_calibration_queries_only"):
        raise ValueError("operating points were not selected on calibration only")
    return data


def _point(spec: dict[str, Any], hops: int) -> OperatingPoint:
    return OperatingPoint(
        str(spec["shape"]),
        int(spec["k"]),
        hops,
        str(spec["predicate_placement"]),
        int(spec["ann_effort"]),
    )


def _tasks(frozen: dict[str, Any], hops: int) -> list[dict[str, OperatingPoint]]:
    tasks = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for analysis in ("iso_plan", "pareto_envelope"):
        for label in ("fast", "balanced", "high_quality"):
            pair = frozen["operating_points"][analysis][label]
            points = {
                system: _point(pair[system], hops)
                for system in ("tridb_live", "polyglot_tuned")
            }
            signature = tuple(
                (system, point.point_id) for system, point in points.items()
            )
            if signature not in seen:
                seen.add(signature)
                tasks.append(points)
    return tasks


def _reusable_keys(
    paths: list[Path],
    evaluation_ids: set[str],
    selected_ids: set[tuple[str, str]],
    repetitions: int,
) -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("query_id") not in evaluation_ids:
                    continue
                system_point = (row["system"], row["point_id"])
                repetition = int(row["repetition"])
                if system_point not in selected_ids or repetition >= repetitions:
                    continue
                if (
                    row.get("status") != "ok"
                    or not row.get("result_validity")
                    or row.get("graph_censored")
                ):
                    raise ValueError(f"{path}:{lineno}: unusable evaluation row")
                key = (row["query_id"], row["point_id"], row["system"], repetition)
                if key in keys:
                    raise ValueError(f"duplicate reusable evaluation key: {key}")
                keys.add(key)
    return keys


def _context(staged_path: Path):
    staged = load_staged_config(staged_path)
    base = load_config(Path(staged["base_config"]))
    dataset_name = staged_dataset(staged, base)
    spec = base["datasets"][dataset_name]
    queries = load_query_specs(Path(spec["queries"]))
    split_cfg = staged["split"]
    split = stratified_query_split(
        queries,
        seed=str(split_cfg["seed"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
    )
    frozen = _load_frozen(Path(staged["outputs"]["operating_points"]))
    return staged, base, dataset_name, spec, queries, split, frozen


def _selected_ids(
    queries: list[Any], frozen: dict[str, Any], systems: list[str] | None = None
) -> set[tuple[str, str]]:
    return {
        (system, point.point_id)
        for query in queries
        for task in _tasks(frozen, query.hop_limit)
        for system, point in task.items()
        if systems is None or system in systems
    }


def _expected(
    queries: list[Any],
    frozen: dict[str, Any],
    repetitions: int,
    systems: list[str] | None = None,
) -> int:
    return sum(
        len(
            {
                (system, point.point_id)
                for task in _tasks(frozen, query.hop_limit)
                for system, point in task.items()
                if systems is None or system in systems
            }
        )
        * repetitions
        for query in queries
    )


def design(
    staged_path: Path, systems: list[str] | None = None
) -> dict[str, Any]:
    staged, base, _, _, queries, split, frozen = _context(staged_path)
    selected_systems = select_systems(base, systems)
    evaluation_ids = set(split["evaluation"])
    queries = [query for query in queries if query.query_id in evaluation_ids]
    repetitions = int(staged["evaluation"]["repetitions"])
    selected_ids = _selected_ids(queries, frozen)
    output = Path(staged["outputs"]["evaluation_observations"])
    paths = [*(Path(value) for value in staged.get("reuse_observations", [])), output]
    done = _reusable_keys(paths, evaluation_ids, selected_ids, repetitions)
    expected = _expected(queries, frozen, repetitions)
    stage_expected = _expected(
        queries, frozen, repetitions, systems=selected_systems
    )
    stage_done = {key for key in done if key[2] in selected_systems}
    return {
        "schema_version": staged["schema_version"],
        "evaluation_queries": len(evaluation_ids),
        "expected_observations": expected,
        "selected_systems": selected_systems,
        "stage_expected_observations": stage_expected,
        "stage_completed_observations": len(stage_done),
        "reused_or_completed_observations": len(done),
        "missing_observations": expected - len(done),
        "would_execute": False,
    }


def run(staged_path: Path, systems: list[str] | None = None) -> dict[str, Any]:
    staged, base, dataset_name, spec, queries, split, frozen = _context(staged_path)
    selected_systems = select_systems(base, systems)
    evaluation_ids = set(split["evaluation"])
    queries = [query for query in queries if query.query_id in evaluation_ids]
    repetitions = int(staged["evaluation"]["repetitions"])
    warmups = int(staged["evaluation"]["warmups"])
    output = Path(staged["outputs"]["evaluation_observations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [*(Path(value) for value in staged.get("reuse_observations", [])), output]
    done = _reusable_keys(
        paths, evaluation_ids, _selected_ids(queries, frozen), repetitions
    )
    expected = _expected(queries, frozen, repetitions)
    staged_sha = sha256_file(staged_path)
    timeout_ms = float(base["measurement"]["soft_timeout_seconds"]) * 1000.0
    stage_done = {key for key in done if key[2] in selected_systems}
    counts = {
        "reused": len(stage_done),
        "written": 0,
        "errors": 0,
        "censored": 0,
    }
    started = time.time()
    failure: str | None = None
    reference = ReferenceDataset(dataset_name, spec)
    backends: dict[str, Any] = {}
    try:
        for system in selected_systems:
            backends[system] = build_backend(
                base["systems"]["backend_factory"][system],
                str(spec["backend_dataset_name"]),
                spec,
            )
        with output.open("a", encoding="utf-8") as handle:
            for query in queries:
                reference.prepare_query(query, {query.hop_limit})
                valid_ids = reference.valid_result_ids(query, query.hop_limit)
                for backend in backends.values():
                    backend.prepare_query(query, {query.hop_limit})
                for task in _tasks(frozen, query.hop_limit):
                    missing_systems = {
                        system
                        for repetition in range(repetitions)
                        for system, point in task.items()
                        if system in selected_systems
                        if (query.query_id, point.point_id, system, repetition)
                        not in done
                    }
                    for system in missing_systems:
                        point = task[system]
                        _set_effort(backends[system], point)
                        for _ in range(warmups):
                            backends[system].execute(
                                query,
                                point.plan,
                                top_n=int(base["semantics"]["output_top_n"]),
                            )
                    for repetition in range(repetitions):
                        for system in system_order(repetition):
                            if system not in selected_systems:
                                continue
                            point = task[system]
                            key = (
                                query.query_id,
                                point.point_id,
                                system,
                                repetition,
                            )
                            if key in done:
                                continue
                            backend = backends[system]
                            _set_effort(backend, point)
                            result = backend.execute(
                                query,
                                point.plan,
                                top_n=int(base["semantics"]["output_top_n"]),
                            )
                            if float(result["latency_ms"]) > timeout_ms:
                                result["status"] = "soft_timeout"
                                result["error_or_censor_reason"] = (
                                    "latency_exceeded_soft_timeout"
                                )
                                counts["censored"] += 1
                            row = normalize_observation(
                                config_sha256=staged_sha,
                                dataset=dataset_name,
                                dataset_valid_for_performance=bool(
                                    spec["valid_for_performance_claims"]
                                ),
                                system=system,
                                query=query,
                                point=point,
                                repetition=repetition,
                                randomized_block=repetition,
                                order_in_block=system_order(repetition).index(system),
                                result=result,
                                valid_result_ids=valid_ids,
                                run_kind="staged_evaluation",
                                allow_headline_claim=True,
                            )
                            _append(handle, row)
                            counts["written"] += 1
                            done.add(key)
                            if (
                                row["status"] != "ok"
                                or not row["result_validity"]
                                or row["graph_censored"]
                            ):
                                raise RuntimeError(
                                    f"evaluation row failed validity gates: {key}"
                                )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        counts["errors"] += 1
    finally:
        for backend in backends.values():
            backend.close()
    manifest = {
        "schema_version": "e1-staged-evaluation-run-v0.2.0",
        "staged_config": artifact_record(staged_path),
        "operating_points": artifact_record(
            Path(staged["outputs"]["operating_points"])
        ),
        "selected_systems": selected_systems,
        "counts": counts,
        "stage_expected_observations": _expected(
            queries, frozen, repetitions, systems=selected_systems
        ),
        "stage_completed_observations": len(
            {key for key in done if key[2] in selected_systems}
        ),
        "expected_observations": expected,
        "completed_observations": len(done),
        "seconds": round(time.time() - started, 3),
        "complete": failure is None and len(done) == expected,
        "failure": failure,
        "valid_for_headline_claims": failure is None and len(done) == expected,
    }
    manifest["stage_complete"] = bool(
        failure is None
        and manifest["stage_completed_observations"]
        == manifest["stage_expected_observations"]
    )
    write_json(output.parent / "evaluation_manifest.json", manifest)
    if len(selected_systems) == 1:
        write_json(
            output.parent / f"evaluation_manifest_{selected_systems[0]}.json",
            manifest,
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-config",
        type=Path,
        default=Path("configs/e1/staged_v0.2.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--system",
        action="append",
        choices=["tridb_live", "polyglot_tuned"],
        help="run one resumable resident-system phase; may be repeated",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                design(args.staged_config, args.system), indent=2, sort_keys=True
            )
        )
        return 0
    manifest = run(args.staged_config, args.system)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["stage_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
