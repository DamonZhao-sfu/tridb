"""Run only missing rows in the staged E1 calibration split."""

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
from .model import enumerate_points, normalize_observation
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


def _good_keys(
    paths: list[Path], calibration_ids: set[str], repetitions: int
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
                if row.get("query_id") not in calibration_ids:
                    continue
                repetition = int(row["repetition"])
                if repetition >= repetitions:
                    continue
                if (
                    row.get("status") != "ok"
                    or not row.get("result_validity")
                    or row.get("graph_censored")
                ):
                    raise ValueError(f"{path}:{lineno}: unusable calibration row")
                key = (row["query_id"], row["point_id"], row["system"], repetition)
                if key in keys:
                    raise ValueError(f"duplicate reusable calibration key: {key}")
                keys.add(key)
    return keys


def design(
    staged_path: Path, systems: list[str] | None = None
) -> dict[str, Any]:
    staged = load_staged_config(staged_path)
    base_path = Path(staged["base_config"])
    base = load_config(base_path)
    selected_systems = select_systems(base, systems)
    dataset_name = staged_dataset(staged, base)
    spec = base["datasets"][dataset_name]
    queries = load_query_specs(Path(spec["queries"]))
    split_cfg = staged["split"]
    split = stratified_query_split(
        queries,
        seed=str(split_cfg["seed"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
    )
    calibration_ids = set(split["calibration"])
    repetitions = int(staged["calibration"]["repetitions"])
    output = Path(staged["outputs"]["calibration_observations"])
    reuse = [Path(value) for value in staged.get("reuse_observations", [])]
    done = _good_keys([*reuse, output], calibration_ids, repetitions)
    all_systems = list(base["systems"]["headline"])
    expected = len(calibration_ids) * 100 * len(all_systems) * repetitions
    stage_expected = (
        len(calibration_ids) * 100 * len(selected_systems) * repetitions
    )
    stage_done = {key for key in done if key[2] in selected_systems}
    return {
        "schema_version": staged["schema_version"],
        "calibration_queries": sorted(calibration_ids),
        "evaluation_queries": split["evaluation"],
        "expected_observations": expected,
        "selected_systems": selected_systems,
        "stage_expected_observations": stage_expected,
        "stage_completed_observations": len(stage_done),
        "reused_or_completed_observations": len(done),
        "missing_observations": expected - len(done),
        "would_execute": False,
    }


def run(staged_path: Path, systems: list[str] | None = None) -> dict[str, Any]:
    staged = load_staged_config(staged_path)
    base_path = Path(staged["base_config"])
    base = load_config(base_path)
    selected_systems = select_systems(base, systems)
    dataset_name = staged_dataset(staged, base)
    spec = base["datasets"][dataset_name]
    reference = ReferenceDataset(dataset_name, spec)
    queries = reference.load_queries(Path(spec["queries"]))
    split_cfg = staged["split"]
    split = stratified_query_split(
        queries,
        seed=str(split_cfg["seed"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
    )
    calibration_ids = set(split["calibration"])
    queries = [query for query in queries if query.query_id in calibration_ids]
    repetitions = int(staged["calibration"]["repetitions"])
    warmups = int(staged["calibration"]["warmups"])
    output = Path(staged["outputs"]["calibration_observations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    reuse = [Path(value) for value in staged.get("reuse_observations", [])]
    done = _good_keys([*reuse, output], calibration_ids, repetitions)
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
                for point in enumerate_points(base, query):
                    missing = [
                        (system, repetition)
                        for repetition in range(repetitions)
                        for system in system_order(repetition)
                        if system in selected_systems
                        if (query.query_id, point.point_id, system, repetition)
                        not in done
                    ]
                    if not missing:
                        continue
                    for system in {system for system, _ in missing}:
                        backend = backends[system]
                        _set_effort(backend, point)
                        for _ in range(warmups):
                            backend.execute(
                                query,
                                point.plan,
                                top_n=int(base["semantics"]["output_top_n"]),
                            )
                    for system, repetition in missing:
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
                            run_kind="calibration_grid",
                            allow_headline_claim=False,
                        )
                        _append(handle, row)
                        counts["written"] += 1
                        key = (query.query_id, point.point_id, system, repetition)
                        done.add(key)
                        if (
                            row["status"] != "ok"
                            or not row["result_validity"]
                            or row["graph_censored"]
                        ):
                            raise RuntimeError(
                                f"calibration row failed validity gates: {key}"
                            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        counts["errors"] += 1
    finally:
        for backend in backends.values():
            backend.close()

    all_systems = list(base["systems"]["headline"])
    expected = len(calibration_ids) * 100 * len(all_systems) * repetitions
    stage_expected = (
        len(calibration_ids) * 100 * len(selected_systems) * repetitions
    )
    stage_completed = len({key for key in done if key[2] in selected_systems})
    manifest = {
        "schema_version": "e1-calibration-run-v0.2.0",
        "staged_config": artifact_record(staged_path),
        "base_config": artifact_record(base_path),
        "split": split,
        "selected_systems": selected_systems,
        "counts": counts,
        "stage_expected_observations": stage_expected,
        "stage_completed_observations": stage_completed,
        "stage_complete": failure is None and stage_completed == stage_expected,
        "expected_observations": expected,
        "completed_observations": len(done),
        "seconds": round(time.time() - started, 3),
        "complete": failure is None and len(done) == expected,
        "failure": failure,
        "valid_for_headline_claims": False,
    }
    write_json(output.parent / "calibration_manifest.json", manifest)
    if len(selected_systems) == 1:
        write_json(
            output.parent / f"calibration_manifest_{selected_systems[0]}.json",
            manifest,
        )
    write_json(Path(staged["outputs"]["split_manifest"]), split)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-config", type=Path, default=Path("configs/e1/staged_v0.2.yaml")
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
