"""Run the six-arm TriDB-only E1 modality ablation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.e0.plan_spread.reference_backend import ReferenceDataset
from experiments.e0.plan_spread.tridb_backend import TriDBLiveDataset
from tools.e0.common import artifact_record, sha256_file, write_json

from .config import load_config
from .evaluation_runner import _load_frozen, _point
from .runner import _set_effort
from .staging import load_staged_config, staged_dataset, stratified_query_split


def _append(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _completed(path: Path, config_sha: str) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("staged_config_sha256") != config_sha:
                raise ValueError(f"{path}:{lineno}: staged config hash differs")
            if row.get("status") != "ok" or row.get("graph_censored"):
                raise ValueError(f"{path}:{lineno}: unusable modality row")
            key = (row["query_id"], row["arm"], int(row["repetition"]))
            if key in keys:
                raise ValueError(f"duplicate modality key: {key}")
            keys.add(key)
    return keys


def _context(staged_path: Path):
    staged = load_staged_config(staged_path)
    base = load_config(Path(staged["base_config"]))
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
    queries = [query for query in queries if query.query_id in set(split["evaluation"])]
    frozen = _load_frozen(Path(staged["outputs"]["operating_points"]))
    return staged, base, dataset_name, spec, reference, queries, frozen


def design(staged_path: Path) -> dict[str, Any]:
    staged, base, _, _, _, queries, _ = _context(staged_path)
    repetitions = int(staged["modality_ablation"]["repetitions"])
    arms = list(base["modality_ablation"]["arms"])
    output = Path(staged["outputs"]["modality_observations"])
    done = _completed(output, sha256_file(staged_path))
    expected = len(queries) * len(arms) * repetitions
    return {
        "schema_version": staged["schema_version"],
        "system": "tridb_live",
        "evaluation_queries": len(queries),
        "arms": arms,
        "expected_observations": expected,
        "completed_observations": len(done),
        "missing_observations": expected - len(done),
        "would_execute": False,
    }


def run(staged_path: Path) -> dict[str, Any]:
    staged, base, dataset_name, spec, reference, queries, frozen = _context(staged_path)
    ablation = staged["modality_ablation"]
    analysis, label = str(ablation["operating_point"]).rsplit("_", 1)
    if analysis != "pareto_envelope" or label != "balanced":
        raise ValueError("v0.2 modality point must be pareto_envelope_balanced")
    point_spec = frozen["operating_points"][analysis][label]["tridb_live"]
    repetitions = int(ablation["repetitions"])
    warmups = int(ablation["warmups"])
    arms = list(base["modality_ablation"]["arms"])
    top_n = int(base["semantics"]["output_top_n"])
    output = Path(staged["outputs"]["modality_observations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_sha = sha256_file(staged_path)
    done = _completed(output, staged_sha)
    expected = len(queries) * len(arms) * repetitions
    counts = {"resumed": len(done), "written": 0, "errors": 0, "censored": 0}
    started = time.time()
    failure: str | None = None
    backend: TriDBLiveDataset | None = None
    try:
        backend = TriDBLiveDataset(str(spec["backend_dataset_name"]), spec)
        with output.open("a", encoding="utf-8") as handle:
            for query in queries:
                reference.prepare_query(query, {query.hop_limit})
                full_valid_ids = reference.valid_result_ids(query, query.hop_limit)
                backend.prepare_query(query, {query.hop_limit})
                point = _point(point_spec, query.hop_limit)
                _set_effort(backend, point)
                for arm in arms:
                    if all(
                        (query.query_id, arm, repetition) in done
                        for repetition in range(repetitions)
                    ):
                        continue
                    for _ in range(warmups):
                        backend.execute_modality(
                            query, point.plan, top_n=top_n, arm=arm
                        )
                    for repetition in range(repetitions):
                        key = (query.query_id, arm, repetition)
                        if key in done:
                            continue
                        result = backend.execute_modality(
                            query, point.plan, top_n=top_n, arm=arm
                        )
                        result_ids = list(result["result_ids"])
                        if any(
                            node_id not in reference.id_to_idx for node_id in result_ids
                        ):
                            raise RuntimeError(
                                f"{key}: result contains unknown node ID"
                            )
                        if result.get("graph_censored"):
                            counts["censored"] += 1
                        valid_count = sum(
                            node_id in full_valid_ids for node_id in result_ids
                        )
                        row = {
                            "schema_version": "e1-modality-observation-v0.2.0",
                            "staged_config_sha256": staged_sha,
                            "dataset": dataset_name,
                            "system": "tridb_live",
                            "query_id": query.query_id,
                            "template": query.template,
                            "arm": arm,
                            "repetition": repetition,
                            "status": result["status"],
                            "graph_censored": bool(result.get("graph_censored")),
                            "termination": result.get("termination"),
                            "client_wall_latency_ns": round(
                                float(result["latency_ms"]) * 1_000_000
                            ),
                            "quality": result["quality"],
                            "result_ids": result_ids,
                            "full_constraint_validity_fraction": (
                                1.0 if not result_ids else valid_count / len(result_ids)
                            ),
                            "candidates": int(
                                result["intermediate_cardinality"].get("candidates")
                                or 0
                            ),
                            "edges_examined": int(
                                result["intermediate_cardinality"].get(
                                    "graph_edges_examined"
                                )
                                or 0
                            ),
                            "operating_point": point.as_dict(),
                            "valid_for_headline_claims": False,
                        }
                        _append(handle, row)
                        counts["written"] += 1
                        done.add(key)
                        if row["status"] != "ok" or row["graph_censored"]:
                            raise RuntimeError(
                                f"modality row failed execution gates: {key}"
                            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        counts["errors"] += 1
    finally:
        if backend is not None:
            backend.close()
    manifest = {
        "schema_version": "e1-modality-run-v0.2.0",
        "staged_config": artifact_record(staged_path),
        "operating_points": artifact_record(
            Path(staged["outputs"]["operating_points"])
        ),
        "counts": counts,
        "expected_observations": expected,
        "completed_observations": len(done),
        "seconds": round(time.time() - started, 3),
        "complete": failure is None and len(done) == expected,
        "failure": failure,
        "valid_for_headline_claims": failure is None and len(done) == expected,
    }
    write_json(output.parent / "modality_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-config",
        type=Path,
        default=Path("configs/e1/staged_v0.2.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(design(args.staged_config), indent=2, sort_keys=True))
        return 0
    manifest = run(args.staged_config)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
