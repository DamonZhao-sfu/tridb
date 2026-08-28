"""Deterministic held-out split and staged E1 configuration helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.e0.plan_spread.model import QuerySpec

STAGED_SCHEMA = "e1-composition-staged-v0.2.0"
STAGED_SCHEMAS = {STAGED_SCHEMA, "e1-composition-staged-mag-v0.3.0"}


def load_query_specs(path: Path) -> list[QuerySpec]:
    """Read query contracts without materializing the reference corpus."""
    return [
        QuerySpec.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_staged_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("schema_version") not in STAGED_SCHEMAS
    ):
        raise ValueError(
            f"staged schema_version must be one of {sorted(STAGED_SCHEMAS)}"
        )
    split = config.get("split") or {}
    if split.get("policy") != "stratified_sha256" or split.get("strata") != "template":
        raise ValueError("staged split must be stratified_sha256 by template")
    fraction = float(split.get("calibration_fraction", 0))
    if not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if not str(split.get("seed", "")):
        raise ValueError("split.seed must be non-empty")
    calibration = config.get("calibration") or {}
    evaluation = config.get("evaluation") or {}
    for section, field in (
        (calibration, "repetitions"),
        (calibration, "warmups"),
        (evaluation, "repetitions"),
        (evaluation, "warmups"),
    ):
        if int(section.get(field, -1)) < 0:
            raise ValueError(f"{field} must be non-negative")
    if int(calibration["repetitions"]) <= 0 or int(evaluation["repetitions"]) < 20:
        raise ValueError("calibration needs repetitions; evaluation needs at least 20")
    return config


def staged_dataset(staged: dict[str, Any], base: dict[str, Any]) -> str:
    """Resolve the staged headline dataset, preserving v0.2's PRIME default."""
    name = str(staged.get("dataset", "stark_prime"))
    override = staged.get("dataset_spec")
    if name not in base.get("datasets", {}) and isinstance(override, dict):
        required = {
            "backend_dataset_name",
            "directed_edges_already_include_reverse",
            "nodes",
            "edges",
            "embeddings",
            "query_embeddings",
            "queries",
            "tridb",
            "live",
        }
        missing = sorted(required - set(override))
        if missing:
            raise ValueError(f"staged dataset_spec missing fields: {missing}")
        base["datasets"][name] = override
    if name not in base.get("datasets", {}):
        raise ValueError(f"staged dataset {name!r} is absent from the base config")
    if not base["datasets"][name].get("valid_for_performance_claims"):
        raise ValueError(f"staged dataset {name!r} is not valid for performance claims")
    return name


def stratified_query_split(
    queries: list[QuerySpec], *, seed: str, calibration_fraction: float
) -> dict[str, Any]:
    """Select calibration IDs by stable hashes within each query template."""
    grouped: dict[str, list[QuerySpec]] = defaultdict(list)
    for query in queries:
        grouped[query.template].append(query)
    calibration: set[str] = set()
    strata: dict[str, dict[str, Any]] = {}
    for template, members in sorted(grouped.items()):
        count = max(1, int(len(members) * calibration_fraction + 0.5))
        ranked = sorted(
            members,
            key=lambda query: hashlib.sha256(
                f"{seed}|{template}|{query.query_id}".encode()
            ).hexdigest(),
        )
        selected = sorted(query.query_id for query in ranked[:count])
        calibration.update(selected)
        strata[template] = {
            "total": len(members),
            "calibration": selected,
            "evaluation": sorted(
                query.query_id for query in members if query.query_id not in calibration
            ),
        }
    all_ids = {query.query_id for query in queries}
    return {
        "calibration": sorted(calibration),
        "evaluation": sorted(all_ids - calibration),
        "strata": strata,
    }


def point_signature(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["shape"]),
        int(row["k"]),
        str(row["predicate_placement"]),
        int(row["ann_effort"]),
    )


def signature_id(signature: tuple[str, int, str, int]) -> str:
    shape, k, placement, effort = signature
    return f"{shape}:k={k}:placement={placement}:ef={effort}"
