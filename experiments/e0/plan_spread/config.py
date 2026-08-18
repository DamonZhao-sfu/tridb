"""Load, validate, and enumerate the frozen E0 plan-space configuration."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from .model import PlanSpec

SHAPES = {"vector_first", "filter_first", "traverse_first"}
PLACEMENTS = {"pre", "during", "post"}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("plan-space config must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") not in {
        "e0-plan-space-v0.2.0",
        "e0-plan-space-v0.3.0",
    }:
        raise ValueError("unsupported plan-space schema_version")
    valid = config.get("valid_placements") or {}
    if set(valid) != SHAPES:
        raise ValueError(f"valid_placements must define exactly {sorted(SHAPES)}")
    for shape, placements in valid.items():
        if not placements or not set(placements) <= PLACEMENTS:
            raise ValueError(f"invalid placements for {shape}: {placements}")
    datasets = config.get("datasets") or {}
    if set(datasets) != {"stark_prime", "openevolve"}:
        raise ValueError("E0 requires exactly stark_prime and openevolve")
    for name, spec in datasets.items():
        dims = spec.get("dimensions") or {}
        if not set(dims.get("shape", [])) <= SHAPES:
            raise ValueError(f"{name}: invalid shape")
        if any(int(value) <= 0 for value in dims.get("k", [])):
            raise ValueError(f"{name}: k must be positive")
        if any(int(value) <= 0 for value in dims.get("hops", [])):
            raise ValueError(f"{name}: hops must be positive")
        for field in ("nodes", "edges", "embeddings", "query_embeddings", "queries"):
            if field not in spec:
                raise ValueError(f"{name}: missing {field}")


def enumerate_plans(config: dict[str, Any], dataset: str) -> list[PlanSpec]:
    spec = config["datasets"][dataset]
    dims = spec["dimensions"]
    valid = config["valid_placements"]
    plans = []
    for shape, k, hops, placement in itertools.product(
        dims["shape"], dims["k"], dims["hops"], dims["predicate_placement"]
    ):
        if placement not in valid[shape]:
            continue
        plans.append(PlanSpec(shape, int(k), int(hops), placement))

    default = config["default_plan"]
    default_plan = PlanSpec(
        str(default["shape"]),
        int(default["k"]),
        int(default["hops"]),
        str(default["predicate_placement"]),
        is_default=True,
    )
    by_id = {plan.plan_id: plan for plan in plans}
    by_id[default_plan.plan_id] = default_plan
    return sorted(
        by_id.values(),
        key=lambda plan: (
            plan.shape,
            plan.k,
            plan.hops,
            plan.predicate_placement,
        ),
    )
