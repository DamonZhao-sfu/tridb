"""Load and validate the frozen E1 composition experiment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "e1-composition-v0.1.0"
HEADLINE_SYSTEMS = {"tridb_live", "polyglot_tuned"}
PLAN_SHAPES = {"vector_first", "filter_first", "traverse_first"}
MODALITY_ARMS = {
    "vector_only",
    "graph_only",
    "relational_only",
    "vector_relational",
    "vector_graph",
    "vector_graph_relational",
}
AUDIT_FLAGS = {
    "co_located",
    "host_network",
    "persistent_connections",
    "batched_id_transfer",
    "predicate_pushdown",
}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E1 config must be a mapping")
    validate_config(config)
    return config


def _require_positive_number(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive number")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    systems = config.get("systems") or {}
    headline = systems.get("headline") or []
    if set(headline) != HEADLINE_SYSTEMS or len(headline) != len(HEADLINE_SYSTEMS):
        raise ValueError(
            "systems.headline must contain exactly tridb_live and polyglot_tuned"
        )
    if any("naive" in str(system).lower() for system in headline):
        raise ValueError("a naive polyglot arm is outside the E1 scope")
    if systems.get("backend_factory") != {
        "tridb_live": "tridb_live",
        "polyglot_tuned": "polyglot_live",
    }:
        raise ValueError("systems.backend_factory must map the two headline systems")

    audit = config.get("polyglot_tuned_audit") or {}
    missing_flags = sorted(AUDIT_FLAGS - set(audit))
    if missing_flags:
        raise ValueError(f"polyglot_tuned_audit missing flags: {missing_flags}")
    disabled_flags = sorted(flag for flag in AUDIT_FLAGS if audit.get(flag) is not True)
    if disabled_flags:
        raise ValueError(f"Polyglot-Tuned audit flags must be true: {disabled_flags}")
    if audit.get("safe_async_policy") != "independent_requests_only":
        raise ValueError("safe_async_policy must be independent_requests_only")
    if set(audit.get("plan_shapes") or []) != PLAN_SHAPES:
        raise ValueError(f"plan_shapes must contain exactly {sorted(PLAN_SHAPES)}")

    comparison = config.get("comparison") or {}
    if set(comparison.get("analyses") or []) != {"iso_plan", "pareto_envelope"}:
        raise ValueError("comparison.analyses must be iso_plan and pareto_envelope")
    matched = comparison.get("matched_quality") or {}
    for field in ("mrr_epsilon", "recall_at_20_epsilon"):
        _require_positive_number(matched.get(field), f"matched_quality.{field}")
        if float(matched[field]) > 1:
            raise ValueError(f"matched_quality.{field} must not exceed 1")
    if comparison.get("latency_percentiles") != [50, 95, 99]:
        raise ValueError("latency_percentiles must be [50, 95, 99]")

    ablation = config.get("modality_ablation") or {}
    if ablation.get("system") != "tridb_live":
        raise ValueError("modality ablation is TriDB-only")
    if set(ablation.get("arms") or []) != MODALITY_ARMS:
        raise ValueError(f"modality arms must contain exactly {sorted(MODALITY_ARMS)}")

    datasets = config.get("datasets") or {}
    if "stark_prime" not in datasets or "openevolve_smoke" not in datasets:
        raise ValueError("datasets must include stark_prime and openevolve_smoke")
    smoke = datasets["openevolve_smoke"]
    if smoke.get("valid_for_performance_claims") is not False:
        raise ValueError("openevolve_smoke must not be valid for performance claims")
    for name, spec in datasets.items():
        for field in ("nodes", "edges", "embeddings", "query_embeddings", "queries"):
            if field not in spec:
                raise ValueError(f"{name}: missing {field}")
        for field in ("backend_dataset_name", "directed_edges_already_include_reverse"):
            if field not in spec:
                raise ValueError(f"{name}: missing {field}")
        if not isinstance(spec.get("tridb"), dict) or not isinstance(
            spec.get("live"), dict
        ):
            raise ValueError(f"{name}: missing live backend configuration")

    search = config.get("search_space") or {}
    if set(search.get("shapes") or []) != PLAN_SHAPES:
        raise ValueError(f"search_space.shapes must contain {sorted(PLAN_SHAPES)}")
    placements = search.get("valid_placements") or {}
    if set(placements) != PLAN_SHAPES:
        raise ValueError("valid_placements must define every plan shape")
    if any(
        not values or not set(values) <= {"pre", "during", "post"}
        for values in placements.values()
    ):
        raise ValueError("invalid predicate placement")
    for field in ("k", "ann_effort"):
        values = search.get(field) or []
        if not values or any(int(value) <= 0 for value in values):
            raise ValueError(f"search_space.{field} must contain positive values")

    measurement = config.get("measurement") or {}
    for field in (
        "pilot_repetitions",
        "warmups",
        "repetitions",
        "mixed_stream_requests",
        "soft_timeout_seconds",
    ):
        _require_positive_number(measurement.get(field), f"measurement.{field}")
    if int(measurement["repetitions"]) < 20:
        raise ValueError("measurement.repetitions must be at least 20 for p95")
    if int(measurement["mixed_stream_requests"]) < 1000:
        raise ValueError("mixed_stream_requests must be at least 1000 for p99")
    if measurement.get("execution") != "sequential_systems":
        raise ValueError("systems must run sequentially to avoid resource interference")


def summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "headline_systems": config["systems"]["headline"],
        "analyses": config["comparison"]["analyses"],
        "modality_ablation_system": config["modality_ablation"]["system"],
        "datasets": sorted(config["datasets"]),
        "status": config["material_passport"]["verification_status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/e1/composition_v0.1.yaml"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(summary(load_config(args.config)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
