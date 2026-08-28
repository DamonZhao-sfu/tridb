"""Stable E1 operating-point and observation contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from experiments.e0.plan_spread.model import PlanSpec, QuerySpec

OBSERVATION_SCHEMA = "e1-composition-observation-v0.1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class OperatingPoint:
    shape: str
    k: int
    hops: int
    predicate_placement: str
    ann_effort: int

    @property
    def plan(self) -> PlanSpec:
        return PlanSpec(self.shape, self.k, self.hops, self.predicate_placement)

    @property
    def point_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()[
            :12
        ]
        return f"{self.shape}-{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "k": self.k,
            "hops": self.hops,
            "predicate_placement": self.predicate_placement,
            "ann_effort": self.ann_effort,
        }


def enumerate_points(config: dict[str, Any], query: QuerySpec) -> list[OperatingPoint]:
    search = config["search_space"]
    points = [
        OperatingPoint(shape, int(k), query.hop_limit, placement, int(effort))
        for shape in search["shapes"]
        for placement in search["valid_placements"][shape]
        for k in search["k"]
        for effort in search["ann_effort"]
    ]
    return sorted(
        points,
        key=lambda point: (
            point.shape,
            point.k,
            point.ann_effort,
            point.predicate_placement,
        ),
    )


def limit_points_balanced(
    points: list[OperatingPoint], limit: int | None
) -> list[OperatingPoint]:
    if limit is None or limit >= len(points):
        return points
    if limit <= 0:
        raise ValueError("point limit must be positive")
    groups: dict[str, list[OperatingPoint]] = {}
    for point in points:
        groups.setdefault(point.shape, []).append(point)
    offsets = {shape: 0 for shape in groups}
    selected = []
    while len(selected) < limit:
        for shape in sorted(groups):
            offset = offsets[shape]
            if offset < len(groups[shape]) and len(selected) < limit:
                selected.append(groups[shape][offset])
                offsets[shape] += 1
        if all(offsets[shape] >= len(groups[shape]) for shape in groups):
            break
    return selected


def normalize_observation(
    *,
    config_sha256: str,
    dataset: str,
    dataset_valid_for_performance: bool,
    system: str,
    query: QuerySpec,
    point: OperatingPoint,
    repetition: int,
    randomized_block: int,
    order_in_block: int,
    result: dict[str, Any],
    valid_result_ids: set[Any],
    run_kind: str = "pilot",
    allow_headline_claim: bool = False,
) -> dict[str, Any]:
    status = str(result.get("status", "error"))
    result_ids = list(result.get("result_ids") or [])
    result_validity = status == "ok" and all(
        node_id in valid_result_ids for node_id in result_ids
    )
    graph_censored = bool(result.get("graph_censored", False))
    error_reason = result.get("error") or result.get("error_or_censor_reason")
    if graph_censored and not error_reason:
        error_reason = "graph_work_budget_censored"
    latency_ms = float(result.get("latency_ms", 0.0))
    valid_for_pilot_latency = bool(
        status == "ok"
        and result.get("valid_for_system_latency_claims", False)
        and result_validity
        and not graph_censored
    )
    quality = dict(result.get("quality") or {})
    required_quality = {"hit_at_1", "hit_at_5", "mrr", "recall_at_20"}
    if status == "ok" and set(quality) != required_quality:
        raise ValueError(f"quality fields must be exactly {sorted(required_quality)}")

    headline_eligible = bool(
        allow_headline_claim
        and dataset_valid_for_performance
        and valid_for_pilot_latency
    )
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "config_sha256": config_sha256,
        "run_kind": run_kind,
        "dataset": dataset,
        "dataset_valid_for_performance_claims": dataset_valid_for_performance,
        "system": system,
        "query_id": query.query_id,
        "query_hop_limit": query.hop_limit,
        "template": query.template,
        "annotation_status": query.annotation_status,
        "point_id": point.point_id,
        **point.as_dict(),
        "repetition": repetition,
        "randomized_block": randomized_block,
        "order_in_block": order_in_block,
        "status": status,
        "valid_for_pilot_latency": valid_for_pilot_latency,
        "valid_for_headline_claims": headline_eligible,
        "error_or_censor_reason": error_reason,
        "client_wall_latency_ns": round(latency_ms * 1_000_000),
        "stage_latency_ms": dict(result.get("stage_latency_ms") or {}),
        "client_request_count": int(result.get("client_query_round_trips", 1)),
        "store_rpc_count": int(
            result.get("store_rpc_count", result.get("round_trips", 0))
        ),
        "cross_store_handoff_count": int(
            result.get(
                "cross_store_handoff_count",
                max(0, int(result.get("round_trips", 0)) - 1),
            )
        ),
        "intermediate_rows_by_boundary": dict(
            result.get("intermediate_rows_by_boundary") or {}
        ),
        "payload_bytes_by_boundary": dict(
            result.get("payload_bytes_by_boundary") or {}
        ),
        "payload_measurement": result.get("payload_measurement"),
        "serialization_ns": round(float(result.get("serialization_ms", 0.0)) * 1e6),
        "serialization_ns_by_boundary": {
            boundary: round(float(value) * 1e6)
            for boundary, value in (
                result.get("serialization_ms_by_boundary") or {}
            ).items()
        },
        "intermediate_cardinality": dict(result.get("intermediate_cardinality") or {}),
        "graph_censored": graph_censored,
        "termination": result.get("termination"),
        "result_validity": result_validity,
        "quality": quality,
        "result_ids": result_ids,
    }
