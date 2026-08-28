from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.e1.composition.analysis import (
    matched_quality_pairs,
    pareto_frontier,
)
from experiments.e1.composition.analyze_grid import representative_pairs
from experiments.e1.composition.config import load_config, validate_config
from experiments.e1.composition.model import (
    OperatingPoint,
    enumerate_points,
    limit_points_balanced,
    normalize_observation,
)
from experiments.e1.composition.calibration_runner import design as calibration_design
from experiments.e1.composition.runner import (
    design_summary,
    select_systems,
    system_order,
)
from experiments.e1.composition.staging import (
    load_staged_config,
    staged_dataset,
    stratified_query_split,
)
from experiments.e0.plan_spread.live_backend import ShipCounter
from experiments.e0.plan_spread.model import QuerySpec


CONFIG = Path("configs/e1/composition_v0.1.yaml")
STAGED_CONFIG = Path("configs/e1/staged_v0.2.yaml")
MAG_STAGED_CONFIG = Path("configs/e1/staged_mag_v0.3.yaml")


def test_frozen_e1_scope_has_only_two_headline_systems():
    config = load_config(CONFIG)
    assert set(config["systems"]["headline"]) == {
        "tridb_live",
        "polyglot_tuned",
    }
    assert config["modality_ablation"]["system"] == "tridb_live"
    assert (
        config["datasets"]["openevolve_smoke"]["valid_for_performance_claims"] is False
    )


def test_config_rejects_naive_headline_arm():
    config = deepcopy(load_config(CONFIG))
    config["systems"]["headline"].append("polyglot_naive")
    with pytest.raises(ValueError, match="exactly"):
        validate_config(config)


def test_config_rejects_polyglot_modality_ablation():
    config = deepcopy(load_config(CONFIG))
    config["modality_ablation"]["system"] = "polyglot_tuned"
    with pytest.raises(ValueError, match="TriDB-only"):
        validate_config(config)


def test_config_rejects_performance_claim_from_tiny_openevolve_smoke():
    config = deepcopy(load_config(CONFIG))
    config["datasets"]["openevolve_smoke"]["valid_for_performance_claims"] = True
    with pytest.raises(ValueError, match="must not be valid"):
        validate_config(config)


def test_pareto_frontier_removes_slower_lower_quality_points():
    points = [
        {"point_id": "fast", "latency_p50_ms": 1.0, "recall_at_20": 0.70},
        {"point_id": "bad", "latency_p50_ms": 2.0, "recall_at_20": 0.60},
        {"point_id": "balanced", "latency_p50_ms": 2.0, "recall_at_20": 0.85},
        {"point_id": "quality", "latency_p50_ms": 4.0, "recall_at_20": 0.95},
    ]
    assert [point["point_id"] for point in pareto_frontier(points)] == [
        "fast",
        "balanced",
        "quality",
    ]


def test_matched_quality_pairs_exclude_unmatched_quality():
    tridb = [
        {
            "point_id": "t-balanced",
            "latency_p50_ms": 2.0,
            "mrr": 0.80,
            "recall_at_20": 0.90,
        }
    ]
    polyglot = [
        {
            "point_id": "p-matched",
            "latency_p50_ms": 5.0,
            "mrr": 0.79,
            "recall_at_20": 0.89,
        },
        {
            "point_id": "p-unmatched",
            "latency_p50_ms": 1.0,
            "mrr": 0.50,
            "recall_at_20": 0.60,
        },
    ]
    pairs = matched_quality_pairs(
        tridb,
        polyglot,
        mrr_epsilon=0.02,
        recall_epsilon=0.02,
    )
    assert len(pairs) == 1
    assert pairs[0]["polyglot_point_id"] == "p-matched"
    assert pairs[0]["polyglot_over_tridb_p50"] == 2.5


def _query() -> QuerySpec:
    return QuerySpec.from_mapping(
        {
            "query_id": "q1",
            "dataset": "synthetic",
            "query_text": "query",
            "anchor_ids": ["a"],
            "answer_ids": ["b"],
            "edge_types": ["edge"],
            "hop_limit": 2,
            "structured_predicate": {},
            "target_entity_type": "node",
            "template": "synthetic",
            "annotation_status": "exact",
        }
    )


def test_operating_points_use_query_hops_and_balance_shapes():
    config = load_config(CONFIG)
    points = enumerate_points(config, _query())
    assert len(points) == 100
    assert {point.hops for point in points} == {2}
    limited = limit_points_balanced(points, 6)
    assert len(limited) == 6
    assert {point.shape for point in limited} == {
        "vector_first",
        "filter_first",
        "traverse_first",
    }


def test_staged_split_is_deterministic_stratified_and_disjoint():
    staged = load_staged_config(STAGED_CONFIG)
    rows = [
        QuerySpec.from_mapping(
            {
                "query_id": f"{template}-{index}",
                "dataset": "synthetic",
                "query_text": "q",
                "anchor_ids": ["a"],
                "answer_ids": ["b"],
                "edge_types": ["edge"],
                "hop_limit": 1,
                "structured_predicate": {},
                "target_entity_type": "node",
                "template": template,
                "annotation_status": "exact",
            }
        )
        for template, size in (("common", 8), ("rare", 2))
        for index in range(size)
    ]
    kwargs = {
        "seed": staged["split"]["seed"],
        "calibration_fraction": staged["split"]["calibration_fraction"],
    }
    first = stratified_query_split(rows, **kwargs)
    second = stratified_query_split(list(reversed(rows)), **kwargs)
    assert first == second
    assert len(first["calibration"]) == 3
    assert not set(first["calibration"]) & set(first["evaluation"])
    assert {value.split("-")[0] for value in first["calibration"]} == {
        "common",
        "rare",
    }


def test_mag_staged_addendum_injects_a_versioned_headline_dataset():
    staged = load_staged_config(MAG_STAGED_CONFIG)
    base = load_config(CONFIG)
    name = staged_dataset(staged, base)
    assert name == "stark_mag"
    assert base["datasets"][name]["backend_dataset_name"] == "stark_mag"
    assert base["datasets"][name]["valid_for_performance_claims"] is True


def test_abba_order_and_dry_run_never_claim_headline_result():
    assert system_order(0) + system_order(1) == [
        "tridb_live",
        "polyglot_tuned",
        "polyglot_tuned",
        "tridb_live",
    ]
    design = design_summary(
        load_config(CONFIG),
        datasets=["stark_prime"],
        query_limit=1,
        point_limit=6,
        repetitions=2,
    )
    assert design["total_observations"] == 24
    assert design["valid_for_headline_claims"] is False

    full = design_summary(
        load_config(CONFIG),
        datasets=["stark_prime"],
        query_limit=None,
        point_limit=None,
        repetitions=30,
        run_kind="full_grid",
    )
    assert full["total_observations"] == 240_000
    assert full["would_be_headline_eligible_on_success"] is True


def test_system_isolated_mag_calibration_preserves_full_expected_grid():
    config = load_config(CONFIG)
    assert select_systems(config, ["polyglot_tuned"]) == ["polyglot_tuned"]
    with pytest.raises(ValueError, match="unique"):
        select_systems(config, ["tridb_live", "tridb_live"])

    design = calibration_design(MAG_STAGED_CONFIG, ["tridb_live"])
    assert design["selected_systems"] == ["tridb_live"]
    assert design["stage_expected_observations"] == 3_000
    assert design["expected_observations"] == 6_000


def test_ship_counter_preserves_boundary_breakdown():
    counter = ShipCounter()
    counter.add("store_to_orchestrator", [1, 2])
    counter.add("store_to_orchestrator", [3])
    counter.add("orchestrator_to_store", [1])
    assert counter.rows == 4
    assert counter.rows_by_boundary == {
        "store_to_orchestrator": 3,
        "orchestrator_to_store": 1,
    }
    assert counter.bytes == sum(counter.bytes_by_boundary.values())


def test_observation_contract_rejects_invalid_ids_for_latency_claim():
    result = {
        "status": "ok",
        "valid_for_system_latency_claims": True,
        "latency_ms": 1.5,
        "quality": {
            "hit_at_1": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "recall_at_20": 0.0,
        },
        "result_ids": ["outside-constraint-set"],
        "client_query_round_trips": 1,
        "store_rpc_count": 2,
        "cross_store_handoff_count": 1,
    }
    row = normalize_observation(
        config_sha256="a" * 64,
        dataset="synthetic",
        dataset_valid_for_performance=True,
        system="polyglot_tuned",
        query=_query(),
        point=OperatingPoint("traverse_first", 20, 2, "during", 100),
        repetition=0,
        randomized_block=0,
        order_in_block=1,
        result=result,
        valid_result_ids={"b"},
    )
    assert row["result_validity"] is False
    assert row["valid_for_pilot_latency"] is False
    assert row["valid_for_headline_claims"] is False


def test_full_grid_observation_becomes_headline_eligible_only_after_gates():
    result = {
        "status": "ok",
        "valid_for_system_latency_claims": True,
        "latency_ms": 1.0,
        "quality": {
            "hit_at_1": 1.0,
            "hit_at_5": 1.0,
            "mrr": 1.0,
            "recall_at_20": 1.0,
        },
        "result_ids": ["b"],
    }
    row = normalize_observation(
        config_sha256="a" * 64,
        dataset="synthetic",
        dataset_valid_for_performance=True,
        system="tridb_live",
        query=_query(),
        point=OperatingPoint("traverse_first", 20, 2, "during", 100),
        repetition=0,
        randomized_block=0,
        order_in_block=0,
        result=result,
        valid_result_ids={"b"},
        run_kind="full_grid",
        allow_headline_claim=True,
    )
    assert row["valid_for_headline_claims"] is True


def test_representative_pairs_ignore_quality_zero_and_keep_matched_positive():
    base = {
        "latency_p50_ms": 1.0,
        "mrr": 0.0,
        "recall_at_20": 0.0,
    }
    tridb = [
        {"point_id": "zero", **base},
        {
            "point_id": "good",
            "latency_p50_ms": 2.0,
            "mrr": 1.0,
            "recall_at_20": 1.0,
        },
    ]
    polyglot = [
        {"point_id": "zero", **base},
        {
            "point_id": "good",
            "latency_p50_ms": 4.0,
            "mrr": 1.0,
            "recall_at_20": 1.0,
        },
    ]
    selected = representative_pairs(
        tridb,
        polyglot,
        mrr_epsilon=0.02,
        recall_epsilon=0.02,
        iso_plan=True,
    )
    assert {row["operating_label"] for row in selected} == {
        "fast",
        "balanced",
        "high_quality",
    }
    assert {row["tridb_point_id"] for row in selected} == {"good"}
