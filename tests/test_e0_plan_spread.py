from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.e0.plan_spread.analyze import reduce_observations
from experiments.e0.plan_spread.config import enumerate_plans, load_config
from experiments.e0.plan_spread.live_backend import _identifier, _relationship
from experiments.e0.plan_spread.model import PlanSpec, QuerySpec
from experiments.e0.plan_spread.reference_backend import ReferenceDataset
from experiments.e0.plan_spread.runner import limit_plans_balanced


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/e0/plan_space_v0.2.yaml"),
        Path("configs/e0/plan_space_v0.3.yaml"),
    ],
)
def test_frozen_plan_counts_and_invalid_placements_are_removed(config_path):
    config = load_config(config_path)
    stark = enumerate_plans(config, "stark_prime")
    evolve = enumerate_plans(config, "openevolve")
    assert len(stark) == 61  # 5 valid shape/placement pairs * 6 k * 2 hops + default
    assert len(evolve) == 101  # 5 * 5 k * 4 hops + default
    assert not any(
        plan.shape == "vector_first" and plan.predicate_placement == "pre"
        for plan in stark
    )
    assert sum(plan.is_default for plan in stark) == 1


def _synthetic_dataset(tmp_path: Path) -> tuple[ReferenceDataset, QuerySpec]:
    nodes = tmp_path / "nodes.parquet"
    edges = tmp_path / "edges.parquet"
    embeddings = tmp_path / "embeddings.parquet"
    query_embeddings = tmp_path / "query_embeddings.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "node_id": ["root", "parent", "answer", "other"],
                "entity_type": ["program"] * 4,
                "generation": [0, 1, 2, 2],
            }
        ),
        nodes,
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "src_id": ["root", "parent", "parent"],
                "dst_id": ["parent", "answer", "other"],
                "edge_type": ["evolved_to"] * 3,
            }
        ),
        edges,
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "node_id": ["root", "parent", "answer", "other"],
                "embedding": [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
            }
        ),
        embeddings,
    )
    pq.write_table(
        pa.Table.from_pydict({"query_id": ["q0"], "embedding": [[1.0, 0.0]]}),
        query_embeddings,
    )
    spec = {
        "nodes": str(nodes),
        "edges": str(edges),
        "embeddings": str(embeddings),
        "query_embeddings": str(query_embeddings),
        "directed_edges_already_include_reverse": False,
    }
    query = QuerySpec.from_mapping(
        {
            "query_id": "q0",
            "dataset": "synthetic",
            "query_text": "descendants",
            "anchor_ids": ["root"],
            "answer_ids": ["answer"],
            "edge_types": ["evolved_to"],
            "hop_limit": 2,
            "structured_predicate": {"generation_gte": 2},
            "target_entity_type": "program",
            "template": "descendant_search",
            "annotation_status": "exact",
        }
    )
    return ReferenceDataset("synthetic", spec), query


def test_all_three_shapes_share_semantics_on_synthetic_graph(tmp_path):
    dataset, query = _synthetic_dataset(tmp_path)
    dataset.prepare_query(query, {2})
    plans = [
        PlanSpec("vector_first", 4, 2, "post"),
        PlanSpec("filter_first", 4, 2, "pre"),
        PlanSpec("traverse_first", 4, 2, "during"),
    ]
    for plan in plans:
        result = dataset.execute(query, plan, top_n=20)
        assert result["status"] == "ok"
        assert result["quality"]["hit_at_1"] == 1.0
        assert result["quality"]["mrr"] == 1.0
        assert result["valid_for_system_latency_claims"] is False


def _observation(
    plan_id: str, latency: float, hit: float, mrr: float, *, default=False
):
    return {
        "dataset": "d",
        "query_id": "q",
        "plan_id": plan_id,
        "shape": "vector_first",
        "k": 8,
        "hops": 2,
        "predicate_placement": "post",
        "is_default": default,
        "template": "t",
        "annotation_status": "a",
        "query_hop_limit": 2,
        "backend": "parquet_reference",
        "valid_for_system_latency_claims": False,
        "status": "ok",
        "latency_ms": latency,
        "quality": {
            "hit_at_1": hit,
            "hit_at_5": hit,
            "mrr": mrr,
            "recall_at_20": hit,
        },
    }


def test_spread_excludes_fast_low_quality_plan_and_grades_default():
    rows = []
    for latency in (1.0, 1.1, 0.9):
        rows.append(_observation("fast-good", latency, 1.0, 1.0))
    for latency in (4.0, 4.1, 3.9):
        rows.append(_observation("default-good", latency, 1.0, 1.0, default=True))
    for latency in (0.1, 0.1, 0.1):
        rows.append(_observation("fast-bad", latency, 0.0, 0.0))
    plan_rows, query_rows = reduce_observations(rows, eps_hit=0.02, eps_mrr=0.02)
    assert len(plan_rows) == 3
    assert len(query_rows) == 1
    assert query_rows[0]["quality_equivalent_plans"] == 2
    assert query_rows[0]["plan_spread"] == 4.0
    assert query_rows[0]["default_suboptimality"] == 4.0


def test_plan_id_ignores_default_marker():
    regular = PlanSpec("vector_first", 8, 2, "post")
    default = PlanSpec("vector_first", 8, 2, "post", is_default=True)
    assert regular.plan_id == default.plan_id
    assert json.loads(json.dumps(default.as_dict()))["is_default"] is True


def test_spread_is_undefined_with_only_one_quality_equivalent_plan():
    rows = [
        _observation("only-good", 1.0, 1.0, 1.0, default=True),
        _observation("fast-bad", 0.1, 0.0, 0.0),
    ]
    _, query_rows = reduce_observations(rows, eps_hit=0.02, eps_mrr=0.02)
    assert query_rows[0]["quality_equivalent_plans"] == 1
    assert query_rows[0]["plan_spread_defined"] is False
    assert query_rows[0]["plan_spread"] is None


def test_live_identifiers_are_sanitized_or_rejected():
    assert _relationship("interacts with") == "interacts_with"
    assert _relationship("evolved_to:reverse") == "evolved_to_reverse"
    with pytest.raises(ValueError):
        _identifier("node; DROP TABLE x")


def test_smoke_plan_limit_samples_every_shape_and_keeps_default():
    config = load_config(Path("configs/e0/plan_space_v0.2.yaml"))
    limited = limit_plans_balanced(enumerate_plans(config, "stark_prime"), 5)
    assert {plan.shape for plan in limited} == {
        "filter_first",
        "traverse_first",
        "vector_first",
    }
    assert len(limited) == 6
    assert sum(plan.is_default for plan in limited) == 1
