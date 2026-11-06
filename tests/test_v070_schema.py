"""T1: TRACE v0.7.0 schema (rank optional incl. "none", no LLM rank, Both / whole fixed)."""

import json

import pytest

from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_runtime import space_patches
from trimem.optimizer import plan_schema as base
from trimem.optimizer import plan_schema_v070 as v7


def test_fields_and_live_domains():
    full = {n: v for n, _, v in v7.FIELDS}
    assert full["rank_impl"] == ("ColBERT", "LLM", "cross-encoder", "none")
    assert {n: v for n, _, v in base.FIELDS}["rank_impl"] == ("ColBERT", "LLM", "cross-encoder")
    assert v7.LIVE_DOMAINS["rank_impl"] == ("ColBERT", "cross-encoder", "none")
    assert v7.LIVE_DOMAINS["graph_direction"] == ("Both",)
    assert v7.LIVE_DOMAINS["retrieval_unit"] == ("whole",)
    space = v7.JointSpaceV070.live()
    assert space.manifest()["schema"] == v7.SCHEMA_VERSION and "semantics" in space.manifest()


def test_llm_rank_refused():
    space = v7.JointSpaceV070.live()
    with pytest.raises(ValueError):
        space.canonicalize({"rank_impl": "LLM"})
    with pytest.raises(ValueError):
        base.JointSpace.live().canonicalize({"rank_impl": "none"})


def test_canonicalize_parity_for_rank_not_none():
    new, old = v7.JointSpaceV070.live(), base.JointSpace.live()
    configs = [c for c in new.sample(60, seed=11) if c["rank_impl"] != "none"]
    assert configs
    for c in configs:
        assert old.canonicalize(c) == new.canonicalize(c)


def test_returned_k_active_with_rank_none():
    c = v7.JointSpaceV070.live().canonicalize({"rank_impl": "none", "returned_k": 8})
    assert c["rank_impl"] == "none" and c["returned_k"] == 8


def test_keys_differ_from_v060():
    c = {"rank_impl": "ColBERT"}
    assert v7.JointSpaceV070.live().key(c) != base.JointSpace.live().key(c)


@pytest.mark.parametrize("order", ["none", "before", "after"])
@pytest.mark.parametrize("rank", ["ColBERT", "none"])
def test_plan_six_combinations(order, rank):
    space = v7.JointSpaceV070.live()
    c = space.canonicalize({"filter_order": order, "rank_impl": rank})
    spec = space.plan(c)
    roles = [n.role for n in spec.nodes]
    expect = ["V"] + (["F_pre"] if order == "before" else []) + ["G"] + (["F_post"] if order == "after" else [])
    assert roles == expect + (["Rank"] if rank != "none" else ["TopK"])
    last = spec.nodes[-1]
    if rank == "none":
        assert last.implementation == "query_cosine" and last.parameters["returned_k"] == c["returned_k"]
        assert spec.describe().endswith("TopK(query_cosine)")
    plan = v7.executor_plan(c)
    assert plan == {"none": "VGR", "before": "P1", "after": "P2"}[order] + ("+Rank" if rank != "none" else "")
    assert v7.has_rank(c) == (rank != "none")


def test_executor_plan_and_ordering_tables():
    assert sorted(v7.EXECUTOR_PLAN.values()) == sorted(["VGR", "P1", "P2", "VGR+Rank", "P1+Rank", "P2+Rank"])
    assert v7.ORDERING == {"none": "query_cosine", "ColBERT": "colbert_score", "cross-encoder": "cross_encoder_score"}


def test_sobol_covers_rank_none():
    ranks = {c["rank_impl"] for c in v7.JointSpaceV070.live().sample(40, seed=5)}
    assert ranks == {"ColBERT", "cross-encoder", "none"}


def test_stock_get_space_refuses_and_patched_accepts():
    m = {"space": json.loads(json.dumps(v7.JointSpaceV070.live(execution_fingerprint="fp").manifest()))}
    with pytest.raises(ValueError, match="Invalid capability for rank_impl"):
        driver.get_space(m)
    with space_patches():
        assert isinstance(driver.get_space(m), v7.JointSpaceV070)
    with pytest.raises(ValueError):
        driver.get_space(m)  # restored
