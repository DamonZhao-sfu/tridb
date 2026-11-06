"""T2: v0.7.0 configuration -> historical executor (VGR / P1 / P2 when rank_impl is "none")."""

import pytest

from tools import trace_joint_live as live
from trimem.experiments.plan_execution_adapter_v070 import compile_config, runner_config
from trimem.experiments.session_retrieval import SessionConfigV2, retrieve
from trimem.optimizer.plan_schema_v070 import JointSpaceV070


def configs(n=60, seed=7):
    return JointSpaceV070.live().sample(n, seed=seed)


def test_parity_with_stock_for_rank_not_none():
    cs = [c for c in configs() if c["rank_impl"] != "none"]
    assert cs
    for c in cs:
        new, old = runner_config(c), live.runner_config(c)
        assert new["v2"].pop("ordering") in ("colbert_score", "cross_encoder_score")
        assert new == old


@pytest.mark.parametrize("order,plan", [("none", "VGR"), ("before", "P1"), ("after", "P2")])
def test_rank_none_maps_to_historical_plans(order, plan):
    c = JointSpaceV070.live().canonicalize({"filter_order": order, "rank_impl": "none", "sem_threshold_pre": 0.8,
                                            "sem_threshold_post": 0.4, "returned_k": 8})
    cfg = runner_config(c)
    assert cfg["plan"] == plan
    assert cfg["v2"]["rank_impl"] == "none" and cfg["v2"]["ordering"] == "query_cosine"
    assert cfg["v2"]["semfilter_threshold"] == {"none": None, "before": 0.8, "after": 0.4}[order]
    assert cfg["memory_config"]["top_k"] == 8
    assert compile_config(c).plan == plan  # SessionConfigV2 accepts it


def test_stock_runner_config_refuses_rank_none():
    with pytest.raises(ValueError):
        live.runner_config({"rank_impl": "none"})


class FakeAdapter:
    """Seeds with cosines; one hop of neighbours; everything allowed."""

    def __init__(self):
        self.sessions = {f"s{i}": {"id": f"s{i}", "cosine": c} for i, c in enumerate(
            [0.9, 0.5, 0.7, 0.7, 0.2, 0.95, 0.1])}

    def seeds(self, count):
        return [self.sessions[k] for k in ("s0", "s1", "s2")][:count]

    def neighbors(self, session, edge, fanout):
        return [self.sessions[k] for k in ("s3", "s4", "s5", "s6")][:fanout]

    def allowed(self, session):
        return True

    def hydrate(self, session):
        return {"id": session["id"], "text": session["id"]}


def test_retrieve_rank_none_orders_by_cosine_and_never_calls_semrank():
    def semrank(cards):
        raise AssertionError("semrank must not run for a rank-less plan")

    config = SessionConfigV2(plan="VGR", seed_count=3, depth=1, fanout=4, top_k=4, edges=("similar",))
    receipt = retrieve(FakeAdapter(), config, semrank=semrank)
    # (-cosine, id): s5 .95, s0 .9, s2 .7, s3 .7, s1 .5, s4 .2, s6 .1
    assert receipt["ranked_ids"] == ["s5", "s0", "s2", "s3", "s1", "s4", "s6"]
    assert receipt["selected_ids"] == ["s5", "s0", "s2", "s3"]
    assert "Rank" not in receipt["plan_order"] and "semantic_scores" not in receipt
