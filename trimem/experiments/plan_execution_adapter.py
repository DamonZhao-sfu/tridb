"""Fail-closed mapping to the existing native-graph session retrieval executor."""

from trimem.optimizer.plan_schema import DIRECTION_EDGES, JointSpace

from .session_retrieval import SessionConfigV2, retrieve


def compile_config(config):
    c = JointSpace.live().canonicalize(config)
    return SessionConfigV2(
        plan={"none": "VGR+Rank", "before": "P1+Rank", "after": "P2+Rank"}[
            c["filter_order"]
        ],
        seed_count=c["seed_count"],
        depth=c["depth"],
        fanout=c["fanout"],
        top_k=c["returned_k"],
        edges=DIRECTION_EDGES[c["graph_direction"]],
        sim_gate=None,
        max_failed=None,
        graph_impl=c["graph_impl"],
        beam_width=c["beam_width"],
    )


def execute_plan(config, adapter, *, judge, ranker):
    """judge(cards) returns calibrated LLM scores in [0,1]; ranker is ColBERT.

    The caller constructs the native adapter with the frozen memory policy.
    Replacing it with a post-hoc selection filter would alter seed semantics.
    """
    c = JointSpace.live().canonicalize(config)
    policy = (
        "success-only-v0"
        if c["trajectory_category"] == "Success"
        else "success+failure-v1"
    )
    if getattr(adapter, "policy", None) != policy:
        raise ValueError("Adapter memory policy does not match trajectory_category")
    threshold = {
        "none": None,
        "before": c["sem_threshold_pre"],
        "after": c["sem_threshold_post"],
    }[c["filter_order"]]

    def semantic(cards):
        scores = judge(cards)
        if set(scores) != {card["id"] for card in cards}:
            raise ValueError("Judge must return exactly one score for every card")
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1
            for v in scores.values()
        ):
            raise ValueError("Invalid semantic judge score")
        return [
            {"id": card["id"], "useful": scores[card["id"]] >= threshold}
            for card in cards
        ]

    receipt = retrieve(adapter, compile_config(c), semfilter=semantic, semrank=ranker)
    receipt["joint_config"] = c
    receipt["typed_plan"] = JointSpace.live().plan(c).describe()
    return receipt
