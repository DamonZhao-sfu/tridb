import math

import pytest

from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_schema import FIELD_NAMES, MASK_FIELDS, JointSpace


def test_vector_selector_removed_and_dimensions():
    space = JointSpace()
    codec = JointCodec(space)
    assert len(FIELD_NAMES) == 18
    assert len(MASK_FIELDS) == 6
    assert len(codec.encode(space.default())) == 24
    assert not {"vector_impl", "failure_slots", "sim_gate"} & set(FIELD_NAMES)
    for name in ["vector_impl", "failure_slots", "sim_gate"]:
        with pytest.raises(ValueError, match="Unknown"):
            space.canonicalize({name: "single"})


def test_conditional_identity_and_typed_plan():
    space = JointSpace()
    codec = JointCodec(space)
    c = space.default(filter_order="before", graph_impl="BFS")
    d = dict(c, beam_width=8, sem_threshold_post=0.8)
    assert codec.key(c) == codec.key(d)
    assert space.canonicalize(d)["beam_width"] is None
    assert [n.role for n in space.plan(c).nodes] == ["V", "F_pre", "G", "Rank"]
    assert space.plan(c).nodes[0].implementation == "task-ann"
    both = space.default(filter_order="both", graph_impl="beam")
    assert [n.role for n in space.plan(both).nodes] == [
        "V",
        "F_pre",
        "G",
        "F_post",
        "Rank",
    ]
    assert codec.key(both) != codec.key(dict(both, beam_width=8))


def test_sample_roundtrip_and_real_spacing():
    space = JointSpace()
    codec = JointCodec(space)
    candidates = space.sample(1000, seed=17)
    assert len(candidates) == 1000
    for c in candidates:
        assert codec.decode(codec.encode(c)) == c
    a = space.default(returned_k=5)
    b = space.default(returned_k=8)
    i = FIELD_NAMES.index("returned_k")
    assert codec.encode(a)[i] == pytest.approx(2 / 7)
    assert codec.encode(b)[i] == pytest.approx(5 / 7)


@pytest.mark.parametrize("value", [16.3, True, math.nan, math.inf])
def test_integer_parameters_reject_illegal_values(value):
    with pytest.raises(ValueError):
        JointSpace().canonicalize({"seed_count": value})


def test_capability_subset_and_strict_decode():
    space = JointSpace.live()
    codec = JointCodec(space)
    for c in space.sample(50, seed=7):
        assert c["graph_impl"] in {"BFS", "best-first", "beam", "MCTS"}
        assert c["rank_impl"] in {"ColBERT", "LLM", "cross-encoder"}
        assert c["filter_order"] in {"none", "before", "after"}
        assert c["graph_direction"] in {"Forward", "Backward", "Both"}
        assert (c["beam_width"] is None) == (c["graph_impl"] != "beam")
        assert (c["retrieval_interval"] is None) == (c["injection_timing"] != "periodic")
    with pytest.raises(ValueError, match="capability"):
        space.canonicalize({"graph_impl": "evolutionary"})
    x = codec.encode(space.default())
    x[FIELD_NAMES.index("seed_count")] = 0.12345
    with pytest.raises(ValueError):
        codec.decode(x)


def test_capability_identity_is_bound_to_configuration():
    a = JointSpace(execution_fingerprint="graph-A")
    b = JointSpace(execution_fingerprint="graph-B")
    assert JointCodec(a).key(a.default()) != JointCodec(b).key(b.default())


def test_optional_filter_masks_and_plan_mapping():
    """Rank is mandatory; filter_order none = VGR+Rank with no threshold field."""
    from trimem.experiments.plan_execution_adapter import compile_config

    space = JointSpace.live()
    c = space.default(filter_order="none")
    assert c["filter_pre_impl"] is None and c["filter_post_impl"] is None
    assert c["sem_threshold_pre"] is None and c["sem_threshold_post"] is None
    assert compile_config(c).plan == "VGR+Rank"
    assert compile_config(c).edges == ("similar", "similar_up")  # Both = historical behaviour
    assert compile_config(space.default(graph_direction="Forward")).edges == ("similar",)
    assert compile_config(space.default(graph_direction="Backward")).edges == ("similar_up",)
    b = compile_config(space.default(graph_impl="beam", beam_width=4))
    assert b.graph_impl == "beam" and b.beam_width == 4
    assert compile_config(space.default(filter_order="before")).plan == "P1+Rank"
    assert compile_config(space.default(filter_order="after")).plan == "P2+Rank"
    roles = [n.role for n in space.plan(c).nodes]
    assert roles == ["V", "G", "Rank"]
    # Engine extension v0.1.0: with the new operators fixed to their historical values the
    # live space is still the 5,040-config three-plan space (brute force on that subset) ...
    import itertools

    fixed = JointSpace({**space.domains, "graph_impl": ("BFS",), "rank_impl": ("ColBERT",),
                        "retrieval_unit": ("whole",), "injection_timing": ("start",),
                        "graph_direction": ("Both",)})
    names = list(fixed.domains)
    keys = {
        fixed.key(dict(zip(names, v)))
        for v in itertools.product(*(fixed.domains[n] for n in names))
    }
    assert len(keys) == 5040
    # ... and the full live space multiplies it by graph 7 (beam x4) x injection 3 (periodic x2)
    # x rank 3 x unit 3 x direction 3 = 567.
    assert 5040 * 7 * 3 * 3 * 3 * 3 == 2_857_680
    codec = JointCodec(space)
    assert codec.decode(codec.encode(c)) == c
