"""v5 declared space: union hop, visit-cap grid, uniform initialization.

No database, GPU, or model requests (the real-PG union-hop EXPLAIN gate lives in
test_trimem_optimizer_native.py).
"""

from __future__ import annotations

import re
from collections import Counter
from types import SimpleNamespace

import pytest
from test_trimem_optimizer_execution import compile_plan, explain_for

from trimem.optimizer.compiler import expand_node_count
from trimem.optimizer.executor import CapabilityError, check_streaming_plan
from trimem.optimizer.schema import CandidateSpec, expansion_bound, stable_hash
from trimem.optimizer.search import SearchData, SearchSettings, suggest
from trimem.optimizer.space import (
    MEDIUM_GRID,
    V5_PLAN_IDS,
    SpaceGrid,
    make_space,
)

UNION = "lineage|similar"


@pytest.mark.parametrize(
    "plan_id", ["vgr_v1", "seed_sem_expand_noscore_v1", "expand_row_sem_noscore_v1"]
)
def test_union_hop_sql_is_union_all_of_both_edge_types(plan_id):
    plan = compile_plan(plan_id, hop_types=("similar", UNION), seed_count=8, fanout=4)
    sql = plan.sql.upper()
    assert sql.count("TRIMEM.EXPAND") == 3
    assert sql.count("UNION ALL") == 1
    assert not re.search(r"UNION(?!\s+ALL)", sql)
    assert all(k not in sql for k in ("DISTINCT", "GROUP BY", "MATERIALIZED"))
    assert "trimem.expand(e1.dst, 'lineage', 4)" in plan.sql
    assert "trimem.expand(e1.dst, 'similar', 4)" in plan.sql
    # The union subquery exposes ``dst`` under the hop alias used downstream.
    assert ") e2\nJOIN attempt d ON d.vid = e2.dst" in plan.sql
    assert plan.expected_custom_nodes.count("TriMEM Expand") == 3
    check_streaming_plan(explain_for(plan), plan)


def test_union_hop_bound_is_product_of_fanout_times_edges():
    assert expansion_bound(8, 4, ("lineage",)) == 8 * 4
    assert expansion_bound(8, 4, ("lineage", "lineage")) == 8 * (4 + 16)
    assert expansion_bound(8, 4, (UNION,)) == 8 * 8
    assert expansion_bound(8, 4, ("similar", UNION)) == 8 * (4 + 4 * 8)
    assert expansion_bound(8, 4, (UNION, "lineage")) == 8 * (8 + 8 * 4)
    action = CandidateSpec("vgr_v1", seed_count=8, fanout=4, hop_types=(UNION, UNION))
    assert action.expansion_bound == 8 * (8 + 64)
    plan = compile_plan("vgr_v1", seed_count=8, fanout=4, hop_types=(UNION, UNION))
    assert plan.emitted_edge_upper_bound == action.expansion_bound
    assert expand_node_count(action) == 4
    with pytest.raises(ValueError, match="edge_emit_cap"):
        CandidateSpec(
            "vgr_v1", seed_count=8, fanout=8, hop_types=(UNION,), edge_emit_cap=127
        )
    with pytest.raises(ValueError, match="unsupported"):
        CandidateSpec("vgr_v1", hop_types=("lineage|lineage_up",))


def test_union_hop_gate_requires_every_expand_and_frontier_outer():
    plan = compile_plan("expand_row_sem_noscore_v1", hop_types=(UNION,), seed_count=8)
    explain = explain_for(plan)
    check_streaming_plan(explain, plan)
    append = explain[0]["Plan"]["Plans"][0]["Plans"][0]["Plans"][1]
    assert append["Node Type"] == "Append"
    append["Plans"].pop()
    with pytest.raises(CapabilityError, match="native operator mismatch"):
        check_streaming_plan(explain, plan)
    # A union-hop Append under a relational (non-frontier) rescan loop fails.
    root = explain_for(plan)[0]["Plan"]
    rescan = {"Node Type": "Nested Loop", "Plans": [{"Node Type": "Index Scan"}, root]}
    with pytest.raises(CapabilityError, match="rescan loop"):
        check_streaming_plan(rescan, plan)
    # Row SemFilter must sit above *all* Expand nodes, including union members.
    seed = compile_plan("seed_sem_expand_noscore_v1", hop_types=(UNION,), seed_count=8)
    with pytest.raises(CapabilityError, match="target/order"):
        check_streaming_plan(explain_for(plan), seed)


def test_default_grid_unchanged_and_medium_grid_count():
    # Pre-v5 default grid hash (v4 code): visit-cap grid must not perturb it.
    default = make_space()
    assert len(default) == 2481
    assert stable_hash([a.to_dict() for a in default.candidates]) == (
        "35bee99167932f9daf6c00d93ee221e27eb17b312b8e939bebafeae879f8e32b"
    )
    space = make_space(MEDIUM_GRID, V5_PLAN_IDS)
    counts = Counter(a.plan_id for a in space.candidates)
    assert counts == {
        "no_memory_v1": 1,
        "vgr_v1": 306,
        "seed_sem_expand_noscore_v1": 306,
        "expand_row_sem_noscore_v1": 306,
    }
    # Per plan: 3 top_k x 3 caps x [d1: 4 seeds x 2 fanouts; d2: 8 x {final,
    # all}; d3: (4 seeds at f4 + 1 at f8; bound 48*(16+256+4096) > 65536) x 2].
    assert len(space) == 919
    assert {a.hop_types for a in space.candidates if a.plan_id != "no_memory_v1"} == {
        (UNION,) * depth for depth in (1, 2, 3)
    }
    assert all(a.frontier == "final" for a in space.candidates if len(a.hop_types) == 1)
    caps = {
        a.candidate_visit_cap for a in space.candidates if a.plan_id != "no_memory_v1"
    }
    assert caps == {64, 256, 1024}
    retrieval = [a for a in space.candidates if a.plan_id != "no_memory_v1"]
    assert {(a.ordering, a.min_gain) for a in retrieval} == {("gain", 1e-9)}
    for action in space.candidates:
        assert action.expansion_bound <= action.edge_emit_cap
        assert action.plan_id == "no_memory_v1" or action.unknown_policy == "drop"
    assert len({space.encode(a) for a in space.candidates}) == len(space)
    with pytest.raises(ValueError, match="no legal configurations"):
        make_space(SpaceGrid(candidate_visit_cap=1, top_ks=(3,)), ("vgr_v1",))


@pytest.mark.parametrize(
    "plan_id", ["vgr_v1", "seed_sem_expand_noscore_v1", "expand_row_sem_noscore_v1"]
)
def test_all_frontier_sql_streams_every_depth_without_dedup(plan_id):
    plan = compile_plan(
        plan_id, hop_types=(UNION,) * 3, seed_count=8, fanout=4, frontier="all"
    )
    sql = plan.sql.upper()
    assert sql.count("TRIMEM.EXPAND") == 6  # each depth expanded once per parent
    assert not re.search(r"UNION(?!\s+ALL)", sql)
    assert all(k not in sql for k in ("DISTINCT", "GROUP BY", "MATERIALIZED", "WITH "))
    assert sql.count("ORDER BY") == 1  # ANN seeds only
    # Hop-1 and hop-2 endpoints are kept alongside the deeper ones.
    assert "SELECT e1.dst AS dst\nUNION ALL" in plan.sql
    assert "SELECT e2.dst AS dst\nUNION ALL" in plan.sql
    assert "JOIN attempt d ON d.vid = f1.dst" in plan.sql
    assert "s.sid AS seed_id" in plan.sql
    assert plan.expected_custom_nodes.count("TriMEM Expand") == 6
    final = compile_plan(plan_id, hop_types=(UNION,) * 3, seed_count=8, fanout=4)
    assert plan.emitted_edge_upper_bound == final.emitted_edge_upper_bound
    assert plan.emitted_edge_upper_bound == 8 * (8 + 64 + 512)
    assert plan.sql_hash != final.sql_hash
    check_streaming_plan(explain_for(plan), plan)


def test_frontier_identity_is_backward_compatible():
    old = CandidateSpec("vgr_v1", hop_types=(UNION, UNION))
    assert "frontier" not in old.to_dict()
    assert CandidateSpec.from_dict(old.to_dict()).frontier == "final"
    one_hop = CandidateSpec("vgr_v1", hop_types=(UNION,), frontier="all")
    assert (
        one_hop.action_hash == CandidateSpec("vgr_v1", hop_types=(UNION,)).action_hash
    )
    deep = CandidateSpec("vgr_v1", hop_types=(UNION, UNION), frontier="all")
    assert deep.to_dict()["frontier"] == "all"
    assert deep.action_hash != old.action_hash
    assert CandidateSpec.from_dict(deep.to_dict()) == deep
    with pytest.raises(ValueError, match="frontier"):
        CandidateSpec("vgr_v1", frontier="bfs")
    # Encoding: absent -> "final"; the feature is only added when "all" occurs.
    mixed = make_space(MEDIUM_GRID, ("vgr_v1",))
    assert "frontier" in mixed.feature_names
    shape = {
        "seed_count": 8,
        "fanout": 4,
        "top_k": 3,
        "ordering": "gain",
        "min_gain": 1e-9,
    }
    final_action = CandidateSpec.from_dict({**old.to_dict(), **shape})
    all_action = CandidateSpec.from_dict({**deep.to_dict(), **shape})
    index = mixed.feature_names.index("frontier")
    assert mixed.encode(final_action)[index] != mixed.encode(all_action)[index]
    assert "frontier" not in make_space().feature_names


def _settings(**kwargs):
    return SearchSettings(
        reference_quality=0,
        reference_tokens=10000,
        seed=42,
        initial_strategy="uniform",
        initial_random=10,
        **kwargs,
    )


def test_uniform_settings_are_validated():
    with pytest.raises(ValueError, match="initial_strategy"):
        SearchSettings(1, 1, initial_strategy="sobol")
    with pytest.raises(ValueError, match="initial_random"):
        SearchSettings(1, 1, initial_random=-1)
    with pytest.raises(ValueError, match="initial_random"):
        SearchSettings(1, 1, initial_random=True)


def test_uniform_init_baseline_then_n_distinct_random_then_bo(monkeypatch):
    space = make_space(MEDIUM_GRID, V5_PLAN_IDS)
    settings = _settings()
    observed: list[SearchData] = []
    pending: list[str] = []
    first = suggest(space, observed, pending, settings)
    assert first.action.plan_id == "no_memory_v1"
    assert first.strategy == "baseline_no_memory"
    # Repeat ask with nothing reserved: same deterministic baseline.
    assert suggest(space, observed, pending, settings) == first
    pending.append(first.action.action_hash)
    picks = []
    for _ in range(10):
        proposal = suggest(space, observed, pending, settings)
        assert proposal.strategy == "uniform_initialization"
        assert proposal.action.plan_id != "no_memory_v1"
        # Deterministic given (seed, #taken): a second ask returns the same.
        assert suggest(space, observed, pending, settings) == proposal
        pending.append(proposal.action.action_hash)
        picks.append(proposal.action)
    assert len({a.action_hash for a in picks}) == 10
    assert len({a.plan_id for a in picks}) > 1
    # Pending reservations and observations count the same way.
    observed = [SearchData(a, 0.1, 1000.0, 0.01, 100.0) for a in [first.action, *picks]]
    calls = []

    def fake(space_, observations, candidates, settings_):
        calls.append(len(candidates))
        return 0, 1.0

    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", fake)
    bo = suggest(space, observed, (), settings)
    assert bo.strategy == "qlognehvi" and calls
    # Another seed draws a different sequence.
    other = suggest(space, [], (first.action.action_hash,), _settings_seed(7))
    assert other.action.action_hash != picks[0].action_hash


def _settings_seed(seed):
    return SearchSettings(
        reference_quality=0,
        reference_tokens=10000,
        seed=seed,
        initial_strategy="uniform",
        initial_random=10,
    )


def test_uniform_init_failed_trials_count_and_no_baseline_when_absent():
    space = make_space(MEDIUM_GRID, ("vgr_v1",))
    settings = _settings()
    proposal = suggest(space, [], (), settings)
    assert proposal.strategy == "uniform_initialization"
    excluded = [a.action_hash for a in space.candidates[:10]]
    after = suggest(space, [], excluded, settings)
    assert after.strategy != "uniform_initialization"


def test_uniform_init_never_falls_through_to_per_plan_stratification(monkeypatch):
    space = make_space(MEDIUM_GRID, V5_PLAN_IDS)
    # initial_per_plan=1 as in the v4-derived A config; seed 11's 10 draws
    # miss expand_row_sem_noscore_v1, which per-plan init would then force.
    settings = SearchSettings(
        reference_quality=0,
        reference_tokens=10000,
        seed=11,
        initial_strategy="uniform",
        initial_random=10,
        initial_per_plan=1,
    )
    taken: list[str] = []
    strategies = []
    plans = set()
    for _ in range(11):
        proposal = suggest(space, [], taken, settings)
        strategies.append(proposal.strategy)
        plans.add(proposal.action.plan_id)
        taken.append(proposal.action.action_hash)
    assert strategies == ["baseline_no_memory"] + ["uniform_initialization"] * 10
    assert len(plans - {"no_memory_v1"}) < 3  # the case under test
    observed = [
        SearchData(
            next(a for a in space.candidates if a.action_hash == h),
            0.1,
            1e3,
            0.01,
            100.0,
        )
        for h in taken
    ]
    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", lambda *args: (0, 1.0))
    for _ in range(3):
        proposal = suggest(space, observed, (), settings)
        assert proposal.strategy == "qlognehvi"
        observed.append(SearchData(proposal.action, 0.1, 1e3, 0.01, 100.0))


def test_ordering_and_min_gain_are_backward_compatible_identity():
    old = CandidateSpec("vgr_v1", hop_types=("lineage",))
    assert "ordering" not in old.to_dict() and "min_gain" not in old.to_dict()
    restored = CandidateSpec.from_dict(old.to_dict())
    assert (restored.ordering, restored.min_gain) == ("stream", 0.0)
    plan = compile_plan("vgr_v1", seed_count=8)
    assert "min_gain" not in plan.sql and "min_gain" not in plan.params
    assert "AND d.is_valid AND d.fitness > p.fitness\n" in plan.sql
    new = CandidateSpec("vgr_v1", ordering="gain", min_gain=1e-9)
    assert new.to_dict()["ordering"] == "gain" and new.to_dict()["min_gain"] == 1e-9
    assert new.action_hash != CandidateSpec("vgr_v1").action_hash
    tight = compile_plan("seed_sem_expand_noscore_v1", seed_count=8, min_gain=1e-9)
    assert tight.sql.count("+ %(min_gain)s::double precision") == 2
    assert tight.params["min_gain"] == 1e-9
    for bad in ({"ordering": "random"}, {"min_gain": -1.0}, {"min_gain": True}):
        with pytest.raises(ValueError):
            CandidateSpec("vgr_v1", **bad)


def test_gain_ranked_pool_ranks_dedups_and_records_pool():
    from test_trimem_optimizer_execution import ExactTestTokenizer, Rows

    from trimem.optimizer.packing import GainRankedPool, pack_stream

    rows = [
        {"id": 3, "fitness": 1.0, "parent_fitness": 0.9, "changes": "a", "code": ""},
        {"id": 1, "fitness": 2.0, "parent_fitness": 1.0, "changes": "b", "code": ""},
        {"id": 3, "fitness": 1.0, "parent_fitness": 0.9, "changes": "a", "code": ""},
        {"id": 2, "fitness": 1.5, "parent_fitness": 0.5, "changes": "c", "code": ""},
        {"id": 4, "fitness": 0.5, "parent_fitness": 0.4, "changes": "d", "code": ""},
    ]
    source = Rows(rows)
    source.receipt = SimpleNamespace(ranked_by="stream", pool_rows=None)
    packed = pack_stream(
        GainRankedPool(source),
        ExactTestTokenizer(),
        tokenizer_id="exact-test-v1",
        top_k=3,
    )
    # gains: 1 -> 1.0, 2 -> 1.0 (tie: id asc), 3 -> 0.1, 4 -> 0.1 (tie: 3 first)
    assert packed.retained_ids == ["1", "2", "3"]
    assert packed.stop_reason == "top_k" and source.closed
    assert source.seen == len(rows) + 1  # the whole bounded pool was drained
    assert (source.receipt.ranked_by, source.receipt.pool_rows) == ("gain", 5)
    assert source.receipt.pool_distinct == 4


def test_ordering_and_min_gain_encoded_only_when_they_vary():
    from trimem.optimizer.space import SearchSpace

    v5 = make_space(MEDIUM_GRID, V5_PLAN_IDS)
    assert "ordering" not in v5.feature_names and "min_gain" not in v5.feature_names
    assert "frontier" in v5.feature_names
    legacy = make_space()
    assert len(legacy.feature_names) == 26  # pre-v5 vector layout
    base = CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3)
    variants = [
        base,
        CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3, ordering="gain"),
        CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3, min_gain=1e-9),
        CandidateSpec(
            "vgr_v1", seed_count=8, fanout=4, top_k=3, ordering="gain", min_gain=1e-9
        ),
        CandidateSpec("no_memory_v1"),
    ]
    mixed = SearchSpace(variants)
    assert {"ordering", "min_gain"} <= set(mixed.feature_names)
    assert len({mixed.encode(a) for a in variants}) == len(variants)
    assert mixed.feature_names.index("ordering") in mixed.categorical_dims
