"""Behavioral checks for conditional Q/T search; optional actual CPU BO smoke."""

from dataclasses import replace

import pytest

from trimem.optimizer.schema import CandidateSpec
from trimem.optimizer.search import (
    SearchData,
    SearchSettings,
    SpaceExhausted,
    objective_arrays,
    suggest,
)
from trimem.optimizer.space import PLAN_IDS, SearchSpace, SpaceGrid, make_space


def small_space(allowed=None):
    return make_space(
        SpaceGrid(
            seed_counts=(8, 24, 48),
            hop_paths=(("lineage",),),
            fanouts=(4,),
            top_ks=(1,),
            unknown_policies=("drop",),
            score_admit_mins=(0.5,),
        ),
        allowed,
    )


def observation(action, quality=0.5, tokens=1000):
    return SearchData(action, quality, tokens, 0.01, 100.0)


def settings(**kwargs):
    return SearchSettings(reference_quality=0, reference_tokens=10000, **kwargs)


def test_default_space_has_only_budget_legal_unique_actions():
    space = make_space()
    assert set(space.plan_ids) == set(PLAN_IDS)
    # Seven plan families incl. the two no-score SemFilter variants.
    assert len(space) < 3000
    assert len({a.action_hash for a in space.candidates}) == len(space)
    for action in space.candidates:
        assert action.expansion_bound <= action.edge_emit_cap
    assert sum(a.plan_id == "no_memory_v1" for a in space.candidates) == 1


def test_budget_filter_cannot_silently_remove_requested_plan_families():
    with pytest.raises(ValueError, match="no legal configurations"):
        make_space(SpaceGrid(edge_emit_cap=1))


def test_fixed_plan_score_threshold_search_is_active_and_nonscore_is_deduplicated():
    grid = SpaceGrid(
        seed_counts=(8,),
        hop_paths=(("lineage",),),
        fanouts=(4,),
        top_ks=(1,),
        unknown_policies=("drop",),
        score_admit_mins=(0.3, 0.5, 0.7),
    )
    joint = make_space(grid)
    fixed = joint.subset(("seed_sem_expand_v1",))
    assert len(fixed) == 3
    assert {a.score_admit_min for a in fixed.candidates} == {0.3, 0.5, 0.7}
    assert len({fixed.encode(a) for a in fixed.candidates}) == 3
    assert len(joint.subset(("vgr_v1",))) == 1
    assert len(joint.subset(("no_memory_v1",))) == 1
    override = make_space(replace(grid, score_admit_min=0.4))
    assert len(override) == len(PLAN_IDS)
    assert {a.score_admit_min for a in override.candidates if a.uses_score} == {0.4}


def test_inactive_knobs_do_not_duplicate_or_change_encoding():
    first = CandidateSpec("no_memory_v1", seed_count=8, top_k=1)
    second = CandidateSpec("no_memory_v1", seed_count=48, top_k=8)
    plain = CandidateSpec("vgr_v1", unknown_policy="drop", score_admit_min=0.2)
    inactive = replace(plain, unknown_policy="keep", score_admit_min=0.8)
    space = SearchSpace([first, second, plain, inactive])
    assert len(space) == 2
    assert space.encode(first) == space.encode(second)
    assert space.encode(plain) == space.encode(inactive)
    assert space.encode(first) != space.encode(plain)


def test_joint_initialization_covers_every_plan_before_second_configuration():
    space = small_space()
    records = []
    picked = []
    for _ in PLAN_IDS:
        proposal = suggest(space, records, (), settings())
        assert proposal.strategy == "stratified_initialization"
        picked.append(proposal.action.plan_id)
        records.append(observation(proposal.action))
    assert picked == list(PLAN_IDS)
    assert len({row.action.action_hash for row in records}) == len(PLAN_IDS)
    next_proposal = suggest(space, records, (), settings())
    assert next_proposal.action.plan_id != "no_memory_v1"
    assert next_proposal.action.action_hash not in {
        r.action.action_hash for r in records
    }


def test_fixed_plan_mask_and_reservations_reuse_same_search_api():
    joint = small_space()
    fixed = joint.subset(("seed_sem_expand_v1",))
    assert len(fixed) == 3
    chosen = suggest(fixed, [], (), settings()).action
    second = suggest(fixed, [], (chosen.action_hash,), settings()).action
    assert second.plan_id == chosen.plan_id == "seed_sem_expand_v1"
    assert second.action_hash != chosen.action_hash
    with pytest.raises(SpaceExhausted):
        suggest(fixed, [], (a.action_hash for a in fixed.candidates), settings())
    with pytest.raises(ValueError):
        joint.subset(("P1",))


def test_observations_outside_fixed_plan_are_not_mixed():
    joint = small_space()
    fixed = joint.subset(("vgr_v1",))
    other = next(a for a in joint.candidates if a.plan_id == "vgr_score_v1")
    with_other = suggest(fixed, [observation(other)], (), settings())
    without = suggest(fixed, [], (), settings())
    assert with_other == without


def test_objective_sign_scales_and_sem_squared_are_explicit():
    row = SearchData(CandidateSpec("no_memory_v1"), -2, 400, 0.16, 25, 4)
    spec = SearchSettings(
        reference_quality=-10,
        reference_tokens=2000,
        quality_scale=2,
        token_scale=100,
    )
    means, variances = objective_arrays([row], spec)
    assert means[0] == [-1, -4]
    assert variances[0] == pytest.approx([0.04, 0.0025])
    assert spec.reference_point == (-5, -20)
    _, noise = objective_arrays([replace(row, quality_variance_of_mean=0)], spec)
    assert noise[0][0] > 0  # a singleton is never claimed to have zero uncertainty


def test_application_failure_quality_can_be_negative_but_not_fake_tokens():
    observation(CandidateSpec("no_memory_v1"), quality=-100, tokens=500)
    with pytest.raises(ValueError, match="negative"):
        observation(CandidateSpec("no_memory_v1"), tokens=-1)
    with pytest.raises(ValueError, match="finite"):
        observation(CandidateSpec("no_memory_v1"), quality=float("nan"))
    with pytest.raises(ValueError, match="aggregate"):
        row = observation(CandidateSpec("no_memory_v1"))
        suggest(small_space(), [row, row], (), settings())


def test_model_failure_is_a_visible_legal_fallback(monkeypatch):
    space = small_space(("vgr_v1",))
    rows = [observation(a) for a in space.candidates[:2]]

    def broken(*args):
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", broken)
    proposal = suggest(space, rows, (), settings())
    assert proposal.strategy == "stratified_fallback"
    assert "RuntimeError: deliberate fit failure" in proposal.diagnostic
    assert proposal.action == space.candidates[2]
    assert proposal.acquisition_value is None
    with pytest.raises(RuntimeError, match="deliberate"):
        suggest(space, rows, (), settings(fallback_on_error=False))


def test_acquisition_selects_whole_action_without_executing_it(monkeypatch):
    space = small_space(("vgr_v1", "expand_row_sem_v1"))
    rows = [
        observation(next(a for a in space.candidates if a.plan_id == plan))
        for plan in space.plan_ids
    ]

    def acquisition(local_space, observations, pool, spec):
        assert local_space is space
        assert len(observations) == 2
        assert {a.plan_id for a in pool} == set(space.plan_ids)
        return len(pool) - 1, -0.15

    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", acquisition)
    proposal = suggest(space, rows, (), settings(initial_per_plan=1))
    assert proposal.strategy == "qlognehvi"
    assert proposal.acquisition_value == -0.15
    assert proposal.action.action_hash not in {row.action.action_hash for row in rows}


def test_repeats_require_explicit_switch_and_pending_exclusion_wins(monkeypatch):
    space = small_space(("vgr_v1",))
    rows = [observation(a) for a in space.candidates]
    with pytest.raises(SpaceExhausted):
        suggest(space, rows, (), settings())
    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", lambda *args: (0, -1))
    excluded = space.candidates[0].action_hash
    proposal = suggest(space, rows, (excluded,), settings(allow_repeats=True))
    assert proposal.action.action_hash != excluded
    assert proposal.strategy == "qlognehvi"


def test_botorch_real_cpu_mixed_space_smoke():
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    space = small_space(("vgr_v1", "expand_row_sem_v1"))
    rows = []
    for plan in space.plan_ids:
        actions = [a for a in space.candidates if a.plan_id == plan]
        rows.extend(
            [observation(actions[0], 0.5, 200), observation(actions[1], 0.8, 800)]
        )
    original_threads = torch.get_num_threads()
    original_rng = torch.random.get_rng_state().clone()
    proposal = suggest(
        space,
        rows,
        (),
        settings(
            initial_per_plan=1,
            max_fit_iterations=5,
            mc_samples=8,
            fallback_on_error=False,
        ),
    )
    assert proposal.strategy == "qlognehvi"
    assert proposal.acquisition_value is not None
    assert proposal.action.action_hash not in {row.action.action_hash for row in rows}
    assert torch.get_num_threads() == original_threads
    assert torch.equal(torch.random.get_rng_state(), original_rng)
