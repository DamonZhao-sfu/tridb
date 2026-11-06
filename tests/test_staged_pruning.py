"""Staged profiling with 3-D Pareto pruning (``trimem.optimizer.staged_pruning``): pure, no I/O."""

from __future__ import annotations

import math

import pytest

from trimem.optimizer import staged_pruning as sp
from trimem.optimizer import unpaired_stats as us

DOM = {
    **{f"airline:{i}": "airline" for i in range(30)},
    **{f"retail:{i}": "retail" for i in range(74)},
    **{f"telecom:{i}": "telecom" for i in range(74)},
}


@pytest.mark.unit
def test_stage_order_is_nested_proportional_and_deterministic():
    ids = sorted(DOM)
    o1 = sp.stage_order(ids, DOM, 7)
    assert sorted(o1) == ids and o1 == sp.stage_order(list(reversed(ids)), DOM, 7)
    assert o1 != sp.stage_order(ids, DOM, 8)
    subs = sp.stage_subsets(o1, (50, 100, 178))
    assert [len(s) for s in subs] == [50, 100, 178]
    assert all(set(a) < set(b) for a, b in zip(subs, subs[1:], strict=False))  # nested
    for (
        s
    ) in subs:  # every prefix within one task per stratum of proportional allocation
        for h, n_h in (("airline", 30), ("retail", 74), ("telecom", 74)):
            got = sum(DOM[i] == h for i in s)
            assert abs(got - len(s) * n_h / 178) <= 1, (len(s), h, got)
    with pytest.raises(ValueError):
        sp.stage_subsets(o1, (50, 100))  # last stage must be the full set
    with pytest.raises(ValueError):
        sp.stage_subsets(o1, (100, 50, 178))
    with pytest.raises(ValueError):
        sp.stage_order(["a", "a"], {"a": "x"}, 1)


@pytest.mark.unit
def test_corner_and_weak_dominance():
    o = sp.Obj(0.10, 1000, 50, quality_se=0.05, tokens_se=100, latency_se=10)
    assert sp.optimistic_corner(o) == pytest.approx((0.198, 804, 30.4))
    assert sp.weakly_dominates((0.2, 800, 30), (0.198, 804, 30.4))
    assert sp.weakly_dominates((0.2, 800, 30), (0.2, 800, 30))  # weak: equal counts
    assert not sp.weakly_dominates((0.19, 700, 20), (0.198, 804, 30.4))


@pytest.mark.unit
def test_decide_prunes_only_dominated_unprotected_nonfinal():
    full = {"b": sp.Obj(0.30, 700, 20), "a": sp.Obj(0.05, 600, 10)}
    bad = sp.Obj(
        0.0, 1000, 50, 0.05, 50, 5, n=50
    )  # corner (0.098, 902, 40.2): dominated by b
    d = sp.decide(bad, full, stage=50, last=False)
    assert d.prune and d.dominated_by == "b" and d.stage == 50
    assert not sp.decide(bad, full, stage=178, last=True).prune
    assert not sp.decide(bad, full, stage=50, last=False, protected=True).prune
    noisy = sp.Obj(
        0.0, 1000, 50, 0.20, 50, 5
    )  # optimistic quality 0.392 > every full point
    assert not sp.decide(noisy, full, stage=50, last=False).prune
    cheap = sp.Obj(0.0, 500, 50, 0.0, 0, 0)  # cheaper in tokens than everything
    assert not sp.decide(cheap, full, stage=50, last=False).prune
    assert set(sp.pareto_front({**full, "c": sp.Obj(0.0, 900, 40)})) == {"a", "b"}


@pytest.mark.unit
def test_budget_never_exceeded_and_savings_buy_configs():
    b = sp.Budget.unpruned(30, 712, 60)
    assert b.total_episodes == 30 * 712
    used = 10 * 712  # round 0 in full
    n, rounds = 10, 0
    while (k := sp.next_batch_size(b, used, n, 5)) > 0:
        # every batch: 2 full, 3 pruned at the first checkpoint (50 tasks x 4 trials)
        used += sum(
            sp.episodes_used({str(i): s}, 4)
            for i, s in enumerate([178, 178, 50, 50, 50][:k])
        )
        n, rounds = n + k, rounds + 1
        assert used <= b.total_episodes
    assert n > 30  # pruning bought extra configurations
    assert sp.next_batch_size(sp.Budget(10, 5, 3), 0, 3, 5) == 0  # config cap
    assert (
        sp.next_batch_size(sp.Budget(10, 5, 9), 6, 0, 5) == 0
    )  # cannot afford one full evaluation
    with pytest.raises(ValueError):
        sp.next_batch_size(b, b.total_episodes + 1, 0, 5)
    assert sp.stage_cost((50, 100, 178), 4) == [200, 200, 312]


@pytest.mark.unit
def test_gp_noise_larger_for_partial_configs():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 22 + [1.0, 0.0]
    strata = ["s"] * len(vals)
    full = us.stratified(vals, strata, {"s": 1.0})
    part = us.stratified(vals[:50], strata[:50], {"s": 1.0})
    of = lambda e: sp.Obj(e.mean, 1000, 100, e.se, 20, 5, e.n)  # noqa: E731
    nf, np_ = sp.gp_noise(of(full)), sp.gp_noise(of(part))
    assert (
        np_[0] > nf[0]
        and math.isclose(nf[1], (20 / 1000) ** 2)
        and math.isclose(nf[2], (5 / 100) ** 2)
    )
    assert sp.gp_noise(sp.Obj(0.1, 1, 1))[0] == 1e-8  # floor
    with pytest.raises(ValueError):
        sp.gp_noise(sp.Obj(0.1, 0, 1))


# ------------------------------------------------------------------ v2 (user decision 2026-10-20)


@pytest.mark.unit
def test_v2_corner_vs_pessimistic_corner_without_any_full_configuration():
    """Round 0: nothing is fully evaluated yet; a stage comparator's PESSIMISTIC corner on the same subset prunes."""
    assert sp.PROTOCOL == "staged-pareto-prune-v2"
    a = sp.Obj(-0.20, 3000, 60, quality_se=0.04, tokens_se=100, latency_se=5, n=50)
    nm = sp.Obj(0.0, 1500, 30, quality_se=0.0, tokens_se=50, latency_se=2, n=50)
    assert sp.optimistic_corner(a) == pytest.approx((-0.1216, 2804, 50.2))
    assert sp.pessimistic_corner(nm) == pytest.approx((0.0, 1598, 33.92))
    d = sp.decide(a, {}, stage=50, last=False, partial={"nomem": nm})
    assert d.prune and d.dominated_by == "nomem" and d.comparators == 1
    assert d.dominators[0][:2] == ("nomem", sp.STAGE_PESSIMISTIC)
    assert d.dominators[0][2] == pytest.approx(sp.pessimistic_corner(nm))
    # the same comparator as a POINT would also dominate, but a noisy one's pessimistic corner does not
    noisy = sp.Obj(0.0, 1500, 30, quality_se=0.08, tokens_se=50, latency_se=2)
    assert not sp.decide(a, {}, stage=50, last=False, partial={"b": noisy}).prune
    assert sp.decide(
        a, {"b": noisy}, stage=50, last=False
    ).prune  # full: point estimate
    # never at the last stage, never protected (NoMemory)
    assert not sp.decide(a, {}, stage=178, last=True, partial={"nomem": nm}).prune
    assert not sp.decide(
        a, {}, stage=50, last=False, protected=True, partial={"nomem": nm}
    ).prune
    # every dominator is recorded, full ones first
    d = sp.decide(
        a,
        {"z": sp.Obj(0.1, 1000, 10)},
        stage=50,
        last=False,
        partial={"nomem": nm, "c": noisy},
    )
    assert [x[0] for x in d.dominators] == ["z", "nomem"] and d.dominated_by == "z"
    assert d.as_dict()["dominators"][0][1] == sp.FULL_POINT


@pytest.mark.unit
def test_plan_guard_keeps_the_best_last_configuration_of_an_uncovered_plan():
    bad = sp.Obj(-0.3, 3000, 60, 0.02, 10, 1)
    worse = sp.Obj(-0.4, 3000, 60, 0.02, 10, 1)

    def pr() -> sp.Decision:
        return sp.Decision(
            True,
            50,
            (0, 0, 0),
            "nomem",
            "dominated",
            (("nomem", sp.STAGE_PESSIMISTIC, (0, 0, 0)),),
        )

    decs = {"a": pr(), "b": pr(), "c": pr(), "d": pr()}
    objs = {"a": worse, "b": bad, "c": bad, "d": bad}
    plan = {"a": "VGR", "b": "VGR", "c": "P2", "d": "P2+Rank"}
    out = sp.guard_plans(decs, objs, plan, covered={"P2+Rank"})
    assert not out["b"].prune and out["b"].guarded and "plan guard" in out["b"].reason
    assert out["a"].prune  # VGR keeps only its best (optimistic quality)
    assert not out["c"].prune and out["c"].guarded  # P2's only configuration
    assert out["d"].prune  # P2+Rank covered by a live / full configuration elsewhere
    assert out["b"].dominators == decs["b"].dominators  # the evidence stays recorded
    # a plan with a surviving configuration at this checkpoint is untouched
    keep = sp.Decision(False, 50, (0, 0, 0), None, "not dominated")
    out = sp.guard_plans({"a": pr(), "b": keep}, objs, plan, covered=set())
    assert out["a"].prune and not out["a"].guarded
