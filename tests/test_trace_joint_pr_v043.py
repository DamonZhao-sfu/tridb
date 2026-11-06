"""v0.4.3: Generalized PR discrete formula, full support, bo_pr budget, re-anchor."""

import math

import pytest
import torch

from trimem.optimizer import probabilistic_plan_search as pps
from trimem.optimizer.plan_schema import JointSpace
from trimem.optimizer.probabilistic_plan_search import (
    PRDistribution,
    PRStart,
    ordinal_levels,
    ordinal_log_probs,
    refine,
)


def test_levels_follow_gp_distance_with_mean_gap_one():
    s = JointSpace.live()
    # returned_k (3,5,8,10) is non-equidistant: gaps 2,3,2 -> 6/7, 9/7, 6/7.
    k = ordinal_levels(s, "returned_k")
    assert torch.allclose(k, torch.tensor([0, 6 / 7, 15 / 7, 3], dtype=torch.double))
    # seed_count is log2-equidistant, so it equals bo_pr's integer scale.
    assert torch.allclose(ordinal_levels(s, "seed_count"), torch.arange(6).double())


@pytest.mark.parametrize("u", [0.0, 0.3, 0.9, 1.5, 2.2, 3.0])
def test_full_support_normalized_and_matches_generalized_pr(u):
    d = torch.tensor([0, 6 / 7, 15 / 7, 3], dtype=torch.double)
    p = ordinal_log_probs(torch.tensor(u, dtype=torch.double), d).exp()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-12)
    assert (p > 0).all()
    # Between d_1 and d_2 only threshold m_1 is unsaturated: paper's D(theta).
    if 6 / 7 <= u <= 15 / 7:
        m = (6 / 7 + 15 / 7) / 2
        upper = 1 / (1 + math.exp(-(u - m) / 0.1))
        assert (p[2] + p[3]).item() == pytest.approx(upper, abs=2e-3)


def test_two_level_field_equals_bo_pr_bernoulli():
    d = torch.tensor([0.0, 1.0], dtype=torch.double)
    for u in (0.2, 0.5, 0.71):
        p = ordinal_log_probs(torch.tensor(u, dtype=torch.double), d).exp()
        assert p[1].item() == pytest.approx(1 / (1 + math.exp(-(u - 0.5) / 0.1)))


def test_initial_neighbor_probability_is_rho():
    s = JointSpace.live()
    d = PRDistribution(s, s.default(seed_count=16), seed=0)
    p = d.field_log_probs("seed_count", "ord").exp()
    j = s.domains["seed_count"].index(16)
    assert p[j].item() == pytest.approx(1 - d.rho, abs=1e-3)
    assert (p[j - 1] + p[j + 1]).item() == pytest.approx(d.rho, abs=1e-3)


def test_bo_pr_decay_schedule(monkeypatch):
    rates = []
    real = torch.optim.Adam

    def spy(params, lr):
        rates.append(lr)
        return real(params, lr=lr)

    monkeypatch.setattr(pps.torch.optim, "Adam", spy)
    s = JointSpace.live()

    class Flat:
        def score(self, configs):
            return [0.0] * len(configs)

    refine(s, [PRStart("t", s.default())], Flat(), steps=4, samples=8, decay=True)
    assert rates == pytest.approx([0.025, 0.025, 0.025 / 2**0.7, 0.025 / 3**0.7])


def historical_space():
    """The 5,040-config space (engine-extension operators fixed to historical values): small
    enough that forbidding almost every configuration forces PR stagnation."""
    live = JointSpace.live()
    return JointSpace({**live.domains, "graph_impl": ("BFS",), "rank_impl": ("ColBERT",),
                       "retrieval_unit": ("whole",), "injection_timing": ("start",),
                       "graph_direction": ("Both",)})


def test_stagnation_reanchors_instead_of_stopping():
    s = historical_space()

    class Flat:
        def score(self, configs):
            return [0.0] * len(configs)

    start = PRStart("t", s.default())
    out = refine(s, [start], Flat(), steps=30, samples=8, forbidden=set())
    run = out["runs"][0]
    assert run["steps"] == 30 and run["stop"] == "step limit"
    # A forbidden-only neighborhood forces stagnation, which must re-anchor.
    every = {s.key(c) for c in s.sample(5000, seed=0, max_draws=20000)}
    keep = sorted(every)[:50]
    out = refine(
        s, [start], Flat(), steps=20, samples=8, forbidden=every - set(keep), patience=2
    )
    run = out["runs"][0]
    assert run["steps"] == 20
    assert run["reanchors"] >= 1
    assert all(e["key"] not in every - set(keep) for e in run["reanchor_log"])


def test_limits_follow_bo_pr():
    s = JointSpace.live()
    with pytest.raises(ValueError, match="v0.4.3"):
        refine(s, [PRStart("t", s.default())], None, steps=201)
    with pytest.raises(ValueError, match="v0.4.3"):
        refine(s, [PRStart("t", s.default())], None, samples=129)


def test_v044_defaults_constant_lr_no_reanchor_no_stop(monkeypatch):
    rates = []
    real = torch.optim.Adam

    def spy(params, lr):
        rates.append(lr)
        return real(params, lr=lr)

    monkeypatch.setattr(pps.torch.optim, "Adam", spy)
    s = historical_space()

    class Flat:
        def score(self, configs):
            return [0.0] * len(configs)

    every = {s.key(c) for c in s.sample(5000, seed=0, max_draws=20000)}
    keep = set(sorted(every)[:50])
    out = refine(
        s, [PRStart("t", s.default())], Flat(), steps=20, samples=8,
        forbidden=every - keep,
    )
    run = out["runs"][0]
    # One persistent Adam at lr 0.025; stagnation is counted, never acted on.
    assert rates == [0.025]
    assert run["steps"] == 20 and run["stop"] == "step limit"
    assert run["reanchors"] == 0 and run["stagnant_steps"] >= 1


def test_no_default_time_limits():
    import inspect

    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.optimizer import joint_search_controller as ctl

    assert inspect.signature(refine).parameters["seconds"].default is None
    assert inspect.signature(ctl.propose_batch).parameters["pr_seconds"].default is None
    budget = pps.ProposalBudget()
    budget.started -= 10**6  # far past any old 60 s limit
    budget.admit(["a"])  # must not raise: only the unique budget applies
    assert '"search": None' in inspect.getsource(driver.guarded_search)
