import pytest
import torch

from trimem.optimizer.plan_schema import JointSpace
from trimem.optimizer.probabilistic_plan_search import (
    PRDistribution,
    ProposalBudget,
    PRStart,
    refine,
)


def test_samples_legal_and_expected_initial_changes():
    torch.set_num_threads(4)
    # E counts active stochastic coordinates, before conditional deactivation.
    s = JointSpace({**JointSpace.live().domains, "filter_order": ("before",)})
    start = s.default()
    d = PRDistribution(s, start, seed=42)
    samples, logp = d.sample(8000)
    active = [n for n, v in start.items() if v is not None and len(s.domains[n]) > 1]
    changes = sum(sum(c[n] != start[n] for n in active) for c in samples) / len(samples)
    assert changes == pytest.approx(min(1.5, 0.3 * len(active)), abs=0.09)
    assert torch.isfinite(logp).all()
    assert all(s.canonicalize(c) == c for c in samples)
    (-logp.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in d.parameters())


def test_conditional_gradients_and_top_endpoint():
    s = JointSpace()
    d = PRDistribution(s, s.default(seed_count=128), seed=3)
    samples, logp = d.sample(128)
    assert all(c["seed_count"] in (64, 128) for c in samples)
    # Force all graph draws to BFS; beam-width score-function gradient is zero.
    fixed = JointSpace({"graph_impl": ("BFS",)})
    d = PRDistribution(fixed, fixed.default(), seed=1)
    _, logp = d.sample(64)
    (-logp.mean()).backward()
    assert d.theta["beam_width"].grad.abs().sum() == 0


class Reward:
    def score(self, configs):
        return [float(c["seed_count"]) / 128 for c in configs]


def test_refinement_resume_weights_and_exclusions():
    s = JointSpace.live()
    start = PRStart("pr-agent", s.default())
    a = refine(
        s,
        [start],
        Reward(),
        seed=2,
        steps=4,
        samples=32,
        forbidden={s.key(start.config)},
    )
    b = refine(
        s,
        [start],
        Reward(),
        seed=2,
        steps=2,
        samples=32,
        forbidden={s.key(start.config)},
    )
    resumed = refine(
        s,
        [start],
        Reward(),
        seed=2,
        steps=4,
        samples=32,
        forbidden={s.key(start.config)},
        checkpoint=b["checkpoint"],
    )
    assert a["candidates"] == resumed["candidates"]
    assert all(s.key(c) != s.key(start.config) for c in a["candidates"])
    assert a["runs"][0]["draws"] == 128
    assert a["runs"][0]["unique_draws"] <= 128


def test_global_unique_budget():
    b = ProposalBudget(max_unique=2, seconds=60)
    b.admit(["a", "a", "b"])
    b.admit(["a"])
    with pytest.raises(RuntimeError, match="budget"):
        b.admit(["c"])


def test_score_function_gradient_matches_exact_four_point_expectation():
    base = JointSpace.live()
    domains = {n: (v[0],) for n, v in base.domains.items()}
    domains.update(filter_order=("before", "after"), seed_count=(4, 8))
    s = JointSpace(domains)
    d = PRDistribution(s, s.default(), seed=17)
    cat = torch.softmax((d.theta["filter_order"] - 0.5) / 0.1, 0)[1]
    high = torch.sigmoid((d.theta["seed_count"] - 0.5) / 0.1)
    expected = cat * high + 0.2 * high
    params = list(d.parameters())
    exact = torch.autograd.grad(expected, params)
    configs, logp = d.sample(16000)
    reward = torch.tensor(
        [
            float(c["filter_order"] == "after" and c["seed_count"] == 8)
            + 0.2 * (c["seed_count"] == 8)
            for c in configs
        ],
        dtype=torch.double,
    )
    estimate = torch.autograd.grad((reward * logp).mean(), params)
    for a, b in zip(exact, estimate):
        assert torch.allclose(a, b, atol=0.08, rtol=0.1)


def test_pr_reaches_small_enumerated_optimum_and_refuses_stale_resume():
    base = JointSpace.live()
    domains = {n: (v[0],) for n, v in base.domains.items()}
    domains.update(seed_count=(4, 8), returned_k=(3, 5))
    s = JointSpace(domains)

    class JointReward:
        identity = "first"

        def score(self, configs):
            return [
                float(c["seed_count"] == 8 and c["returned_k"] == 5) for c in configs
            ]

    starts = [PRStart("pr-agent", s.default(seed_count=4, returned_k=3))]
    reward = JointReward()
    result = refine(s, starts, reward, steps=4, seed=1)
    assert max(reward.score(result["candidates"])) == 1
    reward.identity = "new-pending-or-model"
    with pytest.raises(ValueError, match="checkpoint"):
        refine(s, starts, reward, steps=4, checkpoint=result["checkpoint"])
