import pytest
import torch

from trimem.optimizer.latent_region import (
    H0, H_MAX, H_MIN, BatchBudget, TrustState, anchor_order, batch_gains, box, replay, step,
)
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, Observation


@pytest.mark.parametrize("gains,h,resets", [
    ([1, 1, 1], 0.8, 0), ([1] * 6, 1.0, 0), ([0, 0, 0], 0.2, 0),
    ([0] * 9, 0.05, 0), ([0] * 12, 0.4, 1), ([1, 0, 1, 0, 1, 0], 0.4, 0)])
def test_replay(gains, h, resets):
    s = replay(gains)
    assert s.h == pytest.approx(h)
    assert s.resets == resets


def test_step_is_pure_and_eps_threshold():
    s = TrustState()
    assert step(s, 1e-4).down == 1 and s == TrustState()
    assert (H0, H_MIN, H_MAX) == (0.4, 0.05, 1.0)


def test_box_clamped_and_positive_width():
    b = box(torch.tensor([0.0, 0.5, 1.0], dtype=torch.double), 0.4)
    assert b.shape == (2, 3)
    assert (b >= 0).all() and (b <= 1).all() and (b[1] > b[0]).all()
    assert b[0, 0] == 0.0 and b[1, 2] == 1.0
    assert torch.allclose(b[:, 1], torch.tensor([0.3, 0.7], dtype=torch.double))


def _obs(k, y):
    return Observation({"k": k}, (0,) * 3, tuple(y), (1e-4,) * 3, "p", True)


def test_batch_gains_skip_incomplete_rounds():
    obs = [_obs("a", (1, 0, 0)), _obs("b", (0, 1, 0)), _obs("c", (0, 0, 1))]
    obj = ObjectiveProtocol((0.0,) * 3, (1.0,) * 3)
    rounds = [({"a"}, ["b"]), ({"a", "b"}, ["c", "missing"]), (set(), ["a"])]
    g = batch_gains(rounds, obs, obj, key=lambda o: o.config["k"])
    assert len(g) == 2
    assert g[0] > 0 and g[1] > 0


def test_budget_shares_carry_unused_forward():
    b = BatchBudget(seconds=300.0, calls=1000)
    s, c = b.slot_share(4)
    assert (s, c) == (75.0, 250)
    a, ac = b.anchor_share((s, c), 5)
    assert (a, ac) == (15.0, 50)
    assert b.anchor_share((s, 10_000), 1)[1] == 100  # capped
    b.charge(5.0, 20)  # anchor used less than share
    s2, c2 = b.slot_share(4)
    assert s2 == pytest.approx(295 / 4) and c2 == 980 // 4
    b.charge(1000.0, 0)
    assert b.exhausted()
    assert BatchBudget(seconds=None).slot_share(2)[0] is None
    assert not BatchBudget(seconds=None).exhausted()


def test_anchor_order_frozen_and_slot_dependent():
    a = anchor_order(25, seed=1, round_index=2, slot=0)
    assert sorted(a) == list(range(25))
    assert a == anchor_order(25, seed=1, round_index=2, slot=0)
    assert a != anchor_order(25, seed=1, round_index=2, slot=1)
