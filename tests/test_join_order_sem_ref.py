"""Semantic-operator placement spec tests (Figure-1 P4, ADR-0026).

Pins src/planner/join_order_sem_ref.py, the executable spec the C port
(src/planner/join_order_sem.c) must match bit for bit (test/join_order_sem_test.sql,
scripts/join_order_sem_test.sh). No database needed.
"""

import itertools
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from planner.join_order_sem_ref import (  # noqa: E402
    c_eff,
    place_sem,
    rank,
    sem_batch,
    sem_plan_cost,
)

# --------------------------------------------------------------------------- #
# place_sem ordering properties                                                #
# --------------------------------------------------------------------------- #


def test_cheap_first_at_equal_selectivity():
    assert place_sem([0.5, 0.5, 0.5], [10.0, 1.0, 5.0], [0.0, 0.0, 0.0]) == [1, 2, 0]


def test_selective_first_at_equal_cost():
    assert place_sem([0.9, 0.1, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == [1, 2, 0]


def test_rank_trades_selectivity_against_cost():
    # rank0 = 0.9/100 = 0.009, rank1 = 0.1/1 = 0.1 -> the cheap weak filter goes first.
    assert place_sem([0.1, 0.9], [100.0, 1.0], [0.0, 0.0]) == [1, 0]


def test_free_predicates_first():
    # c_eff == 0 (zero cost, or fully cached with cache_cost 0) -> +inf rank, even at sel 1.
    assert place_sem([0.01, 1.0, 0.5], [1.0, 0.0, 7.0], [0.0, 0.0, 1.0]) == [1, 2, 0]
    assert rank(1.0, 0.0) == math.inf


def test_cache_hits_lower_effective_cost():
    assert c_eff(10.0, 0.5, 0.0) == 5.0
    assert c_eff(10.0, 0.5, 2.0) == 6.0
    # Same sel and cost; the well-cached predicate is cheaper, so it goes first.
    assert place_sem([0.5, 0.5], [10.0, 10.0], [0.0, 0.9]) == [1, 0]
    # A nonzero cache_cost keeps a fully cached predicate from being free.
    assert place_sem([0.5, 0.2], [10.0, 1.0], [1.0, 0.0], cache_cost=5.0) == [1, 0]


def test_stable_ties_keep_input_order():
    assert place_sem([0.5] * 4, [2.0] * 4, [0.0] * 4) == [0, 1, 2, 3]
    assert place_sem([0.5, 0.2, 0.5], [1.0, 3.0, 1.0], [0.0] * 3) == [0, 2, 1]
    assert place_sem([1.0, 0.3, 1.0], [0.0, 1.0, 0.0], [0.0] * 3) == [0, 2, 1]


def test_empty_input():
    assert place_sem([], [], []) == []
    assert sem_plan_cost([], [], [], [], 1000.0) == 0.0


def test_result_is_permutation():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(0, 64)
        sels = [rng.random() for _ in range(n)]
        costs = [rng.choice([0.0, rng.uniform(0, 50)]) for _ in range(n)]
        hits = [rng.random() for _ in range(n)]
        assert sorted(place_sem(sels, costs, hits, rng.uniform(0, 1))) == list(range(n))


# --------------------------------------------------------------------------- #
# sem_plan_cost                                                                #
# --------------------------------------------------------------------------- #


def test_cost_hand_computed():
    sels, costs, hits = [0.5, 0.25], [4.0, 8.0], [0.0, 0.0]
    # order [0,1]: 100*(4 + 8*0.5) = 800; order [1,0]: 100*(8 + 4*0.25) = 900.
    assert sem_plan_cost(sels, costs, hits, [0, 1], 100.0) == 800.0
    assert sem_plan_cost(sels, costs, hits, [1, 0], 100.0) == 900.0
    assert place_sem(sels, costs, hits) == [0, 1]


def test_cost_with_cache():
    # c_eff0 = 10*0.5 + 2*0.5 = 6; c_eff1 = 4*1 + 0 = 4.
    # order [0,1]: 10*(6 + 4*0.5) = 80.
    assert sem_plan_cost([0.5, 0.5], [10.0, 4.0], [0.5, 0.0], [0, 1], 10.0, 2.0) == 80.0


def test_cost_zero_rows():
    assert sem_plan_cost([0.5], [3.0], [0.0], [0], 0.0) == 0.0


def test_rank_order_minimises_cost_over_all_permutations():
    rng = random.Random(20261010)
    for _ in range(300):
        n = rng.randint(1, 6)
        sels = [rng.choice([0.0, 1.0, rng.random()]) for _ in range(n)]
        costs = [rng.choice([0.0, rng.uniform(0.01, 100.0)]) for _ in range(n)]
        hits = [rng.choice([0.0, 1.0, rng.random()]) for _ in range(n)]
        cc = rng.choice([0.0, rng.uniform(0.0, 1.0)])
        n_in = rng.uniform(1.0, 1e6)
        best = min(
            sem_plan_cost(sels, costs, hits, list(p), n_in, cc)
            for p in itertools.permutations(range(n))
        )
        got = sem_plan_cost(
            sels, costs, hits, place_sem(sels, costs, hits, cc), n_in, cc
        )
        assert got <= best * (1 + 1e-12) + 1e-9, (sels, costs, hits, cc, got, best)


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sels", "costs", "hits", "cc"),
    [
        ([0.5], [1.0, 2.0], [0.0], 0.0),  # length mismatch
        ([1.5], [1.0], [0.0], 0.0),  # sel > 1
        ([-0.1], [1.0], [0.0], 0.0),  # sel < 0
        ([math.nan], [1.0], [0.0], 0.0),  # NaN sel
        ([0.5], [-1.0], [0.0], 0.0),  # negative cost
        ([0.5], [math.inf], [0.0], 0.0),  # infinite cost
        ([0.5], [1.0], [1.1], 0.0),  # hit > 1
        ([0.5], [1.0], [0.0], -1.0),  # negative cache_cost
        ([0.5], [1.0], [0.0], math.inf),  # infinite cache_cost
        ([None], [1.0], [0.0], 0.0),  # None element
        ([0.5] * 65, [1.0] * 65, [0.0] * 65, 0.0),  # n > 64
    ],
)
def test_place_sem_rejects_invalid(sels, costs, hits, cc):
    with pytest.raises(ValueError):
        place_sem(sels, costs, hits, cc)


@pytest.mark.parametrize("ord_", [[0], [0, 0], [0, 2], [1, -1], [0, 1, 2]])
def test_plan_cost_rejects_non_permutation(ord_):
    with pytest.raises(ValueError, match="permutation"):
        sem_plan_cost([0.5, 0.5], [1.0, 1.0], [0.0, 0.0], ord_, 10.0)


@pytest.mark.parametrize("n_in", [-1.0, math.inf, math.nan])
def test_plan_cost_rejects_bad_n_in(n_in):
    with pytest.raises(ValueError, match="n_in"):
        sem_plan_cost([0.5], [1.0], [0.0], [0], n_in)


# --------------------------------------------------------------------------- #
# sem_batch                                                                    #
# --------------------------------------------------------------------------- #


def test_batch_basic_and_clamping():
    assert sem_batch(10, 0.1, 1, 1000) == 100
    assert sem_batch(10, 0.3, 1, 1000) == 34  # ceil(33.33...)
    assert sem_batch(1, 1.0, 16, 1000) == 16  # clamped up to min_batch
    assert sem_batch(10, 0.001, 1, 1000) == 1000  # clamped down to max_batch
    assert sem_batch(5, 0.0, 1, 10**9) == 5_000_000  # sel floored at 1e-6
    assert sem_batch(2**31 - 1, 0.0, 1, 2**31 - 1) == 2**31 - 1  # no overflow


def test_batch_done_returns_zero():
    assert sem_batch(0, 0.5, 8, 64) == 0
    assert sem_batch(-3, 0.5, 8, 64) == 0


@pytest.mark.parametrize(
    ("k", "sel", "lo", "hi"),
    [
        (1, 1.5, 1, 10),
        (1, -0.1, 1, 10),
        (1, math.nan, 1, 10),
        (1, 0.5, 0, 10),
        (1, 0.5, 5, 4),
    ],
)
def test_batch_rejects_invalid(k, sel, lo, hi):
    with pytest.raises(ValueError):
        sem_batch(k, sel, lo, hi)
