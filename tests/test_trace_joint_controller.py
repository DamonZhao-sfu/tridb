import torch

from trimem.optimizer.joint_search_controller import (
    greedy_batch,
    initial_configs,
    recombine,
    select_starts,
)
from trimem.optimizer.plan_schema import JointSpace


class Snapshot:
    identity = "test"

    def __init__(self, space, pending=()):
        self.space = space
        self.pending = pending

    def score(self, configs):
        return [
            float(c["seed_count"])
            + (1000 if self.pending and c["filter_order"] == "after" else 0)
            for c in configs
        ]

    def posterior(self, configs):
        return torch.tensor(
            [[c["seed_count"], -c["seed_count"], c["returned_k"]] for c in configs],
            dtype=torch.double,
        )

    def with_pending(self, pending):
        return Snapshot(self.space, pending)


def test_initial_and_starts_groups():
    s = JointSpace.live()
    initial = initial_configs(s, seed=1)
    assert len(initial) == len({s.key(c) for c in initial}) == 10
    assert {c["filter_order"] for c in initial} == {"none", "before", "after"}
    snap = Snapshot(s)
    pool = s.sample(80, seed=4)
    starts = select_starts(
        s, snap, initial[:1], pool, agent=pool[:2], cross=pool[2:4], seed=5
    )
    labels = [p.source for p in starts]
    assert labels[:2] == ["pr-agent"] * 2
    assert "pr-crossover" not in labels
    assert "pr-newplan" in labels and "pr-random" in labels and "pr-pool" in labels
    assert len(starts) <= 25 and len({s.key(p.config) for p in starts}) == len(starts)
    fallback = select_starts(s, snap, initial, pool, cross=pool[:3], seed=5)
    assert [x.source for x in fallback[:3]] == ["pr-crossover"] * 3


def test_pending_greedy_rescores_and_no_duplicates():
    s = JointSpace.live()
    a = s.default(seed_count=128, filter_order="before")
    b = s.default(seed_count=64, filter_order="before")
    c = s.default(seed_count=4, filter_order="after")
    result = greedy_batch(s, Snapshot(s), [a, b, c, a], q=3)
    assert [r["config"] for r in result] == [a, c, b]
    assert len({r["snapshot"] for r in result}) >= 1
    assert all(s.canonicalize(c) == c for c in recombine(s, [a, c], seed=8))


def test_neighborhood_diagnostic_finds_two_field_interaction():
    from trimem.optimizer.joint_search_controller import neighborhood_diagnostic

    s = JointSpace.live()

    class Bonus:
        def score(self, configs):
            # Only the joint move (returned_k=8 AND Success) pays; each alone does not.
            return [
                float(c["returned_k"] == 8 and c["trajectory_category"] == "Success")
                for c in configs
            ]

    mode = s.default(returned_k=3, trajectory_category="All")
    row = neighborhood_diagnostic(s, Bonus(), [mode], set())[0]
    assert row["best_1field_gain"] == 0.0
    assert row["best_2field_gain"] == 1.0
    assert row["max_interaction"] == 1.0
    assert sorted(row["max_interaction_fields"]) == ["returned_k", "trajectory_category"]

    class Additive:  # two independent gains add up: no interaction
        def score(self, configs):
            return [float(c["returned_k"] == 8) + float(c["trajectory_category"] == "Success")
                    for c in configs]

    row = neighborhood_diagnostic(s, Additive(), [mode], set())[0]
    assert row["best_2field_gain"] == 2.0 and abs(row["max_interaction"]) < 1e-12


def test_channel_quota_takes_each_channels_best_once():
    from trimem.optimizer.joint_search_controller import channel_quota_batch

    s = JointSpace.live()

    class Snap:
        identity = "x"
        pending = ()

        def __init__(self, pending=()):
            self.pending = tuple(pending)

        def with_pending(self, p):
            return Snap(p)

        def score(self, configs):
            # seed_count is the score; pending configs make identical seeds worth less
            taken = {c["seed_count"] for c in self.pending}
            return [float(c["seed_count"]) - (100 if c["seed_count"] in taken else 0) for c in configs]

    mk = lambda **k: s.default(**k)
    channels = {
        "llm": [mk(seed_count=64), mk(seed_count=8)],
        "recombination": [mk(seed_count=128, depth=3)],
        "pr": [mk(seed_count=128, depth=2), mk(seed_count=32)],
        "sobol": [mk(seed_count=4)],
    }
    out = channel_quota_batch(s, Snap(), channels, q=4)
    assert [r["channel"] for r in out] == ["recombination", "llm", "pr", "sobol"]
    # pr's best (128) is penalized once recombination's 128 is pending -> pr takes 32
    assert {r["channel"]: r["config"]["seed_count"] for r in out} == {"recombination": 128, "llm": 64, "pr": 32, "sobol": 4}
    assert len({r["key"] for r in out}) == 4 and not any(r["channel_fill"] for r in out)
    # an empty channel is filled from the whole pool and recorded
    out = channel_quota_batch(s, Snap(), {**channels, "llm": []}, q=4,
                              fill=[mk(seed_count=16), *channels["pr"]])
    assert sum(r["channel_fill"] for r in out) == 1 and len({r["key"] for r in out}) == 4
