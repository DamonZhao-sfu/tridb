from bench.agent_memory.gem_oe.run_matrix import pin_cells_to_worker_slots


def _cells(n: int):
    return [(f"math:t{i}", "polyglot", 1.0, 0.5, 42) for i in range(n)]


def test_each_worker_slot_stays_on_one_endpoint() -> None:
    plan, shards = pin_cells_to_worker_slots(
        _cells(7), ["http://gpu0/v1", "http://gpu1/v1"], workers=2
    )

    assert len(plan) == 7
    assert len(shards) == 2
    assert {row[5] for row in shards[0]} == {"http://gpu0/v1"}
    assert {row[5] for row in shards[1]} == {"http://gpu1/v1"}
    assert [len(shard) for shard in shards] == [4, 3]


def test_multiple_slots_may_share_a_replica_without_migrating() -> None:
    _, shards = pin_cells_to_worker_slots(
        _cells(10), ["http://gpu0/v1", "http://gpu1/v1"], workers=4
    )

    assert [{row[5] for row in shard} for shard in shards] == [
        {"http://gpu0/v1"},
        {"http://gpu1/v1"},
        {"http://gpu0/v1"},
        {"http://gpu1/v1"},
    ]
