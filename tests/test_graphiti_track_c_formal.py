from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.graphiti_track_c.adapter import GraphitiTrackCAdapter
from experiments.graphiti_track_c_formal.adapter import (
    AsyncGroupFifoGate,
    FormalGraphitiTrackCAdapter,
)
from experiments.graphiti_track_c_formal.hashing import graphiti_adapter_sha256


def test_clone_driver_rebinds_listener_ports() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "table5_track_c_graphiti_clone_state.sh"
    ).read_text(encoding="utf-8")

    assert "GRAPHITI_NEO4J_BOLT_PORT" in script
    assert "GRAPHITI_NEO4J_HTTP_PORT" in script
    assert "server.bolt.listen_address=:$bolt_port" in script
    assert "server.http.listen_address=:$http_port" in script
    assert "Graphiti cloned config did not bind the requested listener ports" in script


def test_group_fifo_is_ordered_within_one_group() -> None:
    async def scenario() -> None:
        gate = AsyncGroupFifoGate()
        first_ticket, _ = await gate.acquire("conversation-a")
        second = asyncio.create_task(gate.acquire("conversation-a"))
        await asyncio.sleep(0)
        assert not second.done()
        gate.release("conversation-a")
        second_ticket, _ = await asyncio.wait_for(second, timeout=1)
        gate.release("conversation-a")
        assert [first_ticket, second_ticket] == [0, 1]

    asyncio.run(scenario())


def test_group_fifo_allows_different_groups_to_overlap() -> None:
    async def scenario() -> None:
        gate = AsyncGroupFifoGate()
        await gate.acquire("conversation-a")
        _, _ = await asyncio.wait_for(gate.acquire("conversation-b"), timeout=1)
        gate.release("conversation-b")
        gate.release("conversation-a")

    asyncio.run(scenario())


def test_formal_hash_binds_base_and_guard_packages(tmp_path: Path) -> None:
    base = tmp_path / "experiments/graphiti_track_c"
    guard = tmp_path / "experiments/graphiti_track_c_formal"
    base.mkdir(parents=True)
    guard.mkdir(parents=True)
    (base / "adapter.py").write_text("BASE = 1\n", encoding="utf-8")
    (guard / "adapter.py").write_text("GUARD = 1\n", encoding="utf-8")
    original = graphiti_adapter_sha256(tmp_path)

    (guard / "adapter.py").write_text("GUARD = 2\n", encoding="utf-8")

    assert graphiti_adapter_sha256(tmp_path) != original


def test_formal_adapter_records_fifo_receipt_and_releases_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(FormalGraphitiTrackCAdapter)
    adapter._formal_add_gate = AsyncGroupFifoGate()
    adapter._formal_add_receipts = {}
    adapter._formal_add_receipts_lock = threading.Lock()
    item = SimpleNamespace(sample_id="conversation-a", event_id="event-1")

    async def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(GraphitiTrackCAdapter, "_add_async", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        asyncio.run(adapter._add_async(item))

    monkeypatch.setattr(
        GraphitiTrackCAdapter,
        "_add_async",
        lambda *args, **kwargs: asyncio.sleep(0, result={"native": "ok"}),
    )
    asyncio.run(adapter._add_async(item))
    monkeypatch.setattr(
        GraphitiTrackCAdapter,
        "add",
        lambda *args, **kwargs: {"created_memory_count": 1},
    )
    receipt = adapter.add(item, visibility=False)

    assert receipt["write_concurrency_policy"] == (
        "per_group_fifo_cross_group_parallel"
    )
    assert receipt["write_group_id"] == "conversation-a"
    assert receipt["write_group_fifo_ticket"] == 1
    assert receipt["write_group_fifo_wait_ms"] >= 0
