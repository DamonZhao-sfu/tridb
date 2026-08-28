"""Conformance tests for the LLM-free prepared-item Add boundary."""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest

from bench.agent_memory.table5_track_c.adapters.mem0 import Mem0Adapter
from bench.agent_memory.table5_track_c.adapters.memos import MemosAdapter
from bench.agent_memory.table5_track_c.dataset import EventItem


def _event() -> EventItem:
    return EventItem(
        sample_id="conv-test",
        event_id="trackc_native_0010_D1:11",
        session_id="S1",
        timestamp="10:00 AM on 01 January, 2026",
        role="Caroline",
        text='(10:00 AM) Caroline said, "I enjoy hiking."',
        ordinal=100_000_010,
        metadata={"source": "locomo"},
    )


def _bare_mem0(add_impl):
    adapter = object.__new__(Mem0Adapter)
    adapter.memory = SimpleNamespace(
        llm=SimpleNamespace(generate_response=lambda *_args, **_kwargs: "forbidden"),
        add=add_impl,
    )
    adapter._llm_guard_lock = threading.Lock()
    adapter._llm_guard_installed = False
    adapter._forbidden_llm_calls = 0
    return adapter


def test_mem0_prepared_add_uses_infer_false_and_reports_zero_llm() -> None:
    calls = []

    def add(messages, **kwargs):
        calls.append((messages, kwargs))
        return {
            "results": [
                {"id": "memory-1", "memory": messages[0]["content"], "event": "ADD"}
            ]
        }

    adapter = _bare_mem0(add)
    receipt = adapter.add(_event(), visibility=False)

    assert calls[0][1]["infer"] is False
    assert receipt["semantic_boundary"] == "prepared_memory_item_insertion_v1"
    assert receipt["llm_call_count"] == 0
    assert receipt["created_memory_count"] == 1
    assert adapter._forbidden_llm_calls == 0


def test_mem0_prepared_add_fails_closed_if_native_code_calls_llm() -> None:
    adapter = None

    def add(_messages, **_kwargs):
        return adapter.memory.llm.generate_response("must fail")

    adapter = _bare_mem0(add)
    with pytest.raises(RuntimeError, match="forbidden LLM call"):
        adapter.add(_event(), visibility=False)
    assert adapter._forbidden_llm_calls == 1


def test_memos_prepared_add_bypasses_moscore_and_memreader() -> None:
    source = inspect.getsource(MemosAdapter._prepared_item_add)
    assert "text_mem.embedder.embed" in source
    assert "text_mem.add([prepared])" in source
    assert "scope.mos.add(" not in source
    assert "mem_reader.get_memory" not in source


def test_prepared_add_semantics_forbid_construction_llm() -> None:
    for adapter in (Mem0Adapter, MemosAdapter):
        semantics = adapter.add_semantics()
        assert semantics["semantic_boundary"] == "prepared_memory_item_insertion_v1"
        assert semantics["construction_llm_policy"] == "forbidden_fail_closed"
        assert all("llm" not in stage for stage in semantics["included_stages"])
