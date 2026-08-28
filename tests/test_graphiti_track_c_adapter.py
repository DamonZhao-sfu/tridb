from __future__ import annotations

import ast
import asyncio
import subprocess
import threading
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.agent_memory.table5_track_c.dataset import EventItem
from experiments.graphiti_track_c import install_receipt
from experiments.graphiti_track_c.conformance import (
    EVENTS,
    QUERIES,
    _gate_runtime_restart,
    _runtime_identity,
)
from experiments.graphiti_track_c.adapter import (
    GraphitiTrackCAdapter,
    cypher_stage,
    event_uuid,
    reference_time,
    snapshot_adapter_compatibility,
)


_FROZEN_GRAPHITI = Path("/localhome/hza214/agent-memory-table5/src/graphiti")
_FROZEN_GRAPHITI_COMMIT = "993e081a6d7948a0d8851c12a5fbdbeb49fed862"


def _class_method_arguments(path: Path, class_name: str, method_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    member.name == method_name
                ):
                    return {
                        argument.arg
                        for argument in [
                            *member.args.posonlyargs,
                            *member.args.args,
                            *member.args.kwonlyargs,
                        ]
                    }
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def _class_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                member.target.id
                for member in node.body
                if isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
            }
    raise AssertionError(f"missing {class_name} in {path}")


def _event(sample_id: str = "conv-26", event_id: str = "D1:1") -> EventItem:
    return EventItem(
        sample_id=sample_id,
        event_id=event_id,
        session_id="S1",
        timestamp="1:56 pm on 8 May, 2023",
        role="Caroline",
        text="hello",
        ordinal=1_000_001,
        metadata={"turn_number": 1},
    )


def test_event_uuid_is_stable_and_scope_sensitive() -> None:
    first = _event()
    assert event_uuid(first) == event_uuid(first)
    assert event_uuid(first) != event_uuid(_event(sample_id="conv-30"))
    assert event_uuid(first) != event_uuid(_event(event_id="D1:2"))


def test_reference_time_preserves_turn_order() -> None:
    first = _event()
    second = EventItem(
        **{**first.__dict__, "ordinal": 1_000_002, "metadata": {"turn_number": 2}}
    )
    assert reference_time(first).tzinfo == timezone.utc
    assert reference_time(first) < reference_time(second)


def test_partial_resume_reuses_complete_groups_and_repairs_only_partial_clone(
    tmp_path: Path,
) -> None:
    complete = _event(sample_id="conv-26", event_id="D1:1")
    partial_first = _event(sample_id="conv-30", event_id="D1:1")
    partial_second = _event(sample_id="conv-30", event_id="D1:2")
    deleted: list[tuple[str, ...]] = []

    class Driver:
        async def execute_query(
            self, query: str, **kwargs: object
        ) -> tuple[object, ...]:
            if "MATCH (n:Episodic)" in query:
                return (
                    [
                        {
                            "uuid": "existing-complete",
                            "name": "D1:1",
                            "group_id": "conv-26",
                        },
                        {
                            "uuid": "existing-partial",
                            "name": "D1:1",
                            "group_id": "conv-30",
                        },
                    ],
                    None,
                    None,
                )
            deleted.append(tuple(kwargs["group_ids"]))  # type: ignore[arg-type]
            return ([], None, None)

    adapter = object.__new__(GraphitiTrackCAdapter)
    adapter.config = SimpleNamespace(resume_partial_build=True, build_scope_workers=1)
    adapter.manifest_path = tmp_path / "manifest.json"
    adapter._episode_map_path = tmp_path / "episode-map.json"
    adapter._episode_to_event = {}
    adapter._build_stats = {}
    adapter._graphiti = SimpleNamespace(driver=Driver())
    adapter._runtime = SimpleNamespace(run=asyncio.run)

    async def ingest_pending(
        grouped: dict[str, list[EventItem]],
    ) -> list[dict[str, object]]:
        assert set(grouped) == {"conv-30"}
        adapter._episode_to_event["new-partial-replay"] = "D1:2"
        return [{"sample_id": "conv-30", "episodes": 2}]

    adapter._ingest_all = ingest_pending
    result = adapter.ingest_history([complete, partial_first, partial_second])

    assert result["resume"] == {
        "enabled": True,
        "reused_complete_groups": ["conv-26"],
        "repaired_partial_groups": ["conv-30"],
        "rebuilt_groups": ["conv-30"],
    }
    assert deleted == [("conv-30",), ("conv-30",)]
    assert adapter._episode_to_event["existing-complete"] == "D1:1"


def test_cypher_stage_classifies_native_boundaries() -> None:
    assert cypher_stage("CALL db.index.vector.queryNodes('x', 2, $v)")[0] == "vector"
    assert cypher_stage("CALL db.index.fulltext.queryNodes('x', $q)")[0] == "vector"
    assert cypher_stage("MATCH (n) RETURN n")[0] == "graph"
    assert cypher_stage("MATCH (n) SET n.x = 1 RETURN n")[0] == "persistence"


def test_stage_tracing_preserves_query_cypher_parameter() -> None:
    """Graphiti passes both positional Cypher and a ``query=`` parameter."""

    async def passthrough(*args: object, **kwargs: object) -> tuple[object, ...]:
        return (*args, kwargs)

    adapter = object.__new__(GraphitiTrackCAdapter)
    adapter._stage_tracing = False
    adapter._instrumented = False
    adapter._graphiti = SimpleNamespace(
        llm_client=SimpleNamespace(generate_response=passthrough),
        embedder=SimpleNamespace(create=passthrough, create_batch=passthrough),
        driver=SimpleNamespace(execute_query=passthrough),
    )

    adapter.enable_stage_tracing()
    result = asyncio.run(
        adapter._graphiti.driver.execute_query(
            "CALL db.index.fulltext.queryNodes($index, $query)",
            query="memory terms",
        )
    )

    assert result == (
        "CALL db.index.fulltext.queryNodes($index, $query)",
        {"query": "memory terms"},
    )


def test_snapshot_adapter_compatibility_requires_both_pinned_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert snapshot_adapter_compatibility("same", "same")["mode"] == "exact"
    assert not snapshot_adapter_compatibility("old", "new")["compatible"]

    monkeypatch.setenv("GRAPHITI_COMPAT_SNAPSHOT_SHA256", "old")
    monkeypatch.setenv("GRAPHITI_COMPAT_RUNTIME_SHA256", "new")
    allowed = snapshot_adapter_compatibility("old", "new")
    assert allowed["compatible"]
    assert allowed["mode"] == "explicit_instrumentation_only"
    assert not snapshot_adapter_compatibility("old", "changed-again")["compatible"]


def test_install_receipt_measures_exact_neo4j_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "neo4j"
    (home / "bin").mkdir(parents=True)
    for name in ("neo4j", "cypher-shell"):
        (home / "bin" / name).write_text("synthetic", encoding="utf-8")
    observed = {
        "neo4j": "5.26.6",
        "cypher-shell": "Cypher-Shell 5.26.6",
    }
    monkeypatch.setattr(
        install_receipt,
        "_run",
        lambda executable, *_: observed[Path(executable).name],
    )

    identity = install_receipt._backend_identity(home)

    assert identity["neo4j_version"] == "5.26.6"
    assert identity["cypher_shell_version"] == "Cypher-Shell 5.26.6"
    observed["neo4j"] = "5.26.7"
    with pytest.raises(RuntimeError, match="Neo4j version mismatch"):
        install_receipt._backend_identity(home)


def test_conformance_requires_a_distinct_neo4j_systemd_invocation() -> None:
    before = _runtime_identity("a" * 32, 101)
    after = _runtime_identity("b" * 32, 202)

    _gate_runtime_restart(before, after)

    with pytest.raises(RuntimeError, match="InvocationID did not change"):
        _gate_runtime_restart(before, {**after, "systemd_invocation_id": "a" * 32})
    with pytest.raises(RuntimeError, match="invalid Neo4j systemd InvocationID"):
        _runtime_identity("not-an-invocation", 202)
    with pytest.raises(RuntimeError, match="invalid Neo4j MainPID"):
        _runtime_identity("b" * 32, 0)


def test_conformance_uses_two_explicit_named_people_per_group() -> None:
    assert "Asteria Northwind mentors Celeste Eastwind" in EVENTS[0].text
    assert "Borealis Southwind mentors Dorian Westwind" in EVENTS[2].text
    assert QUERIES[0].question == "Who does Asteria Northwind mentor?"
    assert QUERIES[1].question == "Who does Borealis Southwind mentor?"


def test_frozen_graphiti_source_api_contract() -> None:
    """Prove the queued adapter calls match the exact frozen upstream source."""
    if not (_FROZEN_GRAPHITI / ".git").is_dir():
        pytest.skip("frozen Graphiti checkout is an experiment-host prerequisite")

    commit = subprocess.run(
        ["git", "-C", str(_FROZEN_GRAPHITI), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(_FROZEN_GRAPHITI),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == _FROZEN_GRAPHITI_COMMIT
    assert tracked_status == ""

    graphiti = _FROZEN_GRAPHITI / "graphiti_core/graphiti.py"
    bulk = _FROZEN_GRAPHITI / "graphiti_core/utils/bulk_utils.py"
    edges = _FROZEN_GRAPHITI / "graphiti_core/edges.py"
    assert {
        "uri",
        "user",
        "password",
        "llm_client",
        "embedder",
        "cross_encoder",
        "store_raw_episode_content",
        "max_coroutines",
    } <= _class_method_arguments(graphiti, "Graphiti", "__init__")
    assert {"bulk_episodes", "group_id"} <= _class_method_arguments(
        graphiti, "Graphiti", "add_episode_bulk"
    )
    assert {
        "name",
        "episode_body",
        "source_description",
        "reference_time",
        "source",
        "group_id",
        "uuid",
        "update_communities",
    } <= _class_method_arguments(graphiti, "Graphiti", "add_episode")
    assert {"query", "group_ids", "num_results"} <= _class_method_arguments(
        graphiti, "Graphiti", "search"
    )
    assert {
        "name",
        "uuid",
        "content",
        "source_description",
        "source",
        "reference_time",
    } <= _class_fields(bulk, "RawEpisode")
    assert "episodes" in _class_fields(edges, "EntityEdge")


def test_prepared_add_uses_public_namespace_writes_without_high_level_llm_api() -> None:
    adapter_path = Path("experiments/graphiti_track_c/adapter.py")
    tree = ast.parse(
        adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path)
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != "_add_async":
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        high_level_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr in {"add_episode", "add_episode_bulk", "add_triplet"}
        ]
        called_attributes = {
            call.func.attr for call in calls if isinstance(call.func, ast.Attribute)
        }
        assert high_level_calls == []
        assert "save" in called_attributes
        return
    raise AssertionError("missing GraphitiTrackCAdapter._add_async")


def test_graphiti_llm_free_semantics_are_fail_closed_and_honestly_named() -> None:
    search = GraphitiTrackCAdapter.search_semantics()
    add = GraphitiTrackCAdapter.add_semantics()

    assert search["generation_llm_policy"] == "forbidden_fail_closed"
    assert search["native_call"].startswith("Graphiti.search")
    assert add["construction_llm_policy"] == "forbidden_fail_closed"
    assert add["semantic_boundary"] == "prepared_memory_item_insertion_v1"
    assert add["not_equivalent_to"] == (
        "Graphiti.add_episode high-level construction API"
    )


def test_graphiti_llm_guard_rejects_generation_and_model_reranking() -> None:
    adapter = object.__new__(GraphitiTrackCAdapter)
    adapter._llm_guard_lock = threading.Lock()
    adapter._llm_guard_installed = False
    adapter._forbidden_llm_calls = 0
    adapter._forbidden_cross_encoder_calls = 0
    adapter._graphiti = SimpleNamespace(
        llm_client=SimpleNamespace(generate_response=None),
        cross_encoder=SimpleNamespace(rank=None),
    )

    adapter._install_no_llm_guard()

    with pytest.raises(RuntimeError, match="forbidden generation"):
        asyncio.run(adapter._graphiti.llm_client.generate_response("prompt"))
    with pytest.raises(RuntimeError, match="forbidden model reranker"):
        asyncio.run(adapter._graphiti.cross_encoder.rank("q", ["fact"]))
    assert adapter._forbidden_llm_calls == 1
    assert adapter._forbidden_cross_encoder_calls == 1
