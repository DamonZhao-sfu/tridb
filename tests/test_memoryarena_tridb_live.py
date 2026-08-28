"""Small stock-PG live gate for the MemoryArena six-arm adapter."""

from __future__ import annotations

import os

import pytest

from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.memoryarena.cross_session import CrossSessionDriver
from bench.agent_memory.memoryarena.dataset import normalize_rows
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.tridb_backend import TriDBMemoryArenaBackend

DSN = os.environ.get("TRIDB_GEM_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TRIDB_GEM_DSN to run MemoryArena live adapter test"
)
DIM = 8


class StubEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * DIM
            for index, byte in enumerate(text.encode()):
                vector[index % DIM] += (byte % 31) / 31.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


def test_all_six_arms_replay_without_leakage_on_stock_pg() -> None:
    task = normalize_rows(
        [
            {
                "id": 999,
                "questions": ["alpha fact", "combine alpha", "final alpha"],
                "answers": ["alpha", "alpha beta", "alpha beta gamma"],
            }
        ],
        config="progressive_search",
    ).tasks[0]
    memory = TriDBGovernedMemory(
        GemStore.connect(DSN, dim=DIM), embedder=StubEmbedder()
    )
    memory.init_schema()
    try:
        for arm in Arm:
            namespace = f"test_memoryarena_live:{arm.value}"
            scope_id = f"{namespace}:{task.task_uid}"
            memory.store.conn.execute(
                "DELETE FROM gem_unit WHERE scope_id=%s", (scope_id,)
            )
            backend = TriDBMemoryArenaBackend(
                memory,
                task,
                namespace=namespace,
                reinforce=False,
                write_enabled=arm is not Arm.MEMORY_OFF,
            )
            driver = CrossSessionDriver(
                backend,
                run_id=namespace,
                dataset_manifest_sha256="a" * 64,
                arm=arm,
                top_k=10,
                injection_token_budget=100,
                count_tokens=lambda value: len(value.split()),
            )
            for session in task.sessions:
                opened = driver.open_session(session)
                assert all(
                    uid in session.protocol_dependencies
                    for uid in opened.receipt.selected_session_uids
                )
                if session.ordinal > 0 and arm in (Arm.VECTOR_ONLY, Arm.GEM_FUSED):
                    selected = {
                        item.session_uid: item
                        for item in opened.receipt.candidates
                        if item.selected_rank is not None
                    }
                    assert selected
                    assert all(
                        item.semantic_score_source
                        == "pgvector_cosine_similarity_selected_top_k"
                        for item in selected.values()
                    )
                if session.ordinal > 0 and arm is Arm.GEM_FUSED:
                    # tjs_open_graph_examined() is an edge-step probe.  Do not
                    # relabel it as a distinct-node counter in the receipt.
                    assert opened.receipt.visited_nodes is None
                    assert opened.receipt.visited_edges is not None
                driver.complete_session(session, response=session.gold_answer)
            row = memory.store.conn.execute(
                "SELECT count(*) FROM gem_unit WHERE scope_id=%s", (scope_id,)
            ).fetchone()
            expected = 0 if arm is Arm.MEMORY_OFF else len(task.sessions)
            assert int(row[0]) == expected
    finally:
        memory.close()
