"""Unit coverage for Polyglot's task- versus live-node query identity."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.agent_memory.gem_eg.oracle import Predicate, QuerySpec
from bench.agent_memory.gem_oe.retrievers import PolyglotRetriever


pytestmark = pytest.mark.unit


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[QuerySpec, np.ndarray]] = []

    def reuse_query(
        self, spec: QuerySpec, query_vec: np.ndarray
    ) -> tuple[list[str], dict[str, float | int]]:
        self.calls.append((spec, query_vec.copy()))
        return ["historical-node"], {"total_ms": 2.0, "candidates_examined": 1}


class _Gem:
    def __init__(self, seed: str) -> None:
        self.seed = seed
        self._engine = SimpleNamespace(_vectors={"math:task": json.dumps([0.0, 2.0])})
        self.embed_calls = 0
        self.spec_vectors: list[tuple[float, ...] | None] = []

    def _embed_parent(self, parent) -> tuple[tuple[float, ...], float]:
        self.embed_calls += 1
        return (3.0, 4.0), 7.0

    def _spec(
        self,
        task_uid: str,
        parent,
        k: int,
        iteration: int,
        query_vector: tuple[float, ...] | None = None,
    ) -> QuerySpec:
        self.spec_vectors.append(query_vector)
        return QuerySpec(
            query_id="oe.reuse",
            decision_point=f"{task_uid}#it{iteration}",
            seed_uid=task_uid,
            relation="lineage" if self.seed == "node" else "hier",
            hops=6 if self.seed == "node" else 2,
            predicate=Predicate(kind="node", include_tasks=frozenset({task_uid})),
            k=k,
            rank_by="seed_fitness" if self.seed == "node" else "similarity",
            query_vector=query_vector,
            ann_entry_kind="node" if self.seed == "node" else "task",
            ann_m_seeds=3,
        )

    @staticmethod
    def _hydrate(uids: list[str]) -> list[str]:
        return [f"hydrated:{uid}" for uid in uids]


def test_node_seed_uses_live_parent_vector_in_spec_and_backend() -> None:
    gem = _Gem("node")
    backend = _Backend()
    retriever = PolyglotRetriever(backend=backend, gem=gem)

    result = retriever.retrieve(
        task_uid="math:task", parent=SimpleNamespace(), k=1, iteration=4
    )

    assert result == ["hydrated:historical-node"]
    assert gem.embed_calls == 1
    assert gem.spec_vectors == [(3.0, 4.0)]
    spec, vector = backend.calls[0]
    assert spec.query_vector == (3.0, 4.0)
    np.testing.assert_allclose(vector, np.asarray([0.6, 0.8], dtype=np.float32))
    telemetry = retriever.telemetry[0]
    assert telemetry["seed_layer"] == "node"
    assert telemetry["query_vector_source"] == "live_parent"
    assert telemetry["embed_ms"] == 7.0
    assert telemetry["retriever_total_ms"] > 0.0
    assert telemetry["backend_total_ms"] == 2.0
    assert len(telemetry["query_vector_sha256"]) == 64


def test_task_seed_reuses_persisted_task_vector_without_embedding_parent() -> None:
    gem = _Gem("task")
    backend = _Backend()
    retriever = PolyglotRetriever(backend=backend, gem=gem)

    retriever.retrieve(task_uid="math:task", parent=SimpleNamespace(), k=1, iteration=1)

    assert gem.embed_calls == 0
    assert gem.spec_vectors == [None]
    spec, vector = backend.calls[0]
    assert spec.query_vector is None
    np.testing.assert_allclose(vector, np.asarray([0.0, 1.0], dtype=np.float32))
    telemetry = retriever.telemetry[0]
    assert telemetry["seed_layer"] == "task"
    assert telemetry["query_vector_source"] == "persisted_seed"
    assert telemetry["embed_ms"] == 0.0
