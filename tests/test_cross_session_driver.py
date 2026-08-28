from __future__ import annotations

import hashlib

import pytest

from bench.agent_memory.memoryarena.cross_session import (
    BackendSearch,
    CrossSessionDriver,
    RetrievedItem,
)
from bench.agent_memory.memoryarena.dataset import normalize_rows
from bench.agent_memory.memoryarena.oracle import Arm, Candidate


def _task():
    return normalize_rows(
        [
            {
                "id": 1,
                "questions": ["q0", "q1", "q2"],
                "answers": ["a0", "a1", "a2"],
            }
        ],
        config="progressive_search",
    ).tasks[0]


class FakeBackend:
    def __init__(self, task) -> None:
        self.task = task
        self.updates = []
        self.inject_future = False

    def retrieve(self, arm, decision, *, top_k):
        candidates = tuple(
            Candidate(
                session_uid=session.session_uid,
                task_uid=session.task_uid,
                ordinal=session.ordinal,
                semantic_score=1.0 - session.ordinal / 10,
                graph_distance=1,
            )
            for session in self.task.sessions
        )
        eligible = [
            item for item in candidates if item.ordinal < decision.cutoff_ordinal
        ]
        ranked = eligible
        if self.inject_future:
            ranked = [candidates[-1]]
        return BackendSearch(
            snapshot_id=f"snapshot:{decision.cutoff_ordinal}",
            all_candidates=candidates,
            ranked_items=tuple(
                RetrievedItem(item, f"memory-{item.ordinal}") for item in ranked
            ),
            first_row_ms=1.0 if ranked else None,
            time_to_k_ms=2.0,
            candidates_examined=len(candidates),
            visited_nodes=len(ranked),
            visited_edges=max(0, len(ranked) - 1),
            termination_reason="limit",
        )

    def update(self, session, *, response, retrieval_receipt):
        self.updates.append((session.session_uid, response, retrieval_receipt))
        return {"committed": True}


def _driver(backend, *, budget=100):
    return CrossSessionDriver(
        backend,
        run_id="run",
        dataset_manifest_sha256="a" * 64,
        arm=Arm.GEM_FUSED,
        top_k=10,
        injection_token_budget=budget,
        count_tokens=lambda value: len(value.split()),
    )


def test_driver_enforces_retrieve_answer_update_order_and_hashes_injection() -> None:
    task = _task()
    backend = FakeBackend(task)
    driver = _driver(backend)

    first = driver.open_session(task.sessions[0])
    assert first.injection_text == ""
    with pytest.raises(RuntimeError, match="must complete"):
        driver.open_session(task.sessions[1])
    driver.complete_session(task.sessions[0], response="answer zero")

    second = driver.open_session(task.sessions[1])
    assert second.injection_text == "memory-0"
    assert second.receipt.injection_sha256 == hashlib.sha256(b"memory-0").hexdigest()
    driver.complete_session(task.sessions[1], response="answer one")
    assert [item[0] for item in backend.updates] == [
        task.sessions[0].session_uid,
        task.sessions[1].session_uid,
    ]


def test_driver_refuses_skip_update_before_retrieve_and_future_leakage() -> None:
    task = _task()
    backend = FakeBackend(task)
    driver = _driver(backend)
    with pytest.raises(RuntimeError, match="before retrieval"):
        driver.complete_session(task.sessions[0], response="bad")
    with pytest.raises(ValueError, match="non-serial"):
        driver.open_session(task.sessions[1])

    backend.inject_future = True
    with pytest.raises(ValueError, match="retrieval leakage"):
        driver.open_session(task.sessions[0])


def test_driver_applies_one_shared_injection_budget_after_ranking() -> None:
    task = _task()
    backend = FakeBackend(task)
    driver = _driver(backend, budget=1)
    driver.open_session(task.sessions[0])
    driver.complete_session(task.sessions[0], response="a")
    driver.open_session(task.sessions[1])
    driver.complete_session(task.sessions[1], response="b")
    opened = driver.open_session(task.sessions[2])
    assert opened.injection_text == "memory-0"
    assert opened.receipt.injection_tokens == 1
    assert opened.receipt.selected_session_uids == (task.sessions[0].session_uid,)
