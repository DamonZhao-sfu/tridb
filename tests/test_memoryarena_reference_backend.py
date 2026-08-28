from __future__ import annotations

import json

from bench.agent_memory.memoryarena.dataset import normalize_rows
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.reference_backend import ReferenceBackend
from bench.agent_memory.memoryarena.cross_session import CrossSessionDriver


def test_reference_backend_never_selects_uncommitted_history() -> None:
    task = normalize_rows(
        [
            {
                "id": 1,
                "questions": ["q0", "q1", "q2"],
                "answers": ["a0", "a1", "a2"],
            }
        ],
        config="progressive_search",
    ).tasks[0]
    for arm in Arm:
        driver = CrossSessionDriver(
            ReferenceBackend(task),
            run_id=arm.value,
            dataset_manifest_sha256="a" * 64,
            arm=arm,
            count_tokens=lambda value: len(value.split()),
        )
        for session in task.sessions:
            opened = driver.open_session(session)
            assert all(
                uid in session.protocol_dependencies
                for uid in opened.receipt.selected_session_uids
            )
            assert json.dumps(opened.receipt.as_dict())
            driver.complete_session(session, response=session.gold_answer)
