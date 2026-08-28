from __future__ import annotations

from dataclasses import replace

import pytest

from bench.agent_memory.memoryarena.dataset import (
    extract_exact_answer,
    normalize_rows,
)
from bench.agent_memory.memoryarena.metrics import retrieval_metrics
from bench.agent_memory.memoryarena.oracle import (
    Arm,
    Candidate,
    DecisionPoint,
    assert_no_leakage,
    candidates_for_task,
    select,
)
from bench.agent_memory.memoryarena.receipts import (
    build_receipt,
    verify_serialized_receipt,
)


def _corpus():
    return normalize_rows(
        [
            {
                "id": 7,
                "questions": ["first clue", "second clue", "final query"],
                "answers": [
                    "Exact Answer: Alpha",
                    "**Exact Answer:** **Alpha**",
                    "Reasoning\nExact Answer: Alpha\nConfidence: 90%",
                ],
            }
        ],
        config="progressive_search",
        source_revision="deadbeef",
    )


def test_normalization_preserves_prefix_cutoff_and_dependency_boundary() -> None:
    corpus = _corpus()
    task = corpus.tasks[0]
    final = task.sessions[-1]

    assert final.ordinal == 2
    assert final.cutoff_ordinal == 2
    assert final.protocol_dependencies == tuple(
        session.session_uid for session in task.sessions[:2]
    )
    assert final.gold_exact_answer == "Alpha"
    assert "not minimal human evidence" in corpus.dependency_annotation
    assert corpus.session_count == 3
    assert corpus.decision_count == 2


def test_extract_exact_answer_handles_released_markdown_variants() -> None:
    assert extract_exact_answer("Exact Answer: Ada Lovelace") == "Ada Lovelace"
    assert extract_exact_answer("**Exact Answer:** **Ada Lovelace**") == "Ada Lovelace"
    assert extract_exact_answer("no short answer") is None


def test_normalization_rejects_misaligned_sessions() -> None:
    with pytest.raises(ValueError, match="questions !="):
        normalize_rows(
            [{"id": 0, "questions": ["a"], "answers": []}],
            config="progressive_search",
        )


def test_six_arm_reference_keeps_governance_filters_in_every_arm() -> None:
    task = _corpus().tasks[0]
    decision = DecisionPoint.from_session(task.sessions[-1])
    scores = {
        task.sessions[0].session_uid: 0.2,
        task.sessions[1].session_uid: 0.9,
        task.sessions[2].session_uid: 1.0,
    }
    distances = {
        task.sessions[0].session_uid: 2,
        task.sessions[1].session_uid: 1,
        task.sessions[2].session_uid: 0,
    }
    candidates = candidates_for_task(
        task,
        semantic_scores=scores,
        graph_distances=distances,
        stale=frozenset({task.sessions[1].session_uid}),
    )

    selections = {arm: select(arm, decision, candidates, top_k=10) for arm in Arm}
    assert selections[Arm.MEMORY_OFF].selected == ()
    assert [item.ordinal for item in selections[Arm.RECENT_FIFO].selected] == [1, 0]
    assert [item.ordinal for item in selections[Arm.VECTOR_ONLY].selected] == [1, 0]
    assert [item.ordinal for item in selections[Arm.GRAPH_RELATIONAL].selected] == [0]
    assert [item.ordinal for item in selections[Arm.GEM_FUSED].selected] == [0]
    assert [item.ordinal for item in selections[Arm.ORACLE].selected] == [0, 1]
    for selection in selections.values():
        assert_no_leakage(decision, selection.selected)
        assert all(
            item.ordinal < decision.cutoff_ordinal for item in selection.selected
        )


def test_future_or_target_session_is_a_hard_leakage_failure() -> None:
    task = _corpus().tasks[0]
    decision = DecisionPoint.from_session(task.sessions[1])
    target = Candidate(
        session_uid=task.sessions[1].session_uid,
        task_uid=task.task_uid,
        ordinal=1,
    )
    with pytest.raises(ValueError, match="retrieval leakage"):
        assert_no_leakage(decision, [target])


def test_metrics_separate_dependency_quality_staleness_and_harm() -> None:
    task = _corpus().tasks[0]
    decision = DecisionPoint.from_session(task.sessions[-1])
    selected = [
        Candidate(
            session_uid=task.sessions[1].session_uid,
            task_uid=task.task_uid,
            ordinal=1,
            is_stale=True,
        ),
        Candidate(
            session_uid="irrelevant",
            task_uid=task.task_uid,
            ordinal=0,
            is_valid=False,
            is_harmful=True,
        ),
    ]
    metrics = retrieval_metrics(selected, decision)
    assert metrics["dependency_recall_at_10"] == 0.5
    # One of two protocol dependencies is retrieved at rank one.  The ideal list
    # contains both relevant sessions, so this is intentionally below 1.0.
    assert metrics["ndcg_at_10"] == pytest.approx(0.6131471927654584)
    assert metrics["harmful_at_10"] == 0.5
    assert metrics["stale_at_10"] == 0.5
    assert metrics["constraint_valid_fraction"] == 0.0


def test_receipt_is_content_addressed_and_rejects_leakage_or_budget_overrun() -> None:
    task = _corpus().tasks[0]
    decision = DecisionPoint.from_session(task.sessions[-1])
    candidates = candidates_for_task(
        task,
        semantic_scores={
            session.session_uid: session.ordinal for session in task.sessions
        },
        graph_distances={session.session_uid: 1 for session in task.sessions},
    )
    selected = select(Arm.GEM_FUSED, decision, candidates).selected
    receipt = build_receipt(
        run_id="run-1",
        dataset_manifest_sha256="a" * 64,
        snapshot_id="snapshot-1",
        arm=Arm.GEM_FUSED,
        decision=decision,
        all_candidates=candidates,
        selected=selected,
        top_k=10,
        injection_token_budget=128,
        injection_tokens=64,
        injection_text="memory payload",
        first_row_ms=1.5,
        time_to_k_ms=3.0,
        candidates_examined=2,
        visited_nodes=3,
        visited_edges=2,
        termination_reason="limit",
    )
    serialized = receipt.as_dict()
    assert verify_serialized_receipt(serialized)
    assert not verify_serialized_receipt({**serialized, "visited_edges": 999})

    with pytest.raises(ValueError, match="token budget"):
        build_receipt(
            run_id="run-1",
            dataset_manifest_sha256="a" * 64,
            snapshot_id="snapshot-1",
            arm=Arm.GEM_FUSED,
            decision=decision,
            all_candidates=candidates,
            selected=selected,
            top_k=10,
            injection_token_budget=10,
            injection_tokens=11,
            injection_text="too many tokens",
            first_row_ms=1.5,
            time_to_k_ms=3.0,
            candidates_examined=2,
            visited_nodes=3,
            visited_edges=2,
            termination_reason="limit",
        )

    leaked = replace(
        selected[0],
        session_uid="leaked-future",
        ordinal=decision.cutoff_ordinal,
    )
    with pytest.raises(ValueError, match="leakage"):
        build_receipt(
            run_id="run-1",
            dataset_manifest_sha256="a" * 64,
            snapshot_id="snapshot-1",
            arm=Arm.GEM_FUSED,
            decision=decision,
            all_candidates=(*candidates, leaked),
            selected=[leaked],
            top_k=10,
            injection_token_budget=128,
            injection_tokens=64,
            injection_text="leaked payload",
            first_row_ms=1.5,
            time_to_k_ms=3.0,
            candidates_examined=2,
            visited_nodes=3,
            visited_edges=2,
            termination_reason="limit",
        )
