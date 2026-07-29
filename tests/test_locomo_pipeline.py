"""Unit tests for the end-to-end LoCoMo pipeline helpers."""

from __future__ import annotations

from bench.agent_memory.locomo_pipeline import (
    build_metrics_report,
    lexical_f1,
    parse_judge_label,
    retrieval_metrics,
)


def _qa(
    *,
    category=4,
    prediction="blue",
    correct=True,
    context=None,
    evidence=None,
):
    return {
        "question": "What color?",
        "answer": "blue",
        "category": category,
        "evidence": ["D1:1"] if evidence is None else evidence,
        "tridb_prediction_context": (["D1:1", "D2:1"] if context is None else context),
        "tridb_prediction": prediction,
        "tridb_prediction_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "tridb_prediction_latency_seconds": 0.5,
        "tridb_prediction_judge": {
            "label": "CORRECT" if correct else "WRONG",
            "correct": correct,
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            },
            "latency_seconds": 0.25,
        },
    }


def test_parse_judge_label_accepts_json_and_plain_text():
    assert parse_judge_label('{"label": "CORRECT"}') == "CORRECT"
    assert parse_judge_label("WRONG") == "WRONG"


def test_lexical_f1_is_one_for_normalized_exact_answer():
    assert lexical_f1("The blue.", "blue") == 1.0


def test_retrieval_metrics_use_annotated_evidence_ids():
    metrics = retrieval_metrics([_qa()], prediction_key="tridb_prediction")
    assert metrics["@1"]["evidence_recall"] == 1.0
    assert metrics["@1"]["hit_all"] == 1.0


def test_metrics_report_emits_paper_table_category_shape():
    samples = [
        {
            "sample_id": "conv-1",
            "qa": [
                _qa(category=4, correct=True),
                _qa(category=1, correct=False),
                _qa(category=2, correct=True),
                _qa(category=3, correct=True),
            ],
        }
    ]
    report = build_metrics_report(
        samples,
        prediction_key="tridb_prediction",
        categories=(1, 2, 3, 4),
        answer_model="answer-model",
        judge_model="judge-model",
        started_at="2026-01-01T00:00:00+00:00",
        elapsed_seconds=1.0,
    )
    row = report["paper_table_percent"]
    assert row["single"] == 100.0
    assert row["multi"] == 0.0
    assert row["temporal"] == 100.0
    assert row["open"] == 100.0
    assert row["overall"] == 75.0
    assert report["status"] == "completed"
