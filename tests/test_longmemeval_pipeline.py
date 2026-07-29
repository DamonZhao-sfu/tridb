"""Tests for the inject-once/query-many LongMemEval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import bench.agent_memory.longmemeval_pipeline as pipeline
from bench.agent_memory.backend import SearchHit


def _row(history: int, questions: int = 60) -> dict:
    return {
        "context": f"history {history} " * 500,
        "questions": [
            f"Current Date: today\n\nNow Answer the Question: question {history}-{i}?"
            for i in range(questions)
        ],
        "answers": [f"answer {history}-{i}" for i in range(questions)],
        "metadata": {
            "source": "longmemeval_s*",
            "question_ids": [f"q-{history}-{i}" for i in range(questions)],
            "question_types": ["multi-session"] * questions,
            "qa_pair_ids": [f"qa-{history}-{i}" for i in range(questions)],
        },
    }


def test_load_workloads_enforces_paper_shape(tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"data": [_row(i) for i in range(5)]}))
    workloads = pipeline.load_workloads(path)
    assert len(workloads) == 5
    assert [len(workload.questions) for workload in workloads] == [60] * 5
    assert workloads[0].questions[0].question_id == "q-0-0"

    path.write_text(json.dumps({"data": [_row(0, questions=2)]}))
    with pytest.raises(ValueError, match="five histories"):
        pipeline.load_workloads(path)
    assert len(pipeline.load_workloads(path, strict_shape=False)) == 1


def test_load_workloads_rejects_duplicate_question_ids(tmp_path: Path):
    rows = [_row(0, questions=1), _row(1, questions=1)]
    rows[1]["metadata"]["question_ids"][0] = rows[0]["metadata"]["question_ids"][0]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate question_id"):
        pipeline.load_workloads(path, strict_shape=False)


def test_retrieval_query_and_answer_prompt_match_memoryagentbench_shape():
    question = (
        "Current Date: 2026/01/02\n\nNow Answer the Question: What color did I choose?"
    )
    assert pipeline.extract_retrieval_query(question) == "What color did I choose?"
    messages = pipeline.build_answer_messages(question, ["blue memory", "other"])
    assert messages[0] == {
        "role": "system",
        "content": pipeline.SYSTEM_MESSAGE,
    }
    assert "Memory 1:\nblue memory" in messages[1]["content"]
    assert "using a single phrase if possible" in messages[1]["content"]
    assert question in messages[1]["content"]
    fitted, used, tokens = pipeline.fit_answer_prompt(
        question,
        ["blue memory", "other"],
        token_counter=len,
        token_budget=600,
    )
    assert used == 2
    assert tokens <= 600
    assert "blue memory" in fitted[1]["content"]


@pytest.mark.parametrize(
    ("question_type", "expected"),
    [
        ("multi-session", "correct answer"),
        ("temporal-reasoning", "off-by-one"),
        ("knowledge-update", "updated answer"),
        ("single-session-preference", "Rubric"),
    ],
)
def test_official_judge_prompt_variants(question_type: str, expected: str):
    prompt = pipeline.build_judge_prompt(
        question_type,
        "question",
        "answer",
        "response",
        abstention=False,
    )
    assert expected in prompt
    assert pipeline.parse_judge_yes_no("Yes") is True
    assert pipeline.parse_judge_yes_no("no") is False


def test_latency_summary_reports_tail_ratio():
    summary = pipeline.latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == 2.5
    assert summary["p95"] > summary["p50"]
    assert summary["p95_over_p50"] == pytest.approx(summary["p95"] / summary["p50"])


class _FakeChunker:
    def __init__(self, **_kwargs):
        pass

    def chunk(self, _text):
        return ["chunk one", "chunk two"]

    def count(self, text):
        return len(text.split())


class _FakeBackend:
    instance = None

    def __init__(self):
        self.table = "longmemeval_test"
        self.replacements = []
        self.searches = []
        self.closed = False
        _FakeBackend.instance = self

    @classmethod
    def connect(cls, *_args, **_kwargs):
        return cls()

    def init_schema(self):
        return {"ok": True}

    def replace_scope(self, scope_id, units, *, isolated):
        self.replacements.append((scope_id, list(units), isolated))
        return len(units)

    def search(self, scope_id, *, query_embedding, k):
        self.searches.append((scope_id, query_embedding, k))
        return [
            SearchHit(
                id=1,
                scope_id=scope_id,
                external_id=f"{scope_id}_chunk_0000",
                session_id=scope_id,
                kind="chunk",
                role=None,
                content="remembered answer",
                event_time=None,
                event_order=0,
                metadata={},
                score=0.9,
            )
        ]

    def close(self):
        self.closed = True


class _FakeEmbeddingClient:
    def __init__(self, *_args, ledger, **_kwargs):
        self.ledger = ledger

    def discover_models(self):
        return [pipeline.DEFAULT_EMBEDDING_MODEL]

    def encode(self, texts, *, phase):
        kind = (
            "construction_embedding" if phase == "construction" else "query_embedding"
        )
        self.ledger.record(
            kind=kind,
            phase=phase,
            target=pipeline.DEFAULT_EMBEDDING_MODEL,
            items=len(texts),
            elapsed_seconds=0.01,
        )
        return [[1.0, 0.0] for _ in texts]


class _FakeChatClient:
    def __init__(self, *_args, ledger, **_kwargs):
        self.ledger = ledger

    def discover_models(self):
        return [pipeline.DEFAULT_ANSWER_MODEL]

    def stream_chat(self, *, model, **_kwargs):
        import time

        began = time.perf_counter()
        first = began + 0.01
        completed = first + 0.01
        self.ledger.record(
            kind="answer_generation",
            phase="qa",
            target=model,
            items=1,
            elapsed_seconds=0.02,
        )
        return pipeline.StreamingChatResult(
            text="answer",
            usage={"prompt_tokens": 10, "completion_tokens": 1},
            request_started=began,
            first_token_at=first,
            completed_at=completed,
        )


def test_pipeline_constructs_once_then_queries_many(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([_row(0, questions=2)]))
    output_dir = tmp_path / "out"
    monkeypatch.setattr(pipeline, "TiktokenSentenceChunker", _FakeChunker)
    monkeypatch.setattr(pipeline, "TriDBMemoryBackend", _FakeBackend)
    monkeypatch.setattr(pipeline, "OpenAIEmbeddingClient", _FakeEmbeddingClient)
    monkeypatch.setattr(pipeline, "OpenAIChatClient", _FakeChatClient)

    args = argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        source="longmemeval_s*",
        allow_nonstandard_shape=True,
        limit_samples=None,
        limit_questions=None,
        answer_base_url="http://127.0.0.1:8000/v1",
        answer_api_key="EMPTY",
        answer_model=pipeline.DEFAULT_ANSWER_MODEL,
        embedding_base_url="http://127.0.0.1:8001/v1",
        embedding_api_key="EMPTY",
        embedding_model=pipeline.DEFAULT_EMBEDDING_MODEL,
        embedding_batch_size=8,
        embedding_dim=2,
        request_timeout=10.0,
        dsn="postgresql://example",
        table="longmemeval_test",
        chunk_size=4096,
        chunk_tokenizer="gpt-4o-mini",
        prompt_token_budget=36_000,
        skip_judge=True,
        judge_base_url="http://127.0.0.1:8000/v1",
        judge_api_key="EMPTY",
        judge_model=pipeline.DEFAULT_ANSWER_MODEL,
        judge_max_tokens=10,
        top_k=10,
        max_prompt_memories=5,
        answer_max_tokens=32,
        temperature=0.0,
        seed=0,
    )
    summary = pipeline.run_pipeline(args)

    backend = _FakeBackend.instance
    assert backend is not None
    assert len(backend.replacements) == 1
    assert len(backend.searches) == 2
    assert backend.closed
    assert summary["status"] == "completed"
    assert summary["calls"]["paper_model_calls"] == 5
    assert summary["calls"]["by_kind"] == {
        "answer_generation": 2,
        "construction_embedding": 1,
        "query_embedding": 2,
        "tridb_insert": 1,
        "tridb_retrieval": 2,
    }
    assert (output_dir / "run_manifest.json").exists()
    assert len((output_dir / "predictions.jsonl").read_text().splitlines()) == 2
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert all(
        item["timing"]["effective_ttft_seconds"] <= item["timing"]["total_seconds"]
        for item in predictions
    )
