"""Frozen-input and runnable factory contracts; no live provider is contacted."""

import json
import os
from pathlib import Path

import pytest

from trimem.optimizer.runtime import math_ale_factory, validate_inputs
from trimem.optimizer.schema import CandidateSpec
from trimem.optimizer.snapshot import FrozenSnapshot, file_sha256


def snapshot_doc():
    return {
        "schema_version": 1,
        "embedding_dimension": 2,
        "embedding_provenance": {"source": "test fixture, not empirical evidence"},
        "attempts": [
            {
                "id": i,
                "task": "math:fixture",
                "parent_id": i - 1 if i > 1 else None,
                "fitness": float(i),
                "is_valid": True,
                "changes": "change " + str(i),
                # Every edit changes the code, so non-root attempts are real
                # improving edits and can be seeds (roots never are).
                "code": f"pass  # attempt {i}",
                "emb": [1.0, i / 400],
            }
            for i in range(1, 401)
        ],
        "edges": [{"src": i, "dst": i + 1, "type": "lineage"} for i in range(1, 400)],
        "queries": {"checkpoint": {"task": "math:fixture", "query_vector": [1.0, 0.0]}},
    }


@pytest.fixture
def config(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(snapshot_doc()))
    paths = {}
    for name, text in {
        "code": "value = -2\n",
        "spec": "Produce a scored program.",
        "evaluator": "def evaluate(path):\n    return {'combined_score': -1, 'is_valid': True}\n",
    }.items():
        path = tmp_path / (name + ".py")
        path.write_text(text)
        paths[name + "_path"] = str(path)
        paths[name + "_sha256"] = file_sha256(path)
    return {
        "application_mode": "first_decision_then_frozen_policy",
        "graph_policy": "frozen_no_ingest",
        "tokenizer_path": str(tmp_path),
        "trace_directory": str(tmp_path / "traces"),
        "snapshots": {
            "math:fixture": {"path": str(snapshot), "sha256": file_sha256(snapshot)}
        },
        "checkpoints": [
            {
                "task_id": "math:fixture",
                "checkpoint_id": "checkpoint",
                "initial_fitness": -2,
                **paths,
            }
        ],
        "seeds": [42],
        "protocol": {
            "name": "fixture",
            "model_version": "mock",
            "prompt_version": "v1",
            "tokenizer_version": "fixture-tokenizer",
            "token_budget": 500000,
            "horizon_kind": "fixed_steps",
            "max_steps": 2,
        },
        "generation": {
            "model": "mock",
            "base_url": "http://unused/v1",
            "model_context_window": 32768,
            "max_output_tokens": 8192,
        },
        "semantic": {
            "model": "mock",
            "base_url": "http://unused/v1",
            "model_context_window": 32768,
            "max_output_tokens": 16,
        },
        "scorer": {"timeout": 20, "cpu_threads": 1},
        "postgres": {"bindir": "/not-used", "cpuset": "16-19"},
    }


def test_frozen_inputs_hashes_and_relational_parent_integrity(config):
    checkpoints, snapshots = validate_inputs(config)
    assert checkpoints["math:fixture"].initial_fitness == -2
    path = snapshots["math:fixture"].path
    document = json.loads(path.read_text())
    document["attempts"][0]["parent_id"] = 99999
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="hash changed"):
        validate_inputs(config)
    with pytest.raises(ValueError, match="parent is missing"):
        FrozenSnapshot.read(path)


def test_real_launch_refuses_missing_frozen_evaluator_digest(config):
    del config["checkpoints"][0]["evaluator_sha256"]
    with pytest.raises(ValueError, match="digests"):
        validate_inputs(config)


def test_snapshot_public_views_cannot_mutate_a_frozen_graph(config):
    _, snapshots = validate_inputs(config)
    snapshot = snapshots["math:fixture"]
    view = snapshot.document
    view["attempts"][0]["code"] = "corrupted"
    view["queries"]["checkpoint"]["query_vector"][0] = 999
    assert snapshot.document["attempts"][0]["code"] == "pass  # attempt 1"
    assert snapshot.query("checkpoint").query_vector == (1.0, 0.0)


def fake_models(monkeypatch):
    from uuid import uuid4
    from trimem.optimizer.clients import ChatResponse

    class Tokenizer:
        tokenizer_id = "fixture-tokenizer"

        def __init__(self, directory):
            pass

        def encode(self, text, *, add_special_tokens=False):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode()

        def render(self, messages):
            return json.dumps(messages)

        def count_messages(self, messages):
            return len(self.encode(self.render(messages)))

    class Transport:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.model_version = self.model

        def close(self):
            pass

        def complete(self, messages, *, ledger, category="generation", **kwargs):
            incoming = self.prompt_token_count(messages)
            ledger.request_allowance(incoming, 8)
            request_id = uuid4().hex
            ledger.record(request_id, category, incoming, 8)
            answer = (
                '{"verdict": true}'
                if category == "semantic_filter"
                else (
                    '{"score": 8}'
                    if category == "semantic_rank"
                    else "```python\nvalue = -1\n```"
                )
            )
            return ChatResponse(answer, incoming, 8, (request_id,), "stop")

    monkeypatch.setattr("trimem.optimizer.tokenization.LocalChatTokenizer", Tokenizer)
    monkeypatch.setattr("trimem.optimizer.clients.OpenAICompatibleTransport", Transport)


def test_runtime_no_memory_does_not_start_postgres_and_runs_negative_fitness(
    config, monkeypatch
):
    fake_models(monkeypatch)

    def forbidden(**kwargs):
        raise AssertionError("no-memory must not start a DB")

    monkeypatch.setattr("trimem.optimizer.private_pg.PrivatePostgres", forbidden)
    with math_ale_factory(config) as evaluator:
        config["generation"]["max_output_tokens"] = 1
        config["checkpoints"][0]["initial_fitness"] = -999
        result = evaluator.evaluate(CandidateSpec("no_memory_v1"))
    assert result.complete_for_model
    assert result.quality_mean == -1
    assert result.units[0].rollout.steps == 2
    telemetry = result.units[0].rollout.telemetry
    assert telemetry["retrieval"]["receipt"]["actual_executions"] == 0
    assert len(result.units[0].ledger["events"]) == 2
    assert result.total_tokens_mean > 0
    assert len(list(Path(config["trace_directory"]).glob("*.json"))) == 1


@pytest.mark.skipif(
    not os.environ.get("TRIMEM_OPTIMIZER_TEST_PG_PREFIX"),
    reason="private PG gate opt-in",
)
@pytest.mark.parametrize(
    "plan", ["vgr_v1", "vgr_score_v1", "seed_sem_expand_v1", "expand_row_sem_v1"]
)
def test_runtime_private_native_worker_end_to_end(config, monkeypatch, plan):
    fake_models(monkeypatch)
    config["postgres"]["bindir"] = str(
        Path(os.environ["TRIMEM_OPTIMIZER_TEST_PG_PREFIX"])
        / "usr/lib/postgresql/16/bin"
    )
    with math_ale_factory(config) as evaluator:
        result = evaluator.evaluate(
            CandidateSpec(plan, seed_count=8, fanout=4, top_k=3)
        )
    assert result.complete_for_model, result.to_dict()
    receipt = result.units[0].rollout.telemetry["retrieval"]["receipt"]
    assert receipt["actual_executions"] == 1 and receipt["closed"]
    assert receipt["usage_complete"]
    events = result.units[0].ledger["events"]
    if plan != "vgr_v1":
        assert any(event["category"] == "semantic_rank" for event in events)
    if "sem" in plan:
        assert any(event["category"] == "semantic_filter" for event in events)
    assert sum(event["category"] == "generation" for event in events) == 2
