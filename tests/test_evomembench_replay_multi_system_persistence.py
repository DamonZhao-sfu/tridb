from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from bench.agent_memory.evomembench import replay_multi_system
from bench.agent_memory.evomembench.system_protocol import embedding_sha256


class _FakeStore:
    cleanup_calls = 0

    def __init__(self, config) -> None:
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def load_snapshot(self, snapshot):
        return {
            "snapshot_digest": snapshot.digest,
            "units": 1,
            "edges": 0,
            "namespace": self.config.namespace,
            "collection": self.config.collection,
        }

    def query(self, query, *, count_tokens):
        del count_tokens
        return SimpleNamespace(
            selected_ids=("experience-0",),
            injection="same injection",
            injection_tokens=2,
            latency_ms={"total": 1.0},
            intermediate={"final_results": 1},
            probes={"mode": "fixture"},
        )

    def cleanup_owned_namespace(self):
        type(self).cleanup_calls += 1
        return {"scope": "fixture"}


class _FakeTokenizer:
    def __init__(self, path) -> None:
        self.path = str(path)
        self.sha256 = "fixture-tokenizer"
        self.implementation = "fixture"

    def __call__(self, _text: str) -> int:
        return 2


def _args(tmp_path: Path, *, retain: bool) -> argparse.Namespace:
    source = tmp_path / "source"
    (source / "snapshots/full_gem").mkdir(parents=True)
    (source / "snapshots/full_gem/scope.json").write_text("{}\n")
    (source / "queries").mkdir()
    vector = [0.25, -0.5]
    query = {
        "query_id": "query-0",
        "scope_id": "scope-0",
        "cutoff_ordinal": 1,
        "task_embedding": vector,
        "task_embedding_sha256": embedding_sha256(vector),
        "k": 1,
        "m_seeds": 1,
        "hops": 2,
        "graph_work_budget": 65536,
        "graph_scoring": "membership",
        "token_budget": 32,
        "expected_ids": ["experience-0"],
        "expected_injection": "same injection",
    }
    (source / "queries/full_gem.jsonl").write_text(json.dumps(query) + "\n")
    (source / "run_receipt.json").write_text('{"status":"complete"}\n')
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n")
    return argparse.Namespace(
        know_run_dir=str(source),
        output_dir=str(tmp_path / "replay"),
        run_id="fixture-run",
        tokenizer_json=str(tokenizer),
        ann_overfetch=8,
        milvus_hnsw_m=16,
        milvus_ef_construction=200,
        milvus_ef=128,
        graph_work_budget=65536,
        timeout=30.0,
        formal=False,
        authorization="",
        retain_namespaces=retain,
    )


def _install_fakes(monkeypatch) -> None:
    snapshot = SimpleNamespace(scope_id="scope-0", digest="digest-0")
    monkeypatch.setattr(
        replay_multi_system.ExperienceSnapshot,
        "read",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        replay_multi_system, "LiveMultiSystemExperienceStore", _FakeStore
    )
    monkeypatch.setattr(replay_multi_system, "FrozenTokenizerCounter", _FakeTokenizer)
    _FakeStore.cleanup_calls = 0


def test_formal_replay_can_retain_parity_checked_namespaces(
    tmp_path, monkeypatch
) -> None:
    _install_fakes(monkeypatch)

    receipt = replay_multi_system.run(_args(tmp_path, retain=True))

    assert receipt["all_parity_passed"] is True
    assert receipt["namespaces_retained"] is True
    assert len(receipt["retained_namespaces"]) == 1
    assert receipt["load_metrics"][0]["cleanup"] == {
        "performed": False,
        "reason": "retained_by_protocol",
    }
    assert _FakeStore.cleanup_calls == 0


def test_default_replay_still_cleans_owned_namespace(tmp_path, monkeypatch) -> None:
    _install_fakes(monkeypatch)

    receipt = replay_multi_system.run(_args(tmp_path, retain=False))

    assert receipt["namespaces_retained"] is False
    assert receipt["retained_namespaces"] == []
    assert receipt["load_metrics"][0]["cleanup"] == {
        "performed": True,
        "scope": "fixture",
    }
    assert _FakeStore.cleanup_calls == 1
