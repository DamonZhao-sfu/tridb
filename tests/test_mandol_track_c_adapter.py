from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.agent_memory.table5_track_c.dataset import EventItem
from experiments.mandol_track_c.adapter import MandolAdapter, MandolConfig


def _event(sample_id: str, event_id: str) -> EventItem:
    return EventItem(
        sample_id=sample_id,
        event_id=event_id,
        session_id="S1",
        timestamp="2023-05-08",
        role="speaker",
        text="memory",
        ordinal=1,
        metadata={"session_number": 1},
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_sample(root: Path, sample_id: str, event_ids: list[str]) -> Path:
    sample = root / sample_id
    units = [
        {
            "uid": f"{sample_id}_dialogue_{event_id}",
            "raw_data": {"type": "dialogue", "dia_id": event_id},
            "metadata": {"sample_id": sample_id},
        }
        for event_id in event_ids
    ]
    _write_json(
        sample / "manifest.json",
        {
            "version": "1.0",
            "build_state": "complete",
            "pending_session_ids": [],
            "stats": {
                "unit_count": len(units),
                "space_count": 2,
                "edge_count": 1,
            },
        },
    )
    _write_json(
        sample / "config.json",
        {
            "config": {
                "root": sample_id,
                "memory_system_config": {
                    "embedder_model": "Qwen/Qwen3-Embedding-0.6B",
                    "embedder_dim": 1024,
                    "reranker_model": "BAAI/bge-reranker-v2-m3",
                    "llm_model": "Qwen/Qwen3-32B",
                },
            }
        },
    )
    _write_json(sample / "data/units.json", {"units": units})
    for relative in (
        "data/spaces.json",
        "data/graph.json",
        "data/sessions.json",
        "state/processed_state.json",
    ):
        _write_json(sample / relative, {})
    return sample


def _adapter(tmp_path: Path) -> MandolAdapter:
    return MandolAdapter(
        MandolConfig(
            snapshot_dir=str(tmp_path / "target"),
            dataset_path=str(tmp_path / "locomo.json"),
        )
    )


def test_resume_sample_requires_complete_exact_locomo_events(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    sample = _complete_sample(tmp_path / "source", "conv-26", ["D1:1", "D1:2"])
    events = [_event("conv-26", "D1:1"), _event("conv-26", "D1:2")]

    validation = adapter._validate_resume_sample(sample, "conv-26", events)

    assert validation["base_units"] == 2
    assert validation["source_files"] == 7
    assert len(validation["source_sha256"]) == 64

    with pytest.raises(RuntimeError, match="does not match LoCoMo events"):
        adapter._validate_resume_sample(
            sample, "conv-26", [*events, _event("conv-26", "D1:3")]
        )


def test_resume_sample_rejects_pending_high_level_build(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    sample = _complete_sample(tmp_path / "source", "conv-26", ["D1:1"])
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pending_session_ids"] = ["session-1"]
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="has pending sessions"):
        adapter._validate_resume_sample(sample, "conv-26", [_event("conv-26", "D1:1")])


def test_resume_ingest_skips_reused_samples_and_builds_only_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = MandolAdapter(
        MandolConfig(
            snapshot_dir=str(tmp_path / "target"),
            dataset_path=str(tmp_path / "locomo.json"),
            resume_snapshot_source=str(tmp_path / "source"),
            resume_expected_samples=("conv-26",),
        )
    )
    reused_report = {
        "origin": "verified_partial_snapshot",
        "counts": {"units": 1, "spaces": 1, "edges": 0},
    }
    monkeypatch.setattr(
        adapter,
        "_seed_resume_snapshot",
        lambda grouped: (
            {"conv-26": reused_report},
            {
                "policy": "copy validated complete samples; build missing samples",
                "source": str(tmp_path / "source"),
                "reused_samples": ["conv-26"],
            },
        ),
    )
    created: list[str] = []

    class FakeSystem:
        def build_high_level(self, *, mode: str) -> SimpleNamespace:
            assert mode == "auto"
            return SimpleNamespace(status="completed", error_message=None)

        def save(self, path: str) -> dict[str, str]:
            return {"path": path}

    def new_system(sample_id: str) -> FakeSystem:
        created.append(sample_id)
        return FakeSystem()

    monkeypatch.setattr(adapter, "_new_system", new_system)
    monkeypatch.setattr(
        adapter,
        "_write_events",
        lambda system, events: {"base_units": len(events)},
    )
    monkeypatch.setattr(
        adapter,
        "_system_counts",
        lambda system: {"units": 1, "spaces": 1, "edges": 0},
    )

    result = adapter.ingest_history(
        [_event("conv-26", "D1:1"), _event("conv-30", "D1:1")]
    )

    assert created == ["conv-30"]
    assert sorted(result["sample_reports"]) == ["conv-26", "conv-30"]
    assert result["resume"]["reused_samples"] == ["conv-26"]
    assert result["resume"]["built_samples"] == ["conv-30"]


def test_v8_driver_resumes_six_then_gates_qps_on_complete_ten() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (
        repository / "scripts/table5_track_c_mandol_search_resume_gpu1_v8.sh"
    ).read_text(encoding="utf-8")
    manifest = (
        repository
        / "bench/agent_memory/table5_track_c/manifests/search_mandol_qps_1_5_10_v8.json"
    ).read_text(encoding="utf-8")

    assert "snapshot_version=${TRACKC_MANDOL_SNAPSHOT_VERSION:-sw15v4}" in driver
    assert 'export TRACKC_SWEEP_VERSION="$snapshot_version"' in driver
    assert "--resume-snapshot-source" in driver
    for sample_id in ("conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44"):
        assert sample_id in driver
    for sample_id in ("conv-47", "conv-48", "conv-49", "conv-50"):
        assert sample_id in driver
    build_at = driver.index("experiments.mandol_track_c.cli build-search")
    ten_sample_gate_at = driver.index("sample provenance gate failed")
    sweep_at = driver.index("bench.agent_memory.table5_track_c.sweep")
    assert build_at < ten_sample_gate_at < sweep_at
    assert "chmod -R a-w" in driver
    assert "CUDA_VISIBLE_DEVICES=1" in driver
    assert '"qps": [1, 5, 10]' in manifest
    assert '"systems": ["mandol"]' in manifest
