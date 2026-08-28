import hashlib
from pathlib import Path

from experiments.evermemos_track_c.adapter import EverMemOSTrackCConfig
from bench.agent_memory.table5_track_c.dataset import LoCoMoCorpus
from experiments.evermemos_track_c.paper_era_adapter import (
    EverMemOSPaperAdapter,
    EverMemOSPaperConfig,
)


ROOT = Path(__file__).resolve().parents[1]
ANSWER_MODEL = "Qwen/Qwen3-32B"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _env(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def test_evermemos_entry_points_default_to_unified_qwen32() -> None:
    current = EverMemOSTrackCConfig(dataset_path="locomo.json")
    paper = EverMemOSPaperConfig(dataset_path="locomo.json")
    for config in (current, paper):
        assert config.answer_model == ANSWER_MODEL
        assert config.answer_base_url == "http://127.0.0.1:8000/v1"
        assert config.embedding_model == EMBEDDING_MODEL
        assert config.embedding_base_url == "http://127.0.0.1:8001/v1"


def test_evermemos_finalize_drains_ome_before_cascade(monkeypatch) -> None:
    adapter = __import__(
        "experiments.evermemos_track_c.adapter", fromlist=["EverMemOSTrackCAdapter"]
    ).EverMemOSTrackCAdapter(
        EverMemOSTrackCConfig(
            dataset_path="locomo.json",
            drain_timeout=30,
            drain_poll_seconds=0,
        )
    )
    calls: list[str] = []
    ome_responses = iter(
        [
            {"status": "ok", "idle": True, "active_runs": 0},
            {"status": "ok", "idle": True, "active_runs": 0},
        ]
    )

    def fake_ome_drain(_timeout: float) -> dict[str, object]:
        calls.append("ome")
        return next(ome_responses)

    def fake_health() -> dict[str, object]:
        calls.append("cascade")
        return {"cascade": {"pending": 0, "failed_retryable": 0}}

    monkeypatch.setattr(adapter, "_ome_drain", fake_ome_drain)
    monkeypatch.setattr(adapter, "_health", fake_health)
    result = adapter.finalize_build()
    adapter.close()

    assert calls == ["ome", "cascade", "cascade", "ome"]
    assert result["converged"] is True
    assert result["ome"]["idle"] is True
    assert result["final_ome"]["active_runs"] == 0


def test_paper_era_qwen32_environment_matches_adapter_contract() -> None:
    values = _env(ROOT / "experiments/evermemos_track_c/paper_era_qwen32.env")
    assert values["LLM_MODEL"] == ANSWER_MODEL
    assert values["LLM_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert values["VECTORIZE_MODEL"] == EMBEDDING_MODEL
    assert values["VECTORIZE_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert values["LLM_MAX_TOKENS"] == "16000"
    assert values["VECTORIZE_MAX_RETRIES"] == "1"


def test_serial_driver_pins_revisions_and_forbids_system_overlap() -> None:
    script = (ROOT / "scripts/expc_graphiti_evermemos_qwen32_serial.sh").read_text(
        encoding="utf-8"
    )
    assert "answer_model=qwen3.8" not in script.lower()
    assert "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df" in script
    assert "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" in script
    assert "assert_no_evermemos_workload" in script
    assert "assert_no_graphiti_workload" in script
    assert "ever_llm_max_tokens < answer_max_model_len" in script
    assert "ever_vectorize_max_retries" in script
    assert "evermemos_configuration" in script
    assert "EXPC_TARGET_GPUS:-$target_gpu" in script
    assert "EXPC_TENSOR_PARALLEL_SIZE:-1" in script
    assert "target_gpus:" in script
    assert "tensor_parallel_size:" in script
    assert "EXPC_EVER_REUSE_BUILD_RECEIPT" in script
    assert "EXPC_EVER_REUSE_CANONICAL_STATE" in script
    assert "reusing completed EverMemOS build" in script
    main = script.rsplit('cd "$repo"', maxsplit=1)[1]
    assert main.index("run_graphiti_conformance") < main.index("run_evermemos")


def test_gpu1_profile_shares_only_the_exact_embedding_service() -> None:
    env_values = _env(
        ROOT / "experiments/evermemos_track_c/paper_era_qwen32_gpu1_shared.env"
    )
    assert env_values["LLM_MODEL"] == ANSWER_MODEL
    assert env_values["LLM_BASE_URL"] == "http://127.0.0.1:8010/v1"
    assert env_values["VECTORIZE_MODEL"] == EMBEDDING_MODEL
    assert env_values["VECTORIZE_BASE_URL"] == "http://127.0.0.1:8011/v1"
    assert env_values["LLM_MAX_TOKENS"] == "16000"
    assert env_values["VECTORIZE_MAX_RETRIES"] == "1"

    wrapper = (
        ROOT / "scripts/expc_graphiti_evermemos_qwen32_gpu1_shared_v2.sh"
    ).read_text(encoding="utf-8")
    assert "EXPC_TARGET_GPU=1" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_EMBEDDING_UNIT=tridb-expc-embed-gpu1.service" in wrapper
    assert "EXPC_CONCURRENT_BACKGROUND=true" in wrapper
    assert "GPU0" in wrapper


def test_evermemos_only_profile_skips_graphiti_and_uses_fresh_v3_state() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_gpu1_shared_v3.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_TARGET_GPU=1" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_STATE_VERSION=v3" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v3" in wrapper
    assert "GPU0" in wrapper


def test_evermemos_v4_continuation_uses_another_fresh_namespace() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_gpu1_shared_v4.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_STATE_VERSION=v4" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v4" in wrapper
    assert "EXPC_OUTPUT_ROOT=" in wrapper and "_v4" in wrapper


def test_evermemos_v5_uses_fixed_readiness_and_another_fresh_namespace() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_gpu1_shared_v5.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_SKIP_EVERMEMOS=0" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_STATE_VERSION=ever_v5" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v5" in wrapper
    assert "EXPC_OUTPUT_ROOT=" in wrapper and "_v5" in wrapper
    assert "GPU0" in wrapper


def test_evermemos_v6_retries_with_bounded_completion_budget() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_gpu1_shared_v6.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_SKIP_EVERMEMOS=0" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_STATE_VERSION=ever_v6" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v6" in wrapper
    assert "EXPC_OUTPUT_ROOT=" in wrapper and "_v6" in wrapper
    assert "GPU0" in wrapper


def test_evermemos_v7_retries_with_one_vectorization_attempt() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_gpu1_shared_v7.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_SKIP_EVERMEMOS=0" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_STATE_VERSION=ever_v7" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v7" in wrapper
    assert "EXPC_OUTPUT_ROOT=" in wrapper and "_v7" in wrapper
    assert "GPU0" in wrapper


def test_evermemos_v8_uses_two_gpu_tensor_parallel_qwen32() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_tp2_gpu01_v8.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_TARGET_GPU=1" in wrapper
    assert "EXPC_TARGET_GPUS=0,1" in wrapper
    assert "EXPC_TENSOR_PARALLEL_SIZE=2" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper
    assert "EXPC_SKIP_GRAPHITI=1" in wrapper
    assert "EXPC_STATE_VERSION=ever_v8" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v8" in wrapper
    assert "qwen32_tp2_gpu01" in wrapper


def test_evermemos_v9_reuses_completed_v8_canonical_state() -> None:
    wrapper = (ROOT / "scripts/expc_evermemos_qwen32_tp2_gpu01_resume_v9.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_TARGET_GPUS=0,1" in wrapper
    assert "EXPC_TENSOR_PARALLEL_SIZE=2" in wrapper
    assert "EXPC_EVER_REUSE_BUILD_RECEIPT=" in wrapper
    assert "/evermemos/build/build_receipt.json" in wrapper
    assert "EXPC_EVER_REUSE_CANONICAL_STATE=" in wrapper
    assert "/canonical_v8" in wrapper
    assert "EXPC_EVER_STATE_SUFFIX=v9" in wrapper
    assert "2026_08_23_v9" in wrapper


def test_graphiti_v5_profile_skips_evermemos_but_keeps_conformance() -> None:
    wrapper = (ROOT / "scripts/expc_graphiti_qwen32_gpu1_shared_v5.sh").read_text(
        encoding="utf-8"
    )
    assert "EXPC_SKIP_GRAPHITI=0" in wrapper
    assert "EXPC_SKIP_EVERMEMOS=1" in wrapper
    assert "EXPC_STATE_VERSION=v5" in wrapper
    assert "EXPC_MANAGE_EMBEDDING=0" in wrapper

    queue = (ROOT / "scripts/expc_graphiti_after_evermemos_gpu1_v5.sh").read_text(
        encoding="utf-8"
    )
    assert "while systemctl" in queue
    assert "EverMemOS workload is active" in queue
    assert "EverMemOS containers remain" in queue
    assert "expc_graphiti_qwen32_gpu1_shared_v5.sh" in queue


def test_paper_backend_uses_real_milvus_grpc_readiness() -> None:
    script = (ROOT / "scripts/table5_track_c_evermemos_paper_host.sh").read_text(
        encoding="utf-8"
    )
    assert "utility.list_collections" in script
    assert "wait_milvus_grpc 150" in script
    assert "wait_http milvus http://127.0.0.1:29091/healthz" not in script


def test_paper_adapter_rebinds_groups_after_process_restart(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text("frozen dataset fixture\n", encoding="utf-8")
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    corpus = LoCoMoCorpus(
        path=dataset,
        sha256=checksum,
        events_by_sample={"conv-a": (), "conv-b": ()},
        queries_by_sample={"conv-a": (), "conv-b": ()},
    )
    adapter = EverMemOSPaperAdapter(EverMemOSPaperConfig(dataset_path=str(dataset)))
    counts = {"conv-a": 3, "conv-b": 5}
    monkeypatch.setattr(
        adapter,
        "_search_response",
        lambda item, _top_k: {"result": {"total_count": counts[item.sample_id]}},
    )
    monkeypatch.setattr(
        adapter,
        "_source_identity",
        lambda: {"root": "/source", "commit": "a" * 40, "dirty": False},
    )

    assert adapter._groups == set()
    adapter.prepare_reused_build(corpus)
    fingerprint = adapter.snapshot_fingerprint()

    assert fingerprint["group_counts"] == counts
    assert fingerprint["total_count"] == 8
    adapter.close()


def test_paper_era_prepared_add_bypasses_extraction_and_llm() -> None:
    server = (
        ROOT / "experiments/evermemos_track_c/prepared_item_server.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT / "experiments/evermemos_track_c/paper_era_adapter.py"
    ).read_text(encoding="utf-8")

    assert "save_memory_docs(" in server
    assert "EpisodeMemory(" in server
    assert "vectorize_service.get_embedding" in server
    assert "mem_reader" not in server
    assert "memory_extractor" not in server
    assert "llm.generate" not in server
    assert '"llm_call_count": 0' in server
    assert '"/api/v1/benchmark/prepared-items"' in server

    assert 'f"{self._api}/benchmark/prepared-items"' in adapter
    assert '"POST",\n                "/memories"' in adapter  # canonical construction remains separate
    assert "prepared_memory_item_insertion_v1" in adapter


def test_paper_era_add_does_not_require_answer_endpoint() -> None:
    cli = (ROOT / "experiments/evermemos_track_c/paper_era_cli.py").read_text(
        encoding="utf-8"
    )
    assert 'require_answer_endpoint=args.command != "add"' in cli
