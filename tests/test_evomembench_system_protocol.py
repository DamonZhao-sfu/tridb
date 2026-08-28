from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from bench.agent_memory.evomembench import answer_server_admission
from bench.agent_memory.evomembench import embedding_server_admission
from bench.agent_memory.evomembench.cpu_embedding_server import (
    EMBEDDING_DIMENSION,
    MAX_MODEL_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    _validate_inputs,
)

from bench.agent_memory.evomembench.crossep_tool_backend import ToolSampleContext
from bench.agent_memory.evomembench.evaluate_know import seal_existing
from bench.agent_memory.evomembench.gpu_startup_admission import vllm_frontends
from bench.agent_memory.evomembench.finalize_system_artifacts import finalize
from bench.agent_memory.evomembench.long_context import (
    CrossEpToolLongContextMemory,
    LongContextOverflow,
    LongContextSnapshot,
    knowledge_history_item,
    trajectory_history_item,
)
from bench.agent_memory.evomembench.dataset import EvoEpisode, Message
from bench.agent_memory.evomembench.multi_system import (
    MultiSystemConfig,
    _bounded_graph_bfs,
    _milvus_cosine_distance,
)
from bench.agent_memory.evomembench.system_protocol import (
    FORMAL_AUTHORIZATION,
    FrozenTokenizerCounter,
    HistoryItem,
    SystemTrace,
    append_history_to_messages,
    balanced_arm_order,
    compare_parity,
    materialize_long_context,
    require_formal_authorization,
    require_parity,
    verify_trace,
)
from bench.agent_memory.evomembench.validate_calibration_handoff import (
    validate as validate_calibration_handoff,
)
from bench.agent_memory.evomembench.summarize_system_experiment import _track
from bench.agent_memory.evomembench.summarize_system_scale import (
    summarize as summarize_system_scale,
)
from bench.agent_memory.evomembench.summarize_know_quality import (
    summarize as summarize_know_quality,
)
from bench.agent_memory.evomembench.system_snapshot import (
    ExperienceSnapshot,
    SnapshotEdge,
    SnapshotUnit,
)
from bench.agent_memory.evomembench.vllm_metrics import (
    attributable_delta,
    parse_metrics,
)
from bench.agent_memory.evomembench.system_material_passport import (
    _answer_server_gate,
    _database_concurrency_gate,
    _database_preparation_gate,
    _embedding_server_gate,
    _dual_gpu_gate,
    _empirical_claim_gates,
    _formal_count_gate,
    _tool_canonical_source_gate,
    _update_trace_gate,
)
from bench.agent_memory.evomembench.scale_snapshot import (
    SCALE_SNAPSHOT_SCHEMA_VERSION,
    SQLiteExperienceSnapshot,
    _float32_blob,
)


class WordCodec:
    def encode(self, value: str) -> list[str]:
        return value.split()

    def decode(self, values: list[str]) -> str:
        return " ".join(values)

    def __call__(self, value: str) -> int:
        return len(self.encode(value))


def _history() -> list[HistoryItem]:
    return [
        HistoryItem("episode-0", 0, "first prior answer"),
        HistoryItem("episode-1", 1, "second prior answer"),
    ]


def _tool_context(ordinal: int = 2) -> ToolSampleContext:
    return ToolSampleContext(
        sample_id=f"sample-{ordinal}",
        episode_uid=f"episode-{ordinal}",
        ordinal=ordinal,
        environment="travel_api",
        question=[[{"role": "user", "content": "book a flight"}]],
        involved_classes=["TravelAPI"],
        allowed_functions=["book_flight"],
    )


def test_long_context_is_complete_ordered_and_never_truncated() -> None:
    codec = WordCodec()
    ready = materialize_long_context(
        reversed(_history()),
        count_tokens=codec,
        base_prompt_tokens=10,
        context_window_tokens=100,
        reserved_generation_tokens=10,
        safety_tokens=0,
    )
    assert ready.episode_uids == ("episode-0", "episode-1")
    assert ready.text.index("first prior") < ready.text.index("second prior")
    assert not ready.overflow

    overflow = materialize_long_context(
        _history(),
        count_tokens=codec,
        base_prompt_tokens=10,
        context_window_tokens=15,
        reserved_generation_tokens=10,
        safety_tokens=0,
    )
    assert overflow.overflow
    assert overflow.text == ready.text
    assert overflow.history_tokens == ready.history_tokens


def test_long_context_message_append_preserves_base_prompt() -> None:
    messages = [
        {"role": "system", "content": "base safety policy"},
        {"role": "user", "content": "current task"},
    ]
    observed = append_history_to_messages(messages, "prior history")
    assert observed[0]["content"] == "base safety policy\n\nprior history"
    assert messages[0]["content"] == "base safety policy"


def test_know_long_context_keeps_raw_messages_and_response_but_not_rubrics() -> None:
    episode = EvoEpisode(
        context_uid="context",
        episode_uid="episode-0",
        source_task_id="task-0",
        ordinal=0,
        category="knowledge",
        subcategory="rules",
        messages=(
            Message(role="system", content="full background"),
            Message(role="user", content="current raw task"),
        ),
        rubrics=("withheld grading condition",),
        prior_episode_uids=(),
    )
    item = knowledge_history_item(episode, "observed response")
    assert "full background" in item.text
    assert "current raw task" in item.text
    assert "observed response" in item.text
    assert "withheld grading condition" not in item.text


def test_parity_gate_requires_set_order_and_injection_identity() -> None:
    passed = compare_parity(
        expected_ids=["a", "b"],
        observed_ids=["a", "b"],
        expected_injection="same",
        observed_injection="same",
    )
    require_parity(passed)
    failed = compare_parity(
        expected_ids=["a", "b"],
        observed_ids=["b", "a"],
        expected_injection="same",
        observed_injection="same",
    )
    assert failed.set_parity and not failed.order_parity
    with pytest.raises(RuntimeError, match="parity gate failed"):
        require_parity(failed)


def test_frozen_qwen2_counter_matches_declared_pretokenizer(tmp_path) -> None:
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import BPE

    tokenizer = Tokenizer(
        BPE(
            vocab={"<unk>": 0, "d": 1, "1": 2, "d1": 3},
            merges=[("d", "1")],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "add_prefix_space": False})
    )

    counter = FrozenTokenizerCounter(str(tokenizer_path))

    assert counter("d1") == 2
    assert counter.implementation == "qwen2_tokenizer_config"


def test_system_trace_is_content_addressed_and_rejects_unknown_arm() -> None:
    trace = SystemTrace(
        run_id="r",
        track="CrossEp-Know",
        arm="full_gem",
        target_id="target",
        scope_id="scope",
        history_size=2,
        status="complete",
        latency_ms={"retrieval": 1.5},
        tokens={"answer_prompt": 10},
        intermediate={"peak_live_tuples": 4},
        selected_ids=("episode-0",),
    ).as_dict()
    assert verify_trace(trace)
    trace["tokens"]["answer_prompt"] = 11
    assert not verify_trace(trace)
    with pytest.raises(ValueError, match="unknown"):
        SystemTrace(
            run_id="r",
            track="x",
            arm="vector_only",
            target_id="t",
            scope_id="s",
            history_size=0,
            status="complete",
        )


def test_experience_snapshot_roundtrip_is_digest_checked(tmp_path) -> None:
    units = [
        SnapshotUnit(
            uid="experience:e0",
            scope_id="scope",
            node_kind="experience",
            ordinal=0,
            state="active",
            summary="task",
            payload="answer",
            embedding=(1.0, 0.0),
            metadata={"validity_state": "active"},
        ),
        SnapshotUnit(
            uid="feature:f0",
            scope_id="scope",
            node_kind="feature",
            ordinal=None,
            state="active",
            summary="skill",
            payload="skill",
            embedding=(0.0, 1.0),
            metadata={},
        ),
    ]
    snapshot = ExperienceSnapshot.build(
        scope_id="scope",
        units=units,
        edges=[SnapshotEdge("experience:e0", "feature:f0", "USES_SKILL", 1.0)],
    )
    path = tmp_path / "snapshot.json"
    snapshot.write(path)
    assert ExperienceSnapshot.read(path) == snapshot
    payload = json.loads(path.read_text())
    payload["units"][0]["payload"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        ExperienceSnapshot.read(path)


def test_tool_long_context_snapshot_is_readonly_and_strips_recursive_system() -> None:
    context = _tool_context(0)
    item = trajectory_history_item(
        context,
        [
            {"role": "system", "content": "injected old history"},
            {"role": "assistant", "content": "new answer"},
        ],
    )
    assert "new answer" in item.text
    assert "injected old history" not in item.text
    snapshot = LongContextSnapshot.build(
        source_environment="travel_api", records=[item]
    )
    backend = CrossEpToolLongContextMemory(
        source_environment="travel_api",
        count_tokens=WordCodec(),
        context_window_tokens=1000,
        reserved_generation_tokens=10,
        readonly=True,
        snapshot=snapshot,
    )
    backend.set_sample_context(_tool_context(1))
    backend.set_base_prompt_tokens(10)
    backend.begin_sample()
    injection = backend.utilize("ignored")
    backend.update([{"role": "assistant", "content": "must not append"}])
    assert "new answer" in injection
    assert backend.freeze().digest == snapshot.digest
    assert backend.receipts[0]["status"] == "ready"


def test_tool_long_context_overflow_fails_closed() -> None:
    snapshot = LongContextSnapshot.build(
        source_environment="travel_api", records=_history()
    )
    backend = CrossEpToolLongContextMemory(
        source_environment="travel_api",
        count_tokens=WordCodec(),
        context_window_tokens=10,
        reserved_generation_tokens=5,
        readonly=True,
        snapshot=snapshot,
        safety_tokens=0,
    )
    backend.set_sample_context(_tool_context(2))
    backend.set_base_prompt_tokens(5)
    backend.begin_sample()
    with pytest.raises(LongContextOverflow):
        backend.utilize("ignored")
    assert backend.receipts[0]["status"] == "context_overflow"


def test_multi_system_namespace_is_fail_closed() -> None:
    assert MultiSystemConfig(namespace="formal_001").collection == "evomem_formal_001"
    with pytest.raises(ValueError, match="namespace"):
        MultiSystemConfig(namespace="Unsafe-Name")


def test_multi_system_bfs_bounds_transferred_edges_across_hops() -> None:
    adjacency = {
        "a": ["b", "c"],
        "b": ["d", "a"],
        "c": ["e"],
    }

    class Session:
        def run(self, _query: str, **params: object) -> list[dict[str, str]]:
            rows = [
                {"src_uid": src, "dst_uid": dst}
                for src in params["frontier"]
                for dst in adjacency.get(src, [])
            ]
            return rows[: int(params["remaining"])]

    observed = _bounded_graph_bfs(
        Session(), namespace="formal_001", seeds=["a"], hops=2, edge_budget=3
    )
    assert observed.candidates == (("a", 0), ("b", 1), ("c", 1), ("d", 2))
    assert observed.edge_rows_transferred == 3
    assert observed.round_trips == 2
    assert observed.peak_frontier_ids == 2
    assert observed.work_budget_reached


def test_milvus_cosine_similarity_is_converted_to_pgvector_distance() -> None:
    assert _milvus_cosine_distance(1.0) == pytest.approx(0.0)
    assert _milvus_cosine_distance(0.25) == pytest.approx(0.75)
    assert _milvus_cosine_distance(-1.0) == pytest.approx(2.0)


def test_dual_gpu_gate_requires_continuous_clean_ownership(tmp_path: Path) -> None:
    (tmp_path / "gpu_layout_admission.log").write_text(
        "two-GPU ownership gate passed: mode=dual\n"
    )
    (tmp_path / "gpu_layout_continuous.log").write_text(
        "2026-08-25T13:00:00-07:00 exit_status=0 two-GPU ownership gate passed\n"
    )
    clean = _dual_gpu_gate(tmp_path, required=True)
    assert clean["passed"] is True
    assert clean["admission_samples"] == 1
    assert clean["continuous_successful_samples"] == 1

    (tmp_path / "gpu_layout_violation.txt").write_text(
        "exit_status=69\ndetail=unexpected GPU 1 compute PID\n"
    )
    contaminated = _dual_gpu_gate(tmp_path, required=True)
    assert contaminated["passed"] is False
    assert contaminated["violation_detected"] is True


def test_database_gate_requires_continuous_clean_observations(tmp_path: Path) -> None:
    assert _database_concurrency_gate(tmp_path, required=True)["passed"] is False
    (tmp_path / "database_concurrency_continuous.log").write_text(
        '2026-08-25T13:00:00-07:00 exit_status=0 {"passed":true}\n'
    )
    assert _database_concurrency_gate(tmp_path, required=True)["passed"] is True
    (tmp_path / "database_concurrency_violation.txt").write_text("exit_status=69\n")
    assert _database_concurrency_gate(tmp_path, required=True)["passed"] is False


def test_vllm_metrics_delta_requires_exact_request_attribution() -> None:
    def payload(offset: float, count: int) -> str:
        return "\n".join(
            f'vllm:{metric}_{suffix}{{engine="0",model_name="qwen3.8"}} {value}'
            for metric in (
                "time_to_first_token_seconds",
                "request_prefill_time_seconds",
                "request_decode_time_seconds",
                "e2e_request_latency_seconds",
            )
            for suffix, value in (("sum", offset), ("count", count))
        )

    before = parse_metrics(payload(10.0, 5), model="qwen3.8", collection_seconds=0.01)
    after = parse_metrics(payload(11.5, 6), model="qwen3.8", collection_seconds=0.02)
    delta = attributable_delta(before, after, expected_requests=1)
    assert delta["attributable"] is True
    assert delta["request_prefill_time_ms"] == pytest.approx(1500.0)
    assert delta["telemetry_overhead_ms"] == pytest.approx(30.0)

    concurrent = attributable_delta(
        before,
        parse_metrics(payload(12.0, 7), model="qwen3.8"),
        expected_requests=1,
    )
    assert concurrent["attributable"] is False


def test_formal_count_gate_rejects_any_reduced_workload() -> None:
    load = {
        "units": 1,
        "cleanup": {
            "scope": "exact_run_owned_namespace",
            "collection_dropped": True,
            "neo4j_nodes_deleted": 1,
            "postgresql_rows_deleted": 1,
        },
    }
    receipts = {
        "know": {
            "predictions": 2652,
            "cross_episode_decisions_per_arm": 764,
            "database_footprint_after": {"bytes": 1},
        },
        "know_multi_system": {"queries": 764, "load_metrics": [load] * 120},
        "tool": {
            "source_building_evaluations_per_memory_arm": 200,
            "target_evaluations_per_arm": 600,
            "directed_transfer_pairs": 12,
            "database_footprint_before": {"bytes": 1},
            "database_footprint_after": {"bytes": 2},
        },
        "tool_multi_system": {"queries": 600, "load_metrics": [load] * 4},
        "scale": {
            "history_sizes": [1000, 10000, 100000, 1000000],
            "queries": 100,
            "warmups": 10,
            "measured_repetitions": 30,
            "points": [{"multi_system_load": load}] * 4,
        },
    }
    assert _formal_count_gate(receipts)["passed"] is True
    receipts["tool"]["target_evaluations_per_arm"] = 599
    assert _formal_count_gate(receipts)["passed"] is False


def test_tool_canonical_source_gate_checks_every_construction_update(tmp_path) -> None:
    environments = {
        f"env-{environment}": {
            f"source-{environment}-{index}": f"sha-{environment}-{index}"
            for index in range(50)
        }
        for environment in range(4)
    }
    payload = {
        "schema_version": "evomembench_tool_canonical_source_v0.1.0",
        "environments": environments,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    payload["digest"] = digest
    tool = tmp_path / "tool"
    (tool / "updates").mkdir(parents=True)
    (tool / "canonical_source_trajectories.json").write_text(json.dumps(payload))
    rows = [
        {
            "source_environment": environment,
            "source_id": source_id,
            "canonical_trajectory_sha256": sha,
            "status": "committed",
        }
        for environment, sources in environments.items()
        for source_id, sha in sources.items()
    ]
    for arm in ("long_context", "gem_fused"):
        (tool / "updates" / f"{arm}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    receipt = {
        "canonical_source_trajectory_digest": digest,
        "source_building_errors": {
            f"{arm}:env-{environment}": 0
            for arm in ("memory_off", "long_context", "gem_fused")
            for environment in range(4)
        },
    }
    assert _tool_canonical_source_gate(tmp_path, receipt)["passed"] is True
    bad = json.loads((tool / "updates/gem_fused.jsonl").read_text().splitlines()[0])
    bad["canonical_trajectory_sha256"] = "wrong"
    remaining = (tool / "updates/gem_fused.jsonl").read_text().splitlines()[1:]
    (tool / "updates/gem_fused.jsonl").write_text(
        json.dumps(bad) + "\n" + "\n".join(remaining) + "\n"
    )
    assert _tool_canonical_source_gate(tmp_path, receipt)["passed"] is False


def test_update_trace_gate_requires_frozen_formal_counts(tmp_path) -> None:
    expected = {
        "know/updates/long_context.jsonl": 884,
        "know/updates/full_gem.jsonl": 884,
        "tool/updates/long_context.jsonl": 200,
        "tool/updates/gem_fused.jsonl": 200,
        "scale/per_update.jsonl": 1,
    }
    for relative, count in expected.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" * count)
    assert _update_trace_gate(tmp_path)["passed"] is True
    (tmp_path / "tool/updates/gem_fused.jsonl").write_text("{}\n" * 199)
    assert _update_trace_gate(tmp_path)["passed"] is False


def test_database_preparation_gate_requires_exact_probe_surface(tmp_path) -> None:
    functions = {
        name: "/tmp/tjs_pg"
        for name in (
            "tjs_open",
            "tjs_open_candidates_examined",
            "tjs_open_relational_examined",
            "tjs_open_relational_passed",
            "tjs_open_graph_examined",
            "tjs_open_graph_reached",
            "tjs_open_graph_censored",
            "tjs_open_termination_reason",
            "tjs_open_budget_capped",
            "tjs_open_bridges_injected",
        )
    }
    receipt = {
        "status": "complete",
        "tjs_library": {"path": "/tmp/tjs_pg.so", "sha256": "a" * 64},
        "databases": [
            {"label": label, "gem_units": 0, "tjs_functions": functions}
            for label in ("native", "scale")
        ],
    }
    (tmp_path / "database_preparation.json").write_text(json.dumps(receipt))
    assert _database_preparation_gate(tmp_path, local_library_required=True)["passed"]
    del receipt["databases"][0]["tjs_functions"]["tjs_open_relational_passed"]
    (tmp_path / "database_preparation.json").write_text(json.dumps(receipt))
    assert not _database_preparation_gate(tmp_path, local_library_required=True)[
        "passed"
    ]


def test_empirical_claim_gate_cannot_be_replaced_by_protocol_completion() -> None:
    summary = {
        "claim_gates": {
            "1_full_gem_multi_system_parity_100pct": True,
            "2_know_memory_utility": {"memory_utility_pass": True},
            "3_tool_memory_utility": {"memory_utility_pass": True},
            "4_full_gem_long_context_quality_noninferiority": {
                "know": True,
                "tool": True,
            },
            "5_latency": {
                "native": {"a": {"passed": True}},
                "systems_scale": [
                    {"passed": True, "history_size": size}
                    for size in (1000, 10000, 100000, 1000000)
                ],
            },
            "6_peak_application_materialized_ids": {"a": {"passed": True}},
            "7_answer_prompt_token_reduction_with_construction": {
                "a": {"passed": True}
            },
        }
    }
    assert _empirical_claim_gates(summary)["systems_headline_passed"] is True
    summary["claim_gates"]["5_latency"]["native"]["a"]["passed"] = False
    assert _empirical_claim_gates(summary)["systems_headline_passed"] is False


def test_sqlite_scale_snapshot_is_lazy_and_digest_checked(tmp_path) -> None:
    database = tmp_path / "snapshot.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE unit (uid TEXT PRIMARY KEY, scope_id TEXT, node_kind TEXT,"
            " ordinal INTEGER, state TEXT, summary TEXT, payload TEXT,"
            " embedding BLOB, metadata TEXT)"
        )
        connection.execute(
            "CREATE TABLE edge (src_uid TEXT, dst_uid TEXT, rel TEXT, weight REAL)"
        )
        connection.execute(
            "INSERT INTO unit VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "u0",
                "scope",
                "experience",
                0,
                "active",
                "summary",
                "payload",
                _float32_blob((1.0, -0.5)),
                '{"validity_state":"active"}',
            ),
        )
        connection.execute(
            "INSERT INTO edge VALUES (?,?,?,?)", ("u0", "u0", "SELF", 1.0)
        )
    database_sha = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SCALE_SNAPSHOT_SCHEMA_VERSION,
                "scope_id": "scope",
                "database": database.name,
                "database_sha256": database_sha,
                "digest": "semantic-digest",
                "unit_count": 1,
                "edge_count": 1,
                "embedding_dim": 2,
            }
        )
    )
    snapshot = SQLiteExperienceSnapshot.read(manifest)
    assert snapshot.units[0].embedding == pytest.approx((1.0, -0.5))
    assert snapshot.edges[:1][0].rel == "SELF"
    database.write_bytes(database.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="database digest"):
        SQLiteExperienceSnapshot.read(manifest)


def test_formal_label_requires_explicit_protocol_authorization() -> None:
    require_formal_authorization(formal=False, authorization="")
    require_formal_authorization(formal=True, authorization=FORMAL_AUTHORIZATION)
    with pytest.raises(PermissionError, match="authorization"):
        require_formal_authorization(formal=True, authorization="")


def test_balanced_arm_order_is_deterministic_and_position_balanced() -> None:
    arms = ("memory_off", "long_context", "full_gem")
    schedule = [balanced_arm_order(arms, block_index=i, seed=42) for i in range(120)]
    assert schedule == [
        balanced_arm_order(arms, block_index=i, seed=42) for i in range(120)
    ]
    assert all(set(row) == set(arms) for row in schedule)
    for position in range(3):
        assert {arm: sum(row[position] == arm for row in schedule) for arm in arms} == {
            arm: 40 for arm in arms
        }


def test_frozen_system_manifest_hashes_the_plan_and_dataset_manifest() -> None:
    repo = Path(__file__).parents[1]
    manifest = json.loads(
        (
            repo / "bench/agent_memory/evomembench/system_protocol_manifest_v0.1.0.json"
        ).read_text()
    )
    for key in ("plan", "dataset_manifest"):
        path = repo / manifest[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]["sha256"]
    scale = manifest["systems_scale"]["query_manifest"]
    scale_path = repo / scale["path"]
    assert hashlib.sha256(scale_path.read_bytes()).hexdigest() == scale["sha256"]
    assert manifest["models"]["judge_max_tokens"] == 4096
    assert manifest["models"]["embedding_revision"] == (
        embedding_server_admission.REVISION
    )


def test_answer_server_admission_reads_the_live_process_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = "tridb-evomembench-answer-formal-offtarget.service"
    pid = 123
    proc = tmp_path / str(pid)
    proc.mkdir()
    arguments = [
        "python",
        "-m",
        "vllm.entrypoints.cli.main",
        "Qwen/Qwen3.8-27B-FP8",
        "--port",
        "8000",
        "--served-model-name",
        "qwen3.8",
        "--max-model-len",
        "262144",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--override-generation-config",
        '{"temperature":0.0,"top_p":1.0,"top_k":-1,"max_new_tokens":4096}',
    ]
    (proc / "cmdline").write_bytes(b"\0".join(value.encode() for value in arguments))
    (proc / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0\0")
    (proc / "cgroup").write_text(f"0::/user.slice/{unit}\n")
    monkeypatch.setattr(answer_server_admission, "_main_pid", lambda _unit: pid)

    report = answer_server_admission.inspect(unit, proc_root=tmp_path)

    assert report["passed"] is True
    assert report["checks"]["max_new_tokens"] is True
    assert report["observed"]["generation_config"]["max_new_tokens"] == 4096


def test_answer_server_passport_gate_requires_the_audited_cap(tmp_path: Path) -> None:
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "models": {
                    "answer": "qwen3.8",
                    "answer_artifact": "Qwen/Qwen3.8-27B-FP8",
                    "judge_max_tokens": 4096,
                    "context_window_tokens": 262144,
                }
            }
        )
    )
    receipt = {
        "status": "complete",
        "passed": True,
        "unit": "answer.service",
        "main_pid": 123,
        "command_sha256": "a" * 64,
        "checks": {"max_new_tokens": True, "cuda_visible_devices": True},
        "observed": {
            "cuda_visible_devices": "0",
            "model": "Qwen/Qwen3.8-27B-FP8",
            "served_model": "qwen3.8",
            "context_tokens": "262144",
            "generation_config": {"max_new_tokens": 4096},
        },
    }
    (tmp_path / "answer_server_admission.json").write_text(json.dumps(receipt))
    assert _answer_server_gate(tmp_path, required=True)["passed"] is True
    receipt["observed"]["generation_config"]["max_new_tokens"] = 262144
    (tmp_path / "answer_server_admission.json").write_text(json.dumps(receipt))
    assert _answer_server_gate(tmp_path, required=True)["passed"] is False


def test_embedding_server_admission_and_passport_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = "tridb-evomembench-embed-formal-offtarget.service"
    pid = 456
    proc = tmp_path / str(pid)
    proc.mkdir()
    arguments = [
        "python",
        "-m",
        "vllm.entrypoints.cli.main",
        embedding_server_admission.MODEL,
        "--revision",
        embedding_server_admission.REVISION,
        "--tokenizer-revision",
        embedding_server_admission.REVISION,
        "--served-model-name",
        embedding_server_admission.MODEL,
        "--runner",
        "pooling",
        "--port",
        "8011",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.15",
    ]
    (proc / "cmdline").write_bytes(b"\0".join(value.encode() for value in arguments))
    (proc / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=1\0")
    (proc / "cgroup").write_text(f"0::/user.slice/{unit}\n")
    monkeypatch.setattr(embedding_server_admission, "_main_pid", lambda _unit: pid)

    receipt = embedding_server_admission.inspect(unit, proc_root=tmp_path)
    assert receipt["passed"] is True
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "models": {
                    "embedding": embedding_server_admission.MODEL,
                    "embedding_revision": embedding_server_admission.REVISION,
                    "embedding_context_tokens": 32768,
                }
            }
        )
    )
    (tmp_path / "embedding_server_admission.json").write_text(json.dumps(receipt))
    assert _embedding_server_gate(tmp_path, required=True)["passed"] is True

    receipt["observed"]["cuda_visible_devices"] = "0"
    (tmp_path / "embedding_server_admission.json").write_text(json.dumps(receipt))
    assert _embedding_server_gate(tmp_path, required=True)["passed"] is False


def test_lossless_know_grading_seals_upstream_omissions_as_score_zero(
    tmp_path,
) -> None:
    source = tmp_path / "predictions.jsonl"
    output = tmp_path / "graded.jsonl"
    receipt = tmp_path / "graded.receipt.json"
    rows = [
        {"idx": index, "metadata": {"task_id": f"task-{index}"}, "model_output": "x"}
        for index in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output.write_text(
        json.dumps({**rows[0], "score": 1, "grading_rationale": "ok"}) + "\n"
    )
    observed = seal_existing(
        argparse.Namespace(input=str(source), output=str(output), receipt=str(receipt))
    )
    graded = [json.loads(line) for line in output.read_text().splitlines()]
    assert observed["denominator_preserved"] is True
    assert observed["judge_failures_counted_as_score_zero"] == 2
    assert len(graded) == 3
    assert [row["score"] for row in graded] == [1, 0, 0]


def test_calibration_handoff_requires_exact_unique_denominators(tmp_path: Path) -> None:
    arms = ("memory_off", "gem_fused")
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_id": "b2-test",
                "arms": list(arms),
                "predictions": 4,
            }
        )
    )
    for folder in ("predictions", "receipts", "graded"):
        (tmp_path / folder).mkdir()
    for arm in arms:
        rows = [
            {"idx": index, "metadata": {"task_id": f"task-{index}"}}
            for index in range(2)
        ]
        for folder in ("predictions", "graded"):
            (tmp_path / folder / f"{arm}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        (tmp_path / "receipts" / f"{arm}.jsonl").write_text("{}\n{}\n")
        (tmp_path / "graded" / f"{arm}.receipt.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "predictions": 2,
                    "graded": 2,
                    "denominator_preserved": True,
                }
            )
        )

    report = validate_calibration_handoff(tmp_path)
    assert report["passed"] is True
    duplicate = {"idx": 0, "metadata": {"task_id": "task-0"}}
    (tmp_path / "graded" / "gem_fused.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n"
    )
    with pytest.raises(ValueError, match="gem_fused calibration handoff failed"):
        validate_calibration_handoff(tmp_path)


def test_scale_summary_pairs_repetitions_by_frozen_query(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    full_path = trace_dir / "full_gem_1000.jsonl"
    multi_path = trace_dir / "multi_system_1000.jsonl"
    for query, full_ms, multi_ms in (("q:a", 2.0, 4.0), ("q:b", 4.0, 2.0)):
        for repetition in range(2):
            for path, arm, latency, intermediate in (
                (
                    full_path,
                    "full_gem",
                    full_ms,
                    {"peak_application_materialized_ids": 2},
                ),
                (
                    multi_path,
                    "multi_system",
                    multi_ms,
                    {"peak_materialized_ids": 10},
                ),
            ):
                row = SystemTrace(
                    run_id="scale",
                    track="Systems-Scale",
                    arm=arm,
                    target_id=f"{query}:{repetition}",
                    scope_id="scope",
                    history_size=1000,
                    status="complete",
                    latency_ms={"total": latency},
                    tokens={},
                    intermediate=intermediate,
                    selected_ids=(),
                    injection_sha256="digest",
                    probes={"repetition": repetition},
                ).as_dict()
                with path.open("a") as target:
                    target.write(json.dumps(row) + "\n")
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "formal": True,
                "run_id": "scale",
                "queries": 2,
                "warmups": 1,
                "measured_repetitions": 2,
                "points": [
                    {
                        "history_size": 1000,
                        "parity_queries": 2,
                        "long_context": {"status": "complete"},
                        "construction": {},
                        "snapshot": {},
                        "footprint": {},
                        "multi_system_load": {},
                    }
                ],
            }
        )
    )
    summary = summarize_system_scale(tmp_path, bootstrap_repetitions=100)
    point = summary["points"][0]
    assert point["parity"]["all_passed"] is True
    assert point["paired_latency"]["median_ratio_multi_over_full"] == 1.25
    assert point["paired_latency"]["full_gem_faster_query_fraction"] == 0.5
    assert (
        point["peak_application_materialized_ids"]["full_gem_reduction_factor_vs_multi"]
        == 5.0
    )


def test_canonical_system_artifacts_are_materialized_from_verified_traces(
    tmp_path,
) -> None:
    def write_trace(relative: str, arm: str, track: str) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        row = SystemTrace(
            run_id="formal",
            track=track,
            arm=arm,
            target_id=f"{track}:{arm}:0",
            scope_id="scope",
            history_size=1,
            status="complete",
            latency_ms={"total": 1.0},
        ).as_dict()
        path.write_text(json.dumps(row) + "\n")

    write_trace("know/traces/full_gem.jsonl", "full_gem", "CrossEp-Know")
    write_trace(
        "know_multi_system/traces/multi_system.jsonl",
        "multi_system",
        "CrossEp-Know",
    )
    write_trace("tool/system_traces/full_gem.jsonl", "full_gem", "CrossEp-Tool")
    write_trace(
        "tool_multi_system/traces/multi_system.jsonl",
        "multi_system",
        "CrossEp-Tool",
    )
    write_trace("scale/traces/full_gem_1000.jsonl", "full_gem", "Systems-Scale")
    (tmp_path / "know/updates").mkdir()
    (tmp_path / "know/updates/full_gem.jsonl").write_text('{"update":1}\n')
    (tmp_path / "tool/updates").mkdir()
    (tmp_path / "tool/updates/full_gem.jsonl").write_text('{"update":2}\n')
    (tmp_path / "scale/per_update.jsonl").write_text('{"update":3}\n')
    for folder in ("know_multi_system", "tool_multi_system"):
        (tmp_path / folder / "run_receipt.json").write_text(
            json.dumps(
                {
                    "queries": 1,
                    "parity_passed": 1,
                    "parity_fraction": 1.0,
                    "all_parity_passed": True,
                }
            )
        )
    parity_dir = tmp_path / "scale/parity"
    parity_dir.mkdir()
    for size in (1000, 10000, 100000, 1000000):
        (parity_dir / f"scale_{size}.jsonl").write_text(
            json.dumps(
                {
                    "set_parity": True,
                    "order_parity": True,
                    "injection_parity": True,
                    "passed": True,
                }
            )
            + "\n"
        )
    scale_points = [
        {
            "history_size": size,
            "latency_ms": {
                "full_gem": {"p50": float(index + 1)},
                "multi_system": {"p50": float(index + 2)},
            },
        }
        for index, size in enumerate((1000, 10000, 100000, 1000000))
    ]
    (tmp_path / "system_summary.json").write_text(
        json.dumps(
            {
                "latency_semantics": {},
                "tracks": {},
                "quality": {},
                "efficiency_normalized": {},
                "claim_gates": {},
                "unavailable_retrieval_quality_metrics": {},
                "systems_scale": {"points": scale_points},
            }
        )
    )
    (tmp_path / "scale_summary.json").write_text("{}\n")
    (tmp_path / "hardware.json").write_text("{}\n")
    receipt = finalize(tmp_path)
    assert receipt["parity_all_passed"] is True
    assert receipt["per_query_rows"] == 5
    assert receipt["per_update_rows"] == 3
    assert (tmp_path / "figures/scale_latency_p50.svg").is_file()
    assert json.loads((tmp_path / "parity.json").read_text())["all_passed"] is True


def test_system_summary_reconstructs_multi_e2e_only_after_parity() -> None:
    def row(arm: str, retrieval: float, e2e: float | None = None) -> dict:
        latency = {
            "total" if arm == "multi_system" else "memory_or_prompt_assembly": retrieval
        }
        if e2e is not None:
            latency["end_to_end"] = e2e
        return SystemTrace(
            run_id="r",
            track="CrossEp-Know",
            arm=arm,
            target_id="target",
            scope_id="scope",
            history_size=1,
            status="complete",
            latency_ms=latency,
            tokens={"answer_prompt": 20},
            intermediate=(
                {"peak_materialized_ids": 10}
                if arm == "multi_system"
                else {"peak_application_materialized_ids": 2}
            ),
        ).as_dict()

    summary = _track(
        full=[row("full_gem", 10.0, 100.0)],
        long=[row("long_context", 20.0, 120.0)],
        multi=[row("multi_system", 40.0)],
        parity={"queries": 1, "passed": 1, "fraction": 1.0, "all_passed": True},
    )
    reconstructed = summary["arms"]["multi_system"][
        "parity_conditioned_reconstructed_end_to_end_ms"
    ]
    assert reconstructed["p50"] == 130.0


def test_system_latency_excludes_long_context_overflow_but_keeps_token_curve() -> None:
    def row(arm: str, target: str, status: str, e2e: float, tokens: int) -> dict:
        return SystemTrace(
            run_id="r",
            track="CrossEp-Know",
            arm=arm,
            target_id=target,
            scope_id="scope",
            history_size=1,
            status=status,
            latency_ms={
                "memory_or_prompt_assembly": 2.0,
                "end_to_end": e2e,
                "generation": 0.0 if status != "complete" else e2e - 2.0,
            },
            tokens={"answer_prompt": tokens},
            intermediate={"peak_application_materialized_ids": 1},
        ).as_dict()

    full = [
        row("full_gem", "a", "complete", 10.0, 10),
        row("full_gem", "b", "complete", 11.0, 10),
    ]
    long = [
        row("long_context", "a", "complete", 20.0, 20),
        row("long_context", "b", "context_overflow", 1.0, 1000),
    ]
    multi = []
    for target in ("a", "b"):
        multi.append(
            SystemTrace(
                run_id="r",
                track="CrossEp-Know",
                arm="multi_system",
                target_id=target,
                scope_id="scope",
                history_size=1,
                status="complete",
                latency_ms={"total": 5.0},
                tokens={"answer_prompt": 10},
                intermediate={"peak_materialized_ids": 10},
            ).as_dict()
        )
    summary = _track(
        full=full,
        long=long,
        multi=multi,
        parity={"queries": 2, "passed": 2, "fraction": 1.0, "all_passed": True},
    )
    assert summary["comparisons"]["long_context_to_full_gem_end_to_end"]["n"] == 1
    assert (
        summary["comparisons"]["long_context_to_full_gem_answer_prompt_tokens"]["n"]
        == 2
    )
    assert summary["arms"]["long_context"]["measured_end_to_end_ms"]["n"] == 1


def test_know_quality_summary_reports_beneficial_and_harmful_flips(tmp_path) -> None:
    (tmp_path / "graded").mkdir()
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "formal": False,
                "run_id": "quality-test",
                "arms": ["memory_off", "long_context", "full_gem"],
            }
        )
    )
    metadata = [
        {
            "context_id": "c1",
            "task_id": "t0",
            "ordinal": 0,
            "context_category": "rules",
        },
        {
            "context_id": "c1",
            "task_id": "t1",
            "ordinal": 1,
            "context_category": "rules",
        },
        {
            "context_id": "c2",
            "task_id": "t2",
            "ordinal": 1,
            "context_category": "facts",
        },
    ]
    scores = {
        "memory_off": [1, 0, 1],
        "long_context": [1, 1, 1],
        "full_gem": [1, 1, 0],
    }
    for arm, values in scores.items():
        (tmp_path / "graded" / f"{arm}.jsonl").write_text(
            "".join(
                json.dumps({"score": score, "metadata": item}) + "\n"
                for score, item in zip(values, metadata, strict=True)
            )
        )
    summary = summarize_know_quality(tmp_path, bootstrap_repetitions=100)
    effect = summary["full_gem_vs_no_memory"]
    assert effect["beneficial_flips"] == 1
    assert effect["harmful_flips"] == 1
    assert summary["arms"]["full_gem"]["reuse_decisions"] == 2


def test_calibration_watchers_allow_explicit_upstream_recovery_units() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scripts/evomembench_after_tool_queue_s1.sh": "${EVOMEMBENCH_TOOL_QUEUE_UNIT:-tridb-evomembench-after-b2-queue-t1.service}",
        "scripts/evomembench_after_systems_queue_b3.sh": "${EVOMEMBENCH_SYSTEMS_QUEUE_UNIT:-tridb-evomembench-after-tool-queue-s1.service}",
        "scripts/evomembench_formal_offtarget_after_b3.sh": "${EVOMEMBENCH_B3_QUEUE_UNIT:-tridb-evomembench-after-systems-queue-b3.service}",
    }
    for relative, contract in expected.items():
        assert contract in (root / relative).read_text()


def test_b2_tool_queue_revalidates_a_complete_handoff_without_overwriting() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/evomembench_after_b2_queue_t1.sh").read_text()
    assert "sealed_present != 4" in script
    assert 'cmp -s "$handoff" <(' in script
    assert '(cd "$b2_run" && sha256sum -c SHA256SUMS)' in script


def test_gpu_startup_admission_detects_vllm_before_cuda_attach(tmp_path) -> None:
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"/env/python\0/opt/bin/vllm\0serve\0model\0")
    (proc / "cgroup").write_text("0::/app.slice/external-vllm.service\n")
    assert vllm_frontends(tmp_path) == [
        {"pid": 123, "cgroup_unit": "external-vllm.service"}
    ]


def test_cpu_embedding_request_validation_is_bounded_and_fail_closed() -> None:
    assert _validate_inputs("one") == ["one"]
    assert _validate_inputs(("one", "two")) == ["one", "two"]
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_inputs([])
    with pytest.raises(ValueError, match="non-empty"):
        _validate_inputs([""])
    with pytest.raises(ValueError, match="128-item"):
        _validate_inputs(["item"] * 129)
    with pytest.raises(TypeError, match="string or sequence"):
        _validate_inputs(None)  # type: ignore[arg-type]


def test_three_arm_protocol_locks_scope_models_and_two_replica_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "bench/agent_memory/evomembench/know_three_arm_protocol_v0.2.0.json"
        ).read_text()
    )
    assert protocol["arms"] == ["no_memory", "gem", "polyglot"]
    assert set(protocol["forbidden_arms"]) == {
        "recent_fifo",
        "vector_only",
        "graph_relational",
        "long_context",
    }
    models = protocol["models"]
    replicas = models["answer_replica_bindings"]
    assert [(row["gpu"], row["port"]) for row in replicas] == [
        (0, 8000),
        (1, 8002),
    ]
    assert models["embedding"] == MODEL_ID
    assert models["embedding_revision"] == MODEL_REVISION
    assert models["embedding_dimension"] == EMBEDDING_DIMENSION
    assert models["embedding_execution"] == "two_gpu_bfloat16_pooling_replicas"
    assert models["embedding_replica_bindings"] == [
        {"gpu": 0, "port": 8011},
        {"gpu": 1, "port": 8012},
    ]
    assert models["answer_gpu_memory_utilization"] == 0.84
    assert models["embedding_gpu_memory_utilization"] == 0.06
    assert MAX_MODEL_TOKENS == 32768


def test_three_arm_launcher_reserves_both_gpus_for_answer_replicas() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/evomembench_know_three_arm_dp2_v0_2_0.sh").read_text()
    assert "env GPU=0 PORT=8000" in launcher
    assert "env GPU=1 PORT=8002" in launcher
    assert "evomembench_serve_qwen38_replica.sh" in launcher
    assert "env GPU=0 PORT=8011" in launcher
    assert "env GPU=1 PORT=8012" in launcher
    assert "evomembench_serve_embedding_replica.sh" in launcher
    assert "--arms memory_off full_gem" in launcher
    assert "--context-shard-count 2" in launcher
    for arm in ("recent_fifo", "vector_only", "graph_relational", "long_context"):
        assert f"--arms {arm}" not in launcher

    embedding_service = (
        root / "scripts/evomembench_serve_embedding_replica.sh"
    ).read_text()
    assert "--gpu-memory-utilization 0.06" in embedding_service
    assert "0:8011|1:8012" in embedding_service


def test_three_arm_v021_freezes_nul_fix_and_new_run_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root / "bench/agent_memory/evomembench/know_three_arm_protocol_v0.2.1.json"
    )
    protocol = json.loads(protocol_path.read_text())
    plan = root / protocol["plan"]["path"]
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == protocol["plan"]["sha256"]
    assert protocol["arms"] == ["no_memory", "gem", "polyglot"]
    assert protocol["database_text_policy"]["forbidden_input"] == "U+0000"
    assert protocol["database_text_policy"]["replacement"] == "U+FFFD"
    assert protocol["database_text_policy"]["agent_prompt_unchanged"] is True
    assert protocol["database_text_policy"]["judge_answer_unchanged"] is True

    wrapper = (root / "scripts/evomembench_know_three_arm_dp2_v0_2_1.sh").read_text()
    assert "EVOMEMBENCH_KNOW_THREE_ARM_V0_2_1" in wrapper
    assert "evomembench_know_three_arm_v0.2.1_offtarget_20260827" in wrapper
    assert "evomembench_know_three_arm_v021_s0_20260827" in wrapper
    assert "evomembench_know_three_arm_v021_s1_20260827" in wrapper
