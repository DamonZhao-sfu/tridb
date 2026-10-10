from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.agent_memory.evomembench.dataset import normalize_crossep_know
from bench.agent_memory.evomembench.crossep_tool_backend import (
    BankSnapshot,
    CrossEpToolGemMemory,
    MemoryUsageLog,
    ToolSampleContext,
)
from bench.agent_memory.evomembench.experience_graph import (
    build_evaluation_audit_graph,
    build_structural_graph,
)
from bench.agent_memory.evomembench.extractor import extract_observed_trajectory
from bench.agent_memory.evomembench.leakage import (
    EvaluationLeakageError,
    assert_online_payload,
)
from bench.agent_memory.evomembench.manifest import (
    load_manifest,
    manifest_sha256,
    verify_assets,
)
from bench.agent_memory.evomembench.metrics import clustered_paired_effect
from bench.agent_memory.evomembench.material_passport import build as build_passport
from bench.agent_memory.evomembench.injection import (
    HEADER,
    fit_injection,
    pad_to_token_budget,
)
from bench.agent_memory.evomembench.modeling import (
    ExperienceUnit,
    PreparedExperienceStrategy,
    build_experience_filter,
    experience_plan,
    experience_query,
    knowledge_experience,
    sanitize_postgres_text,
)
from bench.agent_memory.evomembench.receipts import (
    build_experience_receipt,
    verify_receipt,
)
from bench.agent_memory.evomembench.run_pilot import (
    TokenCounter,
    _messages,
    _replacement_capacity,
)
from bench.agent_memory.evomembench.run_know_system import _shard_contexts
from bench.agent_memory.evomembench.run_tool import (
    TokenSlotString,
    _SourceConstructionOnlyMemory,
    _jsonable,
    _replay_canonical_source_result,
    _uses_fixed_token_slot,
    _write_system_trace,
)
from bench.agent_memory.evomembench.run_systems import (
    _merge_membership_candidates,
    _percentile,
)
from bench.agent_memory.evomembench.scale import iter_scaled_experiences
from bench.agent_memory.evomembench.summarize import _p95, summarize
from bench.agent_memory.evomembench.summarize_tool import _official_cost_metrics
from bench.agent_memory.evomembench.protocol import ordered_transfer_pairs
from bench.agent_memory.evomembench.task_signature import (
    knowledge_task_signature,
    tool_task_signature,
)
from bench.agent_memory.evomembench.tool_dataset import normalize_crossep_tool
from bench.agent_memory.gem.types import EdgeKind
from bench.agent_memory.gem.types import InteractionEvent, RetrievalMode
from bench.agent_memory.memoryarena.metrics import retrieval_metrics
from bench.agent_memory.memoryarena.oracle import Arm, Candidate, select


def _row(context: str, task: str, rubric: str, category: str = "Rules") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "shared background"},
            {"role": "user", "content": f"solve {task}"},
        ],
        "rubrics": [rubric],
        "metadata": {
            "task_id": task,
            "context_id": context,
            "context_category": category,
            "sub_category": "unit",
        },
    }


def _corpus():
    return normalize_crossep_know(
        [
            _row("ctx-a", "a0", "must cite rule"),
            _row("ctx-b", "b0", "must be safe", "Procedure"),
            _row("ctx-a", "a1", "must cite rule"),
            _row("ctx-a", "a2", "must explain exception"),
        ],
        source_revision="deadbeef",
        source_sha256="a" * 64,
    )


def test_normalization_preserves_per_context_order_and_cutoff() -> None:
    corpus = _corpus()
    ctx = corpus.contexts[0]
    final = ctx.episodes[-1]
    assert [item.source_task_id for item in ctx.episodes] == ["a0", "a1", "a2"]
    assert final.ordinal == 2
    assert final.cutoff_ordinal == 2
    assert final.prior_episode_uids == tuple(
        item.episode_uid for item in ctx.episodes[:2]
    )
    assert final.retrieval_query == "solve a2"
    assert corpus.episode_count == 4
    assert corpus.decision_count == 2


def test_database_experience_text_replaces_nul_and_records_the_boundary() -> None:
    row = _row("ctx-nul", "task-nul", "must remain gradable")
    row["messages"][-1]["content"] = "artifact bytes: A\x00B"
    episode = (
        normalize_crossep_know(
            [row], source_revision="deadbeef", source_sha256="b" * 64
        )
        .contexts[0]
        .episodes[0]
    )

    unit = knowledge_experience(
        episode,
        response="answer bytes: C\x00D",
        scope_id="scope-nul",
    )
    assert "\x00" not in unit.task_signature
    assert "\x00" not in unit.memory_payload
    assert unit.task_signature.count("\ufffd") == 1
    assert unit.memory_payload.count("\ufffd") == 2
    assert unit.metadata["postgres_nul_replacements"] == 3
    assert "\x00" not in json.dumps(
        experience_plan(unit, valid_from="2000-01-01T00:00:00+00:00"),
        ensure_ascii=False,
    )
    assert sanitize_postgres_text("A\x00B") == "A\ufffdB"


def test_experience_unit_rejects_unsanitized_postgres_text() -> None:
    with pytest.raises(ValueError, match="PostgreSQL text contains NUL"):
        ExperienceUnit(
            uid="uid",
            scope_id="scope",
            ordinal=0,
            task_signature="unsafe\x00signature",
            memory_payload="payload",
            source_external_ids=("source",),
            features=(),
        )


def test_unreleased_relevance_is_na_not_a_zero_score_or_fake_oracle() -> None:
    episode = _corpus().contexts[0].episodes[-1]
    decision = episode.decision_point()
    candidates = [
        Candidate(
            session_uid=uid,
            task_uid=episode.context_uid,
            ordinal=ordinal,
            graph_distance=1,
        )
        for ordinal, uid in enumerate(episode.prior_episode_uids)
    ]
    metrics = retrieval_metrics(candidates, decision)
    assert metrics["dependency_recall_at_10"] is None
    assert metrics["ndcg_at_10"] is None
    with pytest.raises(ValueError, match="oracle arm requires"):
        select(Arm.ORACLE, decision, candidates)


def test_know_context_shards_are_disjoint_complete_and_keep_global_indices() -> None:
    contexts = tuple(f"context-{index}" for index in range(120))
    shard0 = _shard_contexts(contexts, shard_count=2, shard_index=0)
    shard1 = _shard_contexts(contexts, shard_count=2, shard_index=1)

    assert len(shard0) == len(shard1) == 60
    assert {index for index, _ in shard0}.isdisjoint({index for index, _ in shard1})
    assert sorted(shard0 + shard1) == list(enumerate(contexts))
    assert [index for index, _ in shard0[:3]] == [0, 2, 4]
    assert [index for index, _ in shard1[:3]] == [1, 3, 5]


@pytest.mark.parametrize(
    ("shard_count", "shard_index", "match"),
    [(0, 0, "positive"), (2, -1, "within"), (2, 2, "within")],
)
def test_know_context_shards_fail_closed(
    shard_count: int, shard_index: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _shard_contexts(
            ("context",),
            shard_count=shard_count,
            shard_index=shard_index,
        )


def test_know_context_shards_reject_empty_assignment() -> None:
    with pytest.raises(ValueError, match="empty"):
        _shard_contexts(("context",), shard_count=2, shard_index=1)


def test_structural_graph_contains_only_auditable_edge_types() -> None:
    graph = build_structural_graph(_corpus().contexts[0])
    edge_kinds = {edge.kind for edge in graph.edges}
    assert edge_kinds == {"BELONGS_TO", "PRECEDES"}
    assert sum(edge.kind == "PRECEDES" for edge in graph.edges) == 2
    assert not any(node.kind == "rubric" for node in graph.nodes)
    assert graph.visibility == "online"

    audit = build_evaluation_audit_graph(_corpus().contexts[0])
    assert {edge.kind for edge in audit.edges} == {"EVALUATED_BY"}
    assert sum(node.kind == "rubric" for node in audit.nodes) == 2
    assert audit.visibility == "evaluation_only"


def test_task_signature_excludes_rubric_and_leakage_gate_fails_closed() -> None:
    episode = _corpus().contexts[0].episodes[0]
    signature = knowledge_task_signature(episode)
    assert "solve a0" in signature
    assert "must cite rule" not in signature
    with pytest.raises(EvaluationLeakageError, match="evaluation-only"):
        assert_online_payload({"online": {"rubrics": ["secret"]}})


def test_context_category_drift_is_rejected() -> None:
    with pytest.raises(ValueError, match="changes category"):
        normalize_crossep_know(
            [
                _row("ctx", "one", "r", "A"),
                _row("ctx", "two", "r", "B"),
            ]
        )


def test_experience_plan_is_bounded_bidirectional_and_evaluation_clean() -> None:
    episode = _corpus().contexts[0].episodes[1]
    unit = knowledge_experience(episode, response="safe prior answer", scope_id="s")
    ops = experience_plan(unit, valid_from="2026-01-01T00:00:00Z")
    links = [item for item in ops if item["kind"] == "link"]
    assert len(links) == 2 * len(unit.features)
    assert {item["edge_kind"] for item in links} == {"association"}
    assert all("rubric" not in str(item).casefold() for item in ops)
    query = experience_query(
        scope_id="s",
        task_signature=unit.task_signature,
        cutoff_ordinal=episode.ordinal,
    )
    assert query.graph_edge_kind is EdgeKind.ASSOCIATION
    assert "experience_ordinal" in query.extra_filter
    assert "validity_state" in build_experience_filter(
        cutoff_ordinal=2, validity_states=["active"]
    )
    with pytest.raises(ValueError, match="unsafe"):
        build_experience_filter(cutoff_ordinal=2, source_phases=["in_env') OR true --"])


def test_prepared_experience_links_consecutive_sessions_in_native_topology() -> None:
    episode = _corpus().contexts[0].episodes[1]
    unit = knowledge_experience(episode, response="safe prior answer", scope_id="s")
    strategy = PreparedExperienceStrategy(unit, valid_from="2026-01-01T00:00:00Z")
    view = SimpleNamespace(
        latest_experience_before=lambda scope_id, ordinal: SimpleNamespace(id=41),
        unit_by_title=lambda scope_id, title: None,
    )
    ops = strategy.plan(
        [
            InteractionEvent(
                scope_id="s",
                external_id=unit.source_external_ids[0],
                content=unit.memory_payload,
            )
        ],
        view,
    )
    temporal = [op for op in ops if op.get("rel") in {"PRECEDES", "FOLLOWS"}]
    assert len(temporal) == 2
    assert temporal[0]["src"] == 41
    assert temporal[1]["dst"] == 41
    assert {op["edge_kind"] for op in temporal} == {"association"}


def test_tool_prompt_loader_quarantines_expected_path() -> None:
    corpus = normalize_crossep_tool(
        [
            {
                "id": "multi_turn_ours_0",
                "question": [[{"role": "user", "content": "move a file"}]],
                "initial_config": {"FS": {}},
                "path": ["FS.mv"],
                "involved_classes": ["FS"],
                "excluded_function": ["cp"],
            }
        ],
        category_by_id={"multi_turn_ours_0": "gorilla_fs"},
    )
    episode = corpus.episodes[0]
    assert episode.category == "gorilla_fs"
    assert "path" not in episode.online_payload()
    assert "FS.mv" not in str(episode.online_payload())
    signature = tool_task_signature(
        question=[
            [{"role": "user", "content": "first visible turn"}],
            [{"role": "user", "content": "future turn"}],
        ],
        involved_classes=["FS"],
    )
    assert "first visible turn" in signature
    assert "future turn" not in signature


def test_observed_extractor_is_deterministic_bounded_and_rubric_free() -> None:
    kwargs = {
        "task_signature": "tool task: move a file",
        "trajectory": [
            {"role": "user", "content": "move /a to /b"},
            {"role": "assistant", "content": "calling mv"},
            {"role": "tool", "content": "ok"},
        ],
        "concept_hints": [("tool", "FS"), ("function", "mv"), ("tool", "FS")],
        "source_external_ids": ["sample-1"],
        "max_chars": 256,
    }
    first = extract_observed_trajectory(**kwargs)
    second = extract_observed_trajectory(**kwargs)
    assert first == second
    assert first.llm_calls == 0
    assert first.source_external_ids == ("sample-1",)
    assert [(item.kind, item.value) for item in first.concepts] == [
        ("tool", "FS"),
        ("function", "mv"),
    ]
    assert len(first.memory_payload) <= 256
    with pytest.raises(EvaluationLeakageError):
        extract_observed_trajectory(
            task_signature="x",
            trajectory=[{"role": "assistant", "content": "x", "score": 1}],
            concept_hints=[],
            source_external_ids=["sample-2"],
        )
    failed = extract_observed_trajectory(
        task_signature="x",
        trajectory=[{"role": "tool", "content": "Permission denied"}],
        concept_hints=[],
        source_external_ids=["sample-3"],
    )
    assert [(item.kind, item.value) for item in failed.concepts] == [
        ("failure_mode", "permission_denied")
    ]


def test_shared_injection_formatter_enforces_item_and_token_budgets() -> None:
    def count(text: str) -> int:
        return len(text.split())

    accepted, rendered, tokens = fit_injection(
        [{"text": "one two"}, {"text": "three four"}, {"text": "five six"}],
        max_items=2,
        token_budget=count(HEADER) + 4,
        count_tokens=count,
    )
    assert len(accepted) == 2
    assert rendered.startswith(HEADER)
    assert "five six" not in rendered
    assert tokens <= count(HEADER) + 4


def test_shared_injection_formatter_bounds_an_oversized_first_experience() -> None:
    counter = TokenCounter()
    item = {"unit_id": 7, "text": "prior task and response " * 500}
    budget = counter(HEADER) + 80
    first = fit_injection(
        [item], max_items=1, token_budget=budget, count_tokens=counter
    )
    second = fit_injection(
        [item], max_items=1, token_budget=budget, count_tokens=counter
    )
    assert first == second
    accepted, rendered, tokens = first
    assert len(accepted) == 1
    assert accepted[0]["injection_truncated"] is True
    assert "experience token-bounded" in rendered
    assert tokens <= budget
    assert rendered.startswith(HEADER)


def test_memory_injection_replaces_equal_system_tokens() -> None:
    row = _row("ctx", "task", "rubric")
    row["messages"][0]["content"] = "system context " * 300
    episode = normalize_crossep_know([row]).contexts[0].episodes[0]
    counter = TokenCounter()
    injection = HEADER + "\nprior experience"
    assert counter(injection) < _replacement_capacity(episode, counter)
    messages, stats = _messages(episode, injection, counter)
    assert stats["injection_policy"] == "replace_equal_total_input_tokens"
    assert stats["system_tokens_original"] == stats["system_tokens_final"]
    assert counter(messages[0]["content"]) == stats["system_tokens_original"]


def test_tool_token_slot_is_exact_and_survives_official_deepcopy() -> None:
    counter = TokenCounter()
    filler, padding = pad_to_token_budget("", token_budget=64, codec=counter)
    slot = TokenSlotString(filler, counter, 64)
    copied = copy.deepcopy(slot)
    replacement, _ = pad_to_token_budget(
        "useful prior experience", token_budget=64, codec=counter
    )
    assert padding == 64
    assert copied + replacement == replacement
    assert counter(copied + replacement) == 64


def test_tool_request_json_normalization_preserves_sdk_message_shape() -> None:
    class SDKMessage:
        def model_dump(self, *, exclude_none: bool) -> dict[str, object]:
            assert exclude_none
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "arguments": {"x": 1}}],
            }

    normalized = _jsonable(
        {"messages": [SDKMessage()], "tools": ({"type": "function"},)}
    )
    assert normalized == {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "arguments": {"x": 1}}],
            }
        ],
        "tools": [{"type": "function"}],
    }
    json.dumps(normalized, allow_nan=False)


def test_tool_fixed_slot_is_only_for_memory_target_retrieval() -> None:
    assert _uses_fixed_token_slot(
        arm="gem_fused", phase="transfer", query_mode="gem_task_seed"
    )
    assert not _uses_fixed_token_slot(
        arm="memory_off", phase="transfer", query_mode="gem_task_seed"
    )
    assert not _uses_fixed_token_slot(
        arm="gem_fused", phase="in_env", query_mode="gem_task_seed"
    )
    assert not _uses_fixed_token_slot(
        arm="long_context", phase="transfer", query_mode="gem_task_seed"
    )


def test_tool_source_construction_adapter_suppresses_retrieval() -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def begin_sample(self) -> None:
            self.calls.append(("begin", None))

        def update(self, trajectory: list[dict]) -> None:
            self.calls.append(("update", trajectory))

        def drain_usage(self) -> dict[str, int]:
            self.calls.append(("drain", None))
            return {"n_update": 1}

    backend = Backend()
    adapter = _SourceConstructionOnlyMemory(backend)
    trajectory = [{"role": "assistant", "content": "canonical"}]
    adapter.begin_sample()
    assert adapter.utilize("must not be retrieved") == ""
    adapter.update(trajectory)
    assert adapter.drain_usage() == {"n_update": 1}
    assert backend.calls == [
        ("begin", None),
        ("update", trajectory),
        ("drain", None),
    ]


def test_vllm_token_counter_counts_the_live_chat_template(monkeypatch) -> None:
    from bench.agent_memory.evomembench.run_pilot import VLLMTokenCounter

    calls: list[dict] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"count": 19, "tokens": list(range(19))}

    counter = VLLMTokenCounter("http://tokenizer", "model", timeout=1)

    def post(url: str, *, json: dict, timeout: float) -> Response:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(counter.session, "post", post)
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "task"},
    ]
    assert (
        counter.count_chat(messages, chat_template_kwargs={"enable_thinking": False})
        == 19
    )
    assert (
        counter.count_chat(messages, chat_template_kwargs={"enable_thinking": False})
        == 19
    )
    assert len(calls) == 1
    assert calls[0]["url"] == "http://tokenizer/tokenize"
    assert calls[0]["json"]["messages"] == messages
    assert calls[0]["json"]["add_generation_prompt"] is True


def test_know_summary_fails_closed_without_replacement_policy(tmp_path) -> None:
    (tmp_path / "run_receipt.json").write_text(
        '{"status":"complete","arms":["memory_off"]}'
    )
    with pytest.raises(ValueError, match="equal-sized token slot"):
        summarize(tmp_path)
    (tmp_path / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "arms": ["memory_off"],
                "injection_policy": "replace_equal_total_input_tokens",
                "answer_max_tokens": 1024,
            }
        )
    )
    with pytest.raises(ValueError, match="4096-token protocol"):
        summarize(tmp_path)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _Connection:
    def execute(self, _sql, _params=None):
        return _Cursor([(7, "prior-episode", 0, "prior tool experience")])


class _FakeToolMemory:
    def __init__(self):
        self.store = SimpleNamespace(conn=_Connection())
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return SimpleNamespace(
            committed=True,
            aborted_reason=None,
            hits=[SimpleNamespace(unit_id=7)],
            probes={"graph_examined": 2, "bridges_injected": 1},
            cost=SimpleNamespace(embed_calls=1, embed_input_tokens=9),
        )


def test_tool_backend_task_seed_and_readonly_snapshot(tmp_path) -> None:
    snapshot = BankSnapshot(
        scope_id="source-bank",
        source_environment="gorilla_fs",
        unit_ids=(7,),
        experience_count=1,
        max_ordinal=0,
        transition_id=4,
        digest="d" * 64,
    )
    snapshot_path = tmp_path / "bank.json"
    CrossEpToolGemMemory.write_snapshot(snapshot_path, snapshot)
    assert CrossEpToolGemMemory.read_snapshot(snapshot_path) == snapshot

    fake = _FakeToolMemory()
    counter = TokenCounter()
    backend = CrossEpToolGemMemory(
        memory=fake,
        scope_id="source-bank",
        source_environment="gorilla_fs",
        count_tokens=counter,
        query_mode="gem_task_seed",
        retrieval_mode=RetrievalMode.FUSED,
        readonly=True,
        snapshot=snapshot,
        token_budget=64,
    )
    context = ToolSampleContext(
        sample_id="target-1",
        episode_uid="target-episode-1",
        ordinal=0,
        environment="travel_api",
        question=[[{"role": "user", "content": "book a flight"}]],
        involved_classes=["TravelAPI"],
        allowed_functions=["book_flight"],
    )
    backend.set_sample_context(context)
    backend.begin_sample()
    injection = backend.utilize("official function-doc query")
    backend.update([{"role": "assistant", "content": "must not be written"}])
    usage = backend.drain_usage()
    assert "prior tool experience" in injection
    assert fake.queries[0].text == context.task_signature
    assert fake.queries[0].graph_edge_kind is EdgeKind.ASSOCIATION
    assert usage.n_utilize == 1
    assert usage.n_update == 1
    assert backend.receipts[0]["readonly"] is True
    assert backend.receipts[0]["source_environment"] == "gorilla_fs"
    assert backend.receipts[0]["target_environment"] == "travel_api"


def test_tool_backend_update_accepts_official_sdk_assistant_message(
    monkeypatch,
) -> None:
    class SDKMessage:
        def model_dump(self, *, exclude_none: bool) -> dict[str, object]:
            assert exclude_none
            return {
                "role": "assistant",
                "content": "completed the tool task",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            }

    captured: dict[str, object] = {}

    def admit(_memory, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            result=SimpleNamespace(
                committed=True,
                aborted_reason=None,
                cost=SimpleNamespace(embed_calls=1, embed_input_tokens=11),
            )
        )

    monkeypatch.setattr(
        "bench.agent_memory.evomembench.crossep_tool_backend.admit_extracted_experience",
        admit,
    )
    backend = CrossEpToolGemMemory(
        memory=_FakeToolMemory(),
        scope_id="source-bank",
        source_environment="gorilla_fs",
        count_tokens=TokenCounter(),
        query_mode="gem_task_seed",
        retrieval_mode=RetrievalMode.FUSED,
        token_budget=64,
    )
    backend.set_sample_context(
        ToolSampleContext(
            sample_id="source-1",
            episode_uid="source-episode-1",
            ordinal=0,
            environment="gorilla_fs",
            question=[[{"role": "user", "content": "write a file"}]],
            involved_classes=["FileSystem"],
            allowed_functions=["write_file"],
        )
    )
    backend.begin_sample()
    backend.update(
        [
            {"role": "system", "content": "fixed memory slot"},
            {"role": "user", "content": "write a file"},
            SDKMessage(),
            {"role": "tool", "content": "success"},
        ]
    )

    extracted = captured["extracted"]
    assert extracted.observed_messages == 3
    assert "assistant: completed the tool task" in extracted.memory_payload
    assert "tool: success" in extracted.memory_payload
    assert "fixed memory slot" not in extracted.memory_payload
    usage = backend.drain_usage()
    assert usage.n_update == 1
    assert usage.n_embed_calls == 1
    assert usage.embedding_tokens == 11


def test_tool_source_construction_trace_allows_empty_retrieval_probes(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    model_trace = {
        "calls": 1,
        "request_bytes": 100,
        "prompt_tokenization_calls": 1,
        "prompt_tokens_recomputed": 20,
        "prompt_token_count_matches_usage": True,
        "cache_detail_calls": 0,
        "cached_tokens": 0,
        "timing_attributable": True,
        "telemetry_overhead_ms": 0.0,
        "prompt_serialization_ms": 1.0,
        "prompt_tokenization_ms": 2.0,
        "model_ttft_ms": 3.0,
        "model_prefill_ms": 4.0,
        "model_decode_ms": 5.0,
        "model_server_end_to_end_ms": 12.0,
    }
    _write_system_trace(
        trace_path,
        args=SimpleNamespace(run_id="tool-trace-test"),
        arm="gem_fused",
        phase="in_env",
        environment="gorilla_fs",
        source_environment="gorilla_fs",
        source_id="source-1",
        history_size=0,
        result={
            "error": None,
            "latency": [0.1],
            "input_token_count": [20],
            "output_token_count": [4],
        },
        receipt=None,
        model_trace=model_trace,
        end_to_end_ms=120.0,
    )
    trace = json.loads(trace_path.read_text())
    assert trace["intermediate"]["peak_operator_items_proxy"] == 0
    assert trace["intermediate"]["final_results"] == 0


def test_tool_source_bank_replays_one_canonical_agent_trajectory() -> None:
    class Backend:
        def __init__(self) -> None:
            self.trajectories: list[list[dict[str, object]]] = []

        def begin_sample(self) -> None:
            return None

        def update(self, trajectory) -> None:
            self.trajectories.append(trajectory)

        def drain_usage(self) -> MemoryUsageLog:
            return MemoryUsageLog(latency_s=0.25, n_embed_calls=1, n_update=1)

    canonical = {
        "id": "source-1",
        "error": None,
        "full_message_history": [
            {"role": "user", "content": "write a file"},
            "ChatCompletionMessage(content='done', role='assistant')",
        ],
        "memory_usage": {"n_update": 0},
    }
    online_trajectory = [
        {"role": "user", "content": "write a file"},
        {"role": "assistant", "content": "done"},
    ]
    backend = Backend()
    replayed = _replay_canonical_source_result(
        canonical,
        canonical_trajectory=online_trajectory,
        backend=backend,
    )

    assert backend.trajectories == [online_trajectory]
    assert replayed["full_message_history"] == canonical["full_message_history"]
    assert replayed["memory_usage"]["n_update"] == 1
    assert replayed["memory_usage"]["n_embed_calls"] == 1
    assert canonical["memory_usage"] == {"n_update": 0}


def test_tool_transfer_matrix_has_all_twelve_ordered_nonself_pairs() -> None:
    pairs = ordered_transfer_pairs()
    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert all(source != target for source, target in pairs)


def test_clustered_paired_effect_is_deterministic_and_counts_harm() -> None:
    rows = [
        ("ctx-a", 0, 1),
        ("ctx-a", 1, 0),
        ("ctx-b", 0, 1),
        ("ctx-b", 1, 1),
    ]
    first = clustered_paired_effect(rows, repetitions=200, seed=7)
    second = clustered_paired_effect(rows, repetitions=200, seed=7)
    assert first == second
    assert first["n"] == 4
    assert first["clusters"] == 2
    assert first["paired_mean_delta"] == 0.25
    assert first["positive_transfer_fraction"] == 0.5
    assert first["negative_transfer_fraction"] == 0.25


def test_scaled_systems_records_are_unique_independent_and_typed() -> None:
    rows = list(
        iter_scaled_experiences(
            count=16,
            primary_scope="scale",
            seed=9,
            forbidden_content_hashes=["a" * 64],
        )
    )
    assert len({row.unit.uid for row in rows}) == 16
    assert len({row.content_sha256 for row in rows}) == 16
    assert {row.kind for row in rows} == {
        "cross_scope_hard_negative",
        "same_domain_semantic_negative",
        "graph_decoy_branch",
        "stale_superseded_version",
    }
    assert all(row.content_sha256 != "a" * 64 for row in rows)
    assert all(row.unit.metadata["systems_only"] is True for row in rows)


def test_scaled_systems_incremental_generation_has_global_ordinals() -> None:
    first = list(iter_scaled_experiences(count=4, primary_scope="scale", seed=9))
    second = list(
        iter_scaled_experiences(count=4, primary_scope="scale", seed=9, start_index=4)
    )
    combined = list(iter_scaled_experiences(count=8, primary_scope="scale", seed=9))
    assert first + second == combined
    assert [row.index for row in combined] == list(range(8))
    assert [row.unit.ordinal for row in combined] == list(range(8))


def test_staged_membership_merge_keeps_bridge_share_and_vector_results() -> None:
    assert _merge_membership_candidates(
        [(1, 0.1), (2, 0.2), (3, 0.3)],
        [(4, 0.15), (2, 0.2), (5, 0.4)],
        k=3,
    ) == [1, 4, 2]
    assert _percentile([1.0, 2.0], 0.95) == 2.0
    assert _p95(range(20)) == 18.0


def test_material_passport_marks_invalid_outcome_as_systems_only(tmp_path) -> None:
    (tmp_path / "run_receipt.json").write_text(
        json.dumps({"status": "complete", "run_id": "invalid-b2"})
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"schema_version": "summary-v1", "outcome_valid": False})
    )
    passport = build_passport(tmp_path)
    assert passport["outcome_valid"] is False
    assert passport["verification_status"] == "executed_systems_only_outcome_invalid"
    assert any("agent-outcome" in item for item in passport["known_limitations"])
    assert len(passport["implementation_manifest_sha256"]) == 64
    assert (
        "bench/agent_memory/evomembench/run_tool.py" in passport["implementation_files"]
    )

    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "evomembench_gem_systems_v0.1.0",
                "outcome_valid": None,
            }
        )
    )
    systems_passport = build_passport(tmp_path)
    assert systems_passport["outcome_valid"] is None
    assert systems_passport["verification_status"] == "executed_systems_only"


def test_tool_cost_summary_excludes_failed_samples_like_official_aggregator() -> None:
    base = {
        "step_count": "3",
        "latency_s_inference": "1",
        "input_tokens_inference": "10",
        "output_tokens_inference": "2",
        "total_tokens_inference": "12",
        "latency_s_memory": "0.2",
        "input_tokens_memory": "0",
        "output_tokens_memory": "0",
        "total_tokens_memory": "0",
        "embedding_tokens": "4",
        "latency_s_total": "1.2",
        "input_tokens_total": "10",
        "output_tokens_total": "2",
        "total_tokens_total": "12",
    }
    metrics = _official_cost_metrics(
        [{**base, "error": ""}, {**base, "step_count": "99", "error": "api"}]
    )
    assert metrics["n_failed"] == 1
    assert metrics["avg_step_count"] == 3.0
    assert metrics["avg_latency_s_total"] == 1.2


def test_gem_schema_has_experience_eligibility_indexes() -> None:
    schema = (
        Path(__file__).parents[1] / "bench/agent_memory/gem/schema.sql"
    ).read_text()
    assert "gem_unit_experience_cutoff_idx" in schema
    assert "gem_unit_experience_protocol_idx" in schema
    assert "experience_ordinal" in schema
    assert "tool_schema_hash" in schema


def test_manifest_and_receipt_are_content_addressed(tmp_path) -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "bench/agent_memory/evomembench/protocol_manifest.json"
    )
    manifest = load_manifest(manifest_path)
    assert len(manifest_sha256(manifest)) == 64
    asset = tmp_path / "asset.jsonl"
    asset.write_text("{}\n")
    local = {
        "assets": [
            {
                "path": "asset.jsonl",
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
        ]
    }
    assert verify_assets(tmp_path, local)["asset.jsonl"]

    receipt = build_experience_receipt(
        run_id="r",
        manifest_sha256="a" * 64,
        track="CrossEp-Know",
        arm="gem_fused",
        target_episode_uid="e2",
        cutoff_ordinal=2,
        task_signature="current task",
        selected_unit_ids=[7],
        selected_episode_uids=["e1"],
        selected_ordinals=[1],
        injection_text="prior experience",
        injection_tokens=2,
        latency_ms=1.5,
        probes={"candidates_examined": 3, "graph_examined": 2},
    )
    assert verify_receipt(receipt.as_dict())
    with pytest.raises(ValueError, match="future"):
        build_experience_receipt(
            run_id="r",
            manifest_sha256="a" * 64,
            track="CrossEp-Know",
            arm="gem_fused",
            target_episode_uid="e2",
            cutoff_ordinal=2,
            task_signature="current task",
            selected_unit_ids=[7],
            selected_episode_uids=["e2"],
            selected_ordinals=[2],
            injection_text="future",
            injection_tokens=1,
            latency_ms=1,
            probes={},
        )
