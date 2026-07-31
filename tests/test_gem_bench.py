"""Unit cover for the [AM] §4.1/§4.2/§4.8 reproduction harness.

No GPU, no database, no vLLM. What is tested here is everything that can be
wrong without one: phase attribution, energy windowing arithmetic, the memory
rendering handed to the generator, and the summary arithmetic that becomes the
paper's tables. The parts that need an engine are exercised by the live suites.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import pytest

from bench.agent_memory import energy as energymod
from bench.agent_memory import serving
from bench.agent_memory.gem.types import Hit, RetrievalMode, UnitState
from bench.agent_memory.gem_bench import points as pointsmod
from bench.agent_memory.gem_bench import report as reportmod
from bench.agent_memory.gem_bench import summarize as summarizemod
from bench.agent_memory.gem_bench.runner import hit_texts


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------


def test_disabled_sampler_reports_unavailable_not_zero():
    sampler = energymod.GpuEnergySampler(enabled=False)
    assert sampler.available is False
    window = sampler.window(0.0, 10.0)
    # The distinction that matters: an unmeasured joule is None, never 0.0.
    assert window.joules is None
    assert window.coverage == 0.0
    assert sampler.describe()["init_error"] == "disabled by caller"


def test_trapezoid_integrates_constant_power():
    stamps = [0.0, 1.0, 2.0, 3.0]
    watts = [100.0, 100.0, 100.0, 100.0]
    assert energymod._trapezoid(stamps, watts, 0.0, 3.0) == pytest.approx(300.0)
    assert energymod._trapezoid(stamps, watts, 0.5, 1.5) == pytest.approx(100.0)


def test_trapezoid_integrates_a_ramp():
    # 0 W -> 100 W over 2 s is 100 J by area, which a midpoint-free sum misses.
    stamps = [0.0, 2.0]
    watts = [0.0, 100.0]
    assert energymod._trapezoid(stamps, watts, 0.0, 2.0) == pytest.approx(100.0)


def test_interpolate_clamps_outside_the_series():
    stamps = [1.0, 2.0]
    values = [10.0, 20.0]
    assert energymod._interpolate(stamps, values, 0.0) == 10.0
    assert energymod._interpolate(stamps, values, 9.0) == 20.0
    assert energymod._interpolate(stamps, values, 1.5) == pytest.approx(15.0)


def _fake_sampler(stamps, watts, millijoules=None, method=energymod.METHOD_POWER):
    sampler = energymod.GpuEnergySampler(enabled=False)
    sampler._stamps = list(stamps)
    sampler._watts = list(watts)
    sampler._millijoules = list(millijoules or [0.0] * len(stamps))
    sampler._handles = [object()]  # available
    sampler._method = method
    return sampler


def test_window_reports_partial_coverage_rather_than_extrapolating():
    sampler = _fake_sampler([1.0, 2.0, 3.0], [100.0, 100.0, 100.0])
    window = sampler.window(0.0, 4.0)
    # Samples span 2 s of a 4 s request; the integral covers only what was seen.
    assert window.joules == pytest.approx(200.0)
    assert window.coverage == pytest.approx(0.5)


def test_window_uses_the_energy_counter_when_available():
    sampler = _fake_sampler(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        millijoules=[0.0, 50_000.0, 125_000.0],
        method=energymod.METHOD_COUNTER,
    )
    window = sampler.window(0.0, 2.0)
    assert window.joules == pytest.approx(125.0)
    assert window.method == energymod.METHOD_COUNTER


def test_window_before_any_sample_is_unavailable_not_zero():
    sampler = _fake_sampler([10.0, 11.0], [50.0, 50.0])
    assert sampler.window(1.0, 2.0).joules is None


def test_sampler_is_a_context_manager_and_stops_cleanly():
    sampler = energymod.GpuEnergySampler(enabled=False)
    with sampler as entered:
        assert entered is sampler
    assert sampler._thread is None


class _FakeNvml:
    """Enough of pynvml to drive the sampler thread without a driver."""

    def __init__(self):
        self.energy = 0

    def nvmlDeviceGetPowerUsage(self, handle):  # noqa: N802 — pynvml's name
        return 50_000  # 50 W in milliwatts

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):  # noqa: N802
        self.energy += 1_000
        return self.energy

    def nvmlShutdown(self):  # noqa: N802
        pass


def _running_sampler(interval=10.0):
    # A deliberately huge interval: if the opening sample were left to the
    # thread's first tick, nothing would be recorded within the test's lifetime.
    sampler = energymod.GpuEnergySampler(interval_seconds=interval, enabled=False)
    sampler._nvml = _FakeNvml()
    sampler._handles = [object()]
    sampler._method = energymod.METHOD_COUNTER
    return sampler


def test_start_samples_immediately_so_the_first_phase_is_measurable():
    sampler = _running_sampler()
    opened = time.perf_counter()
    sampler.start()
    try:
        assert len(sampler._stamps) >= 1
        assert sampler._stamps[0] >= opened
    finally:
        sampler.stop()


def test_stop_closes_the_series_so_the_last_phase_is_not_truncated():
    sampler = _running_sampler()
    sampler.start()
    time.sleep(0.01)
    closed_before_stop = len(sampler._stamps)
    sampler.stop()
    assert len(sampler._stamps) == closed_before_stop + 1


def test_mark_brackets_a_phase_exactly():
    sampler = _running_sampler()
    sampler.start()
    try:
        sampler.mark()
        started = time.perf_counter()
        time.sleep(0.005)
        ended = time.perf_counter()
        sampler.mark()
        window = sampler.window(started, ended)
        # Without the trailing mark the series would stop up to one interval
        # (10 s here) before `ended`, and the window would be truncated.
        assert window.coverage == pytest.approx(1.0)
        assert window.joules is not None
    finally:
        sampler.stop()


def test_mark_is_safe_when_no_gpu_is_present():
    sampler = energymod.GpuEnergySampler(enabled=False)
    stamp = sampler.mark()
    assert stamp > 0
    assert sampler._stamps == []


def test_a_phase_shorter_than_the_poll_interval_still_reports_energy():
    # The defect the live smoke found: a 17 ms construction at a 50 ms poll
    # interval reported joules=None because the series began after it.
    sampler = _running_sampler()
    sampler.start()
    started = time.perf_counter()
    time.sleep(0.005)
    ended = time.perf_counter()
    sampler.stop()
    window = sampler.window(started, ended)
    assert window.joules is not None
    assert window.coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# serving: phase attribution
# ---------------------------------------------------------------------------


class _RecordingEmbeddingClient:
    def __init__(self):
        self.batch_size = 8
        self.calls = []

    def encode(self, texts, *, phase):
        self.calls.append((list(texts), phase))
        return [[0.0, 1.0] for _ in texts]


def test_phased_embedder_tags_the_bound_phase():
    client = _RecordingEmbeddingClient()
    embedder = serving.PhasedEmbedder(client)

    with embedder.phase("construction"):
        embedder.encode(["a"])
    embedder.encode(["b"])  # back to the default

    assert [phase for _, phase in client.calls] == ["construction", "query"]


def test_phased_embedder_refuses_a_conflicting_nested_phase():
    embedder = serving.PhasedEmbedder(_RecordingEmbeddingClient())
    with embedder.phase("construction"):
        with pytest.raises(RuntimeError, match="cross-tagged"):
            with embedder.phase("query"):
                pass


def test_phased_embedder_restores_the_previous_phase_on_error():
    client = _RecordingEmbeddingClient()
    embedder = serving.PhasedEmbedder(client)
    with pytest.raises(ValueError):
        with embedder.phase("construction"):
            raise ValueError("boom")
    embedder.encode(["after"])
    assert client.calls[-1][1] == "query"


def test_ledger_splits_tokens_by_phase_and_drops_the_judge():
    ledger = serving.CallLedger()
    ledger.record(
        kind="construction_llm",
        phase="construction",
        target="m",
        items=1,
        elapsed_seconds=0.1,
        detail={"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
    )
    ledger.record(
        kind="answer_generation",
        phase="qa",
        target="m",
        items=1,
        elapsed_seconds=0.1,
        detail={"usage": {"prompt_tokens": 20, "completion_tokens": 5}},
    )
    ledger.record(
        kind="construction_embedding",
        phase="construction",
        target="e",
        items=4,
        elapsed_seconds=0.1,
        detail={"usage": {"prompt_tokens": 400}},
    )
    ledger.record(
        kind="judge",
        phase="evaluation",
        target="m",
        items=1,
        elapsed_seconds=0.1,
        detail={"usage": {"prompt_tokens": 999, "completion_tokens": 999}},
    )

    tokens = ledger.tokens()
    assert tokens["construction_prompt_tokens"] == 100
    assert tokens["construction_completion_tokens"] == 10
    assert tokens["construction_embed_tokens"] == 400
    assert tokens["qa_prompt_tokens"] == 20
    # A judged token is not a served token; pooling them would inflate Table 3.
    assert tokens["qa_completion_tokens"] == 5

    summary = ledger.summary()
    assert summary["construction_calls"] == 2
    assert summary["qa_calls"] == 1
    assert summary["paper_model_calls"] == 3
    assert summary["judge_calls_excluded_from_paper_total"] == 1


def test_ledger_chat_extractor_bills_construction_not_qa():
    ledger = serving.CallLedger()

    class _Client:
        def chat(self, **kwargs):
            ledger.record(
                kind=kwargs["ledger_kind"],
                phase=kwargs["phase"],
                target=kwargs["model"],
                items=1,
                elapsed_seconds=0.01,
                detail={"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            )
            return "{}", {"prompt_tokens": 7, "completion_tokens": 3}, 0.01

    extractor = serving.LedgerChatExtractor(_Client(), model="m")
    text, usage = extractor.complete("sys", "user")
    assert text == "{}"
    assert usage["prompt_tokens"] == 7
    assert ledger.records[0]["kind"] == "construction_llm"
    assert ledger.records[0]["phase"] == "construction"


def test_judge_protocol_label_never_calls_a_local_judge_the_paper_protocol():
    assert (
        serving.judge_protocol_label(
            enabled=True, model="gpt-4o", base_url="https://api.openai.com/v1"
        )
        == "memoryagentbench_gpt4o"
    )
    assert (
        serving.judge_protocol_label(
            enabled=True, model="Qwen/Qwen3-32B", base_url="http://127.0.0.1:8000/v1"
        )
        == "protocol_variant"
    )
    assert (
        serving.judge_protocol_label(enabled=False, model="gpt-4o", base_url="")
        == "not_run"
    )


def test_pipeline_still_exposes_the_names_it_moved_to_serving():
    # Guards the extraction: the embedRAG arm must keep importing through the
    # shared module rather than re-growing a private copy.
    import bench.agent_memory.tridbBackend.longmemeval_pipeline as pipeline

    for name in (
        "SYSTEM_MESSAGE",
        "StreamingChatResult",
        "build_answer_messages",
        "build_judge_prompt",
        "extract_retrieval_query",
        "fit_answer_prompt",
        "latency_summary",
        "load_workloads",
        "parse_judge_yes_no",
    ):
        assert hasattr(pipeline, name), name
    assert pipeline.fit_answer_prompt is serving.fit_answer_prompt


# ---------------------------------------------------------------------------
# operating points
# ---------------------------------------------------------------------------


def test_every_point_is_labelled_a_proxy():
    for point in pointsmod.POINTS:
        payload = point.to_dict()
        assert payload["paradigm_proxy"] is True
        assert "Not the [AM] system" in payload["proxy_note"]


def test_the_matrix_matches_the_documented_configuration():
    by_key = pointsmod.POINTS_BY_KEY
    assert by_key["II_embedrag"].mode is RetrievalMode.VECTOR
    assert not by_key["II_embedrag"].uses_llm_construction
    assert by_key["IIIa_graphrag_like"].uses_llm_construction
    assert by_key["IV_agentic"].forget is True
    conformant = by_key["gem_conformant"]
    # The GEM row's whole purpose is an ingest-identical delta against II.
    assert conformant.ingest == by_key["II_embedrag"].ingest
    assert (conformant.reinforce, conformant.revise, conformant.forget) == (
        True,
        True,
        True,
    )


def test_resolve_rejects_an_unknown_point():
    assert pointsmod.resolve(None) == list(pointsmod.POINTS)
    assert [p.key for p in pointsmod.resolve(["IV_agentic"])] == ["IV_agentic"]
    with pytest.raises(ValueError, match="unknown operating point"):
        pointsmod.resolve(["nope"])


def test_build_strategy_covers_every_point():
    extractor = SimpleNamespace(complete=lambda system, user: ("{}", {}))
    embedder = SimpleNamespace(encode=lambda texts: [[0.0] for _ in texts])
    seen = set()
    for point in pointsmod.POINTS:
        strategy = pointsmod.build_strategy(
            point,
            extractor=extractor,
            embedder=embedder,
            construction_model="m",
            chunk_tokens=4096,
            embedding_batch_size=64,
        )
        seen.add(strategy.name)
        if point.ingest == pointsmod.INGEST_AGENTIC:
            # [AM] Recommendation 10: the caps are required, never defaulted.
            assert strategy.max_rounds > 0 and strategy.max_tool_calls > 0
    assert seen == {"deterministic", "llm_mediated", "agentic"}


def test_sequential_llm_strategy_demands_an_embedder():
    with pytest.raises(ValueError, match="sequential mode requires an embedder"):
        pointsmod.build_strategy(
            pointsmod.POINTS_BY_KEY["IIIb_mem0_like"],
            extractor=SimpleNamespace(),
            embedder=None,
            construction_model="m",
            chunk_tokens=4096,
            embedding_batch_size=64,
        )


# ---------------------------------------------------------------------------
# memories handed to the generator
# ---------------------------------------------------------------------------


def _hit(unit_id, title, field, value, score=0.5):
    return Hit(
        unit_id=unit_id,
        title=title,
        field_name=field,
        value=value,
        score=score,
        state=UnitState.ACTIVE,
    )


def test_hit_texts_emits_a_bare_chunk_for_the_deterministic_arm():
    # Byte-identical to what the embedRAG pipeline sends, so the G2 comparison
    # measures retrieval rather than prompt formatting.
    hits = [_hit(1, "lme_00_history#0", "content", "the chunk body")]
    assert hit_texts(hits, 5) == ["the chunk body"]


def test_hit_texts_groups_fields_of_one_unit_into_one_memory():
    hits = [
        _hit(7, "Trip to Kyoto", "city", "Kyoto"),
        _hit(7, "Trip to Kyoto", "month", "April"),
        _hit(9, "Budget", "amount", "1200"),
    ]
    texts = hit_texts(hits, 5)
    assert len(texts) == 2
    assert texts[0] == "Trip to Kyoto\ncity: Kyoto\nmonth: April"
    assert texts[1] == "Budget\namount: 1200"


def test_hit_texts_limits_units_not_hits():
    hits = [
        _hit(1, "A", "f1", "v1"),
        _hit(1, "A", "f2", "v2"),
        _hit(2, "B", "f1", "v1"),
        _hit(3, "C", "f1", "v1"),
    ]
    assert len(hit_texts(hits, 2)) == 2


def test_hit_texts_keeps_a_fieldless_unit_as_its_title():
    assert hit_texts([_hit(4, "Bare Topic", None, None)], 5) == ["Bare Topic"]


# ---------------------------------------------------------------------------
# summary arithmetic
# ---------------------------------------------------------------------------


def _args(**overrides):
    base = dict(
        top_k=10,
        max_prompt_memories=5,
        prompt_token_budget=36_000,
        chunk_size=4096,
        term_cond=32,
        skip_judge=False,
        judge_model="Qwen/Qwen3-32B",
        judge_base_url="http://127.0.0.1:8000/v1",
        max_rejection_rate=None,
        agentic_max_rounds=8,
        agentic_max_tool_calls=24,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _timing(ttft, total, retrieval=0.2, generation=None):
    return {
        "effective_ttft_seconds": ttft,
        "total_seconds": total,
        "query_embedding_seconds": 0.05,
        "gem_retrieval_seconds": retrieval,
        "retrieval_phase_seconds": retrieval + 0.05,
        "prompt_assembly_seconds": 0.01,
        "vllm_queue_prefill_seconds": ttft - retrieval - 0.05,
        "generation_seconds": generation
        if generation is not None
        else total - retrieval,
        "decode_seconds": total - ttft,
    }


def _prediction(question_id, ttft, total, joules=None, correct_type="multi-session"):
    return {
        "question_id": question_id,
        "question_type": correct_type,
        "timing": _timing(ttft, total),
        "retrieval_cost": {"db_statements": 3, "embed_calls": 0},
        "energy": {"joules": joules, "coverage": 1.0 if joules is not None else 0.0},
    }


def _summarize(predictions, judge_results, *, point=None, construction=None):
    ledger = serving.CallLedger()
    ledger.record(
        kind="construction_embedding",
        phase="construction",
        target="e",
        items=10,
        elapsed_seconds=1.0,
        detail={"usage": {"prompt_tokens": 1000}},
    )
    for prediction in predictions:
        ledger.record(
            kind="answer_generation",
            phase="qa",
            target="m",
            items=1,
            elapsed_seconds=0.5,
            detail={"usage": {"prompt_tokens": 50, "completion_tokens": 10}},
        )
    construction_records = construction or [
        {
            "construction_seconds": 12.0,
            "ingest_cost": {"embed_calls": 1, "db_statements": 40},
            "capped": False,
            "rejected": 0,
            "energy": {"joules": 600.0, "coverage": 1.0},
        }
    ]
    return summarizemod.summarize_point(
        point=point or pointsmod.POINTS_BY_KEY["II_embedrag"],
        args=_args(),
        predictions=predictions,
        judge_results=judge_results,
        ledger=ledger,
        construction_records=construction_records,
        maintenance_records=[],
        lifecycle_seconds=30.0,
        sampler=energymod.GpuEnergySampler(enabled=False),
    )


def test_section_4_1_excludes_construction_from_per_query_latency():
    predictions = [
        _prediction("q1", 1.0, 2.0),
        _prediction("q2", 1.0, 4.0),
    ]
    summary = _summarize(predictions, [])
    # (2 + 4) / 2 = 3 s per query. The 12 s construction must not appear.
    assert summary["section_4_1"][
        "mean_qa_wallclock_per_query_seconds"
    ] == pytest.approx(3.0)
    assert summary["section_4_2"]["construction_wallclock_seconds"] == pytest.approx(
        12.0
    )


def test_section_4_2_prices_the_whole_lifecycle_in_joules():
    predictions = [
        _prediction("q1", 1.0, 2.0, joules=100.0),
        _prediction("q2", 1.0, 2.0, joules=100.0),
    ]
    judge = [
        {"question_id": "q1", "question_type": "multi-session", "correct": True},
        {"question_id": "q2", "question_type": "multi-session", "correct": False},
    ]
    summary = _summarize(predictions, judge)
    section = summary["section_4_2"]
    assert section["construction_kilojoules"] == pytest.approx(0.6)
    assert section["qa_kilojoules"] == pytest.approx(0.2)
    assert section["total_kilojoules"] == pytest.approx(0.8)
    # One correct answer carries the whole 800 J.
    assert section["joules_per_correct"] == pytest.approx(800.0)


def test_energy_is_none_when_a_window_went_unmeasured():
    predictions = [
        _prediction("q1", 1.0, 2.0, joules=100.0),
        _prediction("q2", 1.0, 2.0, joules=None),
    ]
    summary = _summarize(predictions, [])
    qa = summary["energy"]["qa"]
    assert qa["windows_missing"] == 1
    assert qa["complete"] is False
    # Partial data still reports the measured sum, flagged, rather than a
    # silently-low total presented as complete.
    assert qa["joules"] == pytest.approx(100.0)


def test_no_energy_at_all_leaves_the_table_columns_absent():
    predictions = [_prediction("q1", 1.0, 2.0)]
    summary = _summarize(
        predictions,
        [{"question_id": "q1", "question_type": "multi-session", "correct": True}],
        construction=[
            {
                "construction_seconds": 5.0,
                "ingest_cost": {},
                "capped": False,
                "rejected": 0,
                "energy": {"joules": None, "coverage": 0.0},
            }
        ],
    )
    assert summary["section_4_2"]["total_kilojoules"] is None
    assert summary["section_4_2"]["joules_per_correct"] is None


def test_section_4_8_reports_tail_width_and_the_bound_regime():
    predictions = [_prediction(f"q{i}", 1.0, float(i)) for i in range(1, 21)]
    summary = _summarize(predictions, [])
    section = summary["section_4_8"]
    assert section["qa_p50_seconds"] == pytest.approx(10.5)
    assert section["qa_p95_over_p50"] > 1.0
    assert section["bound_regime"] == "algorithm_bounded"

    agentic = _summarize(predictions, [], point=pointsmod.POINTS_BY_KEY["IV_agentic"])
    assert agentic["section_4_8"]["bound_regime"] == "llm_bounded"
    assert agentic["section_4_8"]["iteration_caps"]["agentic_max_rounds"] == 8


def test_operator_meter_is_reported_beside_the_http_ledger():
    summary = _summarize([_prediction("q1", 1.0, 2.0)], [])
    assert summary["operator_meter"]["construction"]["db_statements"] == 40
    assert summary["operator_meter"]["retrieval"]["db_statements"] == 3
    assert "authoritative" in summary["operator_meter"]["note"]


def test_paper_sections_computes_the_cross_arm_spread():
    fast = _summarize([_prediction("q1", 0.1, 0.5)], [])
    slow = _summarize(
        [_prediction("q1", 5.0, 10.0)],
        [],
        point=pointsmod.POINTS_BY_KEY["IV_agentic"],
    )
    sections = summarizemod.paper_sections([fast, slow])
    spread = sections["section_4_1"]["serving_latency_spread"]
    assert spread["ratio"] == pytest.approx(20.0)
    assert spread["arms"] == 2
    assert (
        "4.7" in sections["section_4_7"] or "not reproduced" in sections["section_4_7"]
    )
    assert "TriDB/GEM operating points only" in sections["scope"]


def test_spread_is_not_computable_from_one_arm():
    only = _summarize([_prediction("q1", 1.0, 2.0)], [])
    sections = summarizemod.paper_sections([only])
    assert sections["section_4_1"]["serving_latency_spread"]["ratio"] is None


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def _manifest():
    return {
        "models": {
            "answer": "Qwen/Qwen3-32B",
            "answer_base_url": "http://127.0.0.1:8000/v1",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_base_url": "http://127.0.0.1:8001/v1",
            "embedding_dim": 1024,
            "judge": "Qwen/Qwen3-32B",
            "judge_base_url": "http://127.0.0.1:8000/v1",
        },
        "evaluation": {"judge_protocol": "protocol_variant"},
        "energy": {"available": False, "init_error": "pynvml not installed"},
    }


def test_report_renders_every_section_and_marks_missing_numbers():
    summary = _summarize([_prediction("q1", 1.0, 2.0)], [])
    sections = summarizemod.paper_sections([summary])
    text = reportmod.render(sections, _manifest())
    assert "## §4.1" in text
    assert "## §4.2" in text
    assert "## §4.8" in text
    # Accuracy was never judged in this fixture; it must print as n/a, not 0.
    assert "n/a" in text
    assert "NOT sampled" in text
    assert "not the paper's H100" in text


def test_report_writes_both_artifacts(tmp_path):
    summary = _summarize([_prediction("q1", 1.0, 2.0)], [])
    sections = summarizemod.paper_sections([summary])
    paths = reportmod.write(sections, _manifest(), tmp_path)
    assert paths["report"].exists()
    assert paths["sections"].exists()
    assert "§4.1" in paths["report"].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _cli(*extra):
    from bench.agent_memory.gem_bench.__main__ import build_parser

    return build_parser().parse_args(
        ["--input", "x.json", "--output-dir", "out", *extra]
    )


def test_cli_defaults_the_judge_to_the_local_answer_endpoint():
    from bench.agent_memory.gem_bench.__main__ import _validate, build_parser

    parser = build_parser()
    args = _cli()
    _validate(args, parser)
    assert args.judge_base_url == serving.DEFAULT_ANSWER_BASE_URL
    assert args.judge_model == args.answer_model
    assert (
        serving.judge_protocol_label(
            enabled=True, model=args.judge_model, base_url=args.judge_base_url
        )
        == "protocol_variant"
    )


def test_cli_rejects_a_second_model_on_a_single_model_endpoint():
    from bench.agent_memory.gem_bench.__main__ import _validate, build_parser

    parser = build_parser()
    args = _cli("--judge-model", "gpt-4o")
    with pytest.raises(SystemExit):
        _validate(args, parser)


def test_cli_rejects_more_prompt_memories_than_retrieved():
    from bench.agent_memory.gem_bench.__main__ import _validate, build_parser

    parser = build_parser()
    args = _cli("--top-k", "3", "--max-prompt-memories", "5")
    with pytest.raises(SystemExit):
        _validate(args, parser)


def test_sampler_thread_does_not_outlive_the_run():
    sampler = energymod.GpuEnergySampler(interval_seconds=0.01, enabled=False)
    sampler.start()
    time.sleep(0.02)
    sampler.stop()
    assert sampler._thread is None
