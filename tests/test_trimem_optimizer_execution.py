"""Selected-only execution contracts; no database, GPU, or model requests."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from trimem.optimizer.compiler import QueryContext, compile_candidate
from trimem.optimizer.executor import (
    CapabilityError,
    NativeExecutor,
    check_streaming_plan,
)
from trimem.optimizer.packing import pack_stream
from trimem.optimizer.schema import CandidateSpec


def compile_plan(plan_id="vgr_v1", **kwargs):
    action = CandidateSpec(plan_id, **kwargs)
    return compile_candidate(action, QueryContext("task-a", (0.1, 0.2), parent_id=12))


def explain_for(plan):
    def wrap(kind, child, **attrs):
        return {"Node Type": kind, "Plans": [child], **attrs}

    root = wrap(
        "Limit",
        {"Node Type": "Index Scan", "Order By": "(emb <=> '[0.1,0.2]'::vector)"},
    )
    if plan.action.filter_target == "seed":
        root = wrap(
            "Custom Scan",
            root,
            **{"Custom Plan Provider": "TriMEM SemFilter", "Batch Size": "1 (bounded)"},
        )
    for hop in plan.action.hop_types:
        expands = [
            {"Node Type": "Custom Scan", "Custom Plan Provider": "TriMEM Expand"}
            for _edge in hop.split("|")
        ]
        inner = (
            expands[0]
            if len(expands) == 1
            else {"Node Type": "Append", "Plans": expands}
        )
        root = {"Node Type": "Nested Loop", "Plans": [root, inner]}
    root = wrap("Limit", root)
    if plan.action.filter_target == "row":
        root = wrap(
            "Custom Scan",
            root,
            **{"Custom Plan Provider": "TriMEM SemFilter", "Batch Size": "1 (bounded)"},
        )
    if plan.action.uses_score:
        root = wrap(
            "Custom Scan",
            root,
            **{"Custom Plan Provider": "TriMEM SemScore", "Batch Size": 1},
        )
    return [{"Plan": root}]


class FakeCursor:
    def __init__(self, connection, name):
        self.connection, self.name = connection, name
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        self.connection.events.append(("close", self.name))

    def execute(self, sql, params):
        self.connection.events.append(("execute", self.name, sql, params))

    def fetchone(self):
        if self.name is None:
            return (self.connection.explain,)
        self.connection.fetches += 1
        item = next(self.connection.rows, None)
        if isinstance(item, Exception):
            raise item
        return item


class FakeConnection:
    info = SimpleNamespace(server_version=160000)

    def __init__(self, explain, rows=()):
        self.explain, self.rows = explain, iter(rows)
        self.events, self.cursors, self.fetches = [], [], 0

    @contextmanager
    def transaction(self, *, force_rollback):
        assert force_rollback is True
        self.events.append(("transaction_open",))
        try:
            yield
        finally:
            self.events.append(("transaction_rollback",))

    def cursor(self, name=None):
        cursor = FakeCursor(self, name)
        self.cursors.append(cursor)
        return cursor


class Collector:
    def __init__(self):
        self.events = []

    def start(self, execution_id, plan):
        self.events.append(("start", execution_id))
        return execution_id

    def finish(self, handle):
        self.events.append(("finish", handle))
        return [{"request_id": handle, "input_tokens": 21, "output_tokens": 3}]


@pytest.mark.parametrize(
    "plan_id", ["vgr_v1", "vgr_score_v1", "seed_sem_expand_v1", "expand_row_sem_v1"]
)
def test_compiler_uses_native_endpoint_path_without_full_materialization(plan_id):
    plan = compile_plan(plan_id, hop_types=("similar", "lineage"), seed_count=8)
    sql = plan.sql.upper()
    assert sql.count("TRIMEM.EXPAND") == 2
    assert (
        sql.count("ORDER BY") == 1
    )  # ANN order only; actual index checked separately.
    assert all(
        keyword not in sql
        for keyword in ("DISTINCT", "UNION", "GROUP BY", "MATERIALIZED", "SORT")
    )
    assert plan.emitted_edge_upper_bound == 8 * (8 + 64)
    check_streaming_plan(explain_for(plan), plan)


@pytest.mark.parametrize(
    "plan_id", ["vgr_v1", "vgr_score_v1", "seed_sem_expand_v1", "expand_row_sem_v1"]
)
def test_seeds_require_change_summary_with_exact_order_iterative_scan(plan_id):
    plan = compile_plan(plan_id, seed_count=8)
    seeds = plan.sql[plan.sql.index("FROM attempt a") : plan.sql.index("LIMIT 8")]
    assert "a.changes IS NOT NULL" in seeds
    assert ("hnsw.iterative_scan", "strict_order") in plan.settings


@pytest.mark.parametrize(
    ("plan_id", "target"),
    [("seed_sem_expand_noscore_v1", "seed"), ("expand_row_sem_noscore_v1", "row")],
)
def test_noscore_semfilter_plans_have_no_stream_score(plan_id, target):
    plan = compile_plan(plan_id, hop_types=("similar", "lineage"), seed_count=8)
    assert plan.action.filter_target == target and not plan.action.uses_score
    assert "sem_score" not in plan.sql and plan.sql.count("trimem.sem_filter") == 1
    assert "score_admit_min" not in plan.action.to_dict()
    assert "score_predicate" not in plan.action.to_dict()
    assert plan.expected_custom_nodes == (
        "TriMEM Expand",
        "TriMEM Expand",
        "TriMEM SemFilter",
    )
    assert "ORDER BY" in plan.sql and plan.sql.upper().count("ORDER BY") == 1
    check_streaming_plan(explain_for(plan), plan)
    scored = compile_plan(
        plan_id.replace("_noscore", ""), hop_types=("similar", "lineage"), seed_count=8
    )
    with pytest.raises(CapabilityError):
        check_streaming_plan(explain_for(scored), plan)


def test_seeds_exclude_noop_edits_identical_to_parent_code():
    plan = compile_plan("vgr_v1", seed_count=8)
    seeds = plan.sql[plan.sql.index("FROM attempt a") : plan.sql.index("LIMIT 8")]
    assert "a.is_valid" in seeds
    # No COALESCE: a root (NULL parent) makes both comparisons NULL -> not a seed.
    assert "COALESCE" not in seeds
    assert (
        "a.code <> (SELECT pa.code FROM attempt pa WHERE pa.id = a.parent_id)" in seeds
    )
    assert (
        "a.fitness > (SELECT pa.fitness FROM attempt pa WHERE pa.id = a.parent_id)"
        in seeds
    )


def test_predicates_and_user_input_are_bound_not_sql_interpolated():
    attack = "'); DROP TABLE attempt; --"
    action = CandidateSpec("expand_row_sem_v1", filter_predicate=attack)
    plan = compile_candidate(action, QueryContext(attack, (0.0, 1.0)))
    assert attack not in plan.sql
    assert plan.params["filter_predicate"] == attack
    assert plan.params["task"] == attack
    with pytest.raises(ValueError):
        QueryContext("x", (float("nan"),))
    with pytest.raises(ValueError):
        compile_plan(seed_count=48, fanout=16, hop_types=("lineage",) * 4)


@pytest.mark.parametrize(
    "node_type",
    [
        "Sort",
        "Incremental Sort",
        "Hash",
        "Hash Join",
        "Aggregate",
        "Materialize",
        "Memoize",
        "Function Scan",
        "Gather",
    ],
)
def test_blocking_or_unverified_physical_nodes_rejected(node_type):
    plan = compile_plan()
    root = {"Node Type": node_type, "Plans": [explain_for(plan)[0]["Plan"]]}
    with pytest.raises(CapabilityError, match="blocking or unverified"):
        check_streaming_plan(root, plan)


def test_missing_ann_or_semantic_order_fails_closed():
    plan = compile_plan("expand_row_sem_v1")
    explain = explain_for(plan)
    text = repr(explain)
    assert "<=>" in text

    def walk(node):
        yield node
        for child in node.get("Plans", []):
            yield from walk(child)

    for node in walk(explain[0]["Plan"]):
        if "Order By" in node:
            del node["Order By"]
    with pytest.raises(CapabilityError, match="ordered index|non-frontier"):
        check_streaming_plan(explain, plan)
    explain = explain_for(plan)
    score = explain[0]["Plan"]
    sem = score["Plans"][0]
    assert sem["Custom Plan Provider"] == "TriMEM SemFilter"
    sem["Batch Size"] = "8 (bounded)"
    with pytest.raises(CapabilityError, match="batch size 1"):
        check_streaming_plan(explain, plan)
    wrong = compile_plan("seed_sem_expand_v1")
    with pytest.raises(CapabilityError, match="target/order"):
        check_streaming_plan(explain_for(plan), wrong)


def test_graph_rescan_on_relational_outer_would_invalidate_edge_bound():
    plan = compile_plan()
    root = explain_for(plan)[0]["Plan"]
    root = {"Node Type": "Nested Loop", "Plans": [{"Node Type": "Index Scan"}, root]}
    with pytest.raises(CapabilityError, match="rescan loop"):
        check_streaming_plan(root, plan)


def test_stream_score_must_follow_filter_even_when_both_are_after_graph():
    plan = compile_plan("expand_row_sem_v1")
    root = explain_for(plan)[0]["Plan"]
    sem = root["Plans"][0]
    raw = sem["Plans"][0]
    root["Plans"] = [raw]
    sem["Plans"] = [root]
    with pytest.raises(CapabilityError, match="before SemFilter"):
        check_streaming_plan(sem, plan)


def test_selected_query_executes_once_and_closes_without_tail_fetch():
    plan = compile_plan()
    conn = FakeConnection(explain_for(plan), ({"id": i} for i in range(100)))
    with NativeExecutor(conn).open_selected(plan) as cursor:
        assert next(cursor)["id"] == 0
        cursor.close("top_k")
    assert conn.fetches == 1
    executions = [e for e in conn.events if e[0] == "execute"]
    assert sum(e[1] is not None for e in executions) == 1
    assert sum(e[2].startswith("EXPLAIN (FORMAT JSON)") for e in executions) == 1
    assert not any("ANALYZE" in e[2] for e in executions)
    assert all(c.closed for c in conn.cursors)
    assert conn.events[-1] == ("transaction_rollback",)
    assert cursor.receipt.actual_executions == 1
    assert cursor.receipt.closed and cursor.receipt.usage_complete
    assert cursor.receipt.edge_visits_measured is None


def test_capability_failure_never_opens_actual_query():
    plan = compile_plan()
    bad = [{"Plan": {"Node Type": "Sort", "Plans": [explain_for(plan)[0]["Plan"]]}}]
    conn = FakeConnection(bad)
    with pytest.raises(CapabilityError), NativeExecutor(conn).open_selected(plan):
        pytest.fail("capability failure must prevent actual execution")
    assert all(cursor.name is None for cursor in conn.cursors)
    assert all(cursor.closed for cursor in conn.cursors)


def test_semantic_usage_is_required_and_collected_after_close_including_failure():
    plan = compile_plan("vgr_score_v1")
    conn = FakeConnection(
        explain_for(plan), [{"id": 1, "sem_score": 0.2}, RuntimeError("model failure")]
    )
    with (
        pytest.raises(ValueError, match="UsageCollector"),
        NativeExecutor(conn).open_selected(plan),
    ):
        pass
    assert not conn.events
    collector = Collector()
    with (
        pytest.raises(RuntimeError, match="model failure"),
        NativeExecutor(conn, collector).open_selected(plan) as cursor,
    ):
        next(cursor)
    assert cursor.receipt.score_rejected == 1
    assert cursor.receipt.closed and cursor.receipt.stop_reason == "fetch_error"
    assert cursor.receipt.usage_complete
    assert cursor.receipt.usage_events[0]["input_tokens"] == 21
    assert [event[0] for event in collector.events] == ["start", "finish"]
    assert all(c.closed for c in conn.cursors)


def test_usage_collection_failure_does_not_become_zero_tokens():
    class BrokenCollector(Collector):
        def finish(self, handle):
            raise RuntimeError("missing provider usage")

    plan = compile_plan("vgr_score_v1")
    conn = FakeConnection(explain_for(plan), [{"id": 1, "sem_score": 0.8}])
    with (
        pytest.raises(RuntimeError, match="missing provider usage"),
        NativeExecutor(conn, BrokenCollector()).open_selected(plan) as cursor,
    ):
        next(cursor)
    assert cursor.receipt.closed and not cursor.receipt.usage_complete
    assert all(c.closed for c in conn.cursors)


def test_no_memory_does_not_access_db():
    plan = compile_candidate(CandidateSpec("no_memory_v1"))
    with NativeExecutor(None).open_selected(plan) as cursor:
        assert list(cursor) == []
    assert cursor.receipt.actual_executions == 0
    assert cursor.receipt.stop_reason == "no_memory"


class ExactTestTokenizer:
    """An explicit test tokenizer, not a production character-count estimate."""

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


class Rows:
    def __init__(self, rows):
        self.rows, self.seen, self.closed = iter(rows), 0, False

    def __iter__(self):
        return self

    def __next__(self):
        self.seen += 1
        return next(self.rows)

    def close(self, reason):
        self.closed, self.reason = True, reason


def pack(rows, **kwargs):
    return pack_stream(
        rows, ExactTestTokenizer(), tokenizer_id="exact-test-v1", **kwargs
    )


def test_8192_exact_token_limit_counts_truncations_without_fetching_tail():
    source = Rows([{"id": "one", "code": "数" * 9000}, {"id": "unconsumed"}])
    result = pack(source, top_k=5)
    assert result.memory_tokens <= 8192
    assert result.memory_tokens == len(ExactTestTokenizer().encode(result.text))
    assert result.truncated_records == 1 and result.prompt_was_truncated
    assert result.events[0].original_tokens > 8192
    assert result.events[0].retained_tokens <= 8192
    assert source.seen == 1 and source.closed


def test_complete_prompt_and_output_reserve_tighten_memory_budget():
    source = Rows([{"id": "one", "code": "x" * 1000}])
    result = pack(
        source,
        top_k=1,
        model_context_window=100,
        max_output_tokens=20,
        render_prompt=lambda text: "SYSTEM:" + text + ":QUESTION",
    )
    assert result.full_prompt_tokens + 20 <= 100
    assert result.memory_tokens < 100
    assert result.truncated_records == 1
    assert source.closed


def test_duplicates_bounded_by_retained_context_and_top_k_stops():
    source = Rows([{"id": 1}, {"id": 1}, {"id": 2}, {"id": "tail"}])
    result = pack(source, top_k=2)
    assert result.retained_ids == ["1", "2"]
    assert result.duplicate_records == 1
    assert source.seen == 3 and source.reason == "top_k"


def test_oversized_unfit_record_is_skipped_and_next_small_record_can_fit():
    source = Rows([{"id": 1, "code": "x" * 50}, {"id": 2, "code": "ok"}])
    result = pack(
        source, top_k=1, memory_token_cap=3, record_formatter=lambda row: row["code"]
    )
    assert result.retained_ids == ["2"] and result.skipped_records == 1
    assert result.events[0].disposition == "skipped_budget"


def test_packer_closes_even_when_tokenizer_or_renderer_fails():
    source = Rows([{"id": 1}])

    def broken_renderer(_):
        raise RuntimeError("tokenizer unavailable")

    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        pack(source, top_k=1, render_prompt=broken_renderer)
    assert source.closed
    source = Rows([{"id": 1}])
    with pytest.raises(ValueError, match="render_prompt"):
        pack(source, top_k=1, model_context_window=100)
    assert source.closed and source.seen == 0


def test_context_exhausted_before_opening_source_is_not_silently_overflowed():
    source = Rows([{"id": 1}])
    with pytest.raises(ValueError, match="base prompt"):
        pack(
            source, top_k=1, model_context_window=5, render_prompt=lambda _: "too long"
        )
    assert source.closed and source.seen == 0


def test_context_packing_does_not_mutate_row_payloads():
    row = {"id": 1, "code": "z" * 1000}
    original = copy.deepcopy(row)
    pack(Rows([row]), top_k=1, memory_token_cap=100)
    assert row == original
