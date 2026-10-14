"""Opt-in, isolated stock-PG integration; no existing DB or LLM is contacted.

TRIMEM_OPTIMIZER_TEST_PG_PREFIX=/absolute/path/.pgprefix16
taskset -c 16-19 .venv/bin/python -m pytest -q tests/test_trimem_optimizer_native.py

Uses installed binaries, including their installed extension version. This is
not a source rebuild or a GX10 fork/CUDA sign-off. All semantic fixture results
are prefilled in the private cache; no worker is started. The private cluster
has a unique Unix socket and no TCP listener, and is removed after testing.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from trimem.optimizer.compiler import QueryContext, compile_candidate
from trimem.optimizer.executor import NativeExecutor, check_streaming_plan
from trimem.optimizer.schema import DEFAULT_PREDICATE, CandidateSpec

psycopg = pytest.importorskip("psycopg")

PREFIX = os.environ.get("TRIMEM_OPTIMIZER_TEST_PG_PREFIX")
pytestmark = pytest.mark.skipif(not PREFIX, reason="isolated native gate is opt-in")


class CacheOnlyCollector:
    """Fixture-only collector: every key exists and there is no model transport.

    A cache miss fails via the short mailbox timeout; it cannot invoke a model.
    This collector is unsuitable for an experiment that runs a semantic worker.
    """

    def start(self, execution_id, plan):
        return execution_id

    def finish(self, handle):
        return []


@pytest.fixture(scope="module")
def private_connection():
    bindir = Path(PREFIX) / "usr/lib/postgresql/16/bin"
    cpuset = os.environ.get("TRIMEM_OPTIMIZER_TEST_CPUSET", "16-19")

    def command(name, *args):
        result = subprocess.run(
            ["taskset", "-c", cpuset, str(bindir / name), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed:\n{result.stdout}")

    with tempfile.TemporaryDirectory(prefix="trimem-opt-native-") as directory:
        root = Path(directory)
        data, sock = root / "data", root / "socket"
        sock.mkdir()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        assert port not in (5432, 55440, 55441)
        started = False
        try:
            command(
                "initdb",
                "-D",
                str(data),
                "-A",
                "trust",
                "--no-locale",
                "--encoding=UTF8",
            )
            options = (
                f"-k {sock} -p {port} -c listen_addresses='' "
                "-c shared_buffers=32MB -c max_connections=8 -c autovacuum=off "
                "-c shared_preload_libraries=tjs_pg,trimem_pg "
                "-c trimem.timeout_ms=100 -c statement_timeout=5000 -c fsync=off"
            )
            command(
                "pg_ctl",
                "-D",
                str(data),
                "-l",
                str(root / "postgres.log"),
                "-o",
                options,
                "-w",
                "start",
            )
            started = True
            with psycopg.connect(
                host=str(sock), port=port, dbname="postgres", autocommit=True
            ) as connection:
                for extension in ("vector", "graph_store_am", "tjs_pg", "trimem_pg"):
                    connection.execute(f"CREATE EXTENSION {extension}")
                connection.execute("""CREATE TABLE attempt(
                    id bigint PRIMARY KEY, vid bigint UNIQUE,
                    task text, cell text, parent_id bigint, fitness float8,
                    is_valid bool, changes text, code text, emb vector(2))""")
                connection.execute("""INSERT INTO attempt
                    SELECT g, graph_store.gph_upsert_vertex(g), 'fixture', NULL,
                           CASE WHEN g>1 THEN g-1 END, g::float8, true,
                           'algorithm change '||g, 'code '||g,
                           ARRAY[cos(g::float8),sin(g::float8)]::vector(2)
                    FROM generate_series(1,400) g""")
                connection.execute(
                    "CREATE INDEX attempt_emb_hnsw ON attempt USING hnsw (emb vector_cosine_ops)"
                )
                for edge, offset in (("lineage", 1), ("similar", 2)):
                    connection.execute(
                        "SELECT graph_store.register_edge_type(%s)", (edge,)
                    )
                    connection.execute(
                        """SELECT graph_store.gph_insert_edge(
                        p.vid, d.vid, graph_store.register_edge_type(%s))
                        FROM attempt p JOIN attempt d ON d.id=p.id+%s""",
                        (edge, offset),
                    )
                connection.execute("SELECT trimem.analyze_graph()")
                connection.execute("ANALYZE attempt")
                for op in ("sem_filter", "sem_score"):
                    connection.execute(
                        """INSERT INTO trimem.sem_cache
                        (key,op,model,prompt_version,predicate,verdict,score)
                        SELECT trimem.key(%(op)s,current_setting('trimem.model'),
                                         'optimizer-v1',%(pred)s,changes),
                               %(op)s,current_setting('trimem.model'),
                               'optimizer-v1',%(pred)s,true,0.8
                        FROM attempt""",
                        {"op": op, "pred": DEFAULT_PREDICATE},
                    )
                yield connection
        finally:
            if started:
                command("pg_ctl", "-D", str(data), "-m", "fast", "-w", "stop")


@pytest.mark.parametrize(
    "plan_id",
    [
        "vgr_v1",
        "vgr_score_v1",
        "seed_sem_expand_v1",
        "expand_row_sem_v1",
        "seed_sem_expand_noscore_v1",
        "expand_row_sem_noscore_v1",
    ],
)
@pytest.mark.parametrize(
    "hop_types",
    [
        ("lineage",),
        ("similar", "lineage"),
        ("lineage|similar",),
        ("similar", "lineage|similar"),
        ("lineage|similar", "lineage"),
        ("lineage|similar",) * 2,
        ("lineage|similar",) * 3,
    ],
)
@pytest.mark.parametrize("frontier", ["final", "all"])
def test_real_native_explain_and_named_cursor_close(
    private_connection, plan_id, hop_types, frontier
):
    action = CandidateSpec(
        plan_id, seed_count=8, fanout=4, top_k=3, hop_types=hop_types, frontier=frontier
    )
    plan = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
    with private_connection.transaction(force_rollback=True):
        for name, value in plan.settings:
            private_connection.execute("SELECT set_config(%s,%s,true)", (name, value))
        doc = private_connection.execute(
            "EXPLAIN (FORMAT JSON) " + plan.sql, plan.params
        ).fetchone()[0]
        check_streaming_plan(doc, plan)
    with NativeExecutor(private_connection, CacheOnlyCollector()).open_selected(
        plan
    ) as cursor:
        rows = []
        for row in cursor:
            rows.append(row)
            if len(rows) >= 3:
                cursor.close("top_k")
                break
    assert len(rows) == 3
    assert cursor.receipt.closed
    assert cursor.receipt.actual_executions == 1
    assert cursor.receipt.rows_fetched == 3
    assert cursor.receipt.usage_complete and cursor.receipt.usage_events == []
    assert cursor.receipt.edge_visits_measured is None
    # No cursor survives early Close; test does not issue another retrieval.
    name = "trimem_joint_" + cursor.receipt.execution_id
    assert (
        private_connection.execute(
            "SELECT count(*) FROM pg_cursors WHERE name=%s", (name,)
        ).fetchone()[0]
        == 0
    )


def test_real_native_seeds_skip_unsummarized_nearest_neighbors(private_connection):
    """More unsummarized near neighbours than hnsw.ef_search (40) must not
    starve the seeds: the iterative scan returns the exact top summarized ones."""
    action = CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3)
    plan = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
    with private_connection.transaction(force_rollback=True):
        ranked = [
            row[0]
            for row in private_connection.execute(
                "SELECT id FROM attempt ORDER BY emb <=> '[1,0]'::vector, id"
            ).fetchall()
        ]
        blanked = ranked[:60]
        private_connection.execute(
            "UPDATE attempt SET changes = NULL WHERE id = ANY(%s)", (blanked,)
        )
        with NativeExecutor(private_connection).open_selected(plan) as cursor:
            rows = list(cursor)
        seeds = {row["seed_id"] for row in rows}
        assert seeds and not seeds & set(blanked)
        # Fixture lineage: each seed g < 400 has exactly one improving child g+1.
        expected = {g for g in ranked[60:68] if g < 400}
        assert seeds == expected
        assert len(rows) == len(expected)


def test_real_native_seeds_skip_noop_edits(private_connection):
    """A seed whose code equals its parent's code (empty diff) is never a seed."""
    action = CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3)
    plan = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
    with private_connection.transaction(force_rollback=True):
        ranked = [
            row[0]
            for row in private_connection.execute(
                "SELECT id FROM attempt ORDER BY emb <=> '[1,0]'::vector, id"
            ).fetchall()
        ]
        noop = [g for g in ranked[:60] if g > 1]
        private_connection.execute(
            """UPDATE attempt d SET code = p.code FROM attempt p
            WHERE p.id = d.parent_id AND d.id = ANY(%s)""",
            (noop,),
        )
        with NativeExecutor(private_connection).open_selected(plan) as cursor:
            rows = list(cursor)
        seeds = {row["seed_id"] for row in rows}
        assert seeds and not seeds & set(noop)
        # Fixture: root 1 has no parent (never a seed); every other g has
        # fitness g > g-1 and is valid, so only the no-op edits drop out.
        eligible = [g for g in ranked if g > 1 and g not in noop][:8]
        assert seeds == {g for g in eligible if g < 400}


def test_real_native_seeds_skip_non_improving_valid_edits(private_connection):
    """A valid real edit whose fitness is <= its parent's is never a seed."""
    action = CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3)
    plan = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
    with private_connection.transaction(force_rollback=True):
        ranked = [
            row[0]
            for row in private_connection.execute(
                "SELECT id FROM attempt ORDER BY emb <=> '[1,0]'::vector, id"
            ).fetchall()
        ]
        # The 8 nearest attempts: half tie their parent, half regress below it.
        demoted = ranked[:8]
        assert 1 not in demoted
        tie, regress = demoted[0::2], demoted[1::2]
        private_connection.execute(
            "UPDATE attempt SET fitness = (id - 1)::float8 WHERE id = ANY(%s)", (tie,)
        )
        private_connection.execute(
            "UPDATE attempt SET fitness = (id - 2)::float8 WHERE id = ANY(%s)",
            (regress,),
        )
        assert (
            private_connection.execute(
                """SELECT count(*) FROM attempt d JOIN attempt p ON p.id = d.parent_id
                WHERE d.id = ANY(%s) AND d.is_valid AND d.code <> p.code
                  AND d.fitness <= p.fitness""",
                (demoted,),
            ).fetchone()[0]
            == 8
        )
        with NativeExecutor(private_connection).open_selected(plan) as cursor:
            rows = list(cursor)
        seeds = {row["seed_id"] for row in rows}
        assert seeds and not seeds & set(demoted)
        eligible = [g for g in ranked if g > 1 and g not in demoted][:8]
        # A seed g yields a row iff its only lineage child g+1 still improves.
        assert seeds == {g for g in eligible if g < 400 and g + 1 not in demoted}
        assert len(rows) == len(seeds)


@pytest.mark.parametrize("token_budget", [200, 10])
def test_owned_worker_real_mailbox_with_mock_http_usage(
    private_connection, token_budget
):
    """Actual cache misses cross PG mailbox -> Worker -> mock HTTP -> receipt."""
    import httpx

    from trimem.optimizer.clients import OpenAICompatibleTransport
    from trimem.optimizer.semantic_usage import (
        ExclusiveClusterLease,
        ExclusiveSemanticUsageCollector,
        MeteredSemanticJudge,
        SemanticBudgetExhausted,
    )

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"verdict":true,"score":8}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    model = private_connection.execute(
        "SELECT current_setting('trimem.model')"
    ).fetchone()[0]
    system_id = private_connection.execute(
        "SELECT system_identifier::text FROM pg_control_system()"
    ).fetchone()[0]
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        transport = OpenAICompatibleTransport(
            model=model,
            base_url="https://fixture.invalid/v1",
            http_client=http,
            prompt_token_count=lambda _: 10,
            model_context_window=100,
            tokenizer_id="fixture-chat-v1",
            temperature=0,
        )
        collector = ExclusiveSemanticUsageCollector(
            lease=ExclusiveClusterLease(
                system_id,
                "postgres",
                private_connection.info.backend_pid,
                "test-owned temporary cluster with one query and one worker",
                True,
                True,
            ),
            connection_factory=lambda: psycopg.connect(
                private_connection.info.dsn, autocommit=True
            ),
            judge_factory=lambda ledger: MeteredSemanticJudge(transport, ledger),
        )
        collector.set_token_budget(token_budget)
        action = CandidateSpec(
            "expand_row_sem_v1",
            seed_count=8,
            fanout=4,
            filter_predicate=f"new owned-worker filter fixture {token_budget}",
            score_predicate=f"new owned-worker score fixture {token_budget}",
        )
        selected = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
        if token_budget == 10:
            with (
                pytest.raises(SemanticBudgetExhausted),
                NativeExecutor(private_connection, collector).open_selected(
                    selected
                ) as cursor,
            ):
                next(cursor)
            assert cursor.receipt.usage_complete
            assert cursor.receipt.usage_events == [] and requests == []
            assert cursor.receipt.stop_reason == "semantic_token_budget"
            assert system_id not in collector._active_clusters
            return
        with NativeExecutor(private_connection, collector).open_selected(
            selected
        ) as cursor:
            row = next(cursor)
            cursor.close("top_k")
    assert row["id"] > 0
    assert len(requests) == 2  # one boolean judgment and one stream score
    assert cursor.receipt.usage_complete
    assert {event["category"] for event in cursor.receipt.usage_events} == {
        "semantic_filter",
        "semantic_rank",
    }
    assert (
        sum(
            event["input_tokens"] + event["output_tokens"]
            for event in cursor.receipt.usage_events
        )
        == 24
    )
    assert all(
        event["execution_id"] == cursor.receipt.execution_id
        for event in cursor.receipt.usage_events
    )
    assert system_id not in collector._active_clusters


@pytest.mark.parametrize(
    "plan_id", ["seed_sem_expand_noscore_v1", "expand_row_sem_noscore_v1"]
)
def test_noscore_semfilter_usage_is_semantic_filter_only(private_connection, plan_id):
    """Cache misses cross PG mailbox -> Worker -> mock HTTP; no StreamScore call."""
    import httpx

    from trimem.optimizer.clients import OpenAICompatibleTransport
    from trimem.optimizer.semantic_usage import (
        ExclusiveClusterLease,
        ExclusiveSemanticUsageCollector,
        MeteredSemanticJudge,
    )

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"verdict":true,"score":8}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    model = private_connection.execute(
        "SELECT current_setting('trimem.model')"
    ).fetchone()[0]
    system_id = private_connection.execute(
        "SELECT system_identifier::text FROM pg_control_system()"
    ).fetchone()[0]
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        transport = OpenAICompatibleTransport(
            model=model,
            base_url="https://fixture.invalid/v1",
            http_client=http,
            prompt_token_count=lambda _: 10,
            model_context_window=100,
            tokenizer_id="fixture-chat-v1",
            temperature=0,
        )
        collector = ExclusiveSemanticUsageCollector(
            lease=ExclusiveClusterLease(
                system_id,
                "postgres",
                private_connection.info.backend_pid,
                "test-owned temporary cluster with one query and one worker",
                True,
                True,
            ),
            connection_factory=lambda: psycopg.connect(
                private_connection.info.dsn, autocommit=True
            ),
            judge_factory=lambda ledger: MeteredSemanticJudge(transport, ledger),
        )
        collector.set_token_budget(200)
        action = CandidateSpec(
            plan_id,
            seed_count=8,
            fanout=4,
            filter_predicate=f"new noscore filter fixture {plan_id}",
        )
        selected = compile_candidate(action, QueryContext("fixture", (1.0, 0.0)))
        assert "sem_score" not in selected.sql
        with NativeExecutor(private_connection, collector).open_selected(
            selected
        ) as cursor:
            row = next(cursor)
            cursor.close("top_k")
    assert row["id"] > 0 and "sem_score" not in row
    assert len(requests) >= 1
    assert cursor.receipt.usage_complete and cursor.receipt.score_rejected == 0
    assert {event["category"] for event in cursor.receipt.usage_events} == {
        "semantic_filter"
    }
    assert len(cursor.receipt.usage_events) == len(requests)
    assert system_id not in collector._active_clusters


class _CharTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


@pytest.mark.parametrize("plan_id", ["vgr_v1", "expand_row_sem_noscore_v1"])
def test_real_native_gain_ranked_pool_with_min_gain(private_connection, plan_id):
    """ordering=gain drains <= visit_cap rows, ranks by gain, packs top_k;
    min_gain drops float-noise "improvements" from seeds and endpoints."""
    from types import SimpleNamespace

    from trimem.optimizer.adapters import NativeRetrievalAdapter
    from trimem.optimizer.ledger import TokenLedger
    from trimem.optimizer.packing import gain_key

    action = CandidateSpec(
        plan_id,
        seed_count=8,
        fanout=4,
        top_k=5,
        candidate_visit_cap=16,
        hop_types=("lineage|similar",) * 2,
        frontier="all",
        ordering="gain",
        min_gain=1e-9,
    )
    query = QueryContext("fixture", (1.0, 0.0))
    adapter = NativeRetrievalAdapter(
        executor=lambda ctx: NativeExecutor(private_connection, CacheOnlyCollector()),
        query_context=lambda ctx, state: query,
        tokenizer=_CharTokenizer(),
        tokenizer_id="fixture",
        render_prompt=lambda ctx, state, memory: "base" + memory,
        model_context_window=100000,
        max_output_tokens=10,
    )
    context = SimpleNamespace(
        protocol=SimpleNamespace(tokenizer_version="fixture"),
        ledger=TokenLedger(100000),
    )
    with private_connection.transaction(force_rollback=True):
        # Varied gains; every 5th attempt is a 1e-12 noise "improvement".
        private_connection.execute("""UPDATE attempt a SET fitness = s.cum
            FROM (SELECT id, sum(CASE WHEN id % 5 = 0 THEN 1e-12
                                      ELSE (id % 3) + 1 END) OVER (ORDER BY id) cum
                  FROM attempt) s WHERE a.id = s.id""")
        result = adapter(action, context, None)
        # Reference: the same stream (stream ordering), first visit_cap rows.
        plan = compile_candidate(
            CandidateSpec(**{**action.__dict__, "ordering": "stream"}), query
        )
        with NativeExecutor(private_connection, CacheOnlyCollector()).open_selected(
            plan
        ) as cursor:
            stream = list(cursor)
    receipt = result.receipt
    assert receipt.ranked_by == "gain" and receipt.closed
    assert receipt.pool_rows == len(stream) == 16
    assert receipt.stop_reason == "candidate_visit_cap"
    assert receipt.pool_distinct == len({r["id"] for r in stream})
    expected = sorted({r["id"]: r for r in stream}.values(), key=gain_key)[:5]
    assert result.packed.retained_ids == [str(r["id"]) for r in expected]
    assert result.packed.stop_reason == "top_k"
    gains = [r["fitness"] - r["parent_fitness"] for r in expected]
    assert gains == sorted(gains, reverse=True) and min(gains) > 1e-9
    assert all(r["id"] % 5 != 0 for r in stream)
    assert "min_gain" in plan.params
