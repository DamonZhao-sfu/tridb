"""TriMEM execution tests: SemFilter mailbox + worker, cache, Expand, Fig. 1(b) end to end.

Same cluster requirements as tests/test_trimem_plan.py (``TRIMEM_DSN``). The worker runs in a
thread with a deterministic MockJudge: an attempt passes iff its id is even.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Iterator

import pytest

psycopg = pytest.importorskip("psycopg")

from test_trimem_plan import FIXTURE, PRED_A, PRED_B, Q_FIG1B  # noqa: E402

from trimem import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_VERSION,
    OP_EXTRACT,
    OP_FILTER,
)
from trimem.keys import sem_key  # noqa: E402
from trimem.llm import MockJudge, parse_label, parse_verdict  # noqa: E402
from trimem.worker import Worker  # noqa: E402

DSN = os.environ.get("TRIMEM_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="TRIMEM_DSN not set")


def even_id(predicate: str, text: str) -> bool:
    m = re.search(r"attempt (\d+)", text)
    return bool(m) and int(m.group(1)) % 2 == 0


def last_digit(predicate: str, text: str) -> float:
    m = re.search(r"attempt (\d+)", text)
    return (int(m.group(1)) % 10) / 10 if m else 0.0


STRATEGIES = ("greedy", "beam search", "simulated annealing")
Q_STRAT = (
    "Which search strategy does this change use? "
    "Answer one of: greedy, beam search, simulated annealing, other"
)


def strategy_of(i: int) -> str | None:
    """Oracle of the mock extractor: ids divisible by 11 get an unparsable reply (NULL)."""
    return None if i % 11 == 0 else STRATEGIES[i % 3]


def strategy(question: str, text: str) -> str | None:
    m = re.search(r"attempt (\d+)", text)
    return strategy_of(int(m.group(1))) if m else None


@pytest.fixture(scope="module")
def db() -> Iterator[str]:
    name = f"trimem_exec_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    dsn = psycopg.conninfo.make_conninfo(DSN, dbname=name)
    try:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(FIXTURE)
        yield dsn
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture(scope="module")
def worker(db: str) -> Iterator[Worker]:
    w = Worker(
        db,
        MockJudge(even_id, score_rule=last_digit, extract_rule=strategy),
        take_ms=100,
    )
    t = threading.Thread(target=w.run, kwargs={"flush_every": 0.2}, daemon=True)
    t.start()
    yield w
    w.stop()
    t.join(timeout=5)


@pytest.fixture()
def conn(db: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(db, autocommit=True) as c:
        c.execute("SET jit = off")
        c.execute("SET trimem.timeout_ms = 20000")
        yield c


def analyze(conn: psycopg.Connection, sql: str) -> dict:
    row = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}").fetchone()
    return row[0][0]["Plan"]


def walk(p: dict) -> Iterator[dict]:
    yield p
    for c in p.get("Plans", []):
        yield from walk(c)


def node(p: dict, provider: str) -> dict:
    return next(n for n in walk(p) if n.get("Custom Plan Provider") == provider)


# ------------------------------------------------------------------ pure Python
def test_parse_verdict_is_strict() -> None:
    assert parse_verdict('{"verdict": true}') is True
    assert parse_verdict('ok {"verdict": false} ') is False
    assert parse_verdict('{"verdict": "yes"}') is None
    assert parse_verdict("true") is None


# ------------------------------------------------------------------ key parity
def test_cache_key_matches_the_c_extension(conn: psycopg.Connection) -> None:
    args = (
        OP_FILTER,
        DEFAULT_MODEL,
        DEFAULT_PROMPT_VERSION,
        PRED_A,
        "attempt 7 outcome ✓",
    )
    (c_key,) = conn.execute("SELECT trimem.key(%s, %s, %s, %s, %s)", args).fetchone()
    assert c_key == sem_key(*args)


# ------------------------------------------------------------------ SemFilter
def test_semfilter_returns_exactly_the_passing_rows(
    conn: psycopg.Connection, worker: Worker
) -> None:
    got = [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM attempt WHERE task = 'task3' AND trimem.sem_filter('{PRED_A}', outcome) "
            "ORDER BY id"
        ).fetchall()
    ]
    want = [i for i in range(1, 4001) if i % 10 == 3 and i % 2 == 0]
    assert got == want  # task3 ids are odd -> none pass
    got = [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM attempt WHERE task = 'task4' AND trimem.sem_filter('{PRED_A}', outcome) "
            "ORDER BY id"
        ).fetchall()
    ]
    assert got == [i for i in range(1, 4001) if i % 10 == 4]


def test_second_run_is_served_from_the_cache(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = f"SELECT id FROM attempt WHERE task = 'task6' AND trimem.sem_filter('{PRED_A}', outcome)"
    first = node(analyze(conn, sql), "TriMEM SemFilter")
    second = node(analyze(conn, sql), "TriMEM SemFilter")
    assert first["Rows Judged"] == 400 and first["Worker Requests"] == 400
    assert second["Worker Requests"] == 0 and second["Cache Hits"] == 400
    assert second["Actual Rows"] == first["Actual Rows"] == 400


def test_limit_stops_judging_early(conn: psycopg.Connection, worker: Worker) -> None:
    sql = (
        f"SELECT id FROM attempt WHERE trimem.sem_filter('{PRED_A}', outcome) "
        "ORDER BY emb <=> (SELECT emb FROM attempt WHERE id = 42) LIMIT 3"
    )
    p = analyze(conn, sql)
    sem = node(p, "TriMEM SemFilter")
    assert p["Actual Rows"] == 3
    batch = int(sem["Batch Size"].split()[0])
    assert sem["Rows Judged"] <= 2 * batch, (
        "streaming: stops after the batch that met LIMIT"
    )


def test_statistics_flow_back_to_the_planner(
    conn: psycopg.Connection, worker: Worker
) -> None:
    conn.execute(
        f"SELECT count(*) FROM attempt WHERE task = 'task8' "
        f"AND trimem.sem_filter('{PRED_A} v2', outcome)"
    ).fetchone()
    conn.execute("SELECT trimem.flush_stats()")
    row = conn.execute(
        "EXPLAIN (FORMAT JSON) SELECT id FROM attempt "
        f"WHERE trimem.sem_filter('{PRED_A} v2', outcome)"
    ).fetchone()
    sem = node(row[0][0]["Plan"], "TriMEM SemFilter")
    assert "(stats, n=400, cached=400)" in sem["Semantic Cost Model"]
    assert "sel=1.000" in sem["Semantic Cost Model"]  # task8 ids are all even


def test_missing_worker_fails_closed(db: str) -> None:
    with psycopg.connect(db, autocommit=True) as c:
        c.execute("SET trimem.timeout_ms = 300")
        c.execute("SET trimem.prompt_version = 'no-worker-will-answer'")
        c.execute(
            "SET trimem.model = 'nobody/no-model'"
        )  # the worker rejects other models
        with pytest.raises(psycopg.Error):
            c.execute(
                f"SELECT id FROM attempt WHERE id < 5 "
                f"AND trimem.sem_filter('{PRED_A}', outcome)"
            ).fetchall()


# ------------------------------------------------------------------ Expand
def test_expand_follows_native_typed_edges(conn: psycopg.Connection) -> None:
    got = sorted(
        r[0]
        for r in conn.execute(
            "SELECT c.id FROM attempt s, LATERAL trimem.expand(s.vid, 'lineage', 16) e "
            "JOIN attempt c ON c.vid = e.dst WHERE s.id = 100"
        ).fetchall()
    )
    assert got == [200, 201]


def test_expand_honours_the_fanout_cap(conn: psycopg.Connection) -> None:
    p = analyze(
        conn,
        "SELECT e.dst FROM attempt s, LATERAL trimem.expand(s.vid, 'similar', 3) e "
        "WHERE s.id = 5",
    )
    ex = node(p, "TriMEM Expand")
    assert p["Actual Rows"] == 3 and ex["Fanout Capped"] == 1


# ------------------------------------------------------------------ Fig. 1(b)
def test_fig1b_end_to_end_matches_an_oracle(
    conn: psycopg.Connection, worker: Worker
) -> None:
    got = [r[0] for r in conn.execute(Q_FIG1B).fetchall()]
    oracle_sql = Q_FIG1B.replace(
        f"AND trimem.sem_filter('{PRED_A}', a.outcome)", "AND a.id % 2 = 0"
    )
    want = [r[0] for r in conn.execute(oracle_sql).fetchall()]
    assert got == want and got
    p = analyze(conn, Q_FIG1B)
    sem = node(p, "TriMEM SemFilter")
    assert sem["Placement"] == "top (above join)"
    assert (
        sem["Rows Judged"]
        == sem["Plans"][0]["Actual Rows"] * sem["Plans"][0]["Actual Loops"]
    )


# ------------------------------------------------------------------ soft semantic operators (0.2.0)
NULL_EVERY_3RD = "CASE WHEN id % 3 = 0 THEN NULL ELSE outcome END"


def test_unknown_as_true_keeps_rows_with_null_input(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT id FROM attempt WHERE id BETWEEN 1 AND 30 "
        f"AND trimem.sem_filter('{PRED_A}', {NULL_EVERY_3RD}, true) ORDER BY id"
    )
    got = [r[0] for r in conn.execute(sql).fetchall()]
    assert got == [i for i in range(1, 31) if i % 3 == 0 or i % 2 == 0]
    sem = node(analyze(conn, sql), "TriMEM SemFilter")
    assert sem["Unknown Input"] == "keep"


def test_unknown_as_false_and_two_arg_form_drop_null_input(
    conn: psycopg.Connection, worker: Worker
) -> None:
    for form in (
        f"trimem.sem_filter('{PRED_A}', {NULL_EVERY_3RD}, false)",
        f"trimem.sem_filter('{PRED_A}', {NULL_EVERY_3RD})",
    ):
        got = [
            r[0]
            for r in conn.execute(
                f"SELECT id FROM attempt WHERE id BETWEEN 1 AND 30 AND {form} ORDER BY id"
            ).fetchall()
        ]
        assert got == [i for i in range(1, 31) if i % 3 != 0 and i % 2 == 0]


def test_sem_score_ranks_without_filtering(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT id, trimem.sem_score('{PRED_B}', outcome) AS s FROM attempt "
        f"WHERE id BETWEEN 101 AND 120 ORDER BY s DESC, id LIMIT 4"
    )
    rows = conn.execute(sql).fetchall()
    assert [r[0] for r in rows] == [109, 119, 108, 118]
    assert [r[1] for r in rows] == [0.9, 0.9, 0.8, 0.8]
    p = analyze(conn, sql)
    sc = node(p, "TriMEM SemScore")
    assert (
        sc["Inputs Scored"] == 20 and sc["Worker Requests"] == 0
    )  # second run: all from cache
    assert sc["Cache Hits"] == 20


def test_sem_score_weights_a_fused_rank_over_a_join(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT c.id FROM attempt s, LATERAL trimem.expand(s.vid, 'lineage', 16) e "
        f"JOIN attempt c ON c.vid = e.dst WHERE s.id BETWEEN 50 AND 59 "
        f"ORDER BY c.fitness * (0.5 + coalesce(trimem.sem_score('{PRED_B}', c.outcome), 0.5)) DESC, c.id LIMIT 5"
    )
    got = [r[0] for r in conn.execute(sql).fetchall()]
    rows = conn.execute(
        "SELECT c.id, c.fitness FROM attempt c WHERE c.id BETWEEN 100 AND 119"
    ).fetchall()
    want = sorted(rows, key=lambda r: (-(r[1] * (0.5 + (r[0] % 10) / 10)), r[0]))[:5]
    assert got == [r[0] for r in want]
    p = analyze(conn, sql)
    sc = node(p, "TriMEM SemScore")
    assert sc["Plans"][0]["Node Type"] in ("Nested Loop", "Hash Join", "Merge Join")


def test_sem_score_outside_its_node_fails_closed(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.Error):
        conn.execute(f"SELECT trimem.sem_score('{PRED_B}', 'x')").fetchall()


def test_sem_score_placement_changes_cost_not_result(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT c.id FROM attempt s, LATERAL trimem.expand(s.vid, 'lineage', 16) e "
        f"JOIN attempt c ON c.vid = e.dst WHERE s.id BETWEEN 60 AND 69 "
        f"ORDER BY c.fitness * (0.5 + coalesce(trimem.sem_score('{PRED_B}', c.outcome), 0.5)) DESC, c.id LIMIT 5"
    )
    results, placements = {}, {}
    for force in ("", "c=base", "c=top"):
        conn.execute(f"SET trimem.force_placement = '{force}'")
        results[force] = [r[0] for r in conn.execute(sql).fetchall()]
        placements[force] = node(analyze(conn, sql), "TriMEM SemScore")["Placement"]
    conn.execute("RESET trimem.force_placement")
    assert results["c=base"] == results["c=top"] == results[""]
    assert placements["c=base"].startswith("base") and placements["c=top"].startswith(
        "top"
    )
    assert placements[""].startswith("top"), (
        "optimizer prefers scoring after the traversal here"
    )


# ------------------------------------------------------------------ SemExtract (0.3.0)
def labels(conn: psycopg.Connection, sql: str) -> list[tuple]:
    return [tuple(r) for r in conn.execute(sql).fetchall()]


def test_parse_label_is_strict() -> None:
    assert parse_label('{"label": "Beam  Search."}') == "beam search"
    assert parse_label('sure: {"label": "greedy"}') == "greedy"
    assert parse_label('{"label": ""}') is None
    assert parse_label('{"label": 3}') is None
    assert parse_label('{"label": "' + "x" * 65 + '"}') is None
    assert parse_label("greedy") is None


def test_extract_cache_key_matches_the_c_extension(conn: psycopg.Connection) -> None:
    args = (OP_EXTRACT, DEFAULT_MODEL, DEFAULT_PROMPT_VERSION, Q_STRAT, "attempt 7 ✓")
    (c_key,) = conn.execute("SELECT trimem.key(%s, %s, %s, %s, %s)", args).fetchone()
    assert c_key == sem_key(*args)


def test_sem_extract_labels_round_trip(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT id, trimem.sem_extract('{Q_STRAT}', outcome) FROM attempt "
        "WHERE id BETWEEN 201 AND 260 ORDER BY id"
    )
    first = node(analyze(conn, sql), "TriMEM SemExtract")
    assert first["Placement"] == "single relation"
    assert first["Extract Question"] == Q_STRAT
    want = [(i, strategy_of(i)) for i in range(201, 261)]
    n_null = sum(1 for _, v in want if v is None)
    assert first["Inputs Extracted"] == 60 and first["Worker Requests"] == 60
    assert first["Unlabeled"] == n_null > 0
    assert labels(conn, sql) == want
    # labels live in sem_cache.extracted; unparsable replies are not cached
    (n_cached,) = conn.execute(
        "SELECT count(*) FROM trimem.sem_cache WHERE op = %s AND predicate = %s "
        "AND extracted IS NOT NULL AND key = ANY(%s)",
        (
            OP_EXTRACT,
            Q_STRAT,
            [
                sem_key(
                    OP_EXTRACT,
                    DEFAULT_MODEL,
                    DEFAULT_PROMPT_VERSION,
                    Q_STRAT,
                    f"attempt {i} outcome",
                )
                for i in range(201, 261)
            ],
        ),
    ).fetchone()
    assert n_cached == 60 - n_null


def test_sem_extract_second_run_issues_no_worker_requests(
    conn: psycopg.Connection, worker: Worker
) -> None:
    # 2102..2111 holds no multiple of 11: every reply is parsable, so every label is cached
    sql = (
        f"SELECT id, trimem.sem_extract('{Q_STRAT}', outcome) FROM attempt "
        "WHERE id BETWEEN 2102 AND 2111"
    )
    first = node(analyze(conn, sql), "TriMEM SemExtract")
    second = node(analyze(conn, sql), "TriMEM SemExtract")
    assert first["Worker Requests"] == 10
    assert second["Worker Requests"] == 0 and second["Cache Hits"] == 10
    assert second["LLM Wait"] == 0
    assert sorted(labels(conn, sql)) == [(i, strategy_of(i)) for i in range(2102, 2112)]


def test_sem_extract_null_input_is_null(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT id, trimem.sem_extract('{Q_STRAT}', {NULL_EVERY_3RD}) FROM attempt "
        "WHERE id BETWEEN 401 AND 430 ORDER BY id"
    )
    assert labels(conn, sql) == [
        (i, None if i % 3 == 0 else strategy_of(i)) for i in range(401, 431)
    ]
    sx = node(analyze(conn, sql), "TriMEM SemExtract")
    unparsable = sum(1 for i in range(401, 431) if i % 3 and strategy_of(i) is None)
    assert sx["NULL Inputs"] == 10 and sx["Inputs Extracted"] == 20
    # second run: every parsable label is cached; only unparsable replies are asked again
    assert sx["Cache Hits"] == 20 - unparsable and sx["Worker Requests"] == unparsable


def test_sem_extract_outside_its_node_fails_closed(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.FeatureNotSupported):
        conn.execute(f"SELECT trimem.sem_extract('{Q_STRAT}', 'x')").fetchall()


def test_sem_extract_label_qual_and_group_by(
    conn: psycopg.Connection, worker: Worker
) -> None:
    ids = range(501, 561)
    got = [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM attempt WHERE id BETWEEN 501 AND 560 "
            f"AND trimem.sem_extract('{Q_STRAT}', outcome) = 'greedy' ORDER BY id"
        ).fetchall()
    ]
    assert got == [i for i in ids if strategy_of(i) == "greedy"]
    p = analyze(
        conn,
        f"SELECT id FROM attempt WHERE id BETWEEN 501 AND 560 "
        f"AND trimem.sem_extract('{Q_STRAT}', outcome) = 'greedy'",
    )
    sx = node(p, "TriMEM SemExtract")
    assert "sem_extract" in sx["Filter"] and sx["Rows Removed by Filter"] == 60 - len(
        got
    )
    groups = labels(
        conn,
        f"SELECT trimem.sem_extract('{Q_STRAT}', outcome) AS s, count(*) FROM attempt "
        "WHERE id BETWEEN 501 AND 560 GROUP BY 1 ORDER BY 1 NULLS LAST",
    )
    want: dict = {}
    for i in ids:
        want[strategy_of(i)] = want.get(strategy_of(i), 0) + 1
    assert groups == sorted(want.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))


JOIN_EXTRACT = (
    "SELECT c.id, trimem.sem_extract('{q}', c.outcome) AS s FROM attempt p, "
    "LATERAL trimem.expand(p.vid, 'lineage', 16) e JOIN attempt c ON c.vid = e.dst "
    "WHERE p.id BETWEEN {lo} AND {hi} {extra} ORDER BY c.id"
)


def test_sem_extract_base_and_top_placement_return_identical_rows(
    conn: psycopg.Connection, worker: Worker
) -> None:
    for extra in ("", f"AND trimem.sem_extract('{Q_STRAT}', c.outcome) <> 'greedy'"):
        sql = JOIN_EXTRACT.format(q=Q_STRAT, lo=70, hi=79, extra=extra)
        results, placements = {}, {}
        for force in ("", "c=base", "c=top"):
            conn.execute(f"SET trimem.force_placement = '{force}'")
            results[force] = labels(conn, sql)
            placements[force] = node(analyze(conn, sql), "TriMEM SemExtract")
        conn.execute("RESET trimem.force_placement")
        want = [
            (i, strategy_of(i))
            for i in range(140, 160)
            if not extra or strategy_of(i) not in (None, "greedy")  # NULL <> x drops
        ]
        assert results["c=base"] == results["c=top"] == results[""] == want
        assert placements["c=base"]["Placement"].startswith("base")
        assert placements["c=top"]["Placement"].startswith("top")
        assert "(forced)" in placements["c=top"]["Placement Decision"]
        assert "extract:c=base" in placements[""]["Placement Decision"]
        assert "extract:c=top" in placements[""]["Placement Decision"]


def test_sem_filter_score_and_extract_compose(
    conn: psycopg.Connection, worker: Worker
) -> None:
    sql = (
        f"SELECT c.id, trimem.sem_extract('{Q_STRAT}', c.outcome) AS strat, "
        f"trimem.sem_score('{PRED_B}', c.outcome) AS sc FROM attempt p, "
        "LATERAL trimem.expand(p.vid, 'lineage', 16) e JOIN attempt c ON c.vid = e.dst "
        f"WHERE p.id BETWEEN 80 AND 89 AND trimem.sem_filter('{PRED_A}', c.outcome) "
        f"AND trimem.sem_extract('{Q_STRAT}', c.outcome) IS DISTINCT FROM 'beam search' "
        "ORDER BY sc DESC, c.id"
    )
    got = labels(conn, sql)
    want = sorted(
        (
            (i, strategy_of(i), (i % 10) / 10)
            for i in range(160, 180)
            if i % 2 == 0 and strategy_of(i) != "beam search"
        ),
        key=lambda r: (-r[2], r[0]),
    )
    assert got == want and got
    p = analyze(conn, sql)
    providers = {n.get("Custom Plan Provider") for n in walk(p)}
    assert {"TriMEM SemFilter", "TriMEM SemScore", "TriMEM SemExtract"} <= providers
    decision = node(p, "TriMEM SemExtract")["Placement Decision"]
    assert "c=" in decision and "score:c=" in decision and "extract:c=" in decision
    # single relation: filter -> extract (with its label qual) -> score, stacked bottom-up
    single = (
        f"SELECT id, trimem.sem_extract('{Q_STRAT}', outcome) AS strat, "
        f"trimem.sem_score('{PRED_B}', outcome) AS sc FROM attempt "
        f"WHERE id BETWEEN 601 AND 640 AND trimem.sem_filter('{PRED_A}', outcome) "
        f"AND trimem.sem_extract('{Q_STRAT}', outcome) = 'greedy' ORDER BY id"
    )
    assert labels(conn, single) == [
        (i, "greedy", (i % 10) / 10)
        for i in range(601, 641)
        if i % 2 == 0 and strategy_of(i) == "greedy"
    ]
    chain = [n.get("Custom Plan Provider") for n in walk(analyze(conn, single))]
    chain = [c for c in chain if c]
    assert chain == ["TriMEM SemScore", "TriMEM SemExtract", "TriMEM SemFilter"]


def test_sem_extract_join_condition_is_rejected(conn: psycopg.Connection) -> None:
    for op in ("<>", "="):
        with pytest.raises(psycopg.errors.FeatureNotSupported):
            conn.execute(
                "SELECT c.id FROM attempt p JOIN attempt c ON c.id = p.id * 2 "
                f"WHERE p.id < 5 AND trimem.sem_extract('{Q_STRAT}', c.outcome) {op} "
                f"trimem.sem_extract('{Q_STRAT}', p.outcome)"
            ).fetchall()


def test_sem_extract_missing_worker_fails_closed(db: str) -> None:
    with psycopg.connect(db, autocommit=True) as c:
        c.execute("SET trimem.timeout_ms = 300")
        c.execute("SET trimem.model = 'nobody/no-model'")  # rejected by the worker
        with pytest.raises(psycopg.Error):
            c.execute(
                f"SELECT trimem.sem_extract('{Q_STRAT}', outcome) FROM attempt WHERE id < 5"
            ).fetchall()


def test_upgrade_from_0_2_0_adds_sem_extract() -> None:
    name = f"trimem_upgrade_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    try:
        dsn = psycopg.conninfo.make_conninfo(DSN, dbname=name)
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute("CREATE EXTENSION graph_store_am")
            c.execute("CREATE EXTENSION trimem_pg VERSION '0.2.0'")
            c.execute("ALTER EXTENSION trimem_pg UPDATE TO '0.3.0'")
            (col,) = c.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = 'trimem' AND table_name = 'sem_cache' "
                "AND column_name = 'extracted'"
            ).fetchone()
            assert col == "text"
            (n,) = c.execute(
                "SELECT count(*) FROM pg_proc WHERE pronamespace = 'trimem'::regnamespace "
                "AND proname IN ('sem_extract', 'complete_extracts', 'sem_extract_support')"
            ).fetchone()
            assert n == 3
            (public_ok,) = c.execute(
                "SELECT has_function_privilege('public', "
                "'trimem.complete_extracts(integer[], bigint[])', 'EXECUTE')"
            ).fetchone()
            assert public_ok is False
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def test_sem_extract_label_qual_on_the_nullable_side_of_an_outer_join(
    conn: psycopg.Connection, worker: Worker
) -> None:
    got = labels(
        conn,
        "SELECT s.id, c.id FROM attempt s LEFT JOIN attempt c ON c.id = s.id * 2 "
        f"AND trimem.sem_extract('{Q_STRAT}', c.outcome) = 'greedy' "
        "WHERE s.id BETWEEN 1 AND 20 ORDER BY s.id",
    )
    assert got == [
        (i, 2 * i if strategy_of(2 * i) == "greedy" else None) for i in range(1, 21)
    ]


def test_sem_extract_relational_ops_on_the_label(
    conn: psycopg.Connection, worker: Worker
) -> None:
    ids = range(701, 761)
    # outer WHERE on a pulled-up subquery's label column becomes a lifted label qual
    got = labels(
        conn,
        f"SELECT id, s FROM (SELECT id, trimem.sem_extract('{Q_STRAT}', outcome) AS s "
        "FROM attempt WHERE id BETWEEN 701 AND 760) x WHERE s = 'beam search' ORDER BY id",
    )
    assert got == [(i, "beam search") for i in ids if strategy_of(i) == "beam search"]
    # per-label cap: top-2 by fitness within each label
    got = labels(
        conn,
        f"SELECT s, id FROM (SELECT id, trimem.sem_extract('{Q_STRAT}', outcome) AS s, "
        f"row_number() OVER (PARTITION BY trimem.sem_extract('{Q_STRAT}', outcome) "
        "ORDER BY fitness DESC, id) AS rn FROM attempt WHERE id BETWEEN 701 AND 760) x "
        "WHERE rn <= 2 AND s IS NOT NULL ORDER BY s, id",
    )
    fit = dict(
        conn.execute(
            "SELECT id, fitness FROM attempt WHERE id BETWEEN 701 AND 760"
        ).fetchall()
    )
    want = []
    for lab in sorted(set(filter(None, (strategy_of(i) for i in ids)))):
        top2 = sorted(
            (i for i in ids if strategy_of(i) == lab), key=lambda i: (-fit[i], i)
        )[:2]
        want += [(lab, i) for i in sorted(top2)]
    assert got == want
    # HAVING over a grouped label
    got = labels(
        conn,
        f"SELECT trimem.sem_extract('{Q_STRAT}', outcome) AS s, count(*) FROM attempt "
        "WHERE id BETWEEN 701 AND 760 GROUP BY 1 "
        f"HAVING trimem.sem_extract('{Q_STRAT}', outcome) <> 'greedy' ORDER BY 1",
    )
    assert got == sorted(
        (lab, sum(1 for i in ids if strategy_of(i) == lab))
        for lab in ("beam search", "simulated annealing")
    )


def test_sem_extract_in_a_correlated_subquery(
    conn: psycopg.Connection, worker: Worker
) -> None:
    # the inner SemExtract node runs (and ends) while the outer projection is evaluated
    got = labels(
        conn,
        f"SELECT p.id, trimem.sem_extract('{Q_STRAT}', p.outcome), "
        f"(SELECT trimem.sem_extract('{Q_STRAT}', c.outcome) FROM attempt c "
        "WHERE c.id = p.id * 2) FROM attempt p WHERE p.id BETWEEN 801 AND 812 ORDER BY p.id",
    )
    assert got == [(i, strategy_of(i), strategy_of(2 * i)) for i in range(801, 813)]
