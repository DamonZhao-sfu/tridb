"""TriMEM SemFilter planner tests: tri-modal (V/G/R) + semantic plans (ADR-0027).

Needs a PG16 cluster with ``shared_preload_libraries = 'tjs_pg,trimem_pg'`` and the
vector / graph_store_am / trimem_pg extensions installed. Point ``TRIMEM_DSN`` at any
database on it; the tests create and drop their own database. Skipped otherwise.

Covers plan Addendum A: N1b cost-based SemFilter placement (base vs above the join) and
N1c the costed native graph Expand node.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import pytest

psycopg = pytest.importorskip("psycopg")

DSN = os.environ.get("TRIMEM_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="TRIMEM_DSN not set")

PRED_A = "transferable idea for improving the parent"
PRED_B = "mentions a timeout"
N_ROWS = 4000

FIXTURE = f"""
CREATE EXTENSION vector;
CREATE EXTENSION graph_store_am;
CREATE EXTENSION trimem_pg;
CREATE TABLE attempt(id int PRIMARY KEY, vid bigint, task text, fitness float8, valid bool,
                     outcome text, emb vector(16));
INSERT INTO attempt
SELECT g, NULL, 'task' || (g % 10), (g % 100) / 100.0, g % 7 <> 0,
       'attempt ' || g || ' outcome', (SELECT array_agg(sin(g * i))::vector(16)
                                       FROM generate_series(1, 16) i)
FROM generate_series(1, {N_ROWS}) g;
UPDATE attempt SET vid = graph_store.gph_upsert_vertex(id);
SELECT count(graph_store.gph_insert_edge(p.vid, c.vid, graph_store.register_edge_type('lineage')))
FROM attempt c JOIN attempt p ON p.id = c.id / 2 WHERE c.id > 1;
-- dense type: every attempt links to 10 others (degree 10)
SELECT count(graph_store.gph_insert_edge(a.vid, b.vid, graph_store.register_edge_type('similar')))
FROM attempt a CROSS JOIN generate_series(1, 10) k
JOIN attempt b ON b.id = 1 + (a.id * 7 + k * 13) % {N_ROWS};
SELECT trimem.analyze_graph();
CREATE INDEX ON attempt(task);
CREATE UNIQUE INDEX ON attempt(vid);
CREATE INDEX attempt_hnsw ON attempt USING hnsw (emb vector_cosine_ops);
ANALYZE attempt;
"""

Q_VRS = f"""
SELECT id FROM attempt
WHERE task = 'task3' AND fitness > 0.3 AND trimem.sem_filter('{PRED_A}', outcome)
ORDER BY emb <=> (SELECT emb FROM attempt WHERE id = 42) LIMIT 5
"""

Q_GRS = f"""
SELECT a.id FROM attempt s,
  LATERAL trimem.expand(s.vid, 'lineage', 16) e
  JOIN attempt a ON a.vid = e.dst
WHERE s.id = ANY('{{100,200,300}}') AND a.valid
  AND trimem.sem_filter('{PRED_A}', a.outcome)
"""

# Figure 1(b): ANN seeds ∪ lookup -> typed expand -> relational filter -> sem_filter -> rank.
Q_FIG1B = f"""
WITH seeds AS (
  (SELECT vid FROM attempt ORDER BY emb <=> (SELECT emb FROM attempt WHERE id = 42) LIMIT 8)
  UNION ALL
  (SELECT vid FROM attempt WHERE id = ANY('{{7,11}}'))
)
SELECT a.id FROM seeds s,
  LATERAL trimem.expand(s.vid, 'lineage', 16) e
  JOIN attempt a ON a.vid = e.dst
WHERE a.valid AND a.fitness > 0.2 AND trimem.sem_filter('{PRED_A}', a.outcome)
ORDER BY a.fitness DESC LIMIT 5
"""

# The join multiplies each seed by ~10 'similar' neighbours: judging the seeds (BASE) is
# ~10x cheaper than judging the joined rows (TOP).
Q_DENSE = f"""
SELECT s.id, e.dst FROM attempt s, LATERAL trimem.expand(s.vid, 'similar', 10) e
WHERE s.task = 'task3' AND trimem.sem_filter('{PRED_A}', s.outcome)
"""


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    name = f"trimem_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    try:
        dsn = psycopg.conninfo.make_conninfo(DSN, dbname=name)
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(FIXTURE)
            c.execute("SET jit = off")
            yield c
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture(autouse=True)
def _clean_settings(conn: psycopg.Connection) -> Iterator[None]:
    yield
    conn.execute("RESET trimem.llm_cost_per_eval")
    conn.execute("RESET trimem.llm_cost_per_batch")
    conn.execute("RESET trimem.enable")
    conn.execute("DELETE FROM trimem.stats")
    conn.execute("SELECT trimem.reset_stats()")


def plan(conn: psycopg.Connection, sql: str) -> dict:
    row = conn.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()
    doc = row[0] if isinstance(row[0], list) else json.loads(row[0])
    return doc[0]["Plan"]


def nodes(p: dict, depth: int = 0) -> Iterator[tuple[int, dict]]:
    yield depth, p
    for child in p.get("Plans", []):
        yield from nodes(child, depth + 1)


def sem_nodes(p: dict) -> list[dict]:
    return [
        n for _, n in nodes(p) if n.get("Custom Plan Provider") == "TriMEM SemFilter"
    ]


def filters_text(p: dict) -> str:
    return " ".join(str(n.get("Filter", "")) for _, n in nodes(p))


# ------------------------------------------------------------------ R + Sem
def test_semantic_qual_is_lifted_into_its_own_node(conn: psycopg.Connection) -> None:
    p = plan(
        conn,
        f"SELECT id FROM attempt WHERE valid AND trimem.sem_filter('{PRED_A}', outcome)",
    )
    sems = sem_nodes(p)
    assert len(sems) == 1
    assert sems[0]["Semantic Predicate"] == PRED_A
    assert "sem_filter" not in filters_text(p), (
        "input scan must not evaluate the semantic qual"
    )
    assert "valid" in filters_text(p), "relational qual stays below the semantic node"


def test_semantic_cost_is_in_the_plan_total(conn: psycopg.Connection) -> None:
    sql = f"SELECT id FROM attempt WHERE task = 'task3' AND trimem.sem_filter('{PRED_A}', outcome)"
    conn.execute("SET trimem.llm_cost_per_eval = 0")
    free = plan(conn, sql)
    conn.execute("SET trimem.llm_cost_per_eval = 2125")
    costly = plan(conn, sql)
    child_rows = sem_nodes(costly)[0]["Plans"][0]["Plan Rows"]
    delta = costly["Total Cost"] - free["Total Cost"]
    assert delta == pytest.approx(child_rows * 2125, rel=1e-6)


def test_statistics_drive_selectivity_and_cost(conn: psycopg.Connection) -> None:
    sql = f"SELECT id FROM attempt WHERE trimem.sem_filter('{PRED_A}', outcome)"
    before = sem_nodes(plan(conn, sql))[0]
    conn.execute(
        "INSERT INTO trimem.stats(model, prompt_version, predicate, n_eval, n_true, n_hit) "
        "VALUES (current_setting('trimem.model'), current_setting('trimem.prompt_version'), "
        "%s, 1000, 100, 900)",
        (PRED_A,),
    )
    after = sem_nodes(plan(conn, sql))[0]
    assert "defaults" in before["Semantic Cost Model"]
    assert "sel=0.100 hit=0.900" in after["Semantic Cost Model"]
    assert after["Plan Rows"] < before["Plan Rows"]
    assert after["Total Cost"] < before["Total Cost"], (
        "cache hits make the node cheaper"
    )


def test_two_predicates_are_stacked_by_rank(conn: psycopg.Connection) -> None:
    # B rejects 90% of rows at the same cost -> higher rank -> applied first (bottom).
    conn.execute(
        "INSERT INTO trimem.stats(model, prompt_version, predicate, n_eval, n_true, n_hit) "
        "SELECT current_setting('trimem.model'), current_setting('trimem.prompt_version'), "
        "p, 1000, t, 0 FROM (VALUES (%s, 900), (%s, 100)) v(p, t)",
        (PRED_A, PRED_B),
    )
    p = plan(
        conn,
        f"SELECT id FROM attempt WHERE trimem.sem_filter('{PRED_A}', outcome) "
        f"AND trimem.sem_filter('{PRED_B}', outcome)",
    )
    top, bottom = sem_nodes(p)
    assert bottom["Semantic Predicate"] == PRED_B
    assert top["Semantic Predicate"] == PRED_A


# ------------------------------------------------------------------ V + R + Sem
def test_ann_index_scan_feeds_the_semantic_node(conn: psycopg.Connection) -> None:
    p = plan(conn, Q_VRS)
    assert p["Node Type"] == "Limit"
    sem = sem_nodes(p)[0]
    child = sem["Plans"][0]
    assert child["Node Type"] == "Index Scan" and child["Index Name"] == "attempt_hnsw"
    assert "Order By" in child, "ANN order is preserved through SemFilter (no Sort)"
    assert not any(n["Node Type"] == "Sort" for _, n in nodes(p))
    assert sem["Batch Size"].startswith("10 "), "LIMIT 5 / sel 0.5 -> batch 10"


def test_limit_uses_semantic_startup_cost(conn: psycopg.Connection) -> None:
    p = plan(conn, Q_VRS)
    sem = sem_nodes(p)[0]
    # Limit pays startup + 5/rows of the run cost; startup includes one judged batch.
    assert sem["Startup Cost"] >= 10 * 2125
    assert p["Total Cost"] < sem["Total Cost"] / 10


# ------------------------------------------------------------------ G + R + Sem
def expand_nodes(p: dict) -> list[dict]:
    return [n for _, n in nodes(p) if n.get("Custom Plan Provider") == "TriMEM Expand"]


def test_graph_expand_is_a_costed_native_node(conn: psycopg.Connection) -> None:
    p = plan(conn, Q_GRS)
    (expand,) = expand_nodes(p)
    assert expand["Edge Type"] == "lineage"
    assert "(stats)" in expand["Degree Model"]
    assert expand["Plan Rows"] <= 2, (
        "degree statistics, not the SRF default of 1000 rows"
    )
    assert not any(n["Node Type"] == "Function Scan" for _, n in nodes(p))


def test_semantic_filter_is_placed_above_the_join(conn: psycopg.Connection) -> None:
    p = plan(conn, Q_GRS)
    sem = sem_nodes(p)[0]
    assert sem["Placement"] == "top (above join)"
    assert sem["Plans"][0]["Node Type"] in ("Nested Loop", "Hash Join", "Merge Join")


def test_placement_flips_to_base_when_the_join_multiplies_rows(
    conn: psycopg.Connection,
) -> None:
    p = plan(conn, Q_DENSE)
    sem = sem_nodes(p)[0]
    assert sem["Placement"] == "base"
    assert sem["Plans"][0]["Relation Name"] == "attempt"
    assert expand_nodes(p)[0]["Edge Type"] == "similar"


def test_placement_follows_the_batch_cost(conn: psycopg.Connection) -> None:
    # Without a per-batch round-trip cost, judging 1-2 rows per nested-loop rescan is no longer
    # penalised; the planner may then keep the semantic filter on the inner scan.
    base = sem_nodes(plan(conn, Q_GRS))[0]
    conn.execute("SET trimem.llm_cost_per_batch = 1e7")
    pricey = sem_nodes(plan(conn, Q_GRS))[0]
    conn.execute("RESET trimem.llm_cost_per_batch")
    assert base["Placement"] == pricey["Placement"] == "top (above join)"
    assert pricey["Total Cost"] > base["Total Cost"]


# ------------------------------------------------------------------ V + G + R + Sem
def test_fig1b_plan_shape(conn: psycopg.Connection) -> None:
    """Rank(Sort) -> SemFilter -> [ANN ∪ Lookup] x Expand x R-filter, as in paper Fig. 1(b)."""
    p = plan(conn, Q_FIG1B)
    order = [n.get("Custom Plan Provider", n["Node Type"]) for _, n in nodes(p)]
    assert order[:3] == ["Limit", "Sort", "TriMEM SemFilter"]
    assert order.index("TriMEM SemFilter") < order.index("TriMEM Expand")
    assert "Append" in order
    text = json.dumps(p)
    assert "attempt_hnsw" in text  # V seed
    assert "attempt_pkey" in text  # R lookup seed
    sem = sem_nodes(p)[0]
    assert sem["Placement"] == "top (above join)"
    assert sem["Batch Size"].startswith("64 "), (
        "ORDER BY above: LIMIT must not shrink the batch"
    )
    assert "valid" in filters_text(p) and "sem_filter" not in filters_text(p)


# ------------------------------------------------------------------ fail-closed contract
def test_direct_evaluation_fails_closed(conn: psycopg.Connection) -> None:
    conn.execute("SET trimem.enable = off")
    with pytest.raises(psycopg.errors.FeatureNotSupported):
        conn.execute(f"SELECT trimem.sem_filter('{PRED_A}', 'x')").fetchall()


def test_or_of_semantic_predicate_is_rejected(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.FeatureNotSupported):
        plan(
            conn,
            f"SELECT id FROM attempt WHERE valid OR trimem.sem_filter('{PRED_A}', outcome)",
        )


def test_non_constant_predicate_is_rejected(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        plan(conn, "SELECT id FROM attempt WHERE trimem.sem_filter(task, outcome)")


# ------------------------------------------------------------------ SemExtract (0.3.0)
Q_LABEL = "Which search strategy? Answer one of: greedy, beam search, other"


def extract_nodes(p: dict) -> list[dict]:
    return [
        n for _, n in nodes(p) if n.get("Custom Plan Provider") == "TriMEM SemExtract"
    ]


def test_label_qual_is_lifted_into_the_extract_node(conn: psycopg.Connection) -> None:
    p = plan(
        conn,
        f"SELECT id, trimem.sem_extract('{Q_LABEL}', outcome) FROM attempt "
        f"WHERE valid AND trimem.sem_extract('{Q_LABEL}', outcome) = 'greedy'",
    )
    (sx,) = extract_nodes(p)
    assert sx["Placement"] == "single relation"
    assert "sem_extract" in sx["Filter"]
    # no cached labels yet: the closed-label-set default, not PG's 0.5% equality default
    assert "label qual sel=0.200" in sx["Semantic Cost Model"]
    below = [n for _, n in nodes(sx["Plans"][0])]
    assert "sem_extract" not in " ".join(str(n.get("Filter", "")) for n in below)


Q_EXTRACT_DENSE = f"""
SELECT s.id, e.dst, trimem.sem_extract('{Q_LABEL}', s.outcome) FROM attempt s,
  LATERAL trimem.expand(s.vid, 'similar', 10) e
WHERE s.task = 'task3' AND trimem.sem_extract('{Q_LABEL}', s.outcome) <> 'other'
"""


def test_extract_placement_is_enumerated_and_forcible(conn: psycopg.Connection) -> None:
    (sx,) = extract_nodes(plan(conn, Q_EXTRACT_DENSE))
    decision = sx["Placement Decision"]
    assert "extract:s=base" in decision and "extract:s=top" in decision
    assert sx["Placement"] == "base", "the join multiplies rows: label the seeds"
    try:
        conn.execute("SET trimem.force_placement = 's=top'")
        (top,) = extract_nodes(plan(conn, Q_EXTRACT_DENSE))
    finally:
        conn.execute("RESET trimem.force_placement")
    assert top["Placement"] == "top (above join, forced)"
    assert "(forced)" in top["Placement Decision"]


def test_label_qual_stays_below_an_outer_join(conn: psycopg.Connection) -> None:
    sql = (
        "SELECT s.id, c.id FROM attempt s LEFT JOIN attempt c ON c.id = s.id * 2 "
        f"AND trimem.sem_extract('{Q_LABEL}', c.outcome) = 'greedy' WHERE s.id < 50"
    )
    try:
        conn.execute("SET trimem.force_placement = 'c=top'")
        (sx,) = extract_nodes(plan(conn, sql))
    finally:
        conn.execute("RESET trimem.force_placement")
    assert sx["Placement"].startswith("base")  # TOP would filter null-extended rows
    assert "extract:c=top" not in sx["Placement Decision"]
