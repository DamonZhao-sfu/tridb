from __future__ import annotations

from pathlib import Path

import pytest

# Throwaway names for the guard integration test below. Deliberately distinct
# from any `e0_*` collection/label/table: a live E1 measurement can be running
# against those at any time (see PolyglotLiveDataset in live_backend.py), and
# this test must never touch them.
_PROBE_COLLECTION = "e1_guard_probe"
_PROBE_LABEL = "E1GuardProbe"
_PROBE_TABLE = "e1_guard_probe"


@pytest.mark.integration
def test_polyglot_returns_nonempty_on_openevolve():
    """Green smoke check, not a regression guard for the v0.2 bug.

    The E0 v0.2 run produced empty result_ids on all 1010 OE cells because its
    polyglot stores were legitimately empty at measurement time (the
    OpenEvolve loader ran after the measurement — see
    results/e0/plan_space/polyglot_live_v0.2/run_manifest.json vs
    data/e0/openevolve_polyglot_load.json timestamps). It was never a
    live_backend.py composition-logic bug: this test passes unmodified,
    with no code change, once the stores are loaded — which disproved that
    hypothesis rather than confirming it. See
    test_polyglot_raises_on_empty_probe_store below for the actual
    regression guard.
    """
    from experiments.e0.plan_spread.config import load_config
    from experiments.e0.plan_spread.live_backend import PolyglotLiveDataset
    from experiments.e0.plan_spread.model import PlanSpec

    config = load_config(Path("configs/e0/plan_space_v0.3.yaml"))
    spec = config["datasets"]["openevolve"]
    backend = PolyglotLiveDataset("openevolve", spec)
    queries = backend.load_queries(Path(spec["queries"]))
    plan = PlanSpec(shape="traverse_first", k=20, hops=2, predicate_placement="post")
    nonempty = 0
    for query in queries:
        backend.prepare_query(query, {2})
        result = backend.execute(query, plan, top_n=20)
        if result["result_ids"]:
            nonempty += 1
    assert nonempty > 0, "every OpenEvolve query still returns an empty result set"


class _StubBackend:
    table = "e0_openevolve_node"
    _parent_cache: dict[str, list[str]] = {}

    def _parents_of(self, anchor_ids):
        return ["p1"]


@pytest.mark.unit
def test_polyglot_predicate_sql_honours_same_parent():
    from experiments.e0.plan_spread.model import QuerySpec

    query = QuerySpec(
        query_id="q",
        dataset="openevolve",
        query_text="t",
        anchor_ids=("a1",),
        answer_ids=("x",),
        edge_types=("evolved_to",),
        hop_limit=2,
        structured_predicate={"same_parent": True},
        target_entity_type="program",
        template="t",
        annotation_status="s",
        require_each_anchor=False,
    )
    from experiments.e0.plan_spread import live_backend

    sql, _params = live_backend.PolyglotLiveDataset._predicate_sql(
        _StubBackend(), query
    )
    assert "parent_id" in sql, f"same_parent not applied, got: {sql}"


@pytest.mark.integration
def test_polyglot_neo_reach_same_parent_narrows_candidates():
    """Behavioural guard for same_parent: it must actually change which
    candidates survive, not merely appear in generated SQL/Cypher text.
    `test_polyglot_predicate_sql_honours_same_parent` above asserts string
    containment on `_predicate_sql`'s output — that assertion cannot fail if
    `_neo_predicate` is broken, `_parents_of` is broken, or the parent
    direction is reversed.

    Deliberately calls `_neo_reach` directly with `apply_predicate=True` vs
    `False`, rather than going through `execute()` on a `traverse_first`/
    `during` plan. `execute()` would not isolate this: for `traverse_first`,
    `_pg_rank` (SQL) always re-applies `_predicate_sql` after the Neo4j
    traversal regardless of placement, so an end-to-end comparison stays
    narrowed even with `_neo_predicate`'s same_parent branch entirely
    deleted — verified by hand before writing this assertion. Calling
    `_neo_reach` directly is the only way to prove the Cypher clause itself
    does the narrowing; `apply_predicate=True` is also exactly what
    `during`-placement `traverse_first` plans pass in `execute()` (see
    live_backend.py), so this exercises the real code path, just without
    the SQL step that would otherwise mask a break in it.

    Read-only: constructs the backend and reads against the shared
    e0_openevolve_* stores, never mutates them, and never reloads.
    """
    from experiments.e0.plan_spread.config import load_config
    from experiments.e0.plan_spread.live_backend import PolyglotLiveDataset

    config = load_config(Path("configs/e0/plan_space_v0.3.yaml"))
    spec = config["datasets"]["openevolve"]
    backend = PolyglotLiveDataset("openevolve", spec)
    try:
        queries = {q.query_id: q for q in backend.load_queries(Path(spec["queries"]))}
        query = queries["oe-001"]
        assert query.structured_predicate.get("same_parent") is True

        with_predicate = backend._neo_reach(
            query, query.hop_limit, restrict_ids=None, apply_predicate=True
        )
        without_predicate = backend._neo_reach(
            query, query.hop_limit, restrict_ids=None, apply_predicate=False
        )
    finally:
        backend.close()

    assert set(with_predicate) <= set(without_predicate), (
        f"with-predicate {with_predicate} is not a subset of "
        f"without-predicate {without_predicate}"
    )
    assert len(with_predicate) < len(without_predicate), (
        "same_parent did not narrow _neo_reach's Cypher-side candidates: "
        f"with={with_predicate} without={without_predicate}"
    )


@pytest.mark.unit
def test_empty_stores_names_every_zero_leg():
    """Pure decision logic, no I/O: which legs are empty given their counts."""
    from experiments.e0.plan_spread.live_backend import _empty_stores

    assert _empty_stores(milvus=0, neo4j=0, postgres=0) == [
        "milvus",
        "neo4j",
        "postgres",
    ]
    assert _empty_stores(milvus=10, neo4j=0, postgres=5) == ["neo4j"]
    assert _empty_stores(milvus=10, neo4j=3, postgres=5) == []
    assert _empty_stores(milvus=0, neo4j=3, postgres=0) == ["milvus", "postgres"]


@pytest.mark.unit
def test_loader_command_is_runnable_for_known_datasets():
    """The empty-store error must name a real command, not just a diagnosis."""
    from experiments.e0.plan_spread.live_backend import _loader_command

    assert _loader_command("openevolve") == "make e0-openevolve-polyglot-load"
    assert _loader_command("stark_prime") == "python -m tools.e0.load_polyglot all"
    # An unknown dataset still gets something runnable, not a dead end.
    unknown = _loader_command("some_future_dataset")
    assert "make e0-openevolve-polyglot-load" in unknown
    assert "python -m tools.e0.load_polyglot all" in unknown


@pytest.mark.integration
def test_polyglot_raises_on_empty_probe_store():
    """Regression guard for the actual v0.2 bug: constructing the backend
    against a store that holds zero rows for the dataset must fail loudly
    instead of silently succeeding (which is exactly what let the v0.2 run
    produce 1,010 well-formed-but-empty observations with no error anywhere).

    Uses dedicated, disposable `e1_guard_probe` / `E1GuardProbe` objects that
    this test creates and drops itself — never any `e0_*` collection, label,
    or table, since a live E1 measurement can be running against those at any
    time and a test-owned DELETE/DROP on shared fixtures is a permanent
    hazard if interrupted mid-flight.
    """
    import psycopg
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    from experiments.e0.plan_spread.config import load_config
    from experiments.e0.plan_spread.live_backend import PolyglotLiveDataset

    config = load_config(Path("configs/e0/plan_space_v0.3.yaml"))
    base_spec = config["datasets"]["openevolve"]
    probe_spec = {
        **base_spec,
        "live": {
            **base_spec["live"],
            "milvus_collection": _PROBE_COLLECTION,
            "neo4j_label": _PROBE_LABEL,
            "postgres_table": _PROBE_TABLE,
        },
    }

    connections.connect(alias="e1_guard_probe", host="127.0.0.1", port="19530")
    if utility.has_collection(_PROBE_COLLECTION, using="e1_guard_probe"):
        utility.drop_collection(_PROBE_COLLECTION, using="e1_guard_probe")
    collection = Collection(
        _PROBE_COLLECTION,
        CollectionSchema(
            [
                FieldSchema(
                    "node_id", DataType.VARCHAR, max_length=64, is_primary=True
                ),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=8),
            ]
        ),
        using="e1_guard_probe",
    )
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 64},
        },
    )
    # No rows inserted into the collection, and no Neo4j nodes are ever
    # created under E1GuardProbe — both are empty by construction. Only the
    # postgres leg needs an explicit empty table (SELECT on a table that
    # doesn't exist raises before our check ever runs).

    pg = psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="tridb_wiki",
        user="postgres",
        password="postgres",
        autocommit=True,
    )
    try:
        with pg.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {_PROBE_TABLE}")
            cursor.execute(f"CREATE TABLE {_PROBE_TABLE} (node_id text PRIMARY KEY)")

        with pytest.raises(RuntimeError, match=r"milvus, neo4j, postgres"):
            PolyglotLiveDataset("e1_guard_probe", probe_spec)
    finally:
        with pg.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {_PROBE_TABLE}")
        pg.close()
        utility.drop_collection(_PROBE_COLLECTION, using="e1_guard_probe")
