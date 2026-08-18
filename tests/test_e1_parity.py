from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_polyglot_returns_nonempty_on_openevolve():
    """Regression: the E0 v0.2 run produced empty result_ids on all 1010 OE cells."""
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


@pytest.mark.integration
def test_polyglot_raises_instead_of_silently_empty_when_store_unloaded():
    """Regression: root cause of the v0.2 all-empty run was a load-order race —
    `e0-plan-live` does not depend on `e0-openevolve-polyglot-load`, so the
    measurement ran ~2 minutes before the OpenEvolve loader populated the
    stores (see results/e0/plan_space/polyglot_live_v0.2/run_manifest.json vs
    data/e0/openevolve_polyglot_load.json timestamps). Every leg was silently
    empty and every query silently returned `result_ids: []`. This asserts the
    fix: constructing the backend against an empty store now fails loudly
    instead of producing well-formed-but-empty observations.
    """
    import psycopg
    from experiments.e0.plan_spread.config import load_config
    from experiments.e0.plan_spread.live_backend import PolyglotLiveDataset
    from tools.e0.load_openevolve_polyglot import load_postgres

    config = load_config(Path("configs/e0/plan_space_v0.3.yaml"))
    spec = config["datasets"]["openevolve"]
    table = spec["live"]["postgres_table"]

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
            cursor.execute(f"DELETE FROM {table}")
        with pytest.raises(RuntimeError, match=r"holds 0 rows"):
            PolyglotLiveDataset("openevolve", spec)
    finally:
        pg.close()
        # Restore from the same parquet source the original load used, so the
        # fixture is byte-for-byte what it was before this test ran.
        load_postgres(drop=True)
