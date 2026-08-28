"""Load the enwiki N-article slice into the multi-store baseline for bench/wiki_fusion.

WHY THIS EXISTS
---------------
`bench/wiki_fusion.py` compares TriDB's single `tjs_open` call against a Milvus -> Neo4j ->
pgvector application-side pipeline. The TriDB side has a loader in-repo
(`tools/wiki_engine_load.py` + `scripts/wiki_engine_load.sh`). The BASELINE side does not:
`tools/wiki_neo4j_load.py` covers only the graph leg (and for a different purpose), and
nothing loads the wiki slice into Milvus or pgvector at all -- that was done by hand on the
Spark. This module is the missing loader, and it exists mainly to make three silent-corruption
hazards impossible rather than merely to move bytes.

HAZARD 1 -- THE INDUCED SUBGRAPH. `bench/wiki_h2h.load_induced_adj` builds the oracle from
edges with `src < N AND dst < N`, and `tools/wiki_engine_load.py` stages the same induced set
into the engine. An edge whose destination lies outside the slice must therefore NOT reach
Neo4j: if it does, the baseline's h-hop reach is a superset of the oracle's, the matched-recall
protocol silently compares two different questions, and the resulting speedup is meaningless.
`tools/wiki_neo4j_load.py --limit` truncates by COUNT, which is not the same predicate.

HAZARD 2 -- ID TYPE. The harness queries Neo4j with `ids=[str(x) for x in seed_ids]` and reads
`int(r["id"])` back, so the Spark's graph stored `Article.id` as a STRING. Loading it as an
integer makes `WHERE a.id IN $ids` match nothing: the baseline returns an empty reach set, scores
recall 0, and TriDB "wins" every hop by default. That is the most dangerous possible bug here --
it produces a large, clean, entirely fake speedup. The id type is therefore explicit
(`--id-type`), defaults to the string form the harness expects, and an integer twin property
(`iid`) with its own index is written alongside, so the fairness cost of a string key can be
measured by re-running the seed lookup against `iid` instead of being assumed away. A string
key IS slower to look up than an integer one, and that cost lands on the BASELINE -- so it must
be quantified before any speedup is quoted, not waved off.

HAZARD 3 -- RELATIONSHIP TYPE. The harness defaults to `[:RELATED]` (WH_NEO4J_REL); the other
loader writes `[:LINKS_TO]`. Mismatched, the traversal matches nothing -- the same fake-win
failure as hazard 2. This loader writes the harness's default and prints the env override.

The `verify` phase re-derives each benchmark query's h-hop reach INSIDE Neo4j and compares it
against the host-side induced adjacency the oracle uses. Counts agreeing is not enough; the
reach sets have to agree, per query, or no number may be quoted.

CLI:
    python -m tools.wiki_baseline_load all --n 200000
    python -m tools.wiki_baseline_load milvus|neo4j|postgres|verify --n 200000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DEFAULTS = {
    "manifest_dir": Path("data/wiki/enwiki"),
    "emb_path": Path("data/wiki/enwiki/emb/dense_id_aligned.npy"),
    "milvus_host": "localhost",
    "milvus_port": "19530",
    "milvus_collection": "wiki_articles",
    "neo4j_uri": "bolt://localhost:7688",
    "neo4j_user": "neo4j",
    "neo4j_password": "testpassword",
    "neo4j_label": "Article",
    "neo4j_rel": "RELATED",
    "pg_host": "127.0.0.1",
    "pg_port": 5434,
    "pg_db": "tridb_wiki",
    "pg_user": "postgres",
    "pg_password": "postgres",
    "pg_table": "wiki_article",
}


# ----------------------------------------------------------------------------------
# Manifest reading -- the induced predicate lives here and nowhere else
# ----------------------------------------------------------------------------------


def read_manifest(manifest_dir: Path) -> dict[str, Any]:
    return json.loads((manifest_dir / "manifest.json").read_text())


def _shard_paths(manifest: dict[str, Any], kind: str) -> list[str]:
    files = manifest["shards"][kind]["files"]
    return [f["path"] if isinstance(f, dict) else f for f in files]


def iter_articles(
    manifest_dir: Path, manifest: dict[str, Any], n: int
) -> Iterator[dict]:
    for shard in _shard_paths(manifest, "articles"):
        path = manifest_dir / shard
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if int(row["id"]) >= n:
                    return
                yield row


def iter_induced_edges(
    manifest_dir: Path, manifest: dict[str, Any], n: int
) -> Iterator[tuple[int, int]]:
    """Edges with BOTH endpoints inside the slice -- the oracle's graph, exactly."""
    for shard in _shard_paths(manifest, "edges"):
        path = manifest_dir / shard
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                tab = line.find("\t")
                if tab < 0:
                    continue
                src = int(line[:tab])
                dst = int(line[tab + 1 :])
                if src < n and dst < n:
                    yield src, dst


def edge_stats(manifest_dir: Path, manifest: dict[str, Any], n: int) -> dict[str, int]:
    """Raw out-edges of the first N articles vs the INDUCED subgraph on those articles.

    Reported together because the two numbers differ a lot and are easy to confuse in a
    write-up: the induced count is the one the benchmark actually loads and traverses.
    """
    raw = induced = 0
    for shard in _shard_paths(manifest, "edges"):
        path = manifest_dir / shard
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                tab = line.find("\t")
                if tab < 0:
                    continue
                src = int(line[:tab])
                if src >= n:
                    continue
                raw += 1
                if int(line[tab + 1 :]) < n:
                    induced += 1
    return {"raw_out_edges": raw, "induced_edges": induced}


def load_embeddings(emb_path: Path, n: int) -> np.ndarray:
    matrix = np.load(emb_path, mmap_mode="r")[:n].astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ----------------------------------------------------------------------------------
# Legs
# ----------------------------------------------------------------------------------


def load_milvus(cfg: dict[str, Any], n: int, *, drop: bool) -> dict[str, Any]:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    vectors = load_embeddings(cfg["emb_path"], n)
    dim = int(vectors.shape[1])
    connections.connect(alias="wb", host=cfg["milvus_host"], port=cfg["milvus_port"])
    name = cfg["milvus_collection"]
    if drop and utility.has_collection(name, using="wb"):
        utility.drop_collection(name, using="wb")
    collection = Collection(
        name,
        CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
            ]
        ),
        using="wb",
    )
    started = time.time()
    batch = 4096
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        collection.insert([list(range(start, stop)), vectors[start:stop].tolist()])
    collection.flush()
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    collection.load()
    return {
        "rows": int(collection.num_entities),
        "expected": n,
        "dim": dim,
        "seconds": round(time.time() - started, 1),
    }


def load_neo4j(
    cfg: dict[str, Any], n: int, *, drop: bool, id_type: str, batch: int = 20_000
) -> dict[str, Any]:
    from neo4j import GraphDatabase

    manifest = read_manifest(cfg["manifest_dir"])
    label = cfg["neo4j_label"]
    rel = cfg["neo4j_rel"]
    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    started = time.time()
    with driver.session() as session:
        if drop:
            while True:
                row = session.run(
                    f"MATCH (a:{label}) WITH a LIMIT 50000 DETACH DELETE a "
                    "RETURN count(*) AS c"
                ).single()
                if not row or row["c"] == 0:
                    break
        session.run(
            f"CREATE CONSTRAINT wiki_article_id IF NOT EXISTS "
            f"FOR (a:{label}) REQUIRE a.id IS UNIQUE"
        )
        session.run(
            f"CREATE INDEX wiki_article_iid IF NOT EXISTS FOR (a:{label}) ON (a.iid)"
        )

        def key(value: int):
            return str(value) if id_type == "string" else int(value)

        rows: list[dict[str, Any]] = []
        n_nodes = 0
        for article in iter_articles(cfg["manifest_dir"], manifest, n):
            rows.append(
                {
                    "id": key(int(article["id"])),
                    "iid": int(article["id"]),
                    "title": article.get("title", ""),
                }
            )
            if len(rows) >= batch:
                session.run(
                    f"UNWIND $rows AS r MERGE (a:{label} {{id: r.id}}) "
                    "SET a.iid = r.iid, a.title = r.title",
                    rows=rows,
                )
                n_nodes += len(rows)
                rows = []
        if rows:
            session.run(
                f"UNWIND $rows AS r MERGE (a:{label} {{id: r.id}}) "
                "SET a.iid = r.iid, a.title = r.title",
                rows=rows,
            )
            n_nodes += len(rows)
        node_seconds = time.time() - started

        pairs: list[dict[str, Any]] = []
        n_edges = 0
        for src, dst in iter_induced_edges(cfg["manifest_dir"], manifest, n):
            pairs.append({"s": key(src), "d": key(dst)})
            if len(pairs) >= batch:
                session.run(
                    f"UNWIND $rows AS r MATCH (s:{label} {{id: r.s}}), "
                    f"(d:{label} {{id: r.d}}) CREATE (s)-[:{rel}]->(d)",
                    rows=pairs,
                )
                n_edges += len(pairs)
                pairs = []
        if pairs:
            session.run(
                f"UNWIND $rows AS r MATCH (s:{label} {{id: r.s}}), "
                f"(d:{label} {{id: r.d}}) CREATE (s)-[:{rel}]->(d)",
                rows=pairs,
            )
            n_edges += len(pairs)

        observed = session.run(
            f"MATCH (:{label})-[r:{rel}]->(:{label}) RETURN count(r) AS c"
        ).single()["c"]
    driver.close()
    return {
        "nodes": n_nodes,
        "edges_sent": n_edges,
        "edges_in_graph": observed,
        "id_type": id_type,
        "relationship": rel,
        "node_seconds": round(node_seconds, 1),
        "seconds": round(time.time() - started, 1),
    }


def load_postgres(cfg: dict[str, Any], n: int, *, drop: bool) -> dict[str, Any]:
    import psycopg

    manifest = read_manifest(cfg["manifest_dir"])
    vectors = load_embeddings(cfg["emb_path"], n)
    dim = int(vectors.shape[1])
    table = cfg["pg_table"]
    conn = psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ("
        " id bigint PRIMARY KEY, title text, embedding vector({dim}))"
    )
    started = time.time()
    titles = {
        int(a["id"]): a.get("title", "")
        for a in iter_articles(cfg["manifest_dir"], manifest, n)
    }
    with cur.copy(f"COPY {table} (id, title, embedding) FROM STDIN") as copy:
        for node_id in range(n):
            literal = "[" + ",".join(f"{x:.6f}" for x in vectors[node_id]) + "]"
            copy.write_row((node_id, titles.get(node_id, ""), literal))
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
    )
    cur.execute(f"ANALYZE {table}")
    cur.execute(f"SELECT count(*) FROM {table}")
    rows = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {
        "rows": rows,
        "expected": n,
        "dim": dim,
        "seconds": round(time.time() - started, 1),
    }


# ----------------------------------------------------------------------------------
# Verification: the reach sets must AGREE, not merely the counts
# ----------------------------------------------------------------------------------


def host_reach(adj: dict[int, list[int]], seeds: list[int], hops: int) -> set[int]:
    """h-hop out-reach over the induced adjacency, the oracle's own definition."""
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: set[int] = set()
        for node in frontier:
            nxt.update(adj.get(node, ()))
        frontier = nxt - seen
        seen |= frontier
        if not frontier:
            break
    return seen - set(seeds)


def verify(
    cfg: dict[str, Any], n: int, *, id_type: str, probes: int, hops: int, seed: int
) -> dict[str, Any]:
    import psycopg
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections

    import sys

    sys.path.insert(0, ".")
    from bench.wiki_h2h import Cfg, load_induced_adj

    manifest = read_manifest(cfg["manifest_dir"])
    counts = edge_stats(cfg["manifest_dir"], manifest, n)

    hcfg = Cfg()
    hcfg.n = n
    hcfg.manifest_dir = cfg["manifest_dir"]
    adj = load_induced_adj(hcfg)
    host_edges = sum(len(v) for v in adj.values())

    connections.connect(alias="wbv", host=cfg["milvus_host"], port=cfg["milvus_port"])
    milvus_rows = int(Collection(cfg["milvus_collection"], using="wbv").num_entities)

    conn = psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    )
    cur = conn.cursor()
    cur.execute(f"SELECT count(*), count(embedding) FROM {cfg['pg_table']}")
    pg_rows, pg_vectors = cur.fetchone()
    cur.close()
    conn.close()

    label, rel = cfg["neo4j_label"], cfg["neo4j_rel"]
    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    rng = np.random.default_rng(seed)
    anchors = sorted(int(x) for x in rng.choice(n, size=probes, replace=False))
    mismatches: list[dict[str, Any]] = []
    with driver.session() as session:
        neo_nodes = session.run(f"MATCH (a:{label}) RETURN count(a) AS c").single()["c"]
        neo_edges = session.run(
            f"MATCH (:{label})-[r:{rel}]->(:{label}) RETURN count(r) AS c"
        ).single()["c"]
        for anchor in anchors:
            key = str(anchor) if id_type == "string" else anchor
            rows = session.run(
                f"MATCH (a:{label})-[:{rel}*1..{hops}]->(b:{label}) "
                "WHERE a.id IN $ids RETURN DISTINCT b.id AS id",
                ids=[key],
            )
            graph_side = {int(r["id"]) for r in rows}
            oracle_side = host_reach(adj, [anchor], hops)
            if graph_side != oracle_side:
                mismatches.append(
                    {
                        "anchor": anchor,
                        "neo4j": len(graph_side),
                        "oracle": len(oracle_side),
                        "only_in_neo4j": len(graph_side - oracle_side),
                        "only_in_oracle": len(oracle_side - graph_side),
                    }
                )
    driver.close()

    checks = {
        "milvus_rows_match": milvus_rows == n,
        "neo4j_nodes_match": neo_nodes == n,
        "neo4j_edges_match_induced": neo_edges == counts["induced_edges"],
        "host_adj_matches_induced": host_edges == counts["induced_edges"],
        "postgres_rows_match": pg_rows == n,
        "postgres_vectors_complete": pg_vectors == n,
        "reach_sets_agree": not mismatches,
    }
    return {
        "schema_version": "wiki-baseline-load-v0.1.0",
        "n": n,
        "edge_counts": counts,
        "host_induced_adj_edges": host_edges,
        "observed": {
            "milvus_rows": milvus_rows,
            "neo4j_nodes": neo_nodes,
            "neo4j_edges": neo_edges,
            "postgres_rows": pg_rows,
            "postgres_vectors": pg_vectors,
        },
        "reach_probe": {"anchors": probes, "hops": hops, "mismatches": mismatches},
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "phase", choices=["all", "milvus", "neo4j", "postgres", "verify", "counts"]
    )
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--id-type", choices=["string", "int"], default="string")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--probe-hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1354)
    parser.add_argument(
        "--out", type=Path, default=Path("data/wiki/baseline_load.json")
    )
    args = parser.parse_args(argv)

    cfg = dict(DEFAULTS)
    drop = not args.keep
    report: dict[str, Any] = {"n": args.n, "id_type": args.id_type}

    if args.phase == "counts":
        manifest = read_manifest(cfg["manifest_dir"])
        stats = edge_stats(cfg["manifest_dir"], manifest, args.n)
        print(
            f"[wiki-baseline] N={args.n}  raw out-edges={stats['raw_out_edges']:,}  "
            f"INDUCED edges={stats['induced_edges']:,}"
        )
        report["edge_counts"] = stats
    if args.phase in ("all", "milvus"):
        report["milvus"] = load_milvus(cfg, args.n, drop=drop)
        print(f"[wiki-baseline] milvus   {report['milvus']}")
    if args.phase in ("all", "neo4j"):
        report["neo4j"] = load_neo4j(cfg, args.n, drop=drop, id_type=args.id_type)
        print(f"[wiki-baseline] neo4j    {report['neo4j']}")
        print(
            f"[wiki-baseline] harness env: WH_NEO4J_REL={cfg['neo4j_rel']} "
            f"WH_NEO4J_LABEL={cfg['neo4j_label']}"
        )
    if args.phase in ("all", "postgres"):
        report["postgres"] = load_postgres(cfg, args.n, drop=drop)
        print(f"[wiki-baseline] postgres {report['postgres']}")
    if args.phase in ("all", "verify"):
        result = verify(
            cfg,
            args.n,
            id_type=args.id_type,
            probes=args.probes,
            hops=args.probe_hops,
            seed=args.seed,
        )
        report["verify"] = result
        print(
            f"[wiki-baseline] raw out-edges={result['edge_counts']['raw_out_edges']:,}  "
            f"INDUCED={result['edge_counts']['induced_edges']:,}"
        )
        for name, ok in result["checks"].items():
            print(f"[wiki-baseline] {'PASS' if ok else 'FAIL'} {name}")
        for row in result["reach_probe"]["mismatches"][:5]:
            print(
                f"    anchor {row['anchor']}: neo4j={row['neo4j']} oracle={row['oracle']}"
            )
        print(f"[wiki-baseline] ready={result['ready']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if "verify" in report and not report["verify"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
