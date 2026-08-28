"""Build the native depth-3 lineage/fitness synopsis used by VFWD and AIVG.

The statistic is physical, not logical: every source node gets a native adjacency
list containing the same descendants as ``eg_lineage`` at depth 1..3, ordered by
``fitness DESC, node_uid ASC``.  Queries can therefore apply their predicate and
return ranked rows one ``Next`` at a time without materialising a reach set or sorting
it at runtime.  ``gem_eg_edge`` remains an audit/idempotence mirror only.

    python3 -m tools.evotrace.build_plan_stats --scope evotrace:349117b0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.store import DEFAULT_DSN, EgStore

STAT_NAME = "lineage_h3_fitness"
STAT_VERSION = "v1"
EDGE_TYPE = "eg_lineage_h3_fit_v1"
MAX_HOPS = 3
ORDERING = "fitness DESC NULLS LAST, node_uid ASC"


def _corpus_hash(normalized: Path) -> str:
    digest = hashlib.sha256()
    for name in ("nodes.jsonl", "lineage_edges.jsonl"):
        path = normalized / name
        digest.update(name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _fitness(corpus: Corpus, uid: str) -> float:
    value = corpus.nodes[uid].get("fitness")
    if value is None or not math.isfinite(float(value)):
        return float("-inf")
    return float(value)


def build(store: EgStore, *, scope_id: str, normalized: Path) -> dict[str, Any]:
    store.init_schema()
    corpus = Corpus.load(normalized)
    vids = store.vertex_ids(scope_id)
    missing = sorted(set(corpus.nodes) - set(vids))
    if missing:
        raise RuntimeError(
            f"scope {scope_id} is missing {len(missing)} normalized nodes; "
            "load the corpus before building plan statistics"
        )

    existing = store.conn.execute(
        "SELECT complete, source_vertices, synopsis_edges, corpus_sha256"
        " FROM gem_eg_plan_stat"
        " WHERE scope_id=%s AND stat_name=%s AND stat_version=%s",
        (scope_id, STAT_NAME, STAT_VERSION),
    ).fetchone()
    corpus_sha = _corpus_hash(normalized)
    if existing and existing[0] and existing[3] == corpus_sha:
        return {
            "status": "already_complete",
            "scope_id": scope_id,
            "source_vertices": int(existing[1]),
            "synopsis_edges": int(existing[2]),
            "corpus_sha256": corpus_sha,
        }
    if existing and existing[3] != corpus_sha:
        raise RuntimeError(
            "a v1 synopsis exists for a different corpus; use a new "
            "stat/edge-type version instead of appending into ordered adjacency"
        )

    store.conn.execute(
        "INSERT INTO gem_eg_plan_stat"
        " (scope_id,stat_name,stat_version,relation,max_hops,ordering,"
        "  source_vertices,synopsis_edges,corpus_sha256,complete,metadata)"
        " VALUES (%s,%s,%s,'lineage',%s,%s,0,0,%s,false,%s)"
        " ON CONFLICT (scope_id,stat_name,stat_version) DO UPDATE SET"
        " complete=false, corpus_sha256=EXCLUDED.corpus_sha256",
        (
            scope_id,
            STAT_NAME,
            STAT_VERSION,
            MAX_HOPS,
            ORDERING,
            corpus_sha,
            json.dumps({"edge_type": EDGE_TYPE, "builder": __name__}),
        ),
    )
    store.conn.commit()

    added = 0
    sources = 0
    # One source per transaction: a crash never publishes a partial adjacency list as
    # complete, and the final receipt is the eligibility gate.
    for source_uid in sorted(corpus.nodes):
        reached = corpus.traverse(source_uid, relation="lineage", hops=MAX_HOPS)
        reached.sort(key=lambda uid: (-_fitness(corpus, uid), uid))
        for dest_uid in reached:
            if store.link(
                vids[source_uid],
                vids[dest_uid],
                relation=STAT_NAME,
                edge_type=EDGE_TYPE,
                rollup_of="eg_lineage",
                provenance=f"{STAT_NAME}:{STAT_VERSION}",
                dataset_revision=corpus_sha,
            ):
                added += 1
        sources += 1
        store.conn.commit()

    total = int(
        store.conn.execute(
            "SELECT count(*) FROM gem_eg_edge e"
            " JOIN gem_eg_vertex v ON v.id=e.src"
            " WHERE v.scope_id=%s AND e.edge_type=%s",
            (scope_id, store.edge_type_id(EDGE_TYPE)),
        ).fetchone()[0]
    )
    expected = sum(
        len(corpus.traverse(uid, relation="lineage", hops=MAX_HOPS))
        for uid in corpus.nodes
    )
    if total != expected:
        raise RuntimeError(
            f"synopsis audit failed: native-edge mirror has {total}, expected {expected}"
        )
    store.conn.execute(
        "UPDATE gem_eg_plan_stat SET source_vertices=%s,synopsis_edges=%s,"
        " complete=true,built_at=now()"
        " WHERE scope_id=%s AND stat_name=%s AND stat_version=%s",
        (sources, total, scope_id, STAT_NAME, STAT_VERSION),
    )
    store.conn.commit()
    return {
        "status": "built",
        "scope_id": scope_id,
        "source_vertices": sources,
        "synopsis_edges": total,
        "added_edges": added,
        "corpus_sha256": corpus_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--scope", required=True)
    parser.add_argument(
        "--normalized", type=Path, default=Path("data/evotrace/normalized")
    )
    args = parser.parse_args(argv)
    store = EgStore.connect(args.dsn)
    try:
        print(
            json.dumps(
                build(store, scope_id=args.scope, normalized=args.normalized), indent=2
            )
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
