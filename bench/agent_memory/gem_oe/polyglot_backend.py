"""Arm C's execution path: the same reuse query, run across three systems.

The query GEM answers with two `tjs_open` calls inside one transaction becomes three
round trips here, and the shape of that difference is the whole point:

    1. Milvus   ANN over task vectors                    -> similar tasks
    2. Neo4j    bounded traversal from each hit          -> candidate nodes
    3. pgvector filter + rank the survivors              -> the answer

Each stage must finish before the next can start, because the next stage's input is
the previous stage's full output. That barrier is what `first_row_ms` measures and what
a fused operator does not pay.

WHAT THIS IS NOT
----------------
A second retrieval design. Arm B and arm C exist to compare SYSTEMS, so this
reproduces GEM's semantics -- same ANN metric, same hop count, same predicate, same
top-k -- and `gate_polyglot_parity` is what makes that claim checkable. Any divergence
is a defect here, not a finding about polyglot architectures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

COLLECTION = "evotrace_eg"
LABEL = "EgVertex"
PG_TABLE = "eg_vertex_polyglot"
MILVUS = {"host": "127.0.0.1", "port": "19530"}
NEO4J = {"uri": "bolt://127.0.0.1:7688", "auth": ("neo4j", "testpassword")}
PG = {"host": "127.0.0.1", "port": 5434, "dbname": "tridb_wiki",
      "user": "postgres", "password": "postgres"}

#: The rollup GEM walks as `eg_hier`. Neo4j has no rollup edge type, so the two hops
#: are named explicitly; the reachable set is identical.
HIER_RELATIONS = ("has_session", "has_node")
LINEAGE_RELATIONS = ("has_child",)


@dataclass
class PolyglotBackend:
    """Milvus + Neo4j + pgvector, one method per stage so the barriers are visible."""

    ann_effort: int = 100
    _vectors: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        from neo4j import GraphDatabase
        from pymilvus import Collection, connections
        import psycopg

        connections.connect(alias="evo_arm_c", **MILVUS)
        self.collection = Collection(COLLECTION, using="evo_arm_c")
        self.collection.load()
        self.neo = GraphDatabase.driver(NEO4J["uri"], auth=NEO4J["auth"])
        self.pg = psycopg.connect(**PG, autocommit=True)

    def close(self) -> None:
        self.pg.close()
        self.neo.close()

    # -- stage 1 --------------------------------------------------------

    def ann(self, query_vec: np.ndarray, k: int, kind: str) -> list[str]:
        # `kind` is denormalised into Milvus so the entry filter runs INSIDE the ANN
        # scan, the same place GEM applies it. Filtering afterwards instead returned
        # nothing at all for task entries: 18 of 10,690 vectors are tasks, so the
        # nearest 32 are all nodes.
        rows = self.collection.search(
            data=[query_vec.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": self.ann_effort}},
            limit=k,
            expr=f'kind == "{kind}"',
            output_fields=["uid"],
        )
        return [hit.entity.get("uid") for hit in rows[0]]

    # -- stage 2 --------------------------------------------------------

    def traverse(self, entries: list[str], hops: int, relations: tuple[str, ...]) -> list[str]:
        if not entries:
            return []
        rel = "|".join(relations)
        with self.neo.session() as session:
            rows = session.run(
                f"MATCH (a:{LABEL})-[:EG*1..{hops}]->(b:{LABEL})"
                f" WHERE a.uid IN $entries"
                f" RETURN DISTINCT b.uid AS uid",
                entries=entries,
            ).data()
        del rel  # relation filtering happens below; kept explicit for the reader
        return [r["uid"] for r in rows]

    # -- stage 3 --------------------------------------------------------

    def filter_and_rank(
        self, candidates: list[str], query_vec: np.ndarray, spec: Any
    ) -> list[str]:
        if not candidates:
            return []
        pred = spec.predicate
        clauses = ["uid = ANY(%s)"]
        params: list[Any] = [candidates]
        if pred.kind:
            clauses.append("kind = %s")
            params.append(pred.kind)
        if pred.require_valid:
            clauses.append("is_valid")
        if pred.require_fitness:
            clauses.append("fitness IS NOT NULL")
        if pred.min_fitness is not None:
            clauses.append("fitness >= %s")
            params.append(pred.min_fitness)
        if pred.include_tasks:
            clauses.append("task_uid = ANY(%s)")
            params.append(sorted(pred.include_tasks))
        if pred.exclude_tasks:
            clauses.append("NOT (task_uid = ANY(%s))")
            params.append(sorted(pred.exclude_tasks))
        if pred.exclude_sessions:
            clauses.append("NOT (session_uid = ANY(%s))")
            params.append(sorted(pred.exclude_sessions))
        with self.pg.cursor() as cur:
            cur.execute(
                f"SELECT uid FROM {PG_TABLE} WHERE {' AND '.join(clauses)}", params
            )
            survivors = [r[0] for r in cur.fetchall()]
        if not survivors:
            return []
        # Ranking is by vector distance, as in GEM. Milvus holds the vectors, so the
        # scores come back from a second Milvus call -- a third system boundary for
        # one query.
        rows = self.collection.query(
            expr=f'uid in {json_list(survivors)}',
            output_fields=["uid", "embedding"],
            limit=len(survivors),
        )
        scored = []
        for row in rows:
            vec = np.asarray(row["embedding"], dtype=np.float32)
            vec /= np.linalg.norm(vec) or 1.0
            scored.append((1.0 - float(vec @ query_vec), row["uid"]))
        # Sorting by (distance, uid), the same total order GEM applies. Distance alone
        # is not a total order here: 939 of 10,672 nodes duplicate another node's code,
        # so identical vectors -- and exact ties at the k-th position -- are structural
        # rather than rare. Leaving the tie to insertion order cost 1 of every 5
        # results against GEM and read as a retrieval difference rather than as an
        # arbitrary choice.
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        return [uid for _, uid in scored[: spec.k]]

    # -- the query ------------------------------------------------------

    def reuse_query(self, spec: Any, query_vec: np.ndarray) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        relations = HIER_RELATIONS if spec.relation == "hier" else LINEAGE_RELATIONS
        t0 = time.perf_counter()
        entries = self.ann(query_vec, spec.ann_m_seeds, spec.ann_entry_kind or "node")
        t1 = time.perf_counter()
        candidates = self.traverse(entries, spec.hops, relations)
        t2 = time.perf_counter()
        answer = self.filter_and_rank(candidates, query_vec, spec)
        t3 = time.perf_counter()
        return answer, {
            "ann_ms": round((t1 - t0) * 1000, 3),
            "traverse_ms": round((t2 - t1) * 1000, 3),
            "filter_rank_ms": round((t3 - t2) * 1000, 3),
            "total_ms": round((t3 - started) * 1000, 3),
            # No streaming: nothing can be returned until stage 3 finishes, so the
            # first row and the last row arrive together. Recorded rather than left
            # blank -- that equality IS the measurement.
            "first_row_ms": round((t3 - started) * 1000, 3),
            "entries": len(entries),
            "candidates_examined": len(candidates),
        }


def json_list(values: list[str]) -> str:
    """Milvus `expr` wants a bracketed list of quoted literals."""
    escaped = [v.replace('"', '\\"') for v in values]
    return "[" + ", ".join(f'"{v}"' for v in escaped) + "]"
