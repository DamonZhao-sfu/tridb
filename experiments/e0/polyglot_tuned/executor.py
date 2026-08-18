"""Polyglot-Tuned executor: one STARK-PRIME query, one plan, measured end to end.

THE QUERY, DECOMPOSED
---------------------
A STARK-PRIME query asks: of the entities of type T that are reachable from anchor A within
`hops` typed edges, which best match this natural-language question? That is three modalities
in one request:

    vector      relevance of a candidate's text to the question
    graph       reachability from the anchor along the typed edge set
    relational  entity_type == T

The anchor is taken as GIVEN (the annotation resolved it; an entity linker would in a real
deployment). What E0 measures is not anchor resolution -- it is the cost of COMPOSING the
three legs, which is what an application author holding three separate engines must do by
hand, in an order they must choose without a cost model.

PLAN SHAPES
-----------
vector_first   Milvus ANN top-k over the WHOLE corpus, then keep only the hits Neo4j
               confirms reachable from the anchor, then rank.
filter_first   pgvector restricts to entity_type = T and does the ANN inside that subset,
               then Neo4j confirms reachability.
traverse_first Neo4j expands from the anchor first (capped at k), then the predicate, then
               the exact vector rerank. The traversal GENERATES the candidates.

D1 originally froze the space to the first two. THAT DECISION WAS REVERSED BY MEASUREMENT,
and the measurement is worth recording because it is a result in itself:

    official answers rank 4,589 / 11,543 / 15,913 / 18,772 / 55,160 in a GLOBAL exact
    vector ranking over the 129,375-node corpus

No ANN top-k with k <= 100 can contain them. With the anchor given, the vector leg is a
RANKER INSIDE the reach set, never a candidate generator -- so vector_first and filter_first
are structurally incapable of answering a STARK-PRIME query, and a plan space containing
only those two has zero feasible plans and therefore no plan spread at all. They are kept in
the space precisely because that failure is the point: they are the plans a vector-first
habit produces, and E0 should show what they cost and what they miss.

`k` MEANS THE SAME THING IN ALL THREE SHAPES: the cap on the intermediate result carried to
the next operator. For the ANN-led shapes that is the ANN top-k; for traverse_first it is the
frontier cap (trevillisPlan's "beam width / expansion strategy"). That is what makes the
latency of different shapes comparable at a given k.

Both end in an exact pgvector rerank so the two shapes are ranked on the same scale and are
therefore comparable at equal quality (docs/e0_plan_space_execution_v0.1.0.md §3.1).

PREDICATE PLACEMENT is where `entity_type = T` is applied:
    pre     pushed into the ANN operator itself (a WHERE alongside the vector order-by)
    during  applied inside the Cypher traversal (WHERE b.entity_type = $t)
    post    applied in the application after the candidate set comes back
Placement changes NOTHING about the answer set and everything about how much data crosses a
store boundary -- which is exactly the mechanism E0 is instrumenting.

TIMING AND FAIRNESS
-------------------
* The query embedding is computed ONCE per query and cached before any plan runs. Every plan
  shares it, so no shape is charged for the embedding.
* Every leg is client-timed as Python wall clock over its real client (pymilvus / neo4j /
  psycopg), so no store gets a server-side timing advantage.
* Connections are pooled and reused across plans (`tuning=tuned`). `tuning=naive` reconnects
  per leg -- the strawman -- so the report can show what tuning bought and prove the tuned
  baseline is not a strawman.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

ID_BYTES = 8  # a node id crossing a store boundary, as an int64


@dataclass(frozen=True)
class PlanSpec:
    shape: str
    k: int
    hops: int
    predicate_placement: str

    @property
    def tag(self) -> str:
        return f"{self.shape}|k{self.k}|h{self.hops}|{self.predicate_placement}"


@dataclass(frozen=True)
class Query:
    query_id: str
    query_text: str
    anchor_ids: tuple[int, ...]
    edge_types: tuple[str, ...]
    hop_limit: int
    target_entity_type: str
    answer_ids: tuple[int, ...]
    annotation_status: str
    template: str

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Query":
        return Query(
            query_id=row["query_id"],
            query_text=row["query_text"],
            anchor_ids=tuple(int(x) for x in row["anchor_ids"]),
            edge_types=tuple(row["edge_types"]),
            hop_limit=int(row["hop_limit"]),
            target_entity_type=row["target_entity_type"],
            answer_ids=tuple(int(x) for x in row["answer_ids"]),
            annotation_status=row.get("annotation_status", "unknown"),
            template=row.get("template", "unknown"),
        )


@dataclass
class PlanResult:
    query_id: str
    plan: PlanSpec
    ranked: list[int] = field(default_factory=list)
    latency_ms: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    round_trips: int = 0
    bytes_shipped: int = 0
    cardinality: dict[str, int] = field(default_factory=dict)
    infeasible: bool = False
    error: str = ""


def rel_type(edge_type: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", edge_type).strip("_").lower()


def vec_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


class PolyglotTuned:
    """The three stores, co-located, with pooled clients — the Polyglot-Tuned baseline."""

    RERANK_LIMIT = 20  # the ranked list every quality metric is computed over

    def __init__(self, cfg: dict[str, Any], *, tuning: str = "tuned") -> None:
        self.cfg = cfg
        self.tuning = tuning
        self._milvus = None
        self._neo4j = None
        self._pg = None

    # -- connections ---------------------------------------------------------------
    def milvus(self):
        from pymilvus import Collection, connections

        if self._milvus is None or self.tuning == "naive":
            connections.connect(
                alias="e0x", host=self.cfg["milvus_host"], port=self.cfg["milvus_port"]
            )
            collection = Collection(self.cfg["milvus_collection"], using="e0x")
            collection.load()
            self._milvus = collection
        return self._milvus

    def neo4j(self):
        from neo4j import GraphDatabase

        if self._neo4j is None or self.tuning == "naive":
            self._neo4j = GraphDatabase.driver(
                self.cfg["neo4j_uri"],
                auth=(self.cfg["neo4j_user"], self.cfg["neo4j_password"]),
            )
        return self._neo4j

    def pg(self):
        import psycopg

        if self._pg is None or self._pg.closed or self.tuning == "naive":
            self._pg = psycopg.connect(
                host=self.cfg["pg_host"],
                port=self.cfg["pg_port"],
                dbname=self.cfg["pg_db"],
                user=self.cfg["pg_user"],
                password=self.cfg["pg_password"],
            )
            self._pg.autocommit = True
        return self._pg

    def close(self) -> None:
        if self._neo4j is not None:
            self._neo4j.close()
        if self._pg is not None and not self._pg.closed:
            self._pg.close()

    # -- legs ----------------------------------------------------------------------
    def ann_global(self, qvec, k: int) -> list[int]:
        """Milvus ANN over the whole corpus (no predicate available in this collection)."""
        hits = self.milvus().search(
            [list(qvec)],
            "embedding",
            {"metric_type": "COSINE", "params": {"ef": max(64, k * 2)}},
            limit=k,
            output_fields=["node_id"],
        )
        return [int(h.id) for h in hits[0]]

    def ann_filtered(self, qvec, k: int, entity_type: str | None) -> list[int]:
        """pgvector ANN, optionally with the relational predicate pushed into the scan."""
        cur = self.pg().cursor()
        table = self.cfg["pg_table"]
        literal = vec_literal(qvec)
        if entity_type is None:
            cur.execute(
                f"SELECT node_id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (literal, k),
            )
        else:
            cur.execute(
                f"SELECT node_id FROM {table} WHERE entity_type = %s "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (entity_type, literal, k),
            )
        rows = [int(r[0]) for r in cur.fetchall()]
        cur.close()
        return rows

    def reachable(
        self,
        anchors: Sequence[int],
        edge_types: Sequence[str],
        hops: int,
        *,
        entity_type: str | None,
        restrict_to: Sequence[int] | None,
        limit: int | None = None,
    ) -> set[int]:
        """Typed variable-length expansion, mirroring the audited reachability semantics."""
        types = "|".join(rel_type(t) for t in edge_types)
        where = []
        params: dict[str, Any] = {"anchors": [int(a) for a in anchors]}
        if entity_type is not None:
            where.append("b.entity_type = $etype")
            params["etype"] = entity_type
        if restrict_to is not None:
            where.append("b.node_id IN $keep")
            params["keep"] = [int(x) for x in restrict_to]
        cypher = (
            f"MATCH (a:{self.cfg['neo4j_label']})-[:{types}*1..{hops}]->"
            f"(b:{self.cfg['neo4j_label']}) "
            "WHERE a.node_id IN $anchors"
            + ("".join(f" AND {c}" for c in where))
            + " RETURN DISTINCT b.node_id AS id"
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        with self.neo4j().session() as session:
            return {int(r["id"]) for r in session.run(cypher, **params)}

    def rerank(self, qvec, candidates: Sequence[int], limit: int) -> list[int]:
        if not candidates:
            return []
        cur = self.pg().cursor()
        cur.execute(
            f"SELECT node_id FROM {self.cfg['pg_table']} WHERE node_id = ANY(%s) "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (list(candidates), vec_literal(qvec), limit),
        )
        rows = [int(r[0]) for r in cur.fetchall()]
        cur.close()
        return rows

    # -- plans ---------------------------------------------------------------------
    def run(self, query: Query, plan: PlanSpec, qvec) -> PlanResult:
        result = PlanResult(query_id=query.query_id, plan=plan)
        started = time.perf_counter()
        try:
            if plan.shape == "vector_first":
                self._vector_first(query, plan, qvec, result)
            elif plan.shape == "filter_first":
                self._filter_first(query, plan, qvec, result)
            elif plan.shape == "traverse_first":
                self._traverse_first(query, plan, qvec, result)
            else:
                raise ValueError(f"unimplemented plan shape {plan.shape!r}")
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            result.infeasible = True
            result.error = f"{type(exc).__name__}: {exc}"
        result.latency_ms = (time.perf_counter() - started) * 1e3
        return result

    def _vector_first(self, query: Query, plan: PlanSpec, qvec, out: PlanResult) -> None:
        # 1. ANN. `pre` placement pushes entity_type into the ANN scan, which only pgvector
        #    can do here (the Milvus collection carries vectors only) -- itself a finding
        #    about what a polyglot stack can and cannot push down.
        t0 = time.perf_counter()
        if plan.predicate_placement == "pre":
            seeds = self.ann_filtered(qvec, plan.k, query.target_entity_type)
        else:
            seeds = self.ann_global(qvec, plan.k)
        t1 = time.perf_counter()
        out.stage_ms["ann"] = (t1 - t0) * 1e3
        out.round_trips += 1
        out.cardinality["seeds"] = len(seeds)
        out.bytes_shipped += len(seeds) * ID_BYTES

        # 2. Graph: keep the ANN hits the anchor can actually reach.
        reached = self.reachable(
            query.anchor_ids,
            query.edge_types,
            plan.hops,
            entity_type=(
                query.target_entity_type if plan.predicate_placement == "during" else None
            ),
            restrict_to=seeds,
        )
        t2 = time.perf_counter()
        out.stage_ms["traverse"] = (t2 - t1) * 1e3
        out.round_trips += 1
        out.cardinality["reached"] = len(reached)
        out.bytes_shipped += (len(seeds) + len(reached)) * ID_BYTES

        # 3. Relational predicate, if it has not been applied yet.
        candidates = list(reached)
        if plan.predicate_placement == "post":
            candidates = self._filter_type(candidates, query.target_entity_type, out)
        t3 = time.perf_counter()
        out.stage_ms["filter"] = (t3 - t2) * 1e3
        out.cardinality["candidates"] = len(candidates)

        out.ranked = self.rerank(qvec, candidates, self.RERANK_LIMIT)
        out.stage_ms["rerank"] = (time.perf_counter() - t3) * 1e3
        out.round_trips += 1
        out.bytes_shipped += len(candidates) * ID_BYTES

    def _filter_first(self, query: Query, plan: PlanSpec, qvec, out: PlanResult) -> None:
        # 1. Relational-first ANN: restrict to the target type, rank inside it.
        t0 = time.perf_counter()
        entity_type = (
            None if plan.predicate_placement == "post" else query.target_entity_type
        )
        seeds = self.ann_filtered(qvec, plan.k, entity_type)
        t1 = time.perf_counter()
        out.stage_ms["ann"] = (t1 - t0) * 1e3
        out.round_trips += 1
        out.cardinality["seeds"] = len(seeds)
        out.bytes_shipped += len(seeds) * ID_BYTES

        # 2. Graph confirmation from the anchor.
        reached = self.reachable(
            query.anchor_ids,
            query.edge_types,
            plan.hops,
            entity_type=(
                query.target_entity_type if plan.predicate_placement == "during" else None
            ),
            restrict_to=seeds,
        )
        t2 = time.perf_counter()
        out.stage_ms["traverse"] = (t2 - t1) * 1e3
        out.round_trips += 1
        out.cardinality["reached"] = len(reached)
        out.bytes_shipped += (len(seeds) + len(reached)) * ID_BYTES

        candidates = list(reached)
        if plan.predicate_placement == "post":
            candidates = self._filter_type(candidates, query.target_entity_type, out)
        t3 = time.perf_counter()
        out.stage_ms["filter"] = (t3 - t2) * 1e3
        out.cardinality["candidates"] = len(candidates)

        out.ranked = self.rerank(qvec, candidates, self.RERANK_LIMIT)
        out.stage_ms["rerank"] = (time.perf_counter() - t3) * 1e3
        out.round_trips += 1
        out.bytes_shipped += len(candidates) * ID_BYTES

    def _traverse_first(self, query: Query, plan: PlanSpec, qvec, out: PlanResult) -> None:
        # 1. The traversal GENERATES candidates, capped at k (the frontier cap). This is the
        #    only shape whose candidate set can contain an answer that is not globally
        #    similar to the question -- which on STARK-PRIME is nearly all of them.
        t0 = time.perf_counter()
        reached = self.reachable(
            query.anchor_ids,
            query.edge_types,
            plan.hops,
            entity_type=(
                query.target_entity_type
                if plan.predicate_placement in ("pre", "during")
                else None
            ),
            restrict_to=None,
            limit=plan.k,
        )
        t1 = time.perf_counter()
        out.stage_ms["traverse"] = (t1 - t0) * 1e3
        out.round_trips += 1
        out.cardinality["seeds"] = 0
        out.cardinality["reached"] = len(reached)
        out.bytes_shipped += len(reached) * ID_BYTES

        candidates = list(reached)
        if plan.predicate_placement == "post":
            candidates = self._filter_type(candidates, query.target_entity_type, out)
        t2 = time.perf_counter()
        out.stage_ms["filter"] = (t2 - t1) * 1e3
        out.cardinality["candidates"] = len(candidates)

        out.ranked = self.rerank(qvec, candidates, self.RERANK_LIMIT)
        out.stage_ms["rerank"] = (time.perf_counter() - t2) * 1e3
        out.round_trips += 1
        out.bytes_shipped += len(candidates) * ID_BYTES

    def _filter_type(
        self, candidates: Sequence[int], entity_type: str, out: PlanResult
    ) -> list[int]:
        if not candidates:
            return []
        cur = self.pg().cursor()
        cur.execute(
            f"SELECT node_id FROM {self.cfg['pg_table']} "
            f"WHERE node_id = ANY(%s) AND entity_type = %s",
            (list(candidates), entity_type),
        )
        rows = [int(r[0]) for r in cur.fetchall()]
        cur.close()
        out.round_trips += 1
        out.bytes_shipped += len(candidates) * ID_BYTES
        return rows


# ----------------------------------------------------------------------------------
# Quality
# ----------------------------------------------------------------------------------


def quality(ranked: Sequence[int], answers: Sequence[int]) -> dict[str, float]:
    """Hit@1 / MRR drive equivalence; Recall@20 is recorded but saturates (§1.2)."""
    answer_set = set(answers)
    hit1 = 1.0 if ranked and ranked[0] in answer_set else 0.0
    mrr = 0.0
    for rank, node_id in enumerate(ranked, start=1):
        if node_id in answer_set:
            mrr = 1.0 / rank
            break
    hit5 = 1.0 if any(n in answer_set for n in ranked[:5]) else 0.0
    recall20 = (
        len(answer_set & set(ranked[:20])) / len(answer_set) if answer_set else 0.0
    )
    return {"hit@1": hit1, "mrr": mrr, "hit@5": hit5, "recall@20": recall20}


def load_queries(path) -> list[Query]:
    return [
        Query.from_row(json.loads(line))
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]
