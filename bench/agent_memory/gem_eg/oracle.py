"""The exact answer, computed without the engine.

Every latency number in this track is gated on the engine agreeing with this module,
so the two must not share code. `tjs_open` is C over a native adjacency method and an
HNSW index; this is numpy over dicts. They share only the normalized JSONL.

WHAT IT MODELS
--------------
A W1 query is vector-seeded graph retrieval:

    seed vector -> (optional) ANN over an entry kind
                -> bounded typed traversal
                -> relational predicate
                -> ranked top-k

The oracle computes that with no approximation and no budget: full traversal, exact
cosine over every candidate. The engine will differ, and the SIZE of that difference —
attributable to HNSW recall, `tjs.graph_work_budget` censoring, and `term_cond` early
termination — is the measurement.

TWO RANKINGS, DELIBERATELY
--------------------------
`tjs_open` ranks by vector distance only (gap G4). The paper's reuse query wants the
*highest-reward* node. These are different queries and the difference is measurable, so
both are first-class here:

``rank_by="similarity"``
    top-k by (cosine distance, node_uid) — what the operator natively returns.
``rank_by="reward"``
    top-k by (-fitness, node_uid) over the eligible set — what the paper asks for, and
    what the engine can only produce as an outer sort over a *bounded* stream. Under
    early termination the engine can therefore miss a high-reward node that sat far
    away in vector space. That loss is the point of measuring it.

Ties break on `node_uid` in both modes. Without a total order "exact match" is not a
well-defined gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from bench.agent_memory.gem_eg.corpus import Corpus

RankBy = Literal["similarity", "reward"]

#: The four query SHAPES. An ablation removes a leg of the fused query, which changes
#: the shape — it is not a parameter setting. Encoding it as `hops=0` (the first
#: attempt) collapsed `vector_only` to the empty set on both sides, where a parity gate
#: passes trivially and every quality metric reads 0.
#:
#:   ann_then_traverse  ANN entry -> typed traversal -> predicate -> rank by similarity
#:                       The full fused query.
#:   ann_only           ANN over the whole filtered table, no traversal. Removes the
#:                       GRAPH leg: what a vector store plus a predicate gives.
#:   task_scoped        Entries chosen by task membership, NOT by ANN, then traversal,
#:                       then predicate, ranked by reward. Removes the VECTOR leg
#:                       entirely: what a graph store plus a predicate gives.
#:   filter_only        Predicate alone, reward-ranked. The relational leg by itself.
#:
#: An earlier `traverse_only` mode was removed because it was mislabelled: it still
#: selected its entries by ANN, so "fused vs traverse_only" compared two RANKING
#: functions (similarity vs reward) while both used the vector leg. It could not answer
#: "does the vector leg help", which is the question the arm's name implied. The
#: ranking question is real and is now an explicit axis -- `rank_by` -- rather than
#: something smuggled into a mode.
Mode = Literal["ann_then_traverse", "ann_only", "task_scoped", "filter_only"]

#: Relations, named exactly as the registered native edge types traverse them.
RELATION_HIER = "hier"
RELATION_LINEAGE = "lineage"
RELATION_CHILD_OF = "child_of"
RELATION_CONTEXT = "context"


@dataclass(frozen=True)
class Predicate:
    """The relational filter `tjs_open` pushes down, as data rather than a lambda.

    Data because it must be rendered into the operator's `filter` string AND applied
    here, and a lambda cannot be rendered. One definition, two consumers.
    """

    kind: str | None = "node"
    require_valid: bool | None = None
    min_fitness: float | None = None
    max_fitness: float | None = None
    exclude_sessions: frozenset[str] = frozenset()
    exclude_tasks: frozenset[str] = frozenset()
    include_tasks: frozenset[str] | None = None
    failure_class: str | None = None
    exclude_nodes: frozenset[str] = frozenset()
    #: Nodes with no reward at all. 81 of 10,672 have `fitness IS NULL` (43 never had
    #: one, 38 had a non-finite one). Admitting them into a reward-ranked answer would
    #: mean inventing an order, so they are excluded whenever a reward bound applies.
    require_fitness: bool = False

    def accepts(self, node: dict[str, Any]) -> bool:
        if self.kind is not None and node.get("_kind", "node") != self.kind:
            return False
        if node["node_uid"] in self.exclude_nodes:
            return False
        if node["session_uid"] in self.exclude_sessions:
            return False
        if node["task_uid"] in self.exclude_tasks:
            return False
        if self.include_tasks is not None and node["task_uid"] not in self.include_tasks:
            return False
        if self.require_valid is not None and node["is_valid"] is not self.require_valid:
            return False
        if self.failure_class is not None and node["failure_class"] != self.failure_class:
            return False
        fitness = node.get("fitness")
        needs_fitness = (
            self.require_fitness or self.min_fitness is not None or self.max_fitness is not None
        )
        if needs_fitness and fitness is None:
            return False
        if self.min_fitness is not None and fitness < self.min_fitness:
            return False
        if self.max_fitness is not None and fitness > self.max_fitness:
            return False
        return True

    def to_sql(self) -> str:
        """The same predicate as a `tjs_open` filter string.

        Rendered against gem_eg_vertex's pushdown columns. Kept beside `accepts` on
        purpose: if the two drift, the parity gate fails loudly, which is the correct
        failure mode.
        """
        clauses: list[str] = []
        if self.kind is not None:
            clauses.append(f"kind = {_lit(self.kind)}")
        if self.require_valid is True:
            clauses.append("is_valid")
        elif self.require_valid is False:
            clauses.append("NOT is_valid")
        if self.min_fitness is not None:
            clauses.append(f"fitness >= {self.min_fitness!r}")
        if self.max_fitness is not None:
            clauses.append(f"fitness <= {self.max_fitness!r}")
        if self.require_fitness or self.min_fitness is not None or self.max_fitness is not None:
            clauses.append("fitness IS NOT NULL")
        if self.exclude_sessions:
            clauses.append(f"session_uid NOT IN ({_lits(self.exclude_sessions)})")
        if self.exclude_tasks:
            clauses.append(f"task_uid NOT IN ({_lits(self.exclude_tasks)})")
        if self.include_tasks is not None:
            clauses.append(f"task_uid IN ({_lits(self.include_tasks)})")
        if self.exclude_nodes:
            clauses.append(f"uid NOT IN ({_lits(self.exclude_nodes)})")
        return " AND ".join(clauses) if clauses else "true"


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _lits(values: Any) -> str:
    return ", ".join(_lit(v) for v in sorted(values))


@dataclass(frozen=True)
class QuerySpec:
    """One W1 query instance at one decision point."""

    query_id: str
    decision_point: str
    seed_uid: str
    relation: str
    hops: int
    predicate: Predicate
    k: int = 10
    mode: Mode = "ann_then_traverse"
    rank_by: RankBy = "similarity"
    #: W1.a only: the traversal starts from tasks found by ANN, not from `seed_uid`.
    ann_entry_kind: str | None = None
    ann_m_seeds: int = 8
    #: Provenance for the report; never used in ranking.
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class OracleResult:
    eligible: tuple[str, ...]
    topk: tuple[str, ...]
    distances: tuple[float, ...]
    entries: tuple[str, ...]
    traversed: int

    @property
    def eligible_set(self) -> frozenset[str]:
        return frozenset(self.eligible)


class Oracle:
    """Exact answers over an in-memory :class:`Corpus`."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    # -- entry points ----------------------------------------------------

    def entries(self, spec: QuerySpec) -> list[str]:
        """Where traversal starts.

        For a task-seeded query this is an exact ANN over the Task vectors — the paper's
        "vector-seeded" step. With 18 Tasks the exhaustive scan IS the exact answer, so
        the oracle and a brute-force ANN coincide here by construction; that is a fact
        about the corpus, not a shortcut.
        """
        if spec.ann_entry_kind is None:
            return [spec.seed_uid]
        query = self.corpus.vector(spec.seed_uid)
        if query is None:
            return []
        if spec.ann_entry_kind == "task":
            pool = sorted(self.corpus.tasks)
        elif spec.ann_entry_kind == "node":
            pool = sorted(self.corpus.nodes)
        else:
            pool = []
        # The SAME exclusions the engine applies at stage 1. Applying them only at
        # stage 2 would let the target's own vertices consume entry slots on one side
        # and not the other, and the gap would surface as unexplained lost recall
        # rather than as a mismatch anyone could locate.
        pred = spec.predicate
        pool = [
            uid
            for uid in pool
            if uid not in pred.exclude_nodes
            and self._entry_task(uid) not in pred.exclude_tasks
            and (pred.include_tasks is None or self._entry_task(uid) in pred.include_tasks)
            and self._entry_session(uid) not in pred.exclude_sessions
        ]
        distances = self.corpus.cosine_distance(query, pool)
        order = sorted(
            range(len(pool)),
            key=lambda i: (float(distances[i]), pool[i]),
        )
        keep = [pool[i] for i in order if np.isfinite(distances[i])]
        return keep[: spec.ann_m_seeds]

    def _entry_task(self, uid: str) -> str | None:
        if uid in self.corpus.tasks:
            return uid
        node = self.corpus.nodes.get(uid)
        return node["task_uid"] if node else None

    def _entry_session(self, uid: str) -> str | None:
        node = self.corpus.nodes.get(uid)
        return node["session_uid"] if node else None

    # -- the query -------------------------------------------------------

    def run(self, spec: QuerySpec) -> OracleResult:
        entries: list[str] = []
        reached: list[str]
        if spec.mode == "filter_only":
            # No vector, no graph: every node the predicate admits.
            reached = sorted(self.corpus.nodes)
        elif spec.mode == "task_scoped":
            # The honest no-vector entry rule: you always know your OWN task, so a
            # system without embeddings would start from it and walk. No ANN anywhere.
            entries = [spec.meta.get("task_uid")] if spec.meta.get("task_uid") else []
            reached = []
            seen_ts: set[str] = set()
            for entry in entries:
                for uid in self.corpus.traverse(entry, relation="hier", hops=2):
                    if uid not in seen_ts:
                        seen_ts.add(uid)
                        reached.append(uid)
        elif spec.mode == "ann_only":
            # The ANN result IS the answer; there is no second stage.
            entries = self.entries(spec)
            reached = list(entries)
        else:
            entries = self.entries(spec)
            reached = []
            seen: set[str] = set()
            for entry in entries:
                for uid in self.corpus.traverse(entry, relation=spec.relation, hops=spec.hops):
                    if uid not in seen:
                        seen.add(uid)
                        reached.append(uid)

        eligible = [
            uid
            for uid in reached
            if uid in self.corpus.nodes and spec.predicate.accepts(self.corpus.nodes[uid])
        ]
        eligible.sort()

        query = self.corpus.vector(spec.seed_uid)
        if query is None:
            distances = np.full(len(eligible), np.inf)
        else:
            distances = self.corpus.cosine_distance(query, eligible)

        if spec.mode in ("task_scoped", "filter_only"):
            # No similarity leg at all. Ordering is by reward then uid, which is the
            # best a store without vectors can do and is what these arms exist to show.
            ranked = sorted(
                ((float(distances[i]), eligible[i]) for i in range(len(eligible))),
                key=lambda pair: (
                    -(self.corpus.nodes[pair[1]]["fitness"] or float("-inf")),
                    pair[1],
                ),
            )
        elif spec.rank_by == "similarity":
            # A vectorless candidate is unrankable, and tjs_open drops it outright
            # (`if (vnull) continue;`). The oracle drops it too, or parity is a lie.
            ranked = [
                (float(distances[i]), eligible[i])
                for i in range(len(eligible))
                if np.isfinite(distances[i])
            ]
            ranked.sort(key=lambda pair: (pair[0], pair[1]))
        else:
            ranked = sorted(
                (
                    (float(distances[i]), eligible[i])
                    for i in range(len(eligible))
                ),
                key=lambda pair: (-(self.corpus.nodes[pair[1]]["fitness"] or float("-inf")), pair[1]),
            )

        topk = tuple(uid for _, uid in ranked[: spec.k])
        return OracleResult(
            eligible=tuple(eligible),
            topk=topk,
            distances=tuple(dist for dist, _ in ranked[: spec.k]),
            entries=tuple(entries),
            traversed=len(reached),
        )


def recall_at_k(observed: list[str], expected: tuple[str, ...]) -> float:
    """|observed ∩ expected| / |expected|, over the oracle's top-k.

    With no baseline system in play this measures the OPERATOR's own approximation
    loss — HNSW recall, graph-budget censoring, and early termination — never a
    cross-system comparison.
    """
    if not expected:
        return 1.0
    return len(set(observed) & set(expected)) / len(expected)
