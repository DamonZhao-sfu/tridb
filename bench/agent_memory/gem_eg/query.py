"""W1 on the engine: vector-seeded graph retrieval through ``tjs_open``.

Every query here is one or two ``tjs_open`` calls and nothing else. That restriction is
the point: assembling the same answer from a pgvector ANN plus a recursive CTE in Python
would measure Python, not the fused operator, and the whole claim is about what the
operator does in one pass.

SHAPE
-----
All three W1 queries share a two-stage form, because the seed always lives in the
*target* session and the target session is exactly what must be excluded from the
results. So the vector cannot be used to rank directly against a subgraph we already
have — it first has to find an entry point in the historical corpus:

    stage 1  seedless tjs_open  ->  ANN entry points (Task or Node), target excluded
    stage 2  filter-first tjs_open from each entry -> bounded typed traversal,
             relational predicate pushed down, top-k, early termination

    W1.a   entry = Task     traverse eg_hier    (Task -> Session -> Node), hops=2
    W1.b   entry = Node     traverse eg_lineage (what happened after a similar state)
    W1.d   entry = Node     traverse eg_child_of (how others reached a state like this)

STREAMING
---------
``tjs_open`` is called in a target-list position (``SELECT tjs_open(...)``), never as a
``FROM``-clause FunctionScan: the extension's own contract is that early termination
under LIMIT is lost in the latter. Results are pulled through a server-side cursor one
row at a time so ``first_row_ms`` is a real measurement of when the first answer became
available, not of when the last one did.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from bench.agent_memory.gem_eg.oracle import QuerySpec
from bench.agent_memory.gem_eg.store import EgStore, vec_literal

#: Query-shape -> the native edge type its stage-2 traversal walks.
RELATION_EDGE_TYPE = {
    "hier": "eg_hier",
    "lineage": "eg_lineage",
    "child_of": "eg_child_of",
    "context": "eg_context",
}


@dataclass(frozen=True)
class Knobs:
    """Engine EXECUTION settings only — never anything that defines the answer.

    The split is structural, not stylistic. ``QuerySpec`` owns what the right answer is
    (k, hops, m_seeds, predicate, relation); ``Knobs`` owns how hard the engine tries to
    find it. The oracle reads only the spec, so a knob can never make the two sides
    compute different questions.

    That separation exists because the first version did NOT have it: ``hops`` lived in
    both, the runner passed 2 while the W1.d spec said 4, and the resulting mismatch
    looked exactly like a plausible operator recall loss.

    These three, and only these three, are why the engine may disagree with the oracle:
      * ``ef_search``         -- HNSW approximation
      * ``graph_work_budget`` -- traversal censoring
      * ``term_cond``         -- early termination
    """

    term_cond: int = 0
    ef_search: int = 100
    graph_work_budget: int = 65536
    graph_scoring: str = "ppr"
    iterative_scan: str = "relaxed_order"


@dataclass
class EngineResult:
    ids: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    #: When the first row of the FINAL (globally ranked) answer became available.
    first_row_ms: float | None = None
    #: When the operator produced its first candidate — tjs_open's own streaming
    #: behaviour, before any cross-entry merge. Kept apart so the merge cannot hide it.
    first_candidate_ms: float | None = None
    total_ms: float = 0.0
    merge_ms: float = 0.0
    candidates_examined: int = 0
    graph_examined: int = 0
    graph_censored: bool = False
    termination_reason: str | None = None
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    calls: int = 0


class W1Engine:
    """Runs the W1 query family against a loaded Experience Graph."""

    TABLE = "gem_eg_vertex"
    ID_COL = "id"

    def __init__(self, store: EgStore, scope_id: str) -> None:
        self.store = store
        self.conn = store.conn
        self.scope_id = scope_id
        self._vid: dict[str, int] = {}
        self._uid: dict[int, str] = {}
        self._vectors: dict[str, str] = {}

    # -- identity -------------------------------------------------------

    def load_identity(self) -> None:
        """uid <-> vid, and the query vectors, fetched once for the whole sweep.

        Per-query lookups would put a round trip inside the timed region and make
        `first_row_ms` a measurement of our own bookkeeping.
        """
        rows = self.conn.execute(
            "SELECT id, uid, embedding::text FROM gem_eg_vertex WHERE scope_id = %s",
            (self.scope_id,),
        ).fetchall()
        for vid, uid, embedding in rows:
            self._vid[uid] = int(vid)
            self._uid[int(vid)] = uid
            if embedding is not None:
                self._vectors[uid] = embedding

    def vid(self, uid: str) -> int | None:
        return self._vid.get(uid)

    def uid(self, vid: int) -> str | None:
        return self._uid.get(vid)

    # -- session settings ------------------------------------------------

    @contextmanager
    def settings(self, knobs: Knobs) -> Iterator[None]:
        """Apply the operator knobs for one measured query, then restore them.

        LOCAL so a failed query cannot leak a setting into the next one and quietly
        move a whole sweep's operating point.
        """
        self.conn.execute("BEGIN")
        # set_config(..., is_local => true) rather than `SET LOCAL`: the latter takes no
        # bind parameters, so a swept value would have to be interpolated into SQL.
        for name, value in (
            ("hnsw.iterative_scan", knobs.iterative_scan),
            ("hnsw.ef_search", knobs.ef_search),
            ("tjs.graph_work_budget", knobs.graph_work_budget),
            ("tjs.graph_scoring", knobs.graph_scoring),
        ):
            self.conn.execute("SELECT set_config(%s, %s, true)", (name, str(value)))
        try:
            yield
        finally:
            self.conn.execute("COMMIT")

    # -- the two stages ---------------------------------------------------

    def _stream(self, sql: str, params: tuple[Any, ...], limit: int) -> tuple[list[int], float | None]:
        """Pull a tjs_open SRF through a server-side cursor, timing the FIRST row.

        A plain execute() would materialise the whole result before returning, which is
        precisely the behaviour this query family exists to avoid measuring.
        """
        out: list[int] = []
        first: float | None = None
        started = time.perf_counter()
        with self.conn.cursor(name="w1_stream") as cursor:
            cursor.itersize = 1
            cursor.execute(sql, params)
            for row in cursor:
                if first is None:
                    first = (time.perf_counter() - started) * 1000.0
                if row[0] is not None:
                    out.append(int(row[0]))
                if len(out) >= limit:
                    break
        return out, first

    def stage1_entries(self, spec: QuerySpec, knobs: Knobs, query_vec: str) -> list[int]:
        """Seedless ANN: the vector-seeded half of "vector-seeded graph retrieval"."""
        entry_filter = self._entry_filter(spec)
        sql = (
            f"SELECT tjs_open('{self.TABLE}'::regclass, %s, %s, 0, 0, '{self.ID_COL}',"
            " %s, %s::vector, NULL, 0)"
        )
        # Over-fetch, then truncate deterministically. Entry selection is ill-posed at
        # a tie: 939 of 10,672 nodes duplicate another node's code, so identical vectors
        # are common and "the m-th nearest entry" can be several different vertices.
        # Picking arbitrarily makes the query non-reproducible — the engine and the
        # oracle then traverse different subtrees and produce different, equally
        # defensible answers, which no parity gate can distinguish from a defect.
        # Ordering by (distance, uid) makes the choice total and repeatable.
        over = max(spec.ann_m_seeds * 4, spec.ann_m_seeds + 16)
        rows = self.conn.execute(
            sql, (over, knobs.term_cond, entry_filter, query_vec)
        ).fetchall()
        candidates = [int(r[0]) for r in rows if r[0] is not None]
        if len(candidates) <= 1:
            return candidates
        ordered = self.conn.execute(
            "SELECT id FROM gem_eg_vertex"
            " WHERE id = ANY(%s) AND embedding IS NOT NULL"
            " ORDER BY embedding <=> %s::vector, uid"
            " LIMIT %s",
            (candidates, query_vec, spec.ann_m_seeds),
        ).fetchall()
        return [int(r[0]) for r in ordered]

    def stage2_expand(
        self, spec: QuerySpec, knobs: Knobs, query_vec: str, entry_vid: int
    ) -> tuple[list[int], float | None]:
        """Filter-first: bounded typed traversal + pushed-down predicate + top-k."""
        edge_type = self.store.edge_type_id(RELATION_EDGE_TYPE[spec.relation])
        sql = (
            f"SELECT tjs_open('{self.TABLE}'::regclass, %s, %s, 0, %s, '{self.ID_COL}',"
            " %s, %s::vector, %s, %s)"
        )
        return self._stream(
            sql,
            (
                spec.k,
                knobs.term_cond,
                spec.hops,
                spec.predicate.to_sql(),
                query_vec,
                entry_vid,
                edge_type,
            ),
            limit=spec.k,
        )

    # -- the query --------------------------------------------------------

    def run(self, spec: QuerySpec, knobs: Knobs) -> EngineResult:
        result = EngineResult()
        query_vec = self._vectors.get(spec.seed_uid)
        if query_vec is None:
            # No seed vector means no query, not an empty answer. Distinguishing the
            # two matters: 6,536 prompts and 121 sessions legitimately have no vector.
            result.termination_reason = "no_seed_vector"
            return result

        started = time.perf_counter()
        with self.settings(knobs):
            if spec.mode == "filter_only":
                entry_vids = []
            elif spec.mode == "task_scoped":
                # Entry by task membership, not by similarity. No ANN call is made.
                task_vid = self.vid(spec.meta.get("task_uid", ""))
                entry_vids = [task_vid] if task_vid is not None else []
            elif spec.ann_entry_kind is not None:
                t0 = time.perf_counter()
                entry_vids = self.stage1_entries(spec, knobs, query_vec)
                result.stage1_ms = (time.perf_counter() - t0) * 1000.0
                result.calls += 1
            else:
                vid = self.vid(spec.seed_uid)
                entry_vids = [vid] if vid is not None else []

            result.entries = [u for u in (self.uid(v) for v in entry_vids) if u]

            if spec.mode in ("ann_only", "filter_only", "task_scoped"):
                result.ids = self._run_ablation(spec, knobs, query_vec, entry_vids)
                result.total_ms = (time.perf_counter() - started) * 1000.0
                result.first_row_ms = result.first_row_ms or result.total_ms
                self._collect_counters(result)
                return result

            seen: set[int] = set()
            pool: list[int] = []
            t1 = time.perf_counter()
            for entry_vid in entry_vids:
                ids, first = self.stage2_expand(spec, knobs, query_vec, entry_vid)
                result.calls += 1
                if first is not None and result.first_candidate_ms is None:
                    # When the OPERATOR produced its first candidate. This is the
                    # streaming property of tjs_open itself, kept separate from the
                    # first row of the final answer below.
                    result.first_candidate_ms = result.stage1_ms + first
                for vid in ids:
                    if vid not in seen:
                        seen.add(vid)
                        pool.append(vid)
            result.stage2_ms = (time.perf_counter() - t1) * 1000.0
            self._collect_counters(result)

            # Merge across entries. EXACT, not a heuristic: hierarchy partitions nodes
            # by Task, so the per-entry candidate sets are disjoint and the union of
            # their top-k necessarily contains the global top-k. With one entry there
            # is nothing to merge and the operator's own order is already final.
            t2 = time.perf_counter()
            if spec.rank_by == "reward":
                # G4 made executable. tjs_open can only RANK by vector distance, so the
                # reward ordering the paper's reuse query actually wants has to be an
                # outer sort over the operator's BOUNDED candidate stream. That bound is
                # the whole point: a high-reward node that sat far away in vector space
                # never enters the pool, so it cannot be sorted back in. The gap between
                # this arm and the oracle is exactly the cost of that.
                rows = self.conn.execute(
                    "SELECT id FROM gem_eg_vertex"
                    " WHERE id = ANY(%s)"
                    " ORDER BY fitness DESC NULLS LAST, uid"
                    " LIMIT %s",
                    (pool, spec.k),
                ).fetchall()
                ordered = [int(r[0]) for r in rows]
                result.first_row_ms = result.stage1_ms + result.stage2_ms + (
                    time.perf_counter() - t2
                ) * 1000.0
            elif len(entry_vids) <= 1:
                ordered = pool[: spec.k]
                if result.first_row_ms is None:
                    result.first_row_ms = result.first_candidate_ms
            else:
                rows = self.conn.execute(
                    "SELECT id FROM gem_eg_vertex"
                    " WHERE id = ANY(%s) AND embedding IS NOT NULL"
                    " ORDER BY embedding <=> %s::vector, uid"
                    " LIMIT %s",
                    (pool, query_vec, spec.k),
                ).fetchall()
                ordered = [int(r[0]) for r in rows]
                # A globally ranked first row cannot exist before the merge, so this is
                # the honest number. `first_candidate_ms` still shows when streaming began.
                result.first_row_ms = result.stage1_ms + result.stage2_ms + (
                    time.perf_counter() - t2
                ) * 1000.0
            result.merge_ms = (time.perf_counter() - t2) * 1000.0
            result.ids = [u for u in (self.uid(v) for v in ordered) if u]

        result.total_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _run_ablation(
        self, spec: QuerySpec, knobs: Knobs, query_vec: str, entry_vids: list[int]
    ) -> list[str]:
        """The single-modality arms, each expressed with the leg it is allowed.

        These are NOT tjs_open calls — that is the point. `ann_only` is what a vector
        store returns, `traverse_only` what a graph store returns, `filter_only` what a
        relational scan returns. Running them through the fused operator would measure
        the fused operator four times.
        """
        pred = spec.predicate.to_sql()
        if spec.mode == "ann_only":
            return [u for u in (self.uid(v) for v in entry_vids[: spec.k]) if u]
        if spec.mode == "filter_only":
            rows = self.conn.execute(
                "SELECT uid FROM gem_eg_vertex WHERE scope_id = %s AND (" + pred + ")"
                " ORDER BY fitness DESC NULLS LAST, uid LIMIT %s",
                (self.scope_id, spec.k),
            ).fetchall()
            return [r[0] for r in rows]
        # task_scoped: native bounded BFS from the Task vertex, then the predicate,
        # ranked by reward. No vector is consulted at any point.
        edge_type = self.store.edge_type_id("eg_hier")
        reached: list[int] = []
        for entry_vid in entry_vids:
            rows = self.conn.execute(
                "SELECT graph_store.gph_traverse_bounded(%s, %s, %s, %s)",
                (entry_vid, 2, edge_type, knobs.graph_work_budget),
            ).fetchall()
            reached.extend(int(r[0]) for r in rows if r[0] is not None)
        if not reached:
            return []
        rows = self.conn.execute(
            "SELECT uid FROM gem_eg_vertex WHERE id = ANY(%s) AND (" + pred + ")"
            " ORDER BY fitness DESC NULLS LAST, uid LIMIT %s",
            (reached, spec.k),
        ).fetchall()
        return [r[0] for r in rows]

    def _collect_counters(self, result: EngineResult) -> None:
        """The operator's own accounting of how much work it did.

        Read inside the same transaction as the query: these are per-backend counters
        reset at each Open, so reading them later would report the wrong call.
        """
        row = self.conn.execute(
            "SELECT tjs_open_candidates_examined(), tjs_open_graph_examined(),"
            " tjs_open_graph_censored(), tjs_open_termination_reason()"
        ).fetchone()
        if row is not None:
            result.candidates_examined = int(row[0] or 0)
            result.graph_examined = int(row[1] or 0)
            result.graph_censored = bool(row[2])
            result.termination_reason = row[3]

    # -- filters ----------------------------------------------------------

    def _entry_filter(self, spec: QuerySpec) -> str:
        """The stage-1 predicate: which vertices may serve as an ANN entry point.

        The target session is excluded HERE as well as in stage 2. Excluding it only at
        stage 2 would let it consume entry slots and silently shrink the candidate pool
        — a leak that shows up as lost recall rather than as wrong answers.
        """
        clauses = [f"kind = '{spec.ann_entry_kind}'"]
        if spec.predicate.exclude_sessions:
            joined = ", ".join(
                "'" + s.replace("'", "''") + "'" for s in sorted(spec.predicate.exclude_sessions)
            )
            clauses.append(f"(session_uid IS NULL OR session_uid NOT IN ({joined}))")
        if spec.predicate.exclude_tasks:
            joined = ", ".join(
                "'" + t.replace("'", "''") + "'" for t in sorted(spec.predicate.exclude_tasks)
            )
            clauses.append(f"(task_uid IS NULL OR task_uid NOT IN ({joined}))")
        return " AND ".join(clauses)


def vector_literal(values: Any) -> str:
    return vec_literal(values)
