"""Streaming physical plans for the GEM+ node-seeded Math workload.

All topology probes use native graph-store iterators.  The relational edge mirror is
read only by eligibility audits, never to answer reachability.  VFWD/AIVG consume a
versioned, fitness-ordered native adjacency synopsis; RREV consumes the fitness index
and probes the native ``eg_child_of`` inverse.  This is what permits exact early
termination under the approved seed-distance/fitness/uid order.
"""

from __future__ import annotations

import heapq
import itertools
import time
from enum import Enum
from typing import TYPE_CHECKING, Iterator

from bench.agent_memory.gem_eg.oracle import (
    SEED_DISTANCE_TIE_EPS,
    QuerySpec,
    logical_spec_hash,
)
from bench.agent_memory.gem_eg.store import vec_literal

if TYPE_CHECKING:
    from bench.agent_memory.gem_eg.query import EngineResult, Knobs, W1Engine

SYNOPSIS_EDGE_TYPE = "eg_lineage_h3_fit_v1"
SYNOPSIS_STAT = "lineage_h3_fitness"
SYNOPSIS_VERSION = "v1"


class PhysicalPlan(str, Enum):
    VFWD = "vfwd"
    RREV = "rrev"
    AIVG = "aivg"

    @classmethod
    def parse(cls, value: str | PhysicalPlan) -> PhysicalPlan:
        return value if isinstance(value, cls) else cls(str(value).lower())


class PlanIneligible(RuntimeError):
    """A correctness/streaming prerequisite is absent; never silently fall back."""


class PhysicalExecutor:
    def __init__(self, engine: W1Engine) -> None:
        self.engine = engine
        self.conn = engine.conn
        self.scope_id = engine.scope_id
        self._cursor_seq = itertools.count()

    def _cursor_name(self, prefix: str) -> str:
        return f"{prefix}_{next(self._cursor_seq)}"

    def _validate_spec(self, spec: QuerySpec) -> None:
        if spec.mode != "ann_then_traverse":
            raise ValueError("physical-plan experiment requires ann_then_traverse")
        if spec.ann_entry_kind != "node":
            raise ValueError("physical-plan experiment requires node ANN entries")
        if spec.relation != "lineage" or spec.hops != 3:
            raise ValueError("v1 synopsis proves only lineage reach at exactly h=3")
        if spec.rank_by != "seed_fitness":
            raise ValueError("physical plans require seed_fitness ranking")
        if spec.query_vector is None:
            raise ValueError(
                "node-artifact physical plans require an external query vector"
            )

    def _require_synopsis(self) -> None:
        try:
            row = self.conn.execute(
                "SELECT complete,max_hops,ordering,synopsis_edges"
                " FROM gem_eg_plan_stat"
                " WHERE scope_id=%s AND stat_name=%s AND stat_version=%s",
                (self.scope_id, SYNOPSIS_STAT, SYNOPSIS_VERSION),
            ).fetchone()
        except Exception as exc:  # table absent is also a closed gate
            raise PlanIneligible(
                "VFWD/AIVG need the native lineage_h3_fitness synopsis; run "
                "python -m tools.evotrace.build_plan_stats"
            ) from exc
        if not row or not row[0] or int(row[1]) != 3 or int(row[3]) < 0:
            raise PlanIneligible("lineage_h3_fitness synopsis is absent or incomplete")
        if str(row[2]) != "fitness DESC NULLS LAST, node_uid ASC":
            raise PlanIneligible(f"unexpected synopsis ordering: {row[2]!r}")

    def _require_reverse(self) -> None:
        lineage = self.engine.store.edge_type_id("eg_lineage")
        child = self.engine.store.edge_type_id("eg_child_of")
        row = self.conn.execute(
            "SELECT count(*) FILTER (WHERE e.edge_type=%s),"
            "       count(*) FILTER (WHERE e.edge_type=%s)"
            " FROM gem_eg_edge e JOIN gem_eg_vertex v ON v.id=e.src"
            " WHERE v.scope_id=%s AND e.edge_type IN (%s,%s)",
            (lineage, child, self.scope_id, lineage, child),
        ).fetchone()
        if not row or int(row[0]) != int(row[1]):
            raise PlanIneligible(
                "RREV inverse-lineage count mismatch; relational-edge fallback is forbidden"
            )
        missing = self.conn.execute(
            "SELECT 1 FROM gem_eg_edge f"
            " JOIN gem_eg_vertex v ON v.id=f.src AND v.scope_id=%s"
            " WHERE f.edge_type=%s AND NOT EXISTS ("
            "   SELECT 1 FROM gem_eg_edge r"
            "   WHERE r.src=f.dst AND r.dst=f.src AND r.edge_type=%s) LIMIT 1",
            (self.scope_id, lineage, child),
        ).fetchone()
        if missing:
            raise PlanIneligible(
                "RREV inverse-lineage audit found a missing native inverse"
            )

    def ensure_ready(self, plan: str | PhysicalPlan) -> None:
        parsed = PhysicalPlan.parse(plan)
        if parsed.value in self.engine._physical_ready:
            return
        if parsed in (PhysicalPlan.VFWD, PhysicalPlan.AIVG):
            self._require_synopsis()
        else:
            self._require_reverse()
        self.engine._physical_ready.add(parsed.value)

    def _entry_filter(self, spec: QuerySpec) -> str:
        return self.engine._entry_filter(spec)

    def _ann_sql(self, spec: QuerySpec) -> str:
        return (
            "SELECT id,uid,embedding <=> %s::vector AS distance"
            " FROM gem_eg_vertex WHERE scope_id=%s AND ("
            + self._entry_filter(spec)
            + ") AND embedding IS NOT NULL"
            " ORDER BY embedding <=> %s::vector,uid LIMIT %s"
        )

    def _ann_all(
        self, spec: QuerySpec, query_vec: str, result: EngineResult
    ) -> list[tuple[int, str, float]]:
        started = time.perf_counter()
        rows = self.conn.execute(
            self._ann_sql(spec),
            (query_vec, self.scope_id, query_vec, spec.ann_m_seeds),
        ).fetchall()
        result.ann_ms += (time.perf_counter() - started) * 1000.0
        result.ann_candidates += len(rows)
        return [(int(r[0]), str(r[1]), float(r[2])) for r in rows]

    def _ann_stream(
        self,
        spec: QuerySpec,
        query_vec: str,
        result: EngineResult,
        prefixes: tuple[int, ...],
    ) -> Iterator[tuple[int, str, float]]:
        with self.conn.cursor(name=self._cursor_name("aivg_ann")) as cursor:
            cursor.itersize = 1
            started = time.perf_counter()
            cursor.execute(
                self._ann_sql(spec),
                (query_vec, self.scope_id, query_vec, spec.ann_m_seeds),
            )
            result.ann_ms += (time.perf_counter() - started) * 1000.0
            while True:
                started = time.perf_counter()
                row = cursor.fetchone()
                result.ann_ms += (time.perf_counter() - started) * 1000.0
                if row is None:
                    return
                result.ann_candidates += 1
                if result.ann_candidates in prefixes:
                    result.ann_prefixes.append(result.ann_candidates)
                yield int(row[0]), str(row[1]), float(row[2])

    def _predicate_probe(
        self, spec: QuerySpec, vid: int, result: EngineResult
    ) -> tuple[str, float] | None:
        started = time.perf_counter()
        row = self.conn.execute(
            "SELECT uid,fitness FROM gem_eg_vertex"
            " WHERE id=%s AND scope_id=%s AND (" + spec.predicate.to_sql() + ")",
            (vid, self.scope_id),
        ).fetchone()
        result.predicate_ms += (time.perf_counter() - started) * 1000.0
        result.predicate_probes += 1
        if row is None:
            return None
        result.predicate_passed += 1
        return str(row[0]), float(row[1])

    def _forward_seed(
        self,
        spec: QuerySpec,
        seed_vid: int,
        result: EngineResult,
        seen: set[int],
        remaining_budget: list[int],
    ) -> Iterator[tuple[str, float]]:
        edge_type = self.engine.store.edge_type_id(SYNOPSIS_EDGE_TYPE)
        sql = "SELECT graph_store.gph_traverse_bounded(%s,1,%s,%s)"
        with self.conn.cursor(name=self._cursor_name("plan_fwd")) as cursor:
            cursor.itersize = 1
            started = time.perf_counter()
            cursor.execute(sql, (seed_vid, edge_type, max(remaining_budget[0], 0)))
            result.graph_ms += (time.perf_counter() - started) * 1000.0
            while remaining_budget[0] > 0:
                started = time.perf_counter()
                row = cursor.fetchone()
                result.graph_ms += (time.perf_counter() - started) * 1000.0
                if row is None:
                    return
                vid = int(row[0])
                remaining_budget[0] -= 1
                result.graph_examined += 1
                result.raw_reached += 1
                probe = self._predicate_probe(spec, vid, result)
                if probe is None:
                    continue
                started = time.perf_counter()
                if vid in seen:
                    result.dedup_hits += 1
                    result.dedup_rank_ms += (time.perf_counter() - started) * 1000.0
                    continue
                seen.add(vid)
                result.distinct_reached += 1
                result.dedup_rank_ms += (time.perf_counter() - started) * 1000.0
                yield probe
        if remaining_budget[0] <= 0:
            result.graph_censored = True

    def _eligible_rows(
        self, spec: QuerySpec, result: EngineResult
    ) -> Iterator[tuple[int, str]]:
        sql = (
            "SELECT id,uid FROM gem_eg_vertex WHERE scope_id=%s AND ("
            + spec.predicate.to_sql()
            + ") ORDER BY fitness DESC NULLS LAST,uid"
        )
        with self.conn.cursor(name=self._cursor_name("rrev_pred")) as cursor:
            cursor.itersize = 1
            started = time.perf_counter()
            cursor.execute(sql, (self.scope_id,))
            result.predicate_ms += (time.perf_counter() - started) * 1000.0
            while True:
                started = time.perf_counter()
                row = cursor.fetchone()
                result.predicate_ms += (time.perf_counter() - started) * 1000.0
                if row is None:
                    return
                result.predicate_probes += 1
                result.predicate_passed += 1
                yield int(row[0]), str(row[1])

    def _reaches_seed_reverse(
        self,
        candidate_vid: int,
        seed_vid: int,
        spec: QuerySpec,
        result: EngineResult,
        remaining_budget: list[int],
    ) -> bool:
        result.reverse_membership_probes += 1
        edge_type = self.engine.store.edge_type_id("eg_child_of")
        with self.conn.cursor(name=self._cursor_name("plan_rev")) as cursor:
            cursor.itersize = 1
            started = time.perf_counter()
            cursor.execute(
                "SELECT graph_store.gph_traverse_bounded(%s,%s,%s,%s)",
                (candidate_vid, spec.hops, edge_type, max(remaining_budget[0], 0)),
            )
            result.graph_ms += (time.perf_counter() - started) * 1000.0
            while remaining_budget[0] > 0:
                started = time.perf_counter()
                row = cursor.fetchone()
                result.graph_ms += (time.perf_counter() - started) * 1000.0
                if row is None:
                    return False
                remaining_budget[0] -= 1
                result.graph_examined += 1
                if int(row[0]) == seed_vid:
                    return True
        result.graph_censored = True
        return False

    def _run_forward_entries(
        self,
        spec: QuerySpec,
        entries: Iterator[tuple[int, str, float]],
        result: EngineResult,
        budget: int,
    ) -> None:
        seen: set[int] = set()
        remaining = [budget]
        active: list[Iterator[tuple[str, float]]] = []
        grouped = self._distance_groups(entries)
        try:
            for group in grouped:
                heap: list[tuple[float, str, int, Iterator[tuple[str, float]]]] = []
                for sequence, (seed_vid, seed_uid, _distance) in enumerate(group):
                    result.entries.append(seed_uid)
                    result.seeds_consumed += 1
                    stream = self._forward_seed(spec, seed_vid, result, seen, remaining)
                    active.append(stream)
                    try:
                        uid, fitness = next(stream)
                    except StopIteration:
                        continue
                    heapq.heappush(heap, (-fitness, uid, sequence, stream))
                # One head per tied seed is sufficient to prove the next global row;
                # state is O(m), never the full reachable set.
                while heap:
                    _negative_fitness, uid, sequence, stream = heapq.heappop(heap)
                    if result.first_candidate_ms is None:
                        result.first_candidate_ms = (
                            time.perf_counter() - self._started
                        ) * 1000.0
                        result.first_row_ms = result.first_candidate_ms
                    result.ids.append(uid)
                    if len(result.ids) >= spec.k:
                        result.termination_reason = "top_k"
                        return
                    try:
                        next_uid, next_fitness = next(stream)
                    except StopIteration:
                        continue
                    heapq.heappush(heap, (-next_fitness, next_uid, sequence, stream))
                if remaining[0] <= 0:
                    result.termination_reason = "graph_budget"
                    return
            result.termination_reason = "seed_exhausted"
        finally:
            # Explicit Close on every opened graph iterator and on the lazy ANN
            # iterator.  Correctness must not depend on CPython refcount timing.
            for stream in active:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            close = getattr(grouped, "close", None)
            if close is not None:
                close()
            close = getattr(entries, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _distance_groups(
        entries: Iterator[tuple[int, str, float]],
    ) -> Iterator[list[tuple[int, str, float]]]:
        group: list[tuple[int, str, float]] = []
        group_distance: float | None = None
        for entry in entries:
            if (
                group
                and group_distance is not None
                and entry[2] - group_distance > SEED_DISTANCE_TIE_EPS
            ):
                yield group
                group = []
                group_distance = None
            if group_distance is None:
                group_distance = entry[2]
            group.append(entry)
        if group:
            yield group

    def run(
        self, spec: QuerySpec, knobs: Knobs, plan: str | PhysicalPlan
    ) -> EngineResult:
        from bench.agent_memory.gem_eg.query import EngineResult

        parsed = PhysicalPlan.parse(plan)
        self._validate_spec(spec)
        self.ensure_ready(parsed)
        result = EngineResult(
            physical_plan=parsed.value,
            logical_spec_hash=logical_spec_hash(spec),
        )
        query_vec = vec_literal(spec.query_vector or ())
        self._started = time.perf_counter()
        with self.engine.settings(knobs):
            if knobs.ann_exact:
                self.conn.execute("SELECT set_config('enable_indexscan','off',true)")
                self.conn.execute("SELECT set_config('enable_bitmapscan','off',true)")
            if parsed is PhysicalPlan.VFWD:
                entries = self._ann_all(spec, query_vec, result)
                self._run_forward_entries(
                    spec, iter(entries), result, knobs.graph_work_budget
                )
            elif parsed is PhysicalPlan.AIVG:
                # One ANN portal is pulled lazily.  Prefixes 1->2->4 govern when more
                # seeds may be requested; lexicographic seed-distance order means a
                # seed must be exhausted before the next can affect the answer.
                # The ANN portal is lazy.  _distance_groups requests only the next
                # prefix plus one lookahead needed to prove a tie boundary, then graph
                # expansion starts; later prefixes are never pulled after top-k.
                self._run_forward_entries(
                    spec,
                    self._ann_stream(spec, query_vec, result, knobs.aivg_seed_prefixes),
                    result,
                    knobs.graph_work_budget,
                )
            else:
                entries = self._ann_all(spec, query_vec, result)
                seen: set[int] = set()
                remaining = [knobs.graph_work_budget]
                for group in self._distance_groups(iter(entries)):
                    seed_vids = {entry[0] for entry in group}
                    result.entries.extend(entry[1] for entry in group)
                    result.seeds_consumed += len(group)
                    for candidate_vid, uid in self._eligible_rows(spec, result):
                        if candidate_vid in seen:
                            continue
                        reaches = False
                        # One reverse iterator can encounter any seed in the tied
                        # group; membership does not require one full probe per seed.
                        result.reverse_membership_probes += 1
                        edge_type = self.engine.store.edge_type_id("eg_child_of")
                        with self.conn.cursor(
                            name=self._cursor_name("plan_rev_group")
                        ) as reverse:
                            reverse.itersize = 1
                            started = time.perf_counter()
                            reverse.execute(
                                "SELECT graph_store.gph_traverse_bounded(%s,%s,%s,%s)",
                                (
                                    candidate_vid,
                                    spec.hops,
                                    edge_type,
                                    max(remaining[0], 0),
                                ),
                            )
                            result.graph_ms += (time.perf_counter() - started) * 1000.0
                            while remaining[0] > 0:
                                started = time.perf_counter()
                                ancestor = reverse.fetchone()
                                result.graph_ms += (
                                    time.perf_counter() - started
                                ) * 1000.0
                                if ancestor is None:
                                    break
                                remaining[0] -= 1
                                result.graph_examined += 1
                                if int(ancestor[0]) in seed_vids:
                                    reaches = True
                                    break
                        if not reaches:
                            if remaining[0] <= 0:
                                break
                            continue
                        started = time.perf_counter()
                        seen.add(candidate_vid)
                        result.distinct_reached += 1
                        result.dedup_rank_ms += (time.perf_counter() - started) * 1000.0
                        if result.first_candidate_ms is None:
                            result.first_candidate_ms = (
                                time.perf_counter() - self._started
                            ) * 1000.0
                            result.first_row_ms = result.first_candidate_ms
                        result.ids.append(uid)
                        if len(result.ids) >= spec.k:
                            result.termination_reason = "top_k"
                            break
                    if len(result.ids) >= spec.k or remaining[0] <= 0:
                        break
                if remaining[0] <= 0:
                    result.graph_censored = True
                    result.termination_reason = "graph_budget"
                elif result.termination_reason is None:
                    result.termination_reason = "seed_exhausted"

        result.total_ms = (time.perf_counter() - self._started) * 1000.0
        result.stage1_ms = result.ann_ms
        result.stage2_ms = result.graph_ms + result.predicate_ms
        classified = (
            result.ann_ms + result.graph_ms + result.predicate_ms + result.dedup_rank_ms
        )
        result.executor_overhead_ms = max(0.0, result.total_ms - classified)
        result.calls = result.seeds_consumed + 1
        return result
