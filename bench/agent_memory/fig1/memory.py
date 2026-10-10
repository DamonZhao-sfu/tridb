"""Figure 1 memory over EvoTrace: tri-modal store + semantic operators under one plan.

Store (one PG database, one WAL): ``gem_eg_vertex`` rows (relational columns + HNSW
vector), lineage edges in the native graph AM, the LLM cache ``gem_sem_cache`` and the
pending set Q ``gem_pending``.

Ingest_pi(x) for one EvoTrace node x:
  core (always eager, deterministic): vertex row + embedding + lineage edges (AM + mirror).
  semantic derivations, eager or deferred per plan pi:
    * ``lesson``   = sem_extract(outcome(x))                      -> vsgt_payload.lesson
    * ``fixed_by`` = sem_match(x, c) over candidates C(x) for a failed x -> eg_fixed_by edges
      C(x) = exact top-k by cosine among same-task, other-session, valid, fitness>0 nodes
      with id < vid(x). Pinning C to ``id < vid(x)`` (the commit snapshot, plan D2) and
      using an exact scan makes C independent of *when* the derivation runs, so eager and
      deferred plans materialize the same state given the cache.

Retrieve_rho(q) for the agent about to improve parent p:
  Seed+Expand+theta_R in the native ``tjs_vsgt`` CustomScan (ANN seeds -> lineage
  descendants, scope/task/session/validity/fitness pushed down), theta_sem over the stream,
  fix hints via one typed graph traversal from p over eg_fixed_by, Assemble under a budget.
  Plans: 'trimodal' (no theta_sem), 'naive' (theta_sem before theta_R, materialized),
  'planned_mat' (planner order, materialized), 'planned' (planner order, streaming with
  early close). Consistency: 'strong' resolves pending derivations of touched items first;
  'stale' reads what is there and reports staleness.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
from psycopg.types.json import Jsonb

from bench.agent_memory.fig1.corpus import Node
from bench.agent_memory.gem_eg.store import EgStore
from bench.agent_memory.semantic.cache import ensure_schema
from bench.agent_memory.semantic.ops import SemOps
from bench.agent_memory.semantic.pending import PendingQueue
from bench.agent_memory.semantic.planner import PredicateLeg, SemPlanner
from bench.agent_memory.semantic.stream import sem_filter_stream

FIXED_BY = "eg_fixed_by"
LESSON_INSTRUCTION = (
    "In one sentence, state what this attempt at the task changed and what its outcome "
    "teaches a later attempt at the same task."
)
FIX_RELATION = (
    "RIGHT is a later or parallel attempt at the same task that avoids the failure "
    "described in LEFT and could serve as a fix example for it."
)
HINT_PREDICATE = (
    "The CANDIDATE past attempt contains a concrete, transferable idea that would help an "
    "agent improve the CURRENT program on the same task."
)
PLANS = ("trimodal", "naive", "planned_mat", "planned")


def vec_text(v: Sequence[float] | np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


@dataclass(frozen=True)
class IngestPlan:
    mode: str = "eager"  # 'eager' | 'deferred'
    trigger: str = "batch"  # deferred only: 'batch' | 'on_read' | 'background'
    batch_every: int = 64
    fix_candidates: int = 5

    def __post_init__(self) -> None:
        if self.mode not in ("eager", "deferred"):
            raise ValueError(f"bad ingest mode {self.mode!r}")
        if self.trigger not in ("batch", "on_read", "background"):
            raise ValueError(f"bad trigger {self.trigger!r}")


@dataclass(frozen=True)
class RetrievePlan:
    plan: str = "planned"
    consistency: str = "strong"  # 'strong' | 'stale'
    k: int = 5
    k_max: int = 48  # bound on the materialized prefix for naive / planned_mat
    m: int = 3
    d: int = 2
    budget_tokens: int = 800
    sel_sem_prior: float = 0.4
    c_llm: float = 0.5  # seconds per semantic evaluation (calibrated from telemetry)
    c_db: float = 2e-6  # seconds per pushed-down relational predicate evaluation

    def __post_init__(self) -> None:
        if self.plan not in PLANS:
            raise ValueError(f"bad retrieve plan {self.plan!r}")
        if self.consistency not in ("strong", "stale"):
            raise ValueError(f"bad consistency {self.consistency!r}")


@dataclass
class Receipt:
    seconds: float = 0.0
    sem: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Fig1Memory:
    def __init__(
        self,
        conn: Any,
        scope: str,
        ops: SemOps,
        *,
        planner: SemPlanner | None = None,
        dim: int = 1024,
    ) -> None:
        self.conn = conn
        self.scope = scope
        self.ops = ops
        self.planner = planner or SemPlanner(conn)
        self.store = EgStore(conn, dim=dim)
        self.pending = PendingQueue(conn)
        self.vid: dict[str, int] = {}
        self.nodes: dict[str, Node] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.fixed_by_type: int | None = None
        self.ingested = 0

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        self.store.init_schema()
        self.conn.execute(
            "ALTER TABLE gem_eg_vertex ADD COLUMN IF NOT EXISTS vsgt_payload jsonb"
        )
        ensure_schema(self.conn)
        row = self.conn.execute(
            "SELECT graph_store.register_edge_type(%s)", (FIXED_BY,)
        ).fetchone()
        self.fixed_by_type = int(row[0])
        self.conn.commit()

    # ----------------------------------------------------------- derivations

    def _fix_candidates(
        self, node: Node, vid: int, k: int
    ) -> list[tuple[int, dict[str, Any]]]:
        """C(x): exact cosine top-k, pinned to the commit snapshot id < vid(x)."""
        rows = self.conn.execute(
            "SELECT id, vsgt_payload->'outcome' FROM gem_eg_vertex"
            " WHERE scope_id = %s AND kind = 'node' AND task_uid = %s"
            "   AND session_uid <> %s AND id < %s AND is_valid AND fitness > 0"
            " ORDER BY (embedding <=> %s::vector) + 0, id LIMIT %s",
            (
                self.scope,
                node.task_uid,
                node.session_uid,
                vid,
                vec_text(self.vectors[node.uid]),
                k,
            ),
        ).fetchall()
        return [(int(r[0]), dict(r[1] or {})) for r in rows]

    def _derive(
        self,
        items: Sequence[tuple[str, str]],
        fix_k: int,
        snap: dict[str, int] | None = None,
    ) -> tuple[dict[str, str], list[tuple[int, int]]]:
        """Run the semantic derivations for (derivation, uid) items as large batches.

        ``snap`` maps uid -> the vid the item has / will get (its commit snapshot); it
        defaults to the stored vid. Returns lessons {uid: text} and fixed_by edges
        [(src_vid, dst_vid)].
        """
        lesson_uids = [u for d, u in items if d == "lesson"]
        fix_uids = [u for d, u in items if d == "fixed_by"]
        lessons: dict[str, str] = {}
        if lesson_uids:
            texts = self.ops.sem_extract(
                LESSON_INSTRUCTION, [self.nodes[u].outcome() for u in lesson_uids]
            )
            lessons = dict(zip(lesson_uids, texts, strict=True))
        edges: list[tuple[int, int]] = []
        if fix_uids:
            pairs: list[tuple[Any, Any]] = []
            owners: list[tuple[int, int]] = []
            for u in fix_uids:
                vid = snap[u] if snap is not None and u in snap else self.vid[u]
                for cand_vid, cand_outcome in self._fix_candidates(
                    self.nodes[u], vid, fix_k
                ):
                    pairs.append((self.nodes[u].outcome(), cand_outcome))
                    owners.append((vid, cand_vid))
            if pairs:
                verdicts = self.ops.sem_match(FIX_RELATION, pairs)
                edges = [o for o, ok in zip(owners, verdicts, strict=True) if ok]
        return lessons, edges

    def _apply(self, lessons: dict[str, str], edges: list[tuple[int, int]]) -> None:
        for uid, text in lessons.items():
            self.conn.execute(
                "UPDATE gem_eg_vertex SET vsgt_payload = vsgt_payload || jsonb_build_object("
                "'lesson', %s::text) WHERE id = %s",
                (text, self.vid[uid]),
            )
        for src, dst in edges:
            inserted = self.conn.execute(
                "INSERT INTO gem_eg_edge (src, dst, edge_type, relation, provenance)"
                " VALUES (%s,%s,%s,'fixed_by','sem_match:v1')"
                " ON CONFLICT (src, dst, edge_type) DO NOTHING RETURNING 1",
                (src, dst, self.fixed_by_type),
            ).fetchone()
            if inserted is not None:
                self.conn.execute(
                    "SELECT graph_store.gph_insert_edge(%s, %s, %s)",
                    (src, dst, self.fixed_by_type),
                )

    def derivations_of(self, node: Node) -> list[tuple[str, str]]:
        out = [("lesson", node.uid)]
        if node.failed:
            out.append(("fixed_by", node.uid))
        return out

    # ----------------------------------------------------------------- ingest

    def ingest(self, node: Node, vector: np.ndarray, plan: IngestPlan) -> Receipt:
        started = time.perf_counter()
        before = self.ops.meter.snapshot()
        self.nodes[node.uid] = node
        self.vectors[node.uid] = vector
        derivs = self.derivations_of(node)

        # Eager LLM work runs BEFORE the writer lock is taken (plan R11). Under the
        # single-writer contract the next vid is known in advance, so the pinned
        # candidate read (id < vid(x)) is the same one a deferred plan will issue.
        eager_lessons: dict[str, str] = {}
        eager_edges: list[tuple[int, int]] = []
        if plan.mode == "eager":
            predicted = int(
                self.conn.execute("SELECT graph_store.gph_allocated_vids()").fetchone()[
                    0
                ]
            )
            eager_lessons, eager_edges = self._derive(
                derivs, plan.fix_candidates, snap={node.uid: predicted}
            )
            self.conn.commit()  # end the read-only transaction before the write one
        vid = self.store.allocate_vertices(1)[0]
        if plan.mode == "eager" and vid != predicted:
            raise RuntimeError(f"vid drift: predicted {predicted}, allocated {vid}")
        self.vid[node.uid] = vid
        payload = {
            "artifact_uid": node.artifact_uid,
            "language": node.language,
            "error_signature": node.error_signature,
            "outcome": node.outcome(),
        }
        self.conn.execute(
            "INSERT INTO gem_eg_vertex (id, scope_id, kind, uid, label, embedding,"
            " embedding_source, embedding_model, task_uid, session_uid, language, status,"
            " is_valid, fitness, iteration, vsgt_payload)"
            " VALUES (%s,%s,'node',%s,%s,%s::vector,'node_artifact',"
            " 'Qwen/Qwen3-Embedding-0.6B',%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                vid,
                self.scope,
                node.uid,
                node.uid[-40:],
                vec_text(vector),
                node.task_uid,
                node.session_uid,
                node.language,
                node.status,
                node.is_valid,
                node.fitness,
                node.iteration,
                Jsonb(payload),
            ),
        )
        if node.parent_uid is not None and node.parent_uid in self.vid:
            pvid = self.vid[node.parent_uid]
            self.store.link(pvid, vid, relation="has_child", edge_type="eg_lineage")
            self.store.link(
                vid,
                pvid,
                relation="child_of",
                edge_type="eg_child_of",
                derived_inverse=True,
            )
        enqueued: list[int] = []
        if plan.mode == "eager":
            self._apply(eager_lessons, eager_edges)
        else:
            for derivation, uid in derivs:
                pid = self.pending.enqueue(
                    self.scope,
                    derivation,
                    uid,
                    trigger=plan.trigger,
                    snapshot_seq=vid,
                    payload={"fix_k": plan.fix_candidates},
                )
                if pid is not None:
                    enqueued.append(pid)
        self.conn.commit()
        self.ingested += 1

        extra: dict[str, Any] = {"vid": vid, "enqueued": len(enqueued)}
        if plan.mode == "deferred" and plan.trigger == "batch":
            if self.ingested % plan.batch_every == 0:
                extra["batch_resolved"] = self.materialize(
                    by="batch", fix_k=plan.fix_candidates
                )
        return Receipt(
            time.perf_counter() - started, self.ops.meter.diff(before), extra
        )

    # ------------------------------------------------------------- materialize

    def materialize(
        self, *, targets: Sequence[str] | None = None, by: str = "batch", fix_k: int = 5
    ) -> int:
        """Resolve pending derivations (all, or only those of ``targets``) in one txn."""
        items = self.pending.open_items(self.scope, targets=targets)
        if not items:
            if targets is not None:
                return 0
            self.conn.commit()
            return 0
        lessons, edges = self._derive(
            [(i.derivation, i.target_ref) for i in items], fix_k
        )
        self._apply(lessons, edges)
        self.pending.resolve([i.id for i in items], by=by)
        self.conn.commit()
        return len(items)

    # --------------------------------------------------------------- retrieve

    def _vsgt_request(
        self, parent: Node, plan: RetrievePlan, *, push_r: bool, k: int
    ) -> dict:
        req: dict[str, Any] = {
            "request_id": "f1" + uuid4().hex[:24],
            "strategy": "seed_cursor_f",
            "table": "gem_eg_vertex",
            "scope": self.scope,
            "task": parent.task_uid,
            "m": plan.m,
            "k": k,
            "d": plan.d,
            "payload_column": "vsgt_payload",
            "exclude_session": parent.session_uid,
        }
        if push_r:
            req["output_valid"] = True
            req["output_min_fitness"] = self._min_fitness(parent)
        else:
            req["output_valid"] = False
            req["output_finite"] = False
        return req

    @staticmethod
    def _min_fitness(parent: Node) -> float:
        # Improve on the parent: valid and strictly better than a failed parent's 0,
        # otherwise at least as good as the parent.
        if parent.failed or parent.fitness is None:
            return 1e-9
        return float(parent.fitness)

    def _theta_r(self, parent: Node, row: dict[str, Any]) -> bool:
        out = (row.get("payload") or {}).get("outcome") or {}
        fit = out.get("fitness")
        return (
            bool(out.get("valid"))
            and fit is not None
            and fit >= self._min_fitness(parent)
        )

    def _current_item(self, parent: Node) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT vsgt_payload->>'lesson' FROM gem_eg_vertex WHERE id = %s",
            (self.vid[parent.uid],),
        ).fetchone()
        item: dict[str, Any] = {"outcome": parent.outcome()}
        if row and row[0]:
            item["lesson"] = row[0]
        return item

    @staticmethod
    def _candidate_item(row: dict[str, Any]) -> dict[str, Any]:
        payload = row.get("payload") or {}
        if payload.get("lesson"):
            return {"lesson": payload["lesson"]}
        return {"outcome": payload.get("outcome")}

    def _row_uid(self, row: dict[str, Any]) -> str:
        return str(row.get("uid") or row.get("id"))

    def _refresh_payloads(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [int(r["id"]) for r in rows]
        fresh = dict(
            self.conn.execute(
                "SELECT id, vsgt_payload FROM gem_eg_vertex WHERE id = ANY(%s)", (ids,)
            ).fetchall()
        )
        return [
            {**r, "payload": fresh.get(int(r["id"]), r.get("payload"))} for r in rows
        ]

    def _strong_prepare(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        uids = [self._row_uid(r) for r in rows]
        if self.pending.count_open(self.scope, targets=uids) == 0:
            return rows
        self._resolved += self._resolve_targets(uids)
        return self._refresh_payloads(rows)

    def _resolve_targets(self, uids: Sequence[str]) -> int:
        items = self.pending.open_items(self.scope, targets=list(uids))
        if not items:
            return 0
        lessons, edges = self._derive(
            [(i.derivation, i.target_ref) for i in items],
            int(items[0].payload.get("fix_k", 5)),
        )
        self._apply(lessons, edges)
        return self.pending.resolve([i.id for i in items], by="on_read")

    def _fix_hints(self, parent: Node) -> list[dict[str, Any]]:
        if not parent.failed:
            return []
        rows = self.conn.execute(
            "SELECT v.id, v.uid, v.vsgt_payload FROM"
            " (SELECT (graph_store.gph_traverse_typed(%s, %s, 0, -1)).dst AS dst) t"
            " JOIN gem_eg_vertex v ON v.id = t.dst ORDER BY v.id",
            (self.vid[parent.uid], self.fixed_by_type),
        ).fetchall()
        return [{"id": int(r[0]), "uid": r[1], "payload": r[2]} for r in rows]

    def retrieve(
        self, parent: Node, plan: RetrievePlan, *, ops: SemOps | None = None
    ) -> tuple[list[dict], Receipt]:
        """``ops`` evaluates theta_sem (default: the derivation ops); on-read resolution
        of pending derivations always runs through the derivation ops ``self.ops``."""
        sem = ops or self.ops
        started = time.perf_counter()
        before = sem.meter.snapshot()
        self._resolved = 0
        extra: dict[str, Any] = {"plan": plan.plan, "consistency": plan.consistency}
        self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")

        if plan.consistency == "strong":
            self._resolved += self._resolve_targets([parent.uid])
        fixes = self._fix_hints(parent)
        current = self._current_item(parent)
        vector = vec_text(self.vectors[parent.uid])
        sql = "SELECT result FROM public.tjs_vsgt(%s::jsonb, %s::vector)"

        def item_of(row: dict[str, Any]) -> dict[str, Any]:
            return {"current": current, "candidate": self._candidate_item(row)}

        prepare = self._strong_prepare if plan.consistency == "strong" else None
        experiences: list[dict[str, Any]]
        if plan.plan == "trimodal":
            req = self._vsgt_request(parent, plan, push_r=True, k=plan.k)
            t0 = time.perf_counter()
            experiences = [
                r[0] for r in self.conn.execute(sql, (Jsonb(req), vector)).fetchall()
            ]
            extra["engine_seconds"] = time.perf_counter() - t0
            extra["rows_pulled"] = len(experiences)
        else:
            sel_r = self._sel_r_estimate(parent)
            hit = self._sem_hit_estimate(sem)
            legs = [
                PredicateLeg("theta_r", sel_r, plan.c_db, 0.0, pushable=True),
                PredicateLeg("theta_sem", plan.sel_sem_prior, plan.c_llm, hit),
            ]
            if plan.plan == "naive":
                order: tuple[str, ...] = ("theta_sem", "theta_r")
            else:
                placement = self.planner.place(legs, n_in=float(plan.k_max))
                order = placement.order
                extra["placement_source"] = placement.source
                extra["expected_cost"] = placement.expected_cost
            extra["order"] = list(order)
            push_r = order[0] == "theta_r"
            if plan.plan == "planned" and push_r:
                req = self._vsgt_request(parent, plan, push_r=True, k=plan.k_max)
                experiences, sr = sem_filter_stream(
                    self.conn,
                    sql,
                    (Jsonb(req), vector),
                    sem,
                    HINT_PREDICATE,
                    item_of,
                    k=plan.k,
                    planner=self.planner,
                    sel_hat=plan.sel_sem_prior,
                    prepare=prepare,
                    map_row=lambda r: r[0],
                )
                extra.update(sr.as_dict())
            else:
                req = self._vsgt_request(parent, plan, push_r=push_r, k=plan.k_max)
                t0 = time.perf_counter()
                rows = [
                    r[0]
                    for r in self.conn.execute(sql, (Jsonb(req), vector)).fetchall()
                ]
                extra["engine_seconds"] = time.perf_counter() - t0
                extra["rows_pulled"] = len(rows)
                if prepare is not None and rows:
                    rows = prepare(rows)
                if push_r:
                    cands = rows
                    verdicts = sem.sem_filter(
                        HINT_PREDICATE, [item_of(r) for r in cands]
                    )
                    kept = [r for r, ok in zip(cands, verdicts, strict=True) if ok]
                else:
                    verdicts = sem.sem_filter(
                        HINT_PREDICATE, [item_of(r) for r in rows]
                    )
                    kept = [
                        r
                        for r, ok in zip(rows, verdicts, strict=True)
                        if ok and self._theta_r(parent, r)
                    ]
                extra["sem_evaluated"] = len(rows)
                experiences = kept[: plan.k]

        hints, used = self._assemble(fixes, experiences, plan.budget_tokens)
        extra.update(
            {
                "fixes": len(fixes),
                "experiences": len(experiences),
                "hints": len(hints),
                "tokens": used,
                "resolved": self._resolved,
                "out_ids": [str(h["uid"]) for h in hints],
            }
        )
        if plan.consistency == "stale":
            touched = [parent.uid] + [self._row_uid(r) for r in experiences]
            extra["staleness"] = self.pending.count_open(self.scope, targets=touched)
        self.conn.commit()
        return hints, Receipt(
            time.perf_counter() - started, sem.meter.diff(before), extra
        )

    def _sel_r_estimate(self, parent: Node) -> float:
        """Selectivity of theta_R among the task's nodes, from the relational store."""
        row = self.conn.execute(
            "SELECT count(*) FILTER (WHERE is_valid AND fitness >= %s), count(*)"
            " FROM gem_eg_vertex WHERE scope_id = %s AND kind = 'node' AND task_uid = %s",
            (self._min_fitness(parent), self.scope, parent.task_uid),
        ).fetchone()
        total = int(row[1] or 0)
        return 1.0 if total == 0 else max(1e-6, min(1.0, int(row[0]) / total))

    @staticmethod
    def _sem_hit_estimate(sem: SemOps) -> float:
        m = sem.meter
        return 0.0 if m.n_sem == 0 else min(1.0, m.hits / m.n_sem)

    @staticmethod
    def _assemble(
        fixes: list[dict[str, Any]],
        experiences: list[dict[str, Any]],
        budget_tokens: int,
    ) -> tuple[list[dict[str, Any]], int]:
        out: list[dict[str, Any]] = []
        used = 0
        seen: set[int] = set()
        for row in [*fixes, *experiences]:
            rid = int(row["id"])
            if rid in seen:
                continue
            payload = row.get("payload") or {}
            text = payload.get("lesson") or str(payload.get("outcome"))
            cost = -(-len(text) // 4)
            if used + cost > budget_tokens:
                break
            used += cost
            seen.add(rid)
            out.append({"id": rid, "uid": row.get("uid"), "text": text})
        return out, used

    # ------------------------------------------------------------- state hash

    def materialized_state(self) -> dict[str, Any]:
        """Canonical Mat(S): lessons by uid and fixed_by edges by uid pair."""
        lessons = dict(
            self.conn.execute(
                "SELECT uid, vsgt_payload->>'lesson' FROM gem_eg_vertex"
                " WHERE scope_id = %s AND kind = 'node' ORDER BY uid",
                (self.scope,),
            ).fetchall()
        )
        edges = self.conn.execute(
            "SELECT a.uid, b.uid FROM gem_eg_edge e JOIN gem_eg_vertex a ON a.id = e.src"
            " JOIN gem_eg_vertex b ON b.id = e.dst"
            " WHERE e.edge_type = %s AND a.scope_id = %s ORDER BY 1, 2",
            (self.fixed_by_type, self.scope),
        ).fetchall()
        am_edges = self.conn.execute(
            "SELECT count(*) FROM gem_eg_vertex v,"
            " LATERAL graph_store.gph_traverse_typed(v.id, %s, 0, -1) t"
            " WHERE v.scope_id = %s AND v.kind = 'node'",
            (self.fixed_by_type, self.scope),
        ).fetchone()[0]
        self.conn.commit()
        return {
            "lessons": lessons,
            "fixed_by": [tuple(e) for e in edges],
            "am_fixed_by_edges": int(am_edges),
        }
