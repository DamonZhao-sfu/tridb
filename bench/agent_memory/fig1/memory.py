"""Figure 1 memory over EvoTrace: tri-modal store + semantic operators under one plan.

Store (one PG database, one WAL): ``gem_eg_vertex`` rows (relational columns + HNSW
vector), lineage edges in the native graph AM, the LLM cache ``gem_sem_cache`` and the
pending set Q ``gem_pending``.

Ingest_pi(x) for one EvoTrace node x:
  core (always eager, deterministic): vertex row + embedding + lineage edges (AM + mirror).
  semantic derivations, eager or deferred per plan pi:
    * ``lesson``   = sem_extract(outcome(x))                       -> vsgt_payload.lesson
    * ``fixed_by`` = sem_match(x, c) over candidates C(x), x failed -> eg_fixed_by edges
      C(x) = exact top-k by cosine among same-task, other-session, valid, fitness>0 nodes
      with id < vid(x).

  Equivalence argument (plan D2). Every input a derivation reads is immutable once
  written in this workload: a vertex's embedding, validity, fitness and ``outcome``
  payload never change (the only update adds ``lesson``, which no derivation reads), and
  vids are allocated monotonically under the single-writer lock. Hence the candidate set
  "id < vid(x)" read by an exact scan is the same whenever the derivation runs, and
  eager / deferred / on-read plans materialize the same state given the cache. With
  concurrent writers or mutable inputs this argument does NOT hold; the code enforces
  the single-writer lock (EgStore.allocate_vertices) and the vid-prediction check.
  The exact candidate scan is a derivation-side read bounded by the task's node count;
  it is not a query operator on the retrieval path.

  Derivation inputs are always read back from the database (never from Python state),
  so a fresh process can resolve a pending queue left by another one.

Retrieve_rho(q) for the agent about to improve parent p. Logical query: let U be the first
k_max rows of the native ``tjs_vsgt`` stream (ANN seeds -> lineage descendants, scope /
task / session pushed down) that satisfy theta_R (valid, fitness >= p's); return the
first k rows of U, in stream order, that satisfy theta_sem, plus fix hints for p (one
typed graph traversal over eg_fixed_by), assembled under a token budget.
  Plans (all compute that same result, all stream through a server-side portal):
    'trimodal'    no theta_sem (baseline), first k rows of U;
    'naive'       theta_sem before theta_R, runs to the universe boundary;
    'planned_mat' order chosen by tridb_place_sem, runs to the universe boundary;
    'planned'     order chosen by tridb_place_sem, closes the portal at k survivors.
  Consistency: 'strong' resolves pending derivations of every item it returns (parent,
  fix-hint destinations, stream rows) before using them, waiting on concurrent resolvers;
  'stale' reads what is there and reports how many returned items were still pending.
"""

from __future__ import annotations

import bisect
import hashlib
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
from bench.agent_memory.semantic.pending import PendingItem, PendingQueue
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
RAW_K = 10000  # VSGT output cap for semantic-first streams (theta_R not pushed down)


def vec_text(v: Sequence[float] | np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def outcome_failed(outcome: dict[str, Any]) -> bool:
    fit = outcome.get("fitness")
    return (not outcome.get("valid")) or fit is None or float(fit) <= 0.0


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
    k_max: int = 48  # size of the candidate universe U
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


@dataclass(frozen=True)
class DerivInput:
    """Everything a derivation reads, as committed (or about to be committed) state."""

    uid: str
    vid: int  # the commit snapshot: candidates are restricted to id < vid
    task: str
    session: str
    outcome: dict[str, Any]
    embedding: str  # vector literal
    fix_k: int


@dataclass
class Receipt:
    seconds: float = 0.0
    sem: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class StrongReadError(RuntimeError):
    """A strong read could not resolve a pending derivation it needed."""


class Fig1Memory:
    def __init__(
        self,
        conn: Any,
        scope: str,
        ops: SemOps,
        *,
        planner: SemPlanner | None = None,
        dim: int = 1024,
        materialize_chunk: int = 256,
    ) -> None:
        self.conn = conn
        self.scope = scope
        self.ops = ops
        self.planner = planner or SemPlanner(conn, cache_cost=2e-4)
        self.store = EgStore(conn, dim=dim)
        self.pending = PendingQueue(conn)
        self.materialize_chunk = materialize_chunk
        self.vid: dict[
            str, int
        ] = {}  # cache of committed uid -> vid (rebuilt on demand)
        self.vectors: dict[str, np.ndarray] = {}  # query vectors for parents (corpus)
        self.fixed_by_type: int | None = None
        self.ingested = 0
        self._task_fit: dict[str, tuple[list[float], int]] = {}  # theta_R statistics
        self._resolved = 0

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

    def _vid_of(self, uid: str) -> int | None:
        if uid not in self.vid:
            row = self.conn.execute(
                "SELECT id FROM gem_eg_vertex WHERE scope_id = %s AND uid = %s",
                (self.scope, uid),
            ).fetchone()
            if row is None:
                return None
            self.vid[uid] = int(row[0])
        return self.vid[uid]

    # ----------------------------------------------------------- derivations

    def _inputs_for(self, items: Sequence[PendingItem]) -> list[tuple[str, DerivInput]]:
        """Rebuild derivation inputs for pending items from committed rows."""
        if not items:
            return []
        rows = self.conn.execute(
            "SELECT uid, id, task_uid, session_uid, vsgt_payload->'outcome', embedding::text"
            " FROM gem_eg_vertex WHERE scope_id = %s AND uid = ANY(%s)",
            (self.scope, list({i.target_ref for i in items})),
        ).fetchall()
        by_uid = {r[0]: r for r in rows}
        out: list[tuple[str, DerivInput]] = []
        for item in items:
            r = by_uid.get(item.target_ref)
            if r is None:
                raise KeyError(f"pending derivation for unknown item {item.target_ref}")
            if int(r[1]) != item.snapshot_seq:
                raise RuntimeError(
                    f"snapshot drift for {item.target_ref}: queued at {item.snapshot_seq},"
                    f" vertex id {r[1]}"
                )
            out.append(
                (
                    item.derivation,
                    DerivInput(
                        uid=r[0],
                        vid=int(r[1]),
                        task=r[2],
                        session=r[3],
                        outcome=dict(r[4] or {}),
                        embedding=r[5],
                        fix_k=int(item.payload.get("fix_k", 5)),
                    ),
                )
            )
        return out

    def _fix_candidates(self, d: DerivInput) -> list[tuple[int, dict[str, Any]]]:
        """C(x): exact cosine top-k among committed rows with id < vid(x)."""
        rows = self.conn.execute(
            "SELECT id, vsgt_payload->'outcome' FROM gem_eg_vertex"
            " WHERE scope_id = %s AND kind = 'node' AND task_uid = %s"
            "   AND session_uid <> %s AND id < %s AND is_valid AND fitness > 0"
            " ORDER BY (embedding <=> %s::vector) + 0, id LIMIT %s",
            (self.scope, d.task, d.session, d.vid, d.embedding, d.fix_k),
        ).fetchall()
        return [(int(r[0]), dict(r[1] or {})) for r in rows]

    def _derive(
        self, items: Sequence[tuple[str, DerivInput]]
    ) -> tuple[dict[str, str], list[tuple[int, int]]]:
        """Run derivations as large batches. Returns lessons {uid: text}, edges [(src, dst)]."""
        lesson_in = [d for kind, d in items if kind == "lesson"]
        fix_in = [d for kind, d in items if kind == "fixed_by"]
        lessons: dict[str, str] = {}
        if lesson_in:
            texts = self.ops.sem_extract(
                LESSON_INSTRUCTION, [d.outcome for d in lesson_in]
            )
            lessons = {d.uid: t for d, t in zip(lesson_in, texts, strict=True)}
        edges: list[tuple[int, int]] = []
        if fix_in:
            pairs: list[tuple[Any, Any]] = []
            owners: list[tuple[int, int]] = []
            for d in fix_in:
                for cand_vid, cand_outcome in self._fix_candidates(d):
                    pairs.append((d.outcome, cand_outcome))
                    owners.append((d.vid, cand_vid))
            if pairs:
                verdicts = self.ops.sem_match(FIX_RELATION, pairs)
                edges = [o for o, ok in zip(owners, verdicts, strict=True) if ok]
        return lessons, edges

    def _apply(self, lessons: dict[str, str], edges: list[tuple[int, int]]) -> None:
        for uid, text in lessons.items():
            self.conn.execute(
                "UPDATE gem_eg_vertex SET vsgt_payload = vsgt_payload ||"
                " jsonb_build_object('lesson', %s::text) WHERE scope_id = %s AND uid = %s",
                (text, self.scope, uid),
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

    @staticmethod
    def derivations_of(node: Node) -> list[str]:
        return ["lesson", "fixed_by"] if node.failed else ["lesson"]

    # ----------------------------------------------------------------- ingest

    def ingest(self, node: Node, vector: np.ndarray, plan: IngestPlan) -> Receipt:
        started = time.perf_counter()
        before = self.ops.meter.snapshot()
        self.vectors[node.uid] = vector
        kinds = self.derivations_of(node)
        vtext = vec_text(vector)

        # Eager LLM work runs BEFORE the writer lock is taken (plan R11). Under the
        # single-writer contract the next vid is known in advance; the allocation below
        # asserts it, so the candidate read (id < vid) is the one a deferred plan issues.
        eager_lessons: dict[str, str] = {}
        eager_edges: list[tuple[int, int]] = []
        predicted = -1
        if plan.mode == "eager":
            predicted = int(
                self.conn.execute("SELECT graph_store.gph_allocated_vids()").fetchone()[
                    0
                ]
            )
            d = DerivInput(
                node.uid,
                predicted,
                node.task_uid,
                node.session_uid,
                node.outcome(),
                vtext,
                plan.fix_candidates,
            )
            eager_lessons, eager_edges = self._derive([(kind, d) for kind in kinds])
            self.conn.commit()  # end the read-only transaction before the write one
        vid = self.store.allocate_vertices(1)[0]
        if plan.mode == "eager" and vid != predicted:
            raise RuntimeError(f"vid drift: predicted {predicted}, allocated {vid}")
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
                vtext,
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
        pvid = self._vid_of(node.parent_uid) if node.parent_uid is not None else None
        if pvid is not None:
            self.store.link(pvid, vid, relation="has_child", edge_type="eg_lineage")
            self.store.link(
                vid,
                pvid,
                relation="child_of",
                edge_type="eg_child_of",
                derived_inverse=True,
            )
        enqueued = 0
        if plan.mode == "eager":
            self._apply(eager_lessons, eager_edges)
        else:
            for kind in kinds:
                pid = self.pending.enqueue(
                    self.scope,
                    kind,
                    node.uid,
                    trigger=plan.trigger,
                    snapshot_seq=vid,
                    payload={"fix_k": plan.fix_candidates},
                )
                enqueued += pid is not None
        self.conn.commit()
        self.vid[node.uid] = vid
        self._note_fitness(node)
        self.ingested += 1

        extra: dict[str, Any] = {"vid": vid, "enqueued": enqueued}
        if plan.mode == "deferred" and plan.trigger == "batch":
            if self.ingested % plan.batch_every == 0:
                extra["batch_resolved"] = self.materialize(by="batch")
        return Receipt(
            time.perf_counter() - started, self.ops.meter.diff(before), extra
        )

    # ------------------------------------------------------------- materialize

    def materialize(self, *, by: str = "batch") -> int:
        """Resolve every pending derivation of the scope, in bounded chunks.

        Each chunk (<= ``materialize_chunk`` items) is one transaction: its LLM batch,
        its writes and the resolution marks commit together. Background resolvers share
        work through SKIP LOCKED.
        """
        total = 0
        while True:
            items = self.pending.open_items(self.scope, limit=self.materialize_chunk)
            if not items:
                self.conn.commit()
                return total
            lessons, edges = self._derive(self._inputs_for(items))
            self._apply(lessons, edges)
            total += self.pending.resolve([i.id for i in items], by=by)
            self.conn.commit()

    def _resolve_strong(self, uids: Sequence[str]) -> int:
        """Resolve the pending derivations of ``uids`` now (strong read).

        Waits on rows held by a concurrent resolver (no SKIP LOCKED) and then verifies
        that nothing needed is still pending; a strong read never proceeds on a guess.
        """
        uids = list(dict.fromkeys(uids))
        if not uids or self.pending.count_open(self.scope, targets=uids) == 0:
            return 0
        items = self.pending.open_items(self.scope, targets=uids, skip_locked=False)
        n = 0
        if items:
            lessons, edges = self._derive(self._inputs_for(items))
            self._apply(lessons, edges)
            n = self.pending.resolve([i.id for i in items], by="on_read")
        left = self.pending.count_open(self.scope, targets=uids)
        if left:
            raise StrongReadError(
                f"{left} derivation(s) still pending after resolution"
            )
        return n

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
        # Improve on the parent: strictly positive for a failed parent, otherwise at least
        # as good as the parent.
        if parent.failed or parent.fitness is None:
            return 1e-9
        return float(parent.fitness)

    def _theta_r(self, parent: Node, row: dict[str, Any]) -> bool:
        """Python twin of the pushed-down predicate (output_valid + output_min_fitness)."""
        out = (row.get("payload") or {}).get("outcome") or {}
        fit = out.get("fitness")
        return (
            bool(out.get("valid"))
            and fit is not None
            and fit >= self._min_fitness(parent)
        )

    def _note_fitness(self, node: Node) -> None:
        fits, total = self._task_fit.get(node.task_uid, ([], 0))
        if node.is_valid and node.fitness is not None:
            bisect.insort(fits, float(node.fitness))
        self._task_fit[node.task_uid] = (fits, total + 1)

    def _sel_r_estimate(self, parent: Node) -> float:
        """theta_R selectivity from incrementally maintained per-task fitness statistics."""
        if parent.task_uid not in self._task_fit:  # e.g. a fresh process: load once
            rows = self.conn.execute(
                "SELECT fitness, is_valid FROM gem_eg_vertex"
                " WHERE scope_id = %s AND kind = 'node' AND task_uid = %s",
                (self.scope, parent.task_uid),
            ).fetchall()
            fits = sorted(float(f) for f, ok in rows if ok and f is not None)
            self._task_fit[parent.task_uid] = (fits, len(rows))
        fits, total = self._task_fit[parent.task_uid]
        if total == 0:
            return 1.0
        passing = len(fits) - bisect.bisect_left(fits, self._min_fitness(parent))
        return max(1e-6, min(1.0, passing / total))

    def _current_item(self, parent_vid: int, parent: Node) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT vsgt_payload->>'lesson' FROM gem_eg_vertex WHERE id = %s",
            (parent_vid,),
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
        if not rows:
            return rows
        n = self._resolve_strong([str(r["uid"]) for r in rows])
        self._resolved += n
        return self._refresh_payloads(rows) if n else rows

    def _fix_hints(self, parent_vid: int, parent: Node) -> list[dict[str, Any]]:
        if not parent.failed:
            return []
        rows = self.conn.execute(
            "SELECT v.id, v.uid, v.vsgt_payload FROM"
            " (SELECT (graph_store.gph_traverse_typed(%s, %s, 0, -1)).dst AS dst) t"
            " JOIN gem_eg_vertex v ON v.id = t.dst ORDER BY v.id",
            (parent_vid, self.fixed_by_type),
        ).fetchall()
        return [{"id": int(r[0]), "uid": r[1], "payload": r[2]} for r in rows]

    def retrieve(
        self, parent: Node, plan: RetrievePlan, *, ops: SemOps | None = None
    ) -> tuple[list[dict], Receipt]:
        """``ops`` evaluates theta_sem (default: the derivation ops); resolution of
        pending derivations always runs through the derivation ops ``self.ops``."""
        sem = ops or self.ops
        started = time.perf_counter()
        before = sem.meter.snapshot()
        self._resolved = 0
        strong = plan.consistency == "strong"
        extra: dict[str, Any] = {"plan": plan.plan, "consistency": plan.consistency}
        self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        parent_vid = self._vid_of(parent.uid)
        if parent_vid is None:
            raise KeyError(f"parent {parent.uid} not ingested")

        if strong:
            self._resolved += self._resolve_strong([parent.uid])
        fixes = self._fix_hints(parent_vid, parent)
        if strong and fixes:
            fixes = self._strong_prepare(fixes)
        current = self._current_item(parent_vid, parent)
        if parent.uid in self.vectors:
            vector = vec_text(self.vectors[parent.uid])
        else:  # a fresh process: the query vector is the parent's stored embedding
            vector = self.conn.execute(
                "SELECT embedding::text FROM gem_eg_vertex WHERE id = %s", (parent_vid,)
            ).fetchone()[0]
        sql = "SELECT result FROM public.tjs_vsgt(%s::jsonb, %s::vector)"

        def item_of(row: dict[str, Any]) -> dict[str, Any]:
            return {"current": current, "candidate": self._candidate_item(row)}

        prepare = self._strong_prepare if strong else None
        if plan.plan == "trimodal":
            req = self._vsgt_request(parent, plan, push_r=True, k=plan.k)
            t0 = time.perf_counter()
            experiences = [r[0] for r in self.conn.execute(sql, (Jsonb(req), vector))]
            extra["engine_seconds"] = time.perf_counter() - t0
            extra["rows_pulled"] = len(experiences)
            if prepare is not None:
                experiences = prepare(experiences)
        else:
            legs = [
                PredicateLeg(
                    "theta_r", self._sel_r_estimate(parent), plan.c_db, 0.0, True
                ),
                PredicateLeg(
                    "theta_sem",
                    plan.sel_sem_prior,
                    plan.c_llm,
                    self._sem_hit_estimate(sem),
                ),
            ]
            if plan.plan == "naive":
                order: tuple[str, ...] = ("theta_sem", "theta_r")
            else:
                placement = self.planner.place(legs, n_in=float(plan.k_max))
                order = placement.order
                extra["placement_source"] = placement.source
                extra["expected_cost"] = placement.expected_cost
            extra["order"] = list(order)
            r_first = order[0] == "theta_r"
            req = self._vsgt_request(
                parent, plan, push_r=r_first, k=plan.k_max if r_first else RAW_K
            )
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
                post_filter=None
                if r_first
                else (lambda row: self._theta_r(parent, row)),
                universe_limit=plan.k_max,
                early_close=plan.plan == "planned",
                prepare=prepare,
                map_row=lambda r: r[0],
            )
            extra.update(sr.as_dict())

        hints, used = self._assemble(fixes, experiences, plan.budget_tokens)
        texts = "\x1e".join(h["text"] for h in hints)
        extra.update(
            {
                "fixes": len(fixes),
                "experiences": len(experiences),
                "hints": len(hints),
                "tokens": used,
                "resolved": self._resolved,
                "out_ids": [str(h["uid"]) for h in hints],
                "out_text_sha": hashlib.sha256(texts.encode()).hexdigest()[:16],
            }
        )
        if not strong:
            touched = [parent.uid] + [str(h["uid"]) for h in hints]
            extra["staleness"] = self.pending.count_open(self.scope, targets=touched)
        self.conn.commit()
        return hints, Receipt(
            time.perf_counter() - started, sem.meter.diff(before), extra
        )

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
        """Canonical Mat(S): lessons by uid, fixed_by pairs in the mirror AND in the AM."""
        lessons = dict(
            self.conn.execute(
                "SELECT uid, vsgt_payload->>'lesson' FROM gem_eg_vertex"
                " WHERE scope_id = %s AND kind = 'node' ORDER BY uid",
                (self.scope,),
            ).fetchall()
        )
        mirror = self.conn.execute(
            "SELECT a.uid, b.uid FROM gem_eg_edge e JOIN gem_eg_vertex a ON a.id = e.src"
            " JOIN gem_eg_vertex b ON b.id = e.dst"
            " WHERE e.edge_type = %s AND a.scope_id = %s ORDER BY 1, 2",
            (self.fixed_by_type, self.scope),
        ).fetchall()
        native = self.conn.execute(
            "SELECT a.uid, b.uid FROM gem_eg_vertex a,"
            " LATERAL graph_store.gph_traverse_typed(a.id, %s, 0, -1) t"
            " JOIN gem_eg_vertex b ON b.id = t.dst"
            " WHERE a.scope_id = %s AND a.kind = 'node' ORDER BY 1, 2",
            (self.fixed_by_type, self.scope),
        ).fetchall()
        self.conn.commit()
        return {
            "lessons": lessons,
            "fixed_by": [tuple(e) for e in mirror],
            "fixed_by_native": [tuple(e) for e in native],
        }
