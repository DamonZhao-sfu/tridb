"""The arms: where an injected inspiration comes from.

Every arm answers the same question -- "given that this run is working on task T and
its parent scored f, which historical attempts should the model see?" -- and returns
the same shape. Arms differ ONLY in which system answers it, so an outcome difference
cannot be attributed to a different question.

The canonical query is W1.a, the shape arXiv:2606.29823 actually specifies for
cross-session reuse: ANN over task specifications, then a bounded typed traversal from
each hit down to nodes, with the relational filter pushed into the traversal. Seeding
from the parent's CODE instead is a different (larger, Node-level) query and is offered
as `seed="node"` for a separate experiment -- never mixed into the same results table.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from openevolve.database import Program
    from bench.agent_memory.gem_oe.memory_database import RetrievedProgram

from bench.agent_memory.gem_eg.oracle import Predicate, QuerySpec, logical_spec_hash
from bench.agent_memory.gem_eg.physical import PhysicalExecutor, PhysicalPlan
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import EgStore

logger = logging.getLogger(__name__)

#: W1.a: ANN over `kind='task'`, then 2 hops of the hierarchy rollup down to nodes.
W1A = {"relation": "hier", "hops": 2, "ann_entry_kind": "task"}
#: W1.b: seeded from a node, walking lineage. Node-level, NOT the paper's entry.
W1B = {"relation": "lineage", "hops": 3, "ann_entry_kind": "node"}


@dataclass
class GemRetriever:
    """Arm B. One `tjs_open` pair per iteration, inside one Postgres transaction."""

    store: EgStore
    scope_id: str
    #: Split policy. `same_task` lets the run see other sessions of its own task;
    #: `cross_task` hides the whole task, so only transfer from other tasks is possible.
    split: str = "same_task"
    seed: str = "task"
    #: Only return attempts that already beat the parent. Retrieving something the
    #: agent has provably already surpassed spends prompt budget to say nothing.
    require_beat_parent: bool = True
    m_seeds: int = 4
    knobs: Knobs = field(default_factory=lambda: Knobs())
    name: str = "gem"
    physical_plan: str | None = None
    embedding_endpoint: str = "http://127.0.0.1:8001/v1/embeddings"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    query_language: str = "python"

    _engine: W1Engine | None = field(default=None, init=False, repr=False)
    #: Per-iteration operator counters, kept so a latency or work claim can be traced
    #: back to the iteration that produced it.
    telemetry: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._engine = W1Engine(self.store, self.scope_id)
        self._engine.load_identity()
        if self.physical_plan is not None:
            self.physical_plan = PhysicalPlan.parse(self.physical_plan).value
            PhysicalExecutor(self._engine).ensure_ready(self.physical_plan)

    # -- query construction ----------------------------------------------

    def _predicate(self, task_uid: str, parent: Program) -> Predicate:
        fitness = parent.metrics.get("combined_score")
        code = getattr(parent, "code", "") or ""
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None
        duplicate_nodes: frozenset[str] = frozenset()
        if digest:
            rows = self.store.conn.execute(
                "SELECT n.node_uid FROM gem_eg_node n"
                " JOIN gem_eg_artifact a ON a.artifact_uid=n.artifact_uid"
                " WHERE a.sha256=%s",
                (digest,),
            ).fetchall()
            duplicate_nodes = frozenset(str(row[0]) for row in rows)
        metadata = getattr(parent, "metadata", {}) or {}
        target_session = metadata.get("session_uid")
        return Predicate(
            kind="node",
            require_valid=True,
            # `require_fitness` on: 81 of 10,672 nodes have no usable reward and can
            # never be ordered against one.
            require_fitness=self.require_beat_parent,
            require_finite_fitness=self.seed == "node",
            fitness_gt=float(fitness)
            if (self.seed == "node" and self.require_beat_parent and fitness is not None)
            else None,
            min_fitness=float(fitness)
            if (
                self.seed != "node"
                and self.require_beat_parent
                and fitness is not None
            )
            else None,
            exclude_sessions=frozenset({str(target_session)})
            if target_session
            else frozenset(),
            exclude_tasks=frozenset({task_uid})
            if self.split == "cross_task"
            else frozenset(),
            include_tasks=frozenset({task_uid}) if self.split == "same_task" else None,
            exclude_nodes=duplicate_nodes,
        )

    def _spec(
        self,
        task_uid: str,
        parent: Program,
        k: int,
        iteration: int,
        query_vector: tuple[float, ...] | None = None,
    ) -> QuerySpec:
        shape = W1A if self.seed == "task" else W1B
        return QuerySpec(
            query_id="oe.reuse",
            decision_point=f"{task_uid}#it{iteration}",
            seed_uid=task_uid,
            relation=str(shape["relation"]),
            hops=int(shape["hops"]),
            predicate=self._predicate(task_uid, parent),
            k=k,
            rank_by="seed_fitness" if self.seed == "node" else "similarity",
            query_vector=query_vector,
            ann_entry_kind=shape["ann_entry_kind"],
            ann_m_seeds=self.m_seeds,
            meta={"arm": self.name, "split": self.split, "iteration": iteration},
        )

    def _embed_parent(self, parent: Program) -> tuple[tuple[float, ...], float]:
        from tools.evotrace.embed import embed_batch, node_query_text

        metrics = getattr(parent, "metrics", {}) or {}
        score = metrics.get("combined_score")
        status = "valid" if score is not None and math.isfinite(float(score)) else "failed"
        metadata = getattr(parent, "metadata", {}) or {}
        text = node_query_text(
            language=getattr(parent, "language", None) or self.query_language,
            status=status,
            changes=getattr(parent, "changes_description", None),
            error=metadata.get("error_signature"),
            payload=getattr(parent, "code", None),
        )
        started = time.perf_counter()
        vectors = embed_batch(
            [text], endpoint=self.embedding_endpoint, model=self.embedding_model
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        if len(vectors) != 1 or len(vectors[0]) != self.store.dim:
            got = 0 if not vectors else len(vectors[0])
            raise RuntimeError(
                f"live node embedding dimension {got}, expected {self.store.dim}"
            )
        return tuple(float(v) for v in vectors[0]), elapsed

    # -- Retriever protocol ----------------------------------------------

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        assert self._engine is not None
        retriever_started = time.perf_counter()
        query_vector: tuple[float, ...] | None = None
        embed_ms = 0.0
        if self.seed == "node":
            query_vector, embed_ms = self._embed_parent(parent)
        spec = self._spec(task_uid, parent, k, iteration, query_vector)
        selected_plan = self.physical_plan
        if self.seed == "node" and selected_plan is None:
            selected_plan = PhysicalPlan.VFWD.value
        result = self._engine.run(spec, self.knobs, selected_plan)
        hydrate_started = time.perf_counter()
        hydrated = self._hydrate(result.ids)
        hydrate_ms = (time.perf_counter() - hydrate_started) * 1000.0
        retriever_total_ms = (time.perf_counter() - retriever_started) * 1000.0
        result.embed_ms = embed_ms
        result.hydrate_ms = hydrate_ms
        result.retriever_total_ms = retriever_total_ms
        classified = (
            embed_ms
            + result.ann_ms
            + result.graph_ms
            + result.predicate_ms
            + result.dedup_rank_ms
            + hydrate_ms
        )
        result.executor_overhead_ms = max(0.0, retriever_total_ms - classified)
        self.telemetry.append(
            {
                "iteration": iteration,
                "returned": len(result.ids),
                "first_row_ms": result.first_row_ms,
                "total_ms": result.total_ms,
                "retriever_total_ms": retriever_total_ms,
                "embed_ms": embed_ms,
                "ann_ms": result.ann_ms,
                "graph_ms": result.graph_ms,
                "predicate_ms": result.predicate_ms,
                "dedup_rank_ms": result.dedup_rank_ms,
                "hydrate_ms": hydrate_ms,
                "executor_overhead_ms": result.executor_overhead_ms,
                "physical_plan": result.physical_plan,
                "logical_spec_hash": result.logical_spec_hash or logical_spec_hash(spec),
                "timing_schema": result.timing_schema,
                "entries": result.entries,
                "seeds_consumed": result.seeds_consumed,
                "ann_candidates": result.ann_candidates,
                "ann_prefixes": result.ann_prefixes,
                "predicate_probes": result.predicate_probes,
                "predicate_passed": result.predicate_passed,
                "reverse_membership_probes": result.reverse_membership_probes,
                "raw_reached": result.raw_reached,
                "distinct_reached": result.distinct_reached,
                "dedup_hits": result.dedup_hits,
                "candidates_examined": result.candidates_examined,
                "graph_examined": result.graph_examined,
                "graph_censored": result.graph_censored,
                "termination_reason": result.termination_reason,
                "min_fitness": spec.predicate.min_fitness,
                "fitness_gt": spec.predicate.fitness_gt,
                "missing_changes": sum(
                    1 for item in hydrated if not item.changes_description.strip()
                ),
            }
        )
        if result.termination_reason == "no_seed_vector":
            # A task with no spec vector is a corpus defect, not an empty answer.
            raise RuntimeError(
                f"task {task_uid} has no seed vector in scope {self.scope_id}; "
                "run tools/evotrace/embed.py --track task"
            )
        return hydrated

    def _hydrate(self, uids: list[str]) -> list[RetrievedProgram]:
        """Fetch the code. Order is the operator's ranking and must be preserved."""
        from bench.agent_memory.gem_oe.memory_database import RetrievedProgram

        if not uids:
            return []
        rows = self.store.conn.execute(
            "SELECT n.node_uid, a.payload, n.language, n.metrics, n.changes,"
            "       n.session_uid, n.task_uid, n.iteration, n.fitness"
            "  FROM gem_eg_node n"
            "  JOIN gem_eg_artifact a ON a.artifact_uid = n.artifact_uid"
            " WHERE n.node_uid = ANY(%s) AND a.payload IS NOT NULL",
            (uids,),
        ).fetchall()
        by_uid = {r[0]: r for r in rows}
        out: list[RetrievedProgram] = []
        for uid in uids:
            row = by_uid.get(uid)
            if row is None:
                # The operator ranked a node whose code we do not hold. Dropping it
                # silently would quietly shrink the injection below the balanced
                # count, so it is logged and counted.
                logger.warning("retrieved node %s has no payload; dropped", uid)
                continue
            metrics = row[3] if isinstance(row[3], dict) else {}
            # 4,443 of the 5,403 math programs carry an edit description; the rest
            # render as `<missing changes_description>` under `--inject-as changes`.
            # Counted rather than dropped: dropping would silently shrink the
            # injection below the requested rate, and the rate is an experimental
            # variable.
            out.append(
                RetrievedProgram(
                    uid=uid,
                    code=row[1],
                    language=row[2] or "python",
                    metrics={
                        k: float(v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    },
                    changes_description=row[4] or "",
                    provenance={
                        "source": self.name,
                        "session_uid": row[5],
                        "task_uid": row[6],
                        "iteration": row[7],
                        "fitness": row[8],
                    },
                )
            )
        return out


@dataclass
class PolyglotRetriever:
    """Arm C. Same question, answered by Milvus + Neo4j + pgvector.

    Deliberately a thin shell over the E0 polyglot adapter rather than a second
    retrieval design: if the two arms computed different answers, an outcome
    difference would say nothing about the systems. The parity gate in
    `tools/evotrace/gate_polyglot_parity.py` is what makes that claim checkable, and
    arm C must not run until it passes -- the E0 polyglot numbers were retracted on
    2026-08-18 precisely because 1,010 cells returned empty result sets that were
    silently scored as zero.
    """

    backend: Any
    gem: GemRetriever
    name: str = "polyglot"
    telemetry: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        spec = self.gem._spec(task_uid, parent, k, iteration)
        # The seed vector comes from GEM's own identity map, so both arms rank against
        # a byte-identical query vector. Re-embedding here would introduce a second
        # source of truth for the same numbers -- the failure that once drove W1.a's
        # parity from 0.963 to 0.000 with no error raised.
        literal = self.gem._engine._vectors.get(spec.seed_uid)
        if literal is None:
            raise RuntimeError(f"no seed vector for {spec.seed_uid}")
        query_vec = np.asarray(json.loads(literal), dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec) or 1.0
        uids, telemetry = self.backend.reuse_query(spec, query_vec)
        self.telemetry.append(
            {"iteration": iteration, "returned": len(uids), **telemetry}
        )
        if not uids and not telemetry.get("candidates_examined"):
            # Zero CANDIDATES means a stage did not connect -- that is an outage
            # and must not be scored as "memory had nothing to say".
            raise RuntimeError(
                f"polyglot examined no candidates at iteration {iteration} for "
                f"{task_uid}; an empty result is treated as an outage, not an answer"
            )
        # Candidates came back but none passed `fitness >= parent_fitness` is a
        # RESULT. Raising on it made this arm lose whole cells that GEM kept: GEM
        # simply returns fewer rows, so on ALE tasks where the predicate is hard
        # to satisfy, four `polyglot@p1.0` cells died at 0, 9, 14 and 0 of 40
        # iterations while their GEM counterparts ran to completion. That is a
        # bias introduced by this harness, not a property of either system, and
        # it would have shown up in the results as polyglot being fragile.
        return self.gem._hydrate(list(uids))


@dataclass
class CogneeRetriever:
    """Arm D. Same question, answered by an off-the-shelf agent-memory system.

    Deliberately NOT held to the parity gate that arms B and C pass. GEM and the
    polyglot stack were verified to return byte-identical result sets against an
    exhaustive oracle, so a difference between them is attributable to the system;
    Cognee builds its own LLM-derived graph and retrieves chunks from it, which is
    a different question answered differently. It earns its own row in the results
    table and must never be folded into a parity claim.

    The reward predicate travels in the request and is applied by the service as a
    post-filter, because Cognee has no scalar pushdown. That is the same place the
    polyglot stack applies it, and the surviving count is recorded per query so a
    silently-shrunk injection cannot be mistaken for "memory had nothing to say".
    """

    endpoint: str
    gem: GemRetriever
    timeout: float = 120.0
    name: str = "cognee"
    telemetry: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        import urllib.request

        spec = self.gem._spec(task_uid, parent, k, iteration)
        # The query text is the TASK SPECIFICATION, the same entry point W1.a uses.
        # Seeding Cognee from the parent's code instead would be a different query,
        # and the arms would then differ by question as well as by system.
        query = self.gem.store.conn.execute(
            "SELECT specification FROM gem_eg_task WHERE task_uid = %s", (task_uid,)
        ).fetchone()
        if not query or not query[0]:
            raise RuntimeError(
                f"task {task_uid} has no specification; cannot query cognee"
            )
        payload = json.dumps(
            {
                "query": query[0][:8000],
                "k": k,
                "task_uid": task_uid,
                "min_fitness": spec.predicate.min_fitness,
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        if "error" in body:
            # Never scored as "memory had nothing": a failed service call that
            # degrades this arm into the no-context arm would invalidate the row.
            raise RuntimeError(
                f"cognee retrieval failed at iteration {iteration}: {body['error']}"
            )
        self.telemetry.append(
            {
                "iteration": iteration,
                "returned": len(body["uids"]),
                **body.get("telemetry", {}),
            }
        )
        telemetry = body.get("telemetry", {})
        if not body["uids"] and not telemetry.get("chunks_returned"):
            # Zero CHUNKS is an outage -- the same rule arm C carries. Measured on
            # the first cognee run, the service answered 200 with zero chunks for
            # every query because the result parsing was wrong, and this arm
            # quietly injected nothing while looking healthy.
            raise RuntimeError(
                f"cognee returned no chunks at iteration {iteration} for "
                f"{task_uid}; an empty result is treated as an outage, not an answer"
            )
        # Chunks came back but nothing survived `fitness >= parent_fitness` is a
        # RESULT, not a fault, and it is the arm's defining difference: Cognee
        # ranks by similarity and the predicate is applied afterwards, so once the
        # agent passes most of the corpus there may be no eligible row among the
        # top chunks. GEM and the polyglot stack push the predicate INTO retrieval
        # and keep returning qualifying rows. Measured on third_autocorr_ineq,
        # where the parent reaches 0.97+: 8 of 11 iterations, with 60 chunks
        # returned and 58 of them mappable every time.
        #
        # Treating it as an outage killed the cell; it is recorded and returned
        # empty instead, so the iteration runs with no injection and the count is
        # visible in the telemetry rather than inferred from a crash.
        # Hydrated from Postgres, NOT from Cognee: the arms must inject the same
        # bytes for the same node, or a quality difference could come from the
        # code the prompt rendered rather than from which node was chosen.
        return self.gem._hydrate(list(body["uids"]))
