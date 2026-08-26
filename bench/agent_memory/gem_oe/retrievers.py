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
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openevolve.database import Program

from bench.agent_memory.gem_eg.oracle import Predicate, QuerySpec
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import EgStore
from bench.agent_memory.gem_oe.memory_database import RetrievedProgram

logger = logging.getLogger(__name__)

#: W1.a: ANN over `kind='task'`, then 2 hops of the hierarchy rollup down to nodes.
W1A = {"relation": "hier", "hops": 2, "ann_entry_kind": "task"}
#: W1.b: seeded from a node, walking lineage. Node-level, NOT the paper's entry.
W1B = {"relation": "lineage", "hops": 3, "ann_entry_kind": None}


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

    _engine: W1Engine | None = field(default=None, init=False, repr=False)
    #: Per-iteration operator counters, kept so a latency or work claim can be traced
    #: back to the iteration that produced it.
    telemetry: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = W1Engine(self.store, self.scope_id)
        self._engine.load_identity()

    # -- query construction ----------------------------------------------

    def _predicate(self, task_uid: str, parent: Program) -> Predicate:
        fitness = parent.metrics.get("combined_score")
        return Predicate(
            kind="node",
            require_valid=True,
            # `require_fitness` on: 81 of 10,672 nodes have no usable reward and can
            # never be ordered against one.
            require_fitness=self.require_beat_parent,
            min_fitness=float(fitness) if (self.require_beat_parent and fitness) else None,
            exclude_tasks=frozenset({task_uid}) if self.split == "cross_task" else frozenset(),
            include_tasks=frozenset({task_uid}) if self.split == "same_task" else None,
        )

    def _spec(self, task_uid: str, parent: Program, k: int, iteration: int) -> QuerySpec:
        shape = W1A if self.seed == "task" else W1B
        return QuerySpec(
            query_id="oe.reuse",
            decision_point=f"{task_uid}#it{iteration}",
            seed_uid=task_uid,
            relation=str(shape["relation"]),
            hops=int(shape["hops"]),
            predicate=self._predicate(task_uid, parent),
            k=k,
            ann_entry_kind=shape["ann_entry_kind"],
            ann_m_seeds=self.m_seeds,
            meta={"arm": self.name, "split": self.split, "iteration": iteration},
        )

    # -- Retriever protocol ----------------------------------------------

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        assert self._engine is not None
        spec = self._spec(task_uid, parent, k, iteration)
        result = self._engine.run(spec, self.knobs)
        self.telemetry.append(
            {
                "iteration": iteration,
                "returned": len(result.ids),
                "first_row_ms": result.first_row_ms,
                "total_ms": result.total_ms,
                "candidates_examined": result.candidates_examined,
                "graph_examined": result.graph_examined,
                "graph_censored": result.graph_censored,
                "termination_reason": result.termination_reason,
                "min_fitness": spec.predicate.min_fitness,
            }
        )
        if result.termination_reason == "no_seed_vector":
            # A task with no spec vector is a corpus defect, not an empty answer.
            raise RuntimeError(
                f"task {task_uid} has no seed vector in scope {self.scope_id}; "
                "run tools/evotrace/embed.py --track task"
            )
        return self._hydrate(result.ids)

    def _hydrate(self, uids: list[str]) -> list[RetrievedProgram]:
        """Fetch the code. Order is the operator's ranking and must be preserved."""
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
        missing_changes = 0
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
            if not (row[4] or "").strip():
                missing_changes += 1
            out.append(
                RetrievedProgram(
                    uid=uid,
                    code=row[1],
                    language=row[2] or "python",
                    metrics={k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float))},
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
        if self.telemetry:
            self.telemetry[-1]["missing_changes"] = missing_changes
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
    telemetry: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

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
        self.telemetry.append({"iteration": iteration, "returned": len(uids), **telemetry})
        if not uids:
            # Never scored as "memory had nothing": an empty result from a multi-system
            # pipeline is far more often a stage that did not connect.
            raise RuntimeError(
                f"polyglot returned no rows at iteration {iteration} for {task_uid}; "
                "an empty result is treated as an outage, not as an answer"
            )
        return self.gem._hydrate(list(uids))
