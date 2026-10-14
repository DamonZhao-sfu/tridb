"""TriMEM as an OpenEvolve memory arm (offline experience graph, read-only during the run).

One retrieval per evolution step, issued through the same SQL the greedy replay uses
(`trimem.replay.memory.retrieve_sql`): ANN seeds over the parent's code embedding ->
native lineage Expand -> telemetry filter (valid, improved over its own parent, >= 0.8 x the
parent's score) -> optional semantic operator -> rank -> LIMIT k. Modes:

* ``trimodal`` — V+G+R only;
* ``hard``     — + ``trimem.sem_filter`` (TriMEM SemFilter, placement chosen by the planner);
* ``rank``     — + ``trimem.sem_score`` weighting the rank (TriMEM SemScore), placement chosen by
  the planner unless ``force`` pins it (``d=base`` semantic first, ``d=top`` traverse first).

The memory is the corpus only: OpenEvolve's own programs live in its database, and
`(cell IS NULL OR cell = '<oe>')` with a cell id nothing was written under keeps the greedy
replay's attempts out. Semantic verdicts/scores need a running ``python -m trimem.worker``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from bench.agent_memory.gem_oe.memory_database import RetrievedProgram
from trimem.replay.memory import Embedder, retrieve_sql, retrieve_sql_mixed, retrieve_sql_multi, retrieve_sql_seedsem

MODES = {"trimodal": False, "hard": True, "rank": "score", "filterrank": "filterrank",
         # seed-SemFilter plans (docs/trimem_math_semfilter_plans_v0.1.0.md): lineage + similar, mean-depth hops
         "seed-vgr": "vgr", "seed-rank": "rank", "seed-p1": "p1", "seed-p2": "p2", "seed-opt": "opt"}
# SemRank(d) always runs after telemetry in these plans; p2 also places SemFilter(s) after it
SEED_FORCE = {"vgr": None, "rank": "d=top", "p1": "d=top", "p2": "sa=top,d=top"}


def _nodes(plan: dict, provider: str):
    if plan.get("Custom Plan Provider") == provider:
        yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child, provider)


def _find(plan: dict, key: str):
    if key in plan:
        return plan[key]
    for child in plan.get("Plans", []):
        hit = _find(child, key)
        if hit:
            return hit
    return None


@dataclass
class TrimemRetriever:
    dsn: str
    mode: str
    force: str | None = None
    prompt_version: str = "oe"
    require_changes: bool = False  # injecting change summaries: rows without one carry nothing
    graph: str | None = None  # None = lineage/down/1/8 (v1); "mixed/down" | "mixed/up" = 2-hop mixed path
    embedding_url: str = "http://127.0.0.1:8011/v1"
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown TriMEM mode {self.mode!r}; expected one of {sorted(MODES)}")
        self.name = (f"trimem-{self.mode}" + (f"[{self.force}]" if self.force else "")
                     + (f"@{self.graph}" if self.graph else ""))
        self.embedder = Embedder(self.embedding_url)
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        for stmt in ("SET jit = off", "SET trimem.timeout_ms = 900000",
                     "SET hnsw.iterative_scan = relaxed_order", "SET hnsw.ef_search = 100",
                     f"SET trimem.prompt_version = '{self.prompt_version}'"):
            self.conn.execute(stmt)
        if self.force:
            self.conn.execute(f"SET trimem.force_placement = '{self.force}'")

    def retrieve(self, *, task_uid: str, parent: Any, k: int, iteration: int) -> list[RetrievedProgram]:
        if k <= 0:
            return []
        started = time.perf_counter()
        fitness = float((parent.metrics or {}).get("combined_score") or 0.0)
        # a NaN / inf parent score (failed evaluation) would make the floor reject every row
        no_floor = not math.isfinite(fitness)
        q, emb_tokens, emb_ms = self.embedder.embed(parent.code)
        params = {"task": task_uid, "cell": f"__oe__:{self.prompt_version}", "pid": -1,
                  "q": "[" + ",".join(f"{x:.6g}" for x in q) + "]",
                  # >= 0.8 x parent, written so it also holds for negative (signed minimisation) ALE scores
                  "floor": -1e300 if no_floor else fitness - 0.2 * abs(fitness)}
        plan_choice = None
        if self.mode.startswith("seed-"):
            sql, plan_choice = self._seed_plan(task_uid, k, params)
        else:
            if self.graph and self.graph.startswith("lineage+similar/"):  # e.g. lineage+similar/3
                sql = retrieve_sql_multi(MODES[self.mode], ("lineage", "similar"), int(self.graph.split("/")[1]),
                                         2500, k)
            elif self.graph:
                sql = retrieve_sql_mixed(MODES[self.mode], self.graph.split("/")[1], 2500, k)
            else:
                sql = retrieve_sql(MODES[self.mode], 2500, k)
            if self.require_changes:
                sql = sql.replace("AND d.code IS NOT NULL", "AND d.code IS NOT NULL AND d.changes IS NOT NULL", 1)
        tok0 = self._tokens()
        explain = self.conn.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params).fetchone()[0][0]
        tok1 = self._tokens()
        rows = self.conn.execute(sql, params).fetchall()
        uids = {r[0]: u for r, u in zip(rows, self._uids([r[0] for r in rows]), strict=True)}
        sem = list(_nodes(explain["Plan"], "TriMEM SemFilter")) + list(_nodes(explain["Plan"], "TriMEM SemScore"))
        self.telemetry.append({
            "iteration": iteration, "k": k, "returned": len(rows),
            "sql_ms": explain["Execution Time"] + explain["Planning Time"],
            "embed_ms": emb_ms, "embed_tokens": emb_tokens,
            "total_ms": (time.perf_counter() - started) * 1000,
            "sem_rows": sum(n.get("Rows Judged", n.get("Inputs Scored", 0)) for n in sem),
            "sem_worker_requests": sum(n.get("Worker Requests", 0) for n in sem),
            "sem_llm_wait_ms": sum(n.get("LLM Wait", 0.0) for n in sem),
            "sem_prompt_tokens": tok1[0] - tok0[0], "sem_completion_tokens": tok1[1] - tok0[1],
            "placement": sem[0].get("Placement") if sem else None,
            "decision": _find(explain["Plan"], "Placement Decision"),
            "plan_choice": plan_choice,
            "edges": sum(n.get("Edges Emitted", 0) for n in _nodes(explain["Plan"], "TriMEM Expand")),
            "ids": [r[0] for r in rows],
            # full source length of every injected program (injected uncut; see truncation report)
            "code_chars": [len(r[4] or "") if len(r[4] or "") < 2500 else len(self._full_code(r[0])) for r in rows],
        })
        return [
            RetrievedProgram(
                uid=uids[r[0]], code=r[4] if len(r[4] or "") < 2500 else self._full_code(r[0]),
                metrics={"combined_score": float(r[1])}, changes_description=r[3] or "",
                provenance={"memory": "trimem", "mode": self.mode, "attempt_id": r[0],
                            "parent_fitness": float(r[2])},
            )
            for r in rows
        ]

    def _depth(self, task_uid: str) -> int:
        """Mean corpus lineage depth of the task (hops of the seed-SemFilter plans)."""
        if not hasattr(self, "_depth_cache"):
            from trimem.objective.space import DEPTH_SQL

            mean, _ = self.conn.execute(DEPTH_SQL, (task_uid,)).fetchone()
            self._depth_cache = max(1, int(mean or 1))
        return self._depth_cache

    def _seed_plan(self, task_uid: str, k: int, params: dict) -> tuple[str, dict]:
        """SQL of a seed-SemFilter plan; for seed-opt, EXPLAIN P1 and P2 and keep the cheaper (plan
        enumeration only -- k, hops and edges are fixed)."""
        hops = self._depth(task_uid)
        plan = MODES[self.mode]
        choice: dict = {"hops": hops, "k": k}
        if plan == "opt":
            costs = {}
            for cand in ("p1", "p2"):
                self.conn.execute(f"SET trimem.force_placement = '{SEED_FORCE[cand]}'")
                costs[cand] = self.conn.execute(
                    "EXPLAIN (FORMAT JSON) " + retrieve_sql_seedsem(cand, hops, k), params).fetchone()[0][0]["Plan"]["Total Cost"]
            plan = min(costs, key=costs.get)
            choice |= {"costs": costs, "chosen": plan}
        force = SEED_FORCE[plan]
        self.conn.execute(f"SET trimem.force_placement = '{force}'" if force else "RESET trimem.force_placement")
        choice["plan"] = plan
        return retrieve_sql_seedsem(plan, hops, k), choice

    def _uids(self, ids: list[int]) -> list[str]:
        if not ids:
            return []
        got = dict(self.conn.execute("SELECT id, uid FROM attempt WHERE id = ANY(%s)", (ids,)).fetchall())
        return [got[i] for i in ids]

    def _full_code(self, attempt_id: int) -> str:
        return self.conn.execute("SELECT code FROM attempt WHERE id = %s", (attempt_id,)).fetchone()[0]

    def _tokens(self) -> tuple[int, int]:
        r = self.conn.execute("SELECT coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0) "
                              "FROM trimem.sem_cache WHERE prompt_version = %s", (self.prompt_version,)).fetchone()
        return int(r[0]), int(r[1])
