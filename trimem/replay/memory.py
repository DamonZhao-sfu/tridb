"""TriMEM memory for the replay: ingest (construction) and retrieval plans, with phase telemetry."""

from __future__ import annotations

import difflib
import hashlib
import re
import time
from dataclasses import dataclass, field

import httpx
import psycopg

from trimem import DEFAULT_MODEL

# SemFilter predicate of the joint SemFilter + SemRank plan (2026-10-12): keep only changes to the search
# algorithm itself; SEM_PREDICATE then weights the ranking of what survives.
FILTER_PREDICATE = (
    "The change modifies the search algorithm itself (for example the neighbourhood, acceptance rule, or search "
    "strategy), not only constants, formatting or input/output handling."
)
SEM_PREDICATE = (
    "The change introduces a concrete algorithmic or mathematical idea (not only a full rewrite "
    "or a small constant tweak) that could help improve another solution to the same problem."
)

RETRIEVE_SQL = """
SELECT d.id, d.fitness, s.fitness AS parent_fitness, d.changes, left(d.code, {code_chars})
FROM (SELECT id, vid, fitness FROM attempt
      WHERE task = %(task)s AND (cell IS NULL OR cell = %(cell)s) AND emb IS NOT NULL
        AND id <> %(pid)s
      ORDER BY emb <=> %(q)s::vector LIMIT 24) s,
     LATERAL trimem.expand(s.vid, 'lineage', 8) e
     JOIN attempt d ON d.vid = e.dst
WHERE d.is_valid AND d.fitness > s.fitness AND d.fitness >= %(floor)s
  AND (d.cell IS NULL OR d.cell = %(cell)s) AND d.code IS NOT NULL
  {sem}
ORDER BY {rank} DESC, d.id LIMIT {k}
"""

# Mixed path (2 chained TriMEM Expand): down = similar (into another session) -> lineage (its
# successful children); up = lineage_up (to the parent) -> similar (the parent's look-alikes in other
# sessions). DISTINCT: a row reached along several paths is injected once.
MIXED_HOPS = {"down": ("similar", "lineage"), "up": ("lineage_up", "similar")}
RETRIEVE_SQL_MIXED = """
SELECT d.id, d.fitness, p.fitness AS parent_fitness, d.changes, left(d.code, {code_chars})
FROM (SELECT DISTINCT e2.dst
      FROM (SELECT id, vid FROM attempt
            WHERE task = %(task)s AND (cell IS NULL OR cell = %(cell)s) AND emb IS NOT NULL
              AND id <> %(pid)s
            ORDER BY emb <=> %(q)s::vector LIMIT 24) s,
           LATERAL trimem.expand(s.vid, '{hop1}', {fanout}) e1,
           LATERAL trimem.expand(e1.dst, '{hop2}', {fanout}) e2) x
     JOIN attempt d ON d.vid = x.dst
     JOIN attempt p ON p.id = d.parent_id
WHERE d.is_valid AND d.fitness > p.fitness AND d.fitness >= %(floor)s
  AND (d.cell IS NULL OR d.cell = %(cell)s) AND d.code IS NOT NULL
  {sem}
ORDER BY {rank} DESC, d.id LIMIT {k}
"""


def retrieve_sql_mixed(semantic: bool | str, direction: str, code_chars: int = 2500, k: int = 5,
                       fanout: int = 8) -> str:
    """Mixed-path variant of retrieve_sql; gain = improvement over the row's own parent."""
    gain = "(d.fitness - p.fitness)"
    sem, rank = "", gain
    if semantic == "filterrank":
        sem = f"AND trimem.sem_filter('{FILTER_PREDICATE}', d.changes)"
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    elif semantic in (True, "filter"):
        sem = f"AND trimem.sem_filter('{SEM_PREDICATE}', d.changes)"
    elif semantic == "score":
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    hop1, hop2 = MIXED_HOPS[direction]
    return RETRIEVE_SQL_MIXED.format(code_chars=code_chars, sem=sem, rank=rank, k=int(k), hop1=hop1,
                                     hop2=hop2, fanout=int(fanout))

# Seed SemFilter of the math seed plans (2026-10-12). The ALE FILTER_PREDICATE names local-search parts
# (neighbourhood, acceptance rule) and rejects every gradient-optimisation change (second_autocorr_ineq:
# 0/24 seeds kept); this domain-neutral form keeps the same intent -- drop pure tuning / formatting / I/O.
SEED_FILTER_PREDICATE = (
    "The change alters the method itself -- the algorithm, optimizer or search procedure, the objective or "
    "constraint formulation, the parametrization or representation, or the construction or initialization "
    "strategy -- rather than only tuning numeric constants or hyperparameters (step counts, learning rates, "
    "restart counts, tolerances), reformatting code, or changing input/output handling.")


def retrieve_sql_multi(semantic: bool | str, edges: tuple[str, ...] = ("lineage", "similar"), max_hops: int = 3,
                       code_chars: int = 2500, k: int = 10, fanout: int = 8) -> str:
    """Multi-edge, multi-hop traversal (2026-10-12 ALE setting): from the 24 ANN seeds, every hop expands
    the DISTINCT frontier along each of `edges` (one fanout-capped TriMEM Expand per edge type); the
    candidate pool is the union of all frontiers up to `max_hops`. Gain = improvement over own parent."""
    gain = "(d.fitness - p.fitness)"
    sem, rank = "", gain
    if semantic == "filterrank":
        sem = f"AND trimem.sem_filter('{FILTER_PREDICATE}', d.changes)"
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    elif semantic in (True, "filter"):
        sem = f"AND trimem.sem_filter('{SEM_PREDICATE}', d.changes)"
    elif semantic == "score":
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    hops, prev = [], "s"
    for h in range(1, max_hops + 1):
        src = "s.vid" if prev == "s" else f"{prev}.dst"
        parts = " UNION ".join(
            f"SELECT e.dst FROM {prev}, LATERAL trimem.expand({src}, '{t}', {int(fanout)}) e" for t in edges)
        hops.append(f"h{h} AS ({parts})")
        prev = f"h{h}"
    union = " UNION ".join(f"SELECT dst FROM h{h}" for h in range(1, max_hops + 1))
    hop_ctes = ",\n".join(hops)  # outside the f-string: Python 3.11 forbids backslashes in expressions
    return f"""
WITH s AS (SELECT id, vid FROM attempt
           WHERE task = %(task)s AND (cell IS NULL OR cell = %(cell)s) AND emb IS NOT NULL AND id <> %(pid)s
           ORDER BY emb <=> %(q)s::vector LIMIT 24),
{hop_ctes},
x AS ({union})
SELECT d.id, d.fitness, p.fitness AS parent_fitness, d.changes, left(d.code, {code_chars})
FROM x JOIN attempt d ON d.vid = x.dst
     JOIN attempt p ON p.id = d.parent_id
WHERE d.is_valid AND d.fitness > p.fitness AND d.fitness >= %(floor)s
  AND (d.cell IS NULL OR d.cell = %(cell)s) AND d.code IS NOT NULL
  {sem}
ORDER BY {rank} DESC, d.id LIMIT {int(k)}
"""


def retrieve_sql_seedsem(plan: str, max_hops: int, k: int = 10, edges: tuple[str, ...] = ("lineage", "similar"),
                         fanout: int = 8, code_chars: int = 2500) -> str:
    """Seed-SemFilter plans (2026-10-12, docs/trimem_math_semfilter_plans_v0.1.0.md).

    plan: vgr  Vector -> Graph -> Telemetry -> Assemble
          rank Vector -> Graph -> Telemetry -> SemRank(d) -> Assemble
          p1   Vector -> SemFilter(s) -> Graph -> Telemetry -> SemRank(d) -> Assemble
          p2   Vector -> Graph -> Telemetry -> SemFilter(s) -> SemRank(d) -> Assemble
    Every frontier row carries the id of the seed it came from, so p1 and p2 select the same rows."""
    gain = "(d.fitness - p.fitness)"
    rank = (f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
            if plan in ("rank", "p1", "p2") else gain)
    seeds = ("SELECT a.id AS sid, a.vid FROM s JOIN attempt a ON a.id = s.id "
             f"WHERE trimem.sem_filter('{SEED_FILTER_PREDICATE}', a.changes)" if plan == "p1"
             else "SELECT s.id AS sid, s.vid FROM s")
    hops, prev = [f"s0 AS ({seeds})"], "s0"
    for h in range(1, max_hops + 1):
        src = f"{prev}.vid" if prev == "s0" else f"{prev}.dst"
        parts = " UNION ".join(
            f"SELECT {prev}.sid, e.dst FROM {prev}, LATERAL trimem.expand({src}, '{t}', {int(fanout)}) e" for t in edges)
        hops.append(f"h{h} AS ({parts})")
        prev = f"h{h}"
    union = " UNION ".join(f"SELECT sid, dst FROM h{h}" for h in range(1, max_hops + 1))
    hop_ctes = ",\n".join(hops)
    seed_join = (f"JOIN attempt sa ON sa.id = x.sid AND trimem.sem_filter('{SEED_FILTER_PREDICATE}', sa.changes)"
                 if plan == "p2" else "")
    return f"""
WITH s AS (SELECT id, vid FROM attempt
           WHERE task = %(task)s AND (cell IS NULL OR cell = %(cell)s) AND emb IS NOT NULL AND id <> %(pid)s
           AND changes IS NOT NULL  -- seeds carry a change summary (the SemFilter's input)
           ORDER BY emb <=> %(q)s::vector LIMIT 24),
{hop_ctes},
x AS ({union})
SELECT d.id, d.fitness, p.fitness AS parent_fitness, d.changes, left(d.code, {code_chars})
FROM x JOIN attempt d ON d.vid = x.dst
     JOIN attempt p ON p.id = d.parent_id
     {seed_join}
WHERE d.is_valid AND d.fitness > p.fitness AND d.fitness >= %(floor)s
  AND (d.cell IS NULL OR d.cell = %(cell)s) AND d.code IS NOT NULL AND d.changes IS NOT NULL
GROUP BY d.id, d.fitness, p.fitness, d.changes, d.code
ORDER BY {rank} DESC, d.id LIMIT {int(k)}
"""


DERIVE_SYSTEM = (
    "You summarise how a program was changed. In at most three sentences, state the concrete "
    "algorithmic or mathematical change from the PARENT program to the CHILD program. No preamble."
)

# Detailed variant (2026-10-13) for the regenerated summaries of trimem/data/regen_change_summaries.py.
DERIVE_SYSTEM_DETAILED = (
    "You summarise how a program was changed, for an engineer who wants to reuse the idea. From the PARENT/CHILD "
    "unified diff, write a detailed summary of at most 1024 characters that states: (1) the concrete algorithmic or "
    "mathematical change — method, objective or constraint formulation, representation or parametrization, "
    "initialization, or search/optimization procedure; (2) which functions or code blocks changed; (3) important "
    "numeric parameters as before -> after (counts, step sizes, iterations, tolerances, coordinates); (4) anything "
    "removed or no longer used. Describe only what the diff shows: do not guess the program's score or call anything "
    "optimal or near-optimal. If the change only reformats code, renames, or changes input/output handling, say "
    "exactly that. Plain prose, no preamble, no markdown."
)


CONTEXT_K = 5  # retrieved items handed to the agent per iteration


def retrieve_sql(
    semantic: bool | str, code_chars: int = 2500, k: int = CONTEXT_K
) -> str:
    """semantic: False (tri-modal), True/"filter" (hard filter, NULL input rejected),
    "keep_unknown" (hard filter, NULL input kept), "score" (no filter; LLM score weights the rank)."""
    gain = "(d.fitness - s.fitness)"
    sem, rank = "", gain
    if semantic == "filterrank":  # SemFilter (search-algorithm change) + SemRank (transferable idea)
        sem = f"AND trimem.sem_filter('{FILTER_PREDICATE}', d.changes)"
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    elif semantic in (True, "filter"):
        sem = f"AND trimem.sem_filter('{SEM_PREDICATE}', d.changes)"
    elif semantic == "keep_unknown":
        sem = f"AND trimem.sem_filter('{SEM_PREDICATE}', d.changes, true)"
    elif semantic == "score":
        rank = f"{gain} * (0.5 + coalesce(trimem.sem_score('{SEM_PREDICATE}', d.changes), 0.5))"
    return RETRIEVE_SQL.format(code_chars=code_chars, sem=sem, rank=rank, k=int(k))


@dataclass
class Phase:
    ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, ms: float, pt: int = 0, ct: int = 0, calls: int = 1) -> None:
        self.ms += ms
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.calls += calls


@dataclass
class Retrieved:
    items: list[dict]
    ms: float
    sem_rows_judged: int = 0
    sem_worker_requests: int = 0
    sem_prompt_tokens: int = 0
    sem_completion_tokens: int = 0
    sem_llm_wait_ms: float = 0.0
    placement: str | None = None
    decision: str | None = None
    edges: int = 0
    deferred: Phase = field(default_factory=Phase)


class Embedder:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8011/v1",
        model: str = "Qwen/Qwen3-Embedding-0.6B",
    ) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.http = httpx.Client(timeout=120)

    def embed(self, text: str) -> tuple[list[float], int, float]:
        t0 = time.perf_counter()
        for attempt in range(
            6
        ):  # the shared GPU server can restart; do not kill the cell
            try:
                r = self.http.post(
                    self.url, json={"model": self.model, "input": text[:2000]}
                )
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 5:
                    raise
                time.sleep(min(60, 2**attempt * 2))
        d = r.json()
        return (
            d["data"][0]["embedding"],
            int((d.get("usage") or {}).get("prompt_tokens") or 0),
            (time.perf_counter() - t0) * 1000,
        )


class Chat:
    def __init__(
        self, base_url: str = "http://127.0.0.1:8000/v1", model: str = DEFAULT_MODEL
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.http = httpx.Client(timeout=900)
        self.last_finish_reason: str | None = None

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int = 0,
        top_p: float = 1.0,
    ) -> tuple[str, int, int, float]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        for attempt in range(4):
            try:
                r = self.http.post(self.url, json=body)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        d = r.json()
        self.last_finish_reason = d["choices"][0].get("finish_reason")
        u = d.get("usage") or {}
        return (
            d["choices"][0]["message"].get("content") or "",
            int(u.get("prompt_tokens") or 0),
            int(u.get("completion_tokens") or 0),
            (time.perf_counter() - t0) * 1000,
        )


def _diff(a: str, b: str, limit: int = 6000) -> str:
    d = "\n".join(
        difflib.unified_diff(
            a.splitlines(), b.splitlines(), "parent", "child", lineterm="", n=1
        )
    )
    return d[:limit]


def ensure_schema(dsn: str) -> None:
    """Replay side tables; created once before the concurrent cells start."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS replay_derive_cache (key text PRIMARY KEY, "
            "summary text, prompt_tokens int, completion_tokens int, ms float8)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS replay_pending (id int PRIMARY KEY, "
            "cell text, enqueued_at timestamptz DEFAULT now())"
        )
        conn.execute("CREATE SEQUENCE IF NOT EXISTS replay_attempt_id START 1000000")


class Memory:
    """One replay cell's view of the TriMEM store."""

    def __init__(
        self,
        dsn: str,
        task: str,
        cell: str,
        *,
        embedder: Embedder,
        chat: Chat,
        deferred: bool,
        semantic: bool,
        placement: str | None,
        derive: bool = True,
        force: str | None = None,
        k: int = CONTEXT_K,
        code_chars: int = 2500,
    ) -> None:
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.task, self.cell = task, cell
        self.embedder, self.chat = embedder, chat
        self.deferred, self.semantic, self.placement = deferred, semantic, placement
        self.code_chars = code_chars
        self.derive = derive
        self.force = force
        self.k = k
        for stmt in (
            "SET jit = off",
            "SET trimem.timeout_ms = 900000",
            "SET hnsw.iterative_scan = relaxed_order",
            "SET hnsw.ef_search = 100",
            f"SET trimem.prompt_version = 'replay-{cell}'",
        ):
            self.conn.execute(stmt)

    # ------------------------------------------------------------------ construction
    def ingest(
        self,
        code: str,
        fitness: float,
        valid: bool,
        parent_id: int | None,
        iteration: int,
        parent_code: str | None,
    ) -> tuple[int, dict]:
        """Store one attempt: embedding (V), row (R), lineage edge (G), change summary (LLM)."""
        tel = {"embed": Phase(), "derive": Phase(), "insert_ms": 0.0}
        emb, etok, ems = self.embedder.embed(code)
        tel["embed"].add(ems, etok)
        t0 = time.perf_counter()
        (aid,) = self.conn.execute(
            "INSERT INTO attempt (id, uid, task, session, iteration, is_valid, fitness, status, judge, "
            "language, code, cell, parent_id, emb) VALUES (nextval('replay_attempt_id'), "
            "%s, %s, %s, %s, %s, %s, 'accepted', NULL, 'python', %s, %s, %s, %s::vector) RETURNING id",
            (
                f"{self.cell}#{iteration}",
                self.task,
                self.cell,
                iteration,
                valid,
                fitness,
                code,
                self.cell,
                parent_id,
                "[" + ",".join(f"{x:.6g}" for x in emb) + "]",
            ),
        ).fetchone()
        self.conn.execute(
            "UPDATE attempt SET vid = graph_store.gph_upsert_vertex(id) WHERE id = %s",
            (aid,),
        )
        if parent_id is not None:
            self.conn.execute(
                "SELECT graph_store.gph_insert_edge(p.vid, c.vid, graph_store.register_edge_type('lineage')) "
                "FROM attempt p, attempt c WHERE p.id = %s AND c.id = %s",
                (parent_id, aid),
            )
        tel["insert_ms"] = (time.perf_counter() - t0) * 1000
        if parent_code is not None and self.derive:
            if self.deferred:
                self.conn.execute(
                    "INSERT INTO replay_pending (id, cell) VALUES (%s, %s)",
                    (aid, self.cell),
                )
            else:
                self._derive(aid, parent_code, code, tel["derive"])
        return aid, tel

    def _derive(self, aid: int, parent_code: str, code: str, phase: Phase) -> None:
        key = hashlib.sha256((parent_code + "\x1f" + code).encode()).hexdigest()
        row = self.conn.execute(
            "SELECT summary FROM replay_derive_cache WHERE key = %s", (key,)
        ).fetchone()
        if row is None:
            text, pt, ct, ms = self.chat.complete(
                DERIVE_SYSTEM,
                f"PARENT/CHILD unified diff:\n{_diff(parent_code, code)}",
                temperature=0.0,
                max_tokens=160,
            )
            summary = re.sub(r"\s+", " ", text).strip()[:600]
            self.conn.execute(
                "INSERT INTO replay_derive_cache VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (key, summary, pt, ct, ms),
            )
            phase.add(ms, pt, ct)
            row = self.conn.execute(
                "SELECT summary FROM replay_derive_cache WHERE key = %s", (key,)
            ).fetchone()
        self.conn.execute(
            "UPDATE attempt SET changes = %s WHERE id = %s", (row[0], aid)
        )

    def resolve_pending(self, phase: Phase) -> int:
        """Strong read: materialise every pending derivation of this cell, FIFO."""
        rows = self.conn.execute(
            "SELECT p.id, c.code, par.code FROM replay_pending p JOIN attempt c ON c.id = p.id "
            "JOIN attempt par ON par.id = c.parent_id WHERE p.cell = %s ORDER BY p.id",
            (self.cell,),
        ).fetchall()
        for aid, code, parent_code in rows:
            self._derive(aid, parent_code, code, phase)
            self.conn.execute("DELETE FROM replay_pending WHERE id = %s", (aid,))
        return len(rows)

    # ------------------------------------------------------------------ retrieval
    def retrieve(
        self, parent_id: int, parent_code: str, parent_fitness: float
    ) -> Retrieved:
        deferred = Phase()
        if self.deferred:
            self.resolve_pending(deferred)
        q, _, _ = self.embedder.embed(parent_code)
        sql = retrieve_sql(self.semantic, self.code_chars, self.k)
        params = {
            "task": self.task,
            "cell": self.cell,
            "pid": parent_id,
            "q": "[" + ",".join(f"{x:.6g}" for x in q) + "]",
            "floor": 0.8 * parent_fitness,
        }
        self.conn.execute("RESET trimem.force_placement")
        if self.force:
            self.conn.execute(f"SET trimem.force_placement = '{self.force}'")
        decision = None
        if self.semantic in (True, "filter", "keep_unknown"):
            plan = self.conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()[
                0
            ][0]["Plan"]
            decision = _find(plan, "Placement Decision")
            if self.placement == "alt" and decision:
                chosen = decision.split(" ")[0]  # e.g. d=top
                alias, where = chosen.split("=")
                self.conn.execute(
                    f"SET trimem.force_placement = '{alias}={'base' if where == 'top' else 'top'}'"
                )
        tok0 = self._sem_tokens()
        explain = self.conn.execute(
            "EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params
        ).fetchone()[0][0]
        ms = explain["Execution Time"] + explain["Planning Time"]
        tok1 = self._sem_tokens()
        rows = self.conn.execute(
            sql, params
        ).fetchall()  # same rows; verdicts now cached
        sem_nodes = list(_nodes(explain["Plan"], "TriMEM SemFilter")) + list(
            _nodes(explain["Plan"], "TriMEM SemScore")
        )
        return Retrieved(
            items=[
                {
                    "id": r[0],
                    "fitness": r[1],
                    "parent_fitness": r[2],
                    "changes": r[3],
                    "code": r[4],
                }
                for r in rows
            ],
            ms=ms,
            sem_rows_judged=sum(
                n.get("Rows Judged", n.get("Inputs Scored", 0)) for n in sem_nodes
            ),
            sem_worker_requests=sum(n.get("Worker Requests", 0) for n in sem_nodes),
            sem_prompt_tokens=tok1[0] - tok0[0],
            sem_completion_tokens=tok1[1] - tok0[1],
            sem_llm_wait_ms=sum(n.get("LLM Wait", 0.0) for n in sem_nodes),
            placement=sem_nodes[0].get("Placement") if sem_nodes else None,
            decision=decision,
            edges=sum(
                n.get("Edges Emitted", 0)
                for n in _nodes(explain["Plan"], "TriMEM Expand")
            ),
            deferred=deferred,
        )

    def _sem_tokens(self) -> tuple[int, int]:
        r = self.conn.execute(
            "SELECT coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0) "
            "FROM trimem.sem_cache WHERE prompt_version = %s",
            (f"replay-{self.cell}",),
        ).fetchone()
        return int(r[0]), int(r[1])


def _nodes(p: dict, provider: str):
    if p.get("Custom Plan Provider") == provider:
        yield p
    for c in p.get("Plans", []):
        yield from _nodes(c, provider)


def _find(p: dict, key: str):
    if key in p:
        return p[key]
    for c in p.get("Plans", []):
        v = _find(c, key)
        if v:
            return v
    return None
