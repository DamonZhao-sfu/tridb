"""Harness search space v7: retrieval-side knobs on top of the v6 probe knobs.

Base knobs come from ``TRIMEM_TRAJ_KNOBS`` (``seed_count``, ``depth``, ``fanout``, ``top_k``,
same contract as ``trajectory_param_cell``). The v7 knobs come from ``TRIMEM_TRAJ_V7`` (JSON):

- ``edges``: edge types expanded per hop, ``L+Cu+S`` (lineage, context_up, similar; default),
  ``L+S``, ``L`` or ``S``.
- ``ordering``: which rows of the bounded pool (at most 64) are kept. ``stream`` (default for
  VGR/P1/P2: SQL order, early termination), ``score`` (default for VGR+SemRank: sem_score desc,
  VGR order on ties), ``gain`` (fitness - parent_fitness desc) and ``gain_x_score``
  (gain * (0.5 + sem_score), VGR+SemRank only; ``trimem.optimizer.packing.gain_key``).
- ``dedup``: ``id`` (default), ``code_hash`` (skip byte-identical code), ``per_seed_cap1`` (at
  most one endpoint per vector seed) or ``mmr0.7`` (maximal marginal relevance over the pool,
  lambda 0.7, cosine on the full-code embedding, relevance = similarity to the parent).
- ``render``: ``full_code`` (default), ``diff_vs_parent`` (unified diff from the attempt's own
  parent to the attempt) or ``intent_only`` (intent card, code omitted).
- ``cadence``: retrieve every ``cadence`` iterations (default 1); in between, the last
  retrieved set is reused without any SQL or semantic call.
- ``sem_fallback``: ``none`` (default) or ``vgr`` (P1/P2 only): when SemFilter leaves nothing to
  inject, the VGR rows of the same knobs are injected instead.

With every v7 knob at its default the selected programs equal the v6 cell's (tested). The
retriever is replaced before ``trajectory_cell`` instruments it, so the audit files are the
same as for every other cell. Shared modules are not edited.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import time
from copy import copy
from dataclasses import asdict
from itertools import islice
from typing import Any

from trimem import trajectory_query as tq

VERSION = "harness-space-v7"
BASE_DEFAULTS = {"seed_count": 24, "depth": 1, "fanout": 4, "top_k": 4}
EDGE_SETS = {
    "L+Cu+S": ("lineage", "context_up", "similar"),
    "L+S": ("lineage", "similar"),
    "L": ("lineage",),
    "S": ("similar",),
}
V7_DEFAULTS = {
    "edges": "L+Cu+S",
    "ordering": None,
    "dedup": "id",
    "render": "full_code",
    "cadence": 1,
    "sem_fallback": "none",
}
V7_VALUES = {
    "edges": tuple(EDGE_SETS),
    "ordering": ("stream", "score", "gain", "gain_x_score"),
    "dedup": ("id", "code_hash", "per_seed_cap1", "mmr0.7"),
    "render": ("full_code", "diff_vs_parent", "intent_only"),
    "cadence": (1, 5),
    "sem_fallback": ("none", "vgr"),
}
GRAPH_PLANS = ("VGR", "P1", "P2")
# Selection semantics (2026-10-15, user decision): every seed is expanded, the complete result
# pool is drained, then ranked (SemRank score when the plan has one, else gain) and the top-k
# taken. This replaces streaming early termination (LIMIT 64 + stop at k). The pool is bounded by
# seed_count x sum_h (|edges| x fanout)^h; FULL_POOL_CAP is a safety stop that is reported.
SELECTION = "full-pool-v1"
FULL_POOL_CAP = 16384  # measured max 8,038 rows (seeds 48, depth 4, fanout 16)
INTENT_ONLY_CODE = "(code omitted: intent-only memory card)"


def base_knobs(raw: str | None = None) -> dict[str, int]:
    raw = json.loads(
        raw if raw is not None else os.environ.get("TRIMEM_TRAJ_KNOBS") or "{}"
    )
    unknown = set(raw) - set(BASE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown trajectory knobs: {sorted(unknown)}")
    out = BASE_DEFAULTS | raw
    for name, value in out.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"knob {name} must be a positive integer")
    if out["depth"] > 4 or out["top_k"] > 16 or out["seed_count"] > 64:
        raise ValueError("knob out of the supported range")
    return out


def default_ordering(plan: str) -> str:
    return "score" if plan == "VGR+SemRank" else "stream"


def v7_knobs(plan: str, raw: str | None = None) -> dict[str, Any]:
    """Validated v7 knobs for ``plan`` (defaults filled; plan-conditional values enforced)."""
    raw = json.loads(
        raw if raw is not None else os.environ.get("TRIMEM_TRAJ_V7") or "{}"
    )
    unknown = set(raw) - set(V7_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown v7 knobs: {sorted(unknown)}")
    out = V7_DEFAULTS | raw
    if out["ordering"] is None:
        out["ordering"] = default_ordering(plan)
    for name, allowed in V7_VALUES.items():
        if out[name] not in allowed or isinstance(out[name], bool):
            raise ValueError(f"v7 knob {name}={out[name]!r} not in {allowed}")
    semrank = plan == "VGR+SemRank"
    if semrank and out["ordering"] == "stream":
        raise ValueError("VGR+SemRank orders by score, gain or gain_x_score")
    if not semrank and out["ordering"] in ("score", "gain_x_score"):
        raise ValueError(
            f"ordering {out['ordering']} needs sem_score (VGR+SemRank only)"
        )
    if out["sem_fallback"] != "none" and plan not in ("P1", "P2"):
        raise ValueError("sem_fallback applies to P1/P2 only")
    return out


def expansion(depth: int, fanout: int, edges: tuple[str, ...]) -> str:
    """LATERAL expansion of each seed: hop i expands every edge type in ``edges`` from hop i-1's
    endpoints. Full-pool selection drains the result, so each hop's frontier is de-duplicated
    (DISTINCT dst) before the next hop: per-seed work is bounded by the reachable set instead of
    the number of paths (depth 4 x fanout 16 produced 1.9M duplicate path rows without it).
    """

    def hop(src: str) -> str:
        return "\n  UNION ALL ".join(
            f"SELECT dst FROM trimem.expand({src},'{edge}',{fanout})" for edge in edges
        )

    level = f"SELECT DISTINCT h1.dst FROM (\n  {hop('s.vid')}\n) h1"
    for i in range(2, depth + 1):
        level = (
            f"SELECT DISTINCT h{i}.dst FROM ({level}) p{i}\n"
            f"CROSS JOIN LATERAL (\n  {hop(f'p{i}.dst')}\n) h{i}"
        )
    return f"CROSS JOIN LATERAL (\n{level}\n) e"


CANONICAL_EXPANSION = (
    "CROSS JOIN LATERAL (\n  "
    + "\n  UNION ALL ".join(
        f"SELECT dst FROM trimem.expand(s.vid,'{edge}',4)"
        for edge in ("lineage", "context_up", "similar")
    )
    + "\n) e"
)


def v7_query(plan: str, track: str, k: dict[str, int], edges: str) -> str:
    """``tq.query`` with seed LIMIT, hop depth/fanout and edge set replaced and the result-row
    ``LIMIT 64`` removed (full-pool selection), as a checked rewrite."""
    sql = tq.query(plan, track)
    if "LIMIT 64" not in sql:
        raise RuntimeError("canonical result LIMIT 64 not found")
    sql = sql.replace("\nLIMIT 64", "\n").replace(" LIMIT 64", "")
    if "LIMIT 64" in sql:
        raise RuntimeError("result LIMIT 64 left in the rewritten query")
    seed_limit = "LIMIT 24"
    if sql.count(seed_limit) != 1:
        raise RuntimeError("canonical seed LIMIT not found exactly once")
    sql = sql.replace(seed_limit, f"LIMIT {k['seed_count']}")
    if plan.split("+")[0] in GRAPH_PLANS:
        canonical = CANONICAL_EXPANSION
        if sql.count(canonical) != 1:
            raise RuntimeError("canonical expansion not found exactly once")
        sql = sql.replace(
            canonical, expansion(k["depth"], k["fanout"], EDGE_SETS[edges])
        )
    return sql


def gain(row: dict[str, Any]) -> float:
    f, p = row.get("fitness"), row.get("parent_fitness")
    if f is None or p is None:
        return float("-inf")
    value = float(f) - float(p)
    return value if math.isfinite(value) else float("-inf")


BLOCKING = {
    "Sort",
    "Incremental Sort",
    "Hash",
    "Hash Join",
    "Materialize",
    "Aggregate",
    "Gather Merge",
}


def plan_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Physical plan nodes for the receipt. Full-pool selection drains the result anyway, so a
    blocking node is recorded (``blocking: true``) instead of failing the retrieval."""
    out: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        rec = {
            k: node[k]
            for k in ("Node Type", "Index Name", "Custom Plan Provider")
            if k in node
        }
        if node.get("Node Type") in BLOCKING:
            rec["blocking"] = True
        out.append(rec)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return out


SEMRANK_PLANS = ("VGR+SemRank", "SemRankOnly", "P1+Rank", "P2+Rank")


def rank_pool(
    rows: list[dict[str, Any]], ordering: str, plan: str
) -> list[dict[str, Any]]:
    """Full-pool ranking: SemRank score for a SemRank plan, else gain; ``gain_x_score`` keeps its
    product rule. Ties keep stream position."""
    if ordering == "gain_x_score":
        return order_pool(rows, ordering)
    if plan in SEMRANK_PLANS:
        for r in rows:
            v = r["sem_score"]
            if (
                not isinstance(v, (float, int))
                or not math.isfinite(v)
                or not 0 <= v <= 1
            ):
                raise ValueError("SemRank needs a finite score in [0, 1]")
        return sorted(rows, key=lambda r: (-float(r["sem_score"]), r["pool_position"]))
    return order_pool(rows, "gain")


def order_pool(rows: list[dict[str, Any]], ordering: str) -> list[dict[str, Any]]:
    """Stable client-side order of a drained pool (rows carry ``pool_position``)."""
    if ordering in ("stream", "score"):
        return list(rows)
    if ordering == "gain":
        return sorted(rows, key=lambda r: (-gain(r), r["pool_position"]))
    if ordering == "gain_x_score":

        def key(r):
            g = gain(r)
            score = r.get("sem_score")
            return (
                -(g * (0.5 + (0.5 if score is None else float(score)))),
                r["pool_position"],
            )

        return sorted(rows, key=key)
    raise ValueError(f"unknown ordering {ordering}")


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb))


def mmr_order(
    rows: list[dict[str, Any]],
    query_vec: list[float],
    embs: dict[int, list[float]],
    lam: float = 0.7,
) -> list[dict[str, Any]]:
    """Greedy MMR: argmax lam*sim(q, d) - (1-lam)*max_{s in chosen} sim(d, s).

    Rows without an embedding keep their relative order after every embedded row.
    """
    pending = [r for r in rows if r["id"] in embs]
    missing = [r for r in rows if r["id"] not in embs]
    rel = {r["id"]: _cos(query_vec, embs[r["id"]]) for r in pending}
    chosen: list[dict[str, Any]] = []
    while pending:

        def score(r):
            red = max((_cos(embs[r["id"]], embs[c["id"]]) for c in chosen), default=0.0)
            return (lam * rel[r["id"]] - (1 - lam) * red, -r["pool_position"])

        best = max(pending, key=score)
        chosen.append(best)
        pending.remove(best)
    return chosen + missing


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def render_code(
    row: dict[str, Any], render: str, parent_code: str | None, language: str
) -> tuple[str, str]:
    """(code shown in the inspiration block, fence language)."""
    if render == "full_code":
        return row["code"], language
    if render == "intent_only":
        return INTENT_ONLY_CODE, "text"
    if render == "diff_vs_parent":
        if parent_code is None:
            return row[
                "code"
            ], language  # no parent code recorded: fall back to full code
        diff = "".join(
            difflib.unified_diff(
                parent_code.splitlines(keepends=True),
                row["code"].splitlines(keepends=True),
                fromfile="parent",
                tofile="inspiration",
                n=3,
            )
        )
        return (diff or row["code"]), ("diff" if diff else language)
    raise ValueError(f"unknown render {render}")


def _vec(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")] if text else []


class Selection:
    """Dedup + token-budget packing shared by the streaming and pooled paths."""

    def __init__(
        self, retriever, *, k: int, dedup: str, render: str, task_uid: str, params: dict
    ):
        from bench.agent_memory.gem_oe.trajectory_prompt import SECTION
        from trimem.objective.store import count_tokens

        self.r, self.k, self.dedup, self.render = retriever, k, dedup, render
        self.task_uid, self.params = task_uid, params
        self.section, self.count_tokens = SECTION, count_tokens
        self.selected, self.blocks, self.rows = [], [], []
        self.seen_ids, self.seen_hash, self.seen_seed = set(), set(), set()
        self.used = self.skipped = self.rows_seen = 0
        self.dedup_skips = 0

    def offer(self, row: dict[str, Any]) -> bool:
        """Consider one row; True once k programs are selected."""
        from bench.agent_memory.gem_oe.memory_database import RetrievedProgram
        from bench.agent_memory.gem_oe.trajectory_prompt import render_inspiration
        from bench.agent_memory.gem_oe.trajectory_retriever import evidence_card

        self.rows_seen += 1
        if row["id"] in self.seen_ids:
            return False
        self.seen_ids.add(row["id"])
        card = evidence_card(row)
        if not row.get("code"):
            return False
        digest = code_sha(row["code"])
        if self.dedup == "code_hash" and digest in self.seen_hash:
            self.dedup_skips += 1
            return False
        seed = row.get("seed_id")
        if (
            self.dedup == "per_seed_cap1"
            and seed is not None
            and seed in self.seen_seed
        ):
            self.dedup_skips += 1
            return False
        language = row.get("language") or (
            "cpp" if self.task_uid.startswith("ale:") else "python"
        )
        parent_code = (
            self.r._parent_code(row["id"]) if self.render == "diff_vs_parent" else None
        )
        shown, fence = render_code(row, self.render, parent_code, language)
        block = render_inspiration(len(self.selected) + 1, shown, card, fence)
        tokens = self.count_tokens(self.section + "\n\n".join(self.blocks + [block]))
        if tokens > self.r.memory_tokens:
            self.skipped += 1
            return False
        self.seen_hash.add(digest)
        if seed is not None:
            self.seen_seed.add(seed)
        self.selected.append(
            RetrievedProgram(
                uid=row["uid"],
                code=shown,
                language=fence,
                metrics={"combined_score": float(row["fitness"])},
                changes_description=card,
                provenance={
                    "memory": "trajectory",
                    "attempt_id": row["id"],
                    "context_sha256": self.params["current_context_sha256"],
                    "assembly": f"{VERSION}:{self.render}",
                    "source_code_sha256": digest,
                    "retrieved_fields": {
                        key: value
                        for key, value in row.items()
                        if key not in ("code", "evidence")
                    },
                },
            )
        )
        self.rows.append(row)
        self.blocks.append(block)
        self.used = tokens
        return len(self.selected) >= self.k


def install() -> None:
    """Replace TrajectoryRetriever.retrieve with the v7 retrieval (idempotent)."""
    from bench.agent_memory.gem_oe import memory_database, trajectory_prompt
    from bench.agent_memory.gem_oe import trajectory_retriever as tr

    if getattr(tr.TrajectoryRetriever, "_v7_knobs", None) is not None:
        return
    base = base_knobs()

    def _parent_code(self, attempt_id):
        row = self.conn.execute(
            "SELECT pa.code FROM attempt a JOIN attempt pa ON pa.id=a.parent_id WHERE a.id=%s",
            (attempt_id,),
        ).fetchone()
        return row["code"] if row else None

    def _embeddings(self, ids):
        if not ids:
            return {}
        rows = self.conn.execute(
            "SELECT id, emb::text AS emb FROM attempt WHERE id = ANY(%s)", (list(ids),)
        ).fetchall()
        return {r["id"]: _vec(r["emb"]) for r in rows if r["emb"]}

    def retrieve(self, *, task_uid, parent, k, iteration):
        return retrieve_v7(
            self, task_uid=task_uid, parent=parent, k=k, iteration=iteration, base=base
        )

    tr.TrajectoryRetriever._parent_code = _parent_code
    tr.TrajectoryRetriever._embeddings = _embeddings
    tr.TrajectoryRetriever.retrieve = retrieve
    trajectory_prompt.MAX_INSPIRATIONS = base["top_k"]
    original_sample = memory_database.GemMemoryDatabase.sample_from_island

    def sample_from_island(self, island_id, num_inspirations=None):
        # Inspiration slots = top_k (run_arm pins num_diverse_programs to 4 for this store).
        self._max_injected = base["top_k"]
        return original_sample(self, island_id, base["top_k"])

    memory_database.GemMemoryDatabase.sample_from_island = sample_from_island
    tr.TrajectoryRetriever._param_knobs = base
    tr.TrajectoryRetriever._v7_knobs = True


def retrieve_v7(self, *, task_uid, parent, k, iteration, base, v7=None, plan=None):  # noqa: C901 - mirrors tr.retrieve
    """``v7``/``plan`` override the process-level knobs for this call (per-iteration router)."""
    from bench.agent_memory.gem_oe import trajectory_retriever as tr
    from trimem.code_embedding import FullCodeEmbedder

    if k <= 0:
        return []
    plan = plan or self.plan
    v7 = v7_knobs(plan) if v7 is None else v7_knobs(plan, json.dumps(v7))
    config_key = json.dumps({"plan": plan, **base, **v7}, sort_keys=True)
    requested_k = k
    k = min(k, base["top_k"])
    cache = getattr(self, "_v7_cache", None)
    if (
        v7["cadence"] > 1
        and cache
        and cache.get("config") == config_key
        and iteration - cache["iteration"] < v7["cadence"]
    ):
        selected = list(cache["selected"])
        self.last_inspirations = selected
        receipt = dict(
            cache["receipt"],
            iteration=iteration,
            parent_id=parent.id,
            cache_hit=True,
            cached_from_iteration=cache["iteration"],
        )
        self.telemetry.append(receipt)
        if self.out:
            with (self.out / "trajectory_retrieval.jsonl").open("a") as f:
                f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        return selected
    task = self.conn.execute(
        "SELECT specification FROM task WHERE task=%s", (task_uid,)
    ).fetchone()
    if not task or not task["specification"]:
        raise ValueError("trajectory retrieval needs the actual task specification")
    if self.database is not None:
        artifacts = self.database.get_artifacts(parent.id)
        if artifacts:
            parent = copy(parent)
            parent.metadata = dict(parent.metadata or {}) | {
                "evaluation_feedback": artifacts
            }
    context = tr.parent_context(
        task_uid,
        task["specification"],
        parent,
        iteration,
        self._remaining(),
        self.objective,
    )
    if self.embedder is None:
        self.embedder = FullCodeEmbedder()
    # Plain floats: receipts are serialized by run_arm with json.dumps (no numpy types).
    vector = [float(x) for x in self.embedder.embed(context.program_code)]
    params = tq.bind_context(
        {
            "task": task_uid,
            "exclude_session": None,
            "q": "[" + ",".join(map(str, vector)) + "]",
            "predicate": tq.FILTER_CRITERION,
            "score_predicate": tq.SCORE_CRITERION,
        },
        context,
    )
    tq.check_request_size(
        params["score_predicate"], params["current_context"], self.max_bytes
    )
    if self.out:
        folder = self.out / "retrieval_contexts"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{iteration:04d}.json").write_text(
            json.dumps(asdict(context), ensure_ascii=False, indent=2)
        )

    def run(plan):
        sql = v7_query(plan, "emb", base, v7["edges"])
        planned = self.conn.execute(
            "EXPLAIN (FORMAT JSON) DECLARE trajectory_check CURSOR FOR " + sql, params
        ).fetchone()["QUERY PLAN"][0]["Plan"]
        nodes = plan_nodes(planned)
        sel = Selection(
            self,
            k=k,
            dedup=v7["dedup"],
            render=v7["render"],
            task_uid=task_uid,
            params=params,
        )
        pool, seen, streamed = [], set(), 0
        with (
            self.conn.transaction(),
            self.conn.cursor(
                name=f"trajectory_{iteration}_{plan.replace('+', '_')}"
            ) as cur,
        ):
            cur.itersize = 256
            cur.execute(sql, params)
            for row in islice(cur, FULL_POOL_CAP + 1):
                streamed += 1
                if streamed > FULL_POOL_CAP or row["id"] in seen or not row.get("code"):
                    continue
                seen.add(row["id"])
                pool.append(dict(row, pool_position=streamed - 1))
        ordered = rank_pool(pool, v7["ordering"], plan)
        if v7["dedup"] == "mmr0.7":
            ordered = mmr_order(
                ordered, vector, self._embeddings([r["id"] for r in ordered])
            )
        for row in ordered:
            if sel.offer(row):
                break
        pool_size = {
            "distinct": len(pool),
            "rows_streamed": min(streamed, FULL_POOL_CAP),
            "truncated": streamed > FULL_POOL_CAP,
        }
        return sel, nodes, pool_size

    started = time.perf_counter()
    sel, nodes, pool_size = run(plan)
    fallback_used, primary = False, None
    # Fallback only when SemFilter passed nothing (zero rows reached packing), not when rows
    # passed but none fit the memory budget.
    if sel.rows_seen == 0 and v7["sem_fallback"] == "vgr":
        primary = {
            "rows_seen": sel.rows_seen,
            "skipped_for_budget": sel.skipped,
            "pool_size": pool_size,
        }
        sel, nodes, pool_size = run("VGR")
        fallback_used = True
    selected = sel.selected
    self.last_inspirations = selected
    # Sub-metrics (diagnostics only): similarity of the injected code to the parent.
    sims = []
    embs = self._embeddings([r["id"] for r in sel.rows])
    for r in sel.rows:
        if r["id"] in embs:
            sims.append(_cos(vector, embs[r["id"]]))
    seeds = [r.get("seed_id") for r in sel.rows if r.get("seed_id") is not None]
    hashes = [code_sha(r["code"]) for r in sel.rows]
    receipt = {
        "iteration": iteration,
        "parent_id": parent.id,
        "plan": plan,
        "space_version": VERSION,
        "v7_knobs": v7,
        "base_knobs": base,
        "returned": len(selected),
        "requested_k": requested_k,
        "effective_k": k,
        "max_inspiration_programs": base["top_k"],
        "context_sha256": params["current_context_sha256"],
        "embedding": self.embedder.last_receipt,
        "memory_tokens": sel.used,
        "memory_token_budget": self.memory_tokens,
        "skipped_for_budget": sel.skipped,
        "dedup_skips": sel.dedup_skips,
        "truncated_records": 0,
        "post_retrieval_summary_llm_calls": 0,
        "rows_seen": sel.rows_seen,
        "pool_size": pool_size,
        "selection": SELECTION,
        "physical_plan_nodes": nodes,
        "fallback_used": fallback_used,
        "primary_plan_stats": primary,
        "cache_hit": False,
        "selected_attempt_ids": [r["id"] for r in sel.rows],
        "selected_gains": [gain(r) for r in sel.rows],
        "selected_sem_scores": [r.get("sem_score") for r in sel.rows],
        "selected_seed_ids": seeds,
        "distinct_seed_share": (len(set(seeds)) / len(seeds)) if seeds else None,
        "duplicate_code_share": (1 - len(set(hashes)) / len(hashes))
        if hashes
        else None,
        "mean_sim_to_parent": (sum(sims) / len(sims)) if sims else None,
        "v7_select_ms": (time.perf_counter() - started) * 1000,
        "remaining_token_budget": context.remaining_token_budget,
        "remaining_budget_scope": "generation LLM usage; excludes semantic retrieval calls",
    }
    self.telemetry.append(receipt)
    if self.out:
        with (self.out / "trajectory_retrieval.jsonl").open("a") as f:
            f.write(json.dumps(receipt, ensure_ascii=False, default=str) + "\n")
    if v7["cadence"] > 1:
        self._v7_cache = {
            "iteration": iteration,
            "selected": list(selected),
            "receipt": receipt,
            "config": config_key,
        }
    return selected
