"""Per-stage sub-metrics of a profiled configuration (agent-env datasets), for the LLM analyzer.

RAG-PE reads stage diagnostics (context recall / precision, faithfulness, the bound stage) to
propose targeted edits; they steer candidate generation only and are never GP inputs or
objectives. This module derives the equivalent diagnostics for trajectory retrieval from what
every profiled episode already stores in ``opt.session_retrieval`` (retrieval receipt, generation
record, query context): panel means per configuration.
"""

from __future__ import annotations

import statistics as st

from trimem.experiments.semantic_rubric import FILTER_THRESHOLD

DESCRIPTIONS = {
    "pool_size": "seed: candidate sessions in the bounded pool after seeding and expansion (coverage)",
    "seed_sessions": "seed: sessions attached to the vector-seed tasks before expansion",
    "sel_cosine": "select: mean cosine between the query task and the injected sessions' tasks (relevance)",
    "type_match": "select: share of injected sessions with the query's task type (ALFWorld task family / "
    "ScienceWorld task) - a proxy for context precision",
    "object_match": "select: ALFWorld only - share of injected sessions that manipulate the query's object",
    "card_steps": "select: mean length (steps) of the injected trajectories",
    "empty_rate": "select: share of queries that received no memory",
    "filter_keep": "filter: share of judged candidates the SemFilter kept (P2 plans only)",
    "sem_tokens": "filter: semantic-operator tokens per query",
    "memory_chars": "prompt: characters of rendered memory injected per query",
    "llm_calls": "generate: LLM calls (agent turns) per episode; fewer = the agent finished faster",
    "max_round_rate": "generate: share of episodes that hit the run's turn limit (max_round; 30 unless the run "
    "records episode_budget max-round-v1) (stuck / looping agent)",
    "tokens_per_call": "generate: generation tokens per LLM call (prompt growth)",
}


def _query_type(dataset: str, task_key: str) -> tuple[str, str | None]:
    """(task type, object) of the query task, parsed from its store key."""
    if (
        dataset == "alfworld"
    ):  # alfworld:train/<type>-<Object>-<Mrecep>-<Recep>-<scene>/<trial>
        family = task_key.split("/")[1]
        parts = family.split("-")
        return parts[0], parts[1]
    return task_key.split(":", 1)[1].split("/")[
        0
    ], None  # sciworld:<task name>/<variation>


def episode_submetrics(
    row: dict, dataset: str, max_round: int = 30
) -> dict[str, float]:
    """``max_round``: the run's turn limit (max-round-v1; default 30 = unchanged); a longer recorded episode is
    counted at the limit (its derived-truncation view)."""
    rec, gen = row["receipt"], row["generation"] or {}
    cards = rec.get("cards") or []
    qtype, qobj = _query_type(dataset, row["query_context"]["task_key"])
    out = {
        "pool_size": float(len(rec.get("pool_ids") or [])),
        "seed_sessions": float(len(rec.get("seed_ids") or [])),
        "empty_rate": float(not cards),
        "sem_tokens": float(gen.get("semantic_tokens") or 0),
        "memory_chars": float(gen.get("memory_chars") or 0),
        "llm_calls": float(min(gen.get("llm_calls") or 0, max_round)),
        "max_round_rate": float((gen.get("llm_calls") or 0) >= max_round),
    }
    if gen.get("llm_calls"):
        out["tokens_per_call"] = (
            float(gen.get("generation_tokens") or 0) / gen["llm_calls"]
        )
    if cards:
        out["sel_cosine"] = st.mean(float(c.get("cosine") or 0) for c in cards)
        out["type_match"] = st.mean(float(c.get("task_type") == qtype) for c in cards)
        out["card_steps"] = st.mean(float(c.get("n_steps") or 0) for c in cards)
        if qobj is not None:
            out["object_match"] = st.mean(
                float(str(c.get("object") or "").lower() == qobj.lower()) for c in cards
            )
    dec = rec.get("semantic_decisions") or []
    if dec:
        out["filter_keep"] = st.mean(
            float((d.get("score") or 0) >= FILTER_THRESHOLD) for d in dec
        )
    return out


def config_submetrics(
    rows: list[dict], dataset: str, max_round: int = 30
) -> dict[str, float]:
    """Panel means of the episode sub-metrics (keys present in at least half of the episodes)."""
    per = [episode_submetrics(r, dataset, max_round) for r in rows]
    keys = {k for p in per for k in p}
    return {
        k: st.mean(p[k] for p in per if k in p)
        for k in sorted(keys)
        if sum(k in p for p in per) >= len(per) / 2
    }
