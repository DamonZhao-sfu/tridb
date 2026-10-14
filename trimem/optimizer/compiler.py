"""Compile one selected, explicitly versioned native streaming retrieval strategy.

By default the graph path emits endpoints at exactly ``len(hop_types)`` hops;
``frontier="all"`` streams the union of every depth (no dedup, depth-first per
seed), not the historical sorted P1/P2 queries. Graph topology
is accessed exclusively through the in-process adjacency ``trimem.expand``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .schema import EDGE_TYPES, CandidateSpec, expansion_bound, hop_edges


@dataclass(frozen=True)
class QueryContext:
    task: str
    query_vector: tuple[float, ...]
    parent_id: int | None = None
    cell: str | None = None
    fitness_floor: float | None = None

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be nonempty")
        if not self.query_vector or not all(
            math.isfinite(v) for v in self.query_vector
        ):
            raise ValueError("query_vector must contain finite coordinates")
        if self.fitness_floor is not None and not math.isfinite(self.fitness_floor):
            raise ValueError("fitness_floor must be finite or None")


@dataclass(frozen=True)
class NativePlanSpec:
    action: CandidateSpec
    sql: str | None
    params: dict[str, Any]
    settings: tuple[tuple[str, str], ...]
    emitted_edge_upper_bound: int
    expected_custom_nodes: tuple[str, ...]
    capability_version: str = "native_endpoint_stream_v1"

    @property
    def sql_hash(self) -> str:
        raw = json.dumps(
            {"sql": self.sql, "params": self.params, "settings": self.settings},
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()


def expand_node_count(action: CandidateSpec) -> int:
    """Physical ``TriMEM Expand`` nodes: one per edge type per hop."""
    if action.plan_id == "no_memory_v1":
        return 0
    return sum(len(hop_edges(hop)) for hop in action.hop_types)


def _hop_from(source: str, hop: str, fanout: int) -> str:
    """One hop as a lateral FROM item exposing ``dst`` (alias added by caller)."""
    edges = hop_edges(hop)
    # CandidateSpec validates each hop against a finite registered catalog.
    if any(edge not in EDGE_TYPES for edge in edges) or len(set(edges)) != len(edges):
        raise ValueError(f"unsupported native edge type: {hop}")
    if len(edges) == 1:
        return f"trimem.expand({source}, '{hop}', {fanout})"
    # Union hop: one streaming Append of per-type Expand nodes. UNION ALL only;
    # UNION would add a blocking dedup (HashAgg/Sort+Unique).
    branches = "\n  UNION ALL\n  ".join(
        f"SELECT dst FROM trimem.expand({source}, '{edge}', {fanout})" for edge in edges
    )
    return f"(\n  {branches}\n)"


def _all_depths_sql(action: CandidateSpec) -> str:
    """frontier="all": emit every depth 1..d in one streaming nested-lateral chain.

    After hop i, ``f{i}`` yields the hop-i endpoint itself, then (UNION ALL)
    everything reachable from it in hops i+1..d. Each depth is expanded once
    per parent, exactly as in the final-frontier chain, so Expand nodes and the
    emitted-edge bound are identical; only the hop-i endpoints are also kept.
    Output is depth-first per seed; there is no DISTINCT/sort/hash.
    """
    hops, fanout = action.hop_types, action.fanout

    def keep_and_descend(index: int) -> str:
        # Hop-index endpoint itself, then everything below it.
        return (
            f"\nSELECT e{index}.dst AS dst\nUNION ALL\n"
            f"{descend(index + 1, f'e{index}.dst')}\n"
        )

    def descend(index: int, source: str) -> str:
        expand = f"{_hop_from(source, hops[index - 1], fanout)} e{index}"
        if index == len(hops):
            return f"SELECT e{index}.dst AS dst FROM {expand}"
        return (
            f"SELECT f{index}.dst AS dst FROM {expand}\n"
            f"CROSS JOIN LATERAL ({keep_and_descend(index)}) f{index}"
        )

    first = f"CROSS JOIN LATERAL {_hop_from('s.vid', hops[0], fanout)} e1"
    return f"{first}\nCROSS JOIN LATERAL ({keep_and_descend(1)}) f1"


def compile_candidate(
    action: CandidateSpec, context: QueryContext | None = None
) -> NativePlanSpec:
    """Pure compilation: no candidate execution, DB access, or model request.

    SQL barriers preserve the *logical* seed/filter/endpoint order; the actual
    EXPLAIN tree must additionally pass ``check_streaming_plan`` before Open.
    ``edge_emit_cap`` bounds matched/emitted edges, not physical adjacency slot
    inspections (the native pull API can skip nonmatching/dead edge slots).
    """
    # Also validates objects that callers reconstructed outside the space builder.
    action = CandidateSpec.from_dict(action.to_dict())
    if action.plan_id == "no_memory_v1":
        return NativePlanSpec(action, None, {}, (), 0, ())
    if context is None:
        raise ValueError("retrieval requires QueryContext")

    edge_bound = expansion_bound(action.seed_count, action.fanout, action.hop_types)
    if edge_bound > action.edge_emit_cap:
        raise ValueError(
            f"conservative emitted-edge bound {edge_bound} exceeds {action.edge_emit_cap}; "
            "reduce seeds/fanout/hops or explicitly increase edge_emit_cap"
        )

    params = {
        "task": context.task,
        "cell": context.cell,
        "parent_id": context.parent_id,
        "floor": context.fitness_floor,
        "q": "[" + ",".join(format(v, ".17g") for v in context.query_vector) + "]",
        "filter_predicate": action.filter_predicate,
        "score_predicate": action.score_predicate,
        "keep_unknown": action.unknown_policy == "keep",
    }
    # min_gain > 0 tightens "improving" against float-noise gains; absent (0)
    # keeps the exact pre-v5 SQL text and parameters.
    gain_floor = ""
    if action.min_gain > 0:
        params["min_gain"] = float(action.min_gain)
        gain_floor = " + %(min_gain)s::double precision"
    # LIMIT, OFFSET 0 and one-row semantic batches are intentional streaming
    # barriers, not materialized CTEs. ANN ORDER BY must use an ordered index.
    # Seeds must carry a change summary and be an improving edit -- valid and
    # strictly better than their own parent (the telemetry rule applied to
    # endpoints) -- for every plan. This excludes no-op edits (code
    # byte-identical to the parent's), so a seed SemFilter never judges an
    # empty/"diff is empty" text, and dead all-zero lineages, whose 1-hop
    # lineage frontier is empty. Roots (no parent) are never seeds. The HNSW
    # iterative scan (strict_order, see settings) keeps exact distance order
    # under the filter.
    seeds = f"""SELECT a.id AS sid, a.vid, a.changes
FROM attempt a
WHERE a.task = %(task)s AND (a.cell IS NULL OR a.cell = %(cell)s)
  AND a.emb IS NOT NULL AND a.changes IS NOT NULL AND a.is_valid
  AND a.code <> (SELECT pa.code FROM attempt pa WHERE pa.id = a.parent_id)
  AND a.fitness > (SELECT pa.fitness FROM attempt pa WHERE pa.id = a.parent_id){gain_floor}
  AND (%(parent_id)s::bigint IS NULL OR a.id <> %(parent_id)s)
ORDER BY a.emb <=> %(q)s::vector LIMIT {action.seed_count}"""
    if action.filter_target == "seed":
        seeds = f"""SELECT seed.* FROM ({seeds}) seed
WHERE trimem.sem_filter(%(filter_predicate)s, seed.changes, %(keep_unknown)s)
OFFSET 0"""

    if action.returns_all_depths:
        expand_sql, source = _all_depths_sql(action), "f1.dst"
    else:
        expansions = []
        source = "s.vid"
        for index, hop in enumerate(action.hop_types, 1):
            expansions.append(
                f"CROSS JOIN LATERAL {_hop_from(source, hop, action.fanout)} e{index}"
            )
            source = f"e{index}.dst"
        expand_sql = "\n".join(expansions)
    rows = f"""SELECT d.id, d.fitness, p.fitness AS parent_fitness,
       d.changes, d.code, s.sid AS seed_id
FROM ({seeds}) s
{expand_sql}
JOIN attempt d ON d.vid = {source}
JOIN attempt p ON p.id = d.parent_id
WHERE d.task = %(task)s AND d.is_valid AND d.fitness > p.fitness{gain_floor}
  AND (%(floor)s::double precision IS NULL OR d.fitness >= %(floor)s)
  AND (d.cell IS NULL OR d.cell = %(cell)s)
  AND d.code IS NOT NULL
LIMIT {action.candidate_visit_cap}"""
    if action.filter_target == "row":
        rows = f"""SELECT endpoint.* FROM ({rows}) endpoint
WHERE trimem.sem_filter(%(filter_predicate)s, endpoint.changes, %(keep_unknown)s)
OFFSET 0"""
    if action.uses_score:
        rows = f"""SELECT evidence.*,
       trimem.sem_score(%(score_predicate)s, evidence.changes) AS sem_score
FROM ({rows}) evidence
LIMIT {action.candidate_visit_cap}"""

    settings = (
        ("enable_hashjoin", "off"),
        ("enable_mergejoin", "off"),
        ("enable_material", "off"),
        ("enable_memoize", "off"),
        ("enable_seqscan", "off"),
        ("max_parallel_workers_per_gather", "0"),
        ("join_collapse_limit", "1"),
        ("from_collapse_limit", "1"),
        ("plan_cache_mode", "force_custom_plan"),
        ("hnsw.iterative_scan", "strict_order"),
        ("trimem.min_batch", "1"),
        ("trimem.max_batch", "1"),
        ("trimem.placement", "top"),
        ("trimem.force_placement", ""),
        ("trimem.prompt_version", action.predicate_version),
    )
    expected = ["TriMEM Expand"] * expand_node_count(action)
    if action.filter_target:
        expected.append("TriMEM SemFilter")
    if action.uses_score:
        expected.append("TriMEM SemScore")
    return NativePlanSpec(action, rows, params, settings, edge_bound, tuple(expected))
