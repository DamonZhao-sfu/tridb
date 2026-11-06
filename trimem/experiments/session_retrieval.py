"""Session-level, full-pool trajectory retrieval with dataset adapters.

Graph adjacency stays in TriDB. Bounded frontier deduplication and final ranking
are application operations, never blocking engine operators. A receipt retains
all candidate paths, exclusions and semantic decisions, including empty results.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from typing import Protocol

# Plan families (plan-p1rank-v1, user decision 2026-10-20). P1 / P1+Rank filter the vector SEEDS before graph
# expansion (descendants are not re-filtered); P2 / P2+Rank filter the final (capped) pool; "+Rank" plans order the
# (filtered) pool with the ``semrank`` scorer (ColBERT in the v2 spaces) before top-k.
SEED_FILTER_PLANS = frozenset({"P1", "P1+Rank"})
POOL_FILTER_PLANS = frozenset({"P2", "P2+Rank"})
SEMFILTER_PLANS = SEED_FILTER_PLANS | POOL_FILTER_PLANS
RANK_PLANS = frozenset({"VGR+SemRank", "VGR+Rank", "P2+Rank", "P1+Rank"})
SEM_FALLBACKS = (
    "none",
    "fill-topk",
)  # semfilter-v2b (user 2026-10-23): fill a short SemFilter stage to top_k
V2_PLANS = frozenset(
    {"VGR", "P2", "VGR+Rank", "P2+Rank", "P1+Rank", "P1"}
)  # P1: plan-p1-v1 (user 2026-10-22)
# Operator order recorded in every receipt (``plan_order``). V = vector seeds, SemFilter = LLM judge, G = graph
# expansion, R = relational isolation (seeds and every hop) + the v2 pool cap, Rank = semrank, TopK = selection.
PLAN_ORDER = {
    "VGR": ("V", "G", "R", "TopK"),
    "VGR+Rank": ("V", "G", "R", "Rank", "TopK"),
    "VGR+SemRank": ("V", "G", "R", "Rank", "TopK"),
    "P1": ("V", "SemFilter", "G", "R", "TopK"),
    "P1+Rank": ("V", "SemFilter", "G", "R", "Rank", "TopK"),
    "P2": ("V", "G", "R", "SemFilter", "TopK"),
    "P2+Rank": ("V", "G", "R", "SemFilter", "Rank", "TopK"),
}


@dataclass(frozen=True)
class SessionConfig:
    plan: str = "VGR"
    seed_count: int = 8
    depth: int = 1
    fanout: int = 4
    top_k: int = 3
    edges: tuple[str, ...] = ("similar",)

    def __post_init__(self):
        if self.plan not in {"VGR", "P1", "P2", "VGR+SemRank", "VGR+Rank", "P2+Rank"}:
            raise ValueError("Unsupported session plan")
        if not (
            1 <= self.seed_count <= 32
            and 1 <= self.depth <= 3
            and 1 <= self.fanout <= 16
            and 1 <= self.top_k <= 5
        ):
            raise ValueError("Configuration outside the bounded search domain")
        if not self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("At least one distinct graph relation is required")


# Session space v2 (agent-env pool profiling, user 2026-10-19): wider knob ranges plus protocol constants.
POOL_CAP = 256  # candidate pool after expansion, truncated by query cosine (fixed protocol constant, not searched)
EXPAND_LIMIT = (
    4096  # hard guard on the expanded pool (sessions); expansion stops adding beyond it
)
# Graph traversal algorithms (engine extensions v0.1.0 section 1, user 2026-11-03). Scoring signal = query cosine.
GRAPH_IMPLS = ("BFS", "best-first", "beam", "MCTS")
BEAM_WIDTHS = (1, 2, 4, 8)
MCTS_C = math.sqrt(2)  # UCB1 exploration constant


@dataclass(frozen=True)
class SessionConfigV2(SessionConfig):
    """v2 configuration: seed_count <= 128, depth <= 4, fanout <= 64, top_k <= 10, plus
    ``pool_cap`` (candidate pool truncated by query cosine before SemFilter / Rank / top-k),
    ``sim_gate`` (inject nothing when the best pool cosine is below it), ``max_failed`` (at most this many
    FAILED cards in the final top-k; None = no cap). v1 configurations (``SessionConfig``) are unchanged."""

    pool_cap: int = POOL_CAP
    expand_limit: int = EXPAND_LIMIT
    sim_gate: float | None = None
    max_failed: int | None = None
    # graph_impl: one of GRAPH_IMPLS (BFS = the original hop-by-hop expansion).
    # beam_width: required (in BEAM_WIDTHS) iff graph_impl == "beam", else None.
    graph_impl: str = "BFS"
    beam_width: int | None = None

    def __post_init__(self):
        if self.plan not in V2_PLANS:  # P1+Rank: plan-p1rank-v1 (user 2026-10-20)
            raise ValueError("Unsupported session plan")
        if not (
            1 <= self.seed_count <= 128
            and 1 <= self.depth <= 4
            and 1 <= self.fanout <= 64
            and 1 <= self.top_k <= 10
            and 1 <= self.pool_cap <= self.expand_limit
        ):
            raise ValueError("Configuration outside the bounded search domain")
        if self.max_failed is not None and not 0 <= self.max_failed <= self.top_k:
            raise ValueError("max_failed outside [0, top_k]")
        if self.sim_gate is not None and not -1.0 <= self.sim_gate <= 1.0:
            raise ValueError("sim_gate outside [-1, 1]")
        if not self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("At least one distinct graph relation is required")
        if self.graph_impl not in GRAPH_IMPLS:
            raise ValueError(f"Unsupported graph_impl {self.graph_impl!r}")
        if self.graph_impl == "beam":
            # type check first: True == 1 and 2.0 == 2 would otherwise pass the membership test
            if type(self.beam_width) is not int or self.beam_width not in BEAM_WIDTHS:
                raise ValueError(f"beam requires beam_width in {BEAM_WIDTHS}")
        elif self.beam_width is not None:
            raise ValueError("beam_width is only valid with graph_impl='beam'")


def select_top_k(
    candidates: list[dict], top_k: int, max_failed: int | None
) -> list[dict]:
    """First ``top_k`` ranked sessions; with ``max_failed`` a FAILED session (success is False) is skipped
    once that many failed sessions are already selected (successful / unknown ones are never skipped)."""
    if max_failed is None:
        return candidates[:top_k]
    out, failed = [], 0
    for s in candidates:
        if len(out) >= top_k:
            break
        if s.get("success") is False:
            if failed >= max_failed:
                continue
            failed += 1
        out.append(s)
    return out


class SessionAdapter(Protocol):
    def seeds(self, count: int) -> list[dict]: ...
    def neighbors(self, session: dict, edge: str, fanout: int) -> list[dict]: ...
    def allowed(self, session: dict) -> bool: ...
    def hydrate(self, session: dict) -> dict: ...


def _cos_key(s: dict) -> tuple:
    """Deterministic tie-break order for every non-BFS traversal: query cosine desc, then id asc."""
    return (-float(s["cosine"]), s["id"])


def _expand_node(
    adapter: SessionAdapter,
    config: SessionConfigV2,
    receipt: dict,
    pool: dict,
    source: dict,
    hop: int,
) -> tuple[list[dict], bool]:
    """One node expansion shared by best-first / beam / MCTS.

    Fetches ``fanout`` neighbors per edge (in ``config.edges`` order). A disallowed neighbor goes to
    ``excluded_ids`` and is never added. Every allowed edge is appended to ``paths`` at ``hop + 1`` (as in BFS,
    including edges to sessions already in the pool); a session already in the pool is never re-added.
    Strict ``expand_limit``: before a new session would make the pool exceed it, stop and return truncated=True.
    Returns (newly added sessions in discovery order, truncated)."""
    new: list[dict] = []
    for edge in config.edges:
        for neighbor in adapter.neighbors(source, edge, config.fanout):
            if not adapter.allowed(neighbor):
                receipt["excluded_ids"].append(neighbor["id"])
                continue
            if neighbor["id"] not in pool:
                if len(pool) >= config.expand_limit:
                    return new, True
                pool[neighbor["id"]] = neighbor
                new.append(neighbor)
            receipt["paths"].append(
                {
                    "src": source["id"],
                    "dst": neighbor["id"],
                    "edge": edge,
                    "hop": hop + 1,
                }
            )
    return new, False


def _expand_best_first(adapter, config, seeds, receipt) -> tuple[dict, bool]:
    """Global priority queue of discovered-but-unexpanded sessions keyed by query cosine (seeds at hop 0).

    Pops the best session, expands it, and queues each newly added session only if it can still be expanded
    (hop + 1 < depth; a session at hop == depth enters the pool but is never expanded). Budget: at most
    ``seed_count * depth`` node expansions."""
    pool = {s["id"]: s for s in seeds}
    heap = [(*_cos_key(s), 0) for s in seeds]
    heapq.heapify(heap)
    budget = config.seed_count * config.depth
    expanded: list = []
    truncated = False
    while heap and len(expanded) < budget:
        _, sid, hop = heapq.heappop(heap)
        expanded.append(sid)
        new, truncated = _expand_node(adapter, config, receipt, pool, pool[sid], hop)
        if truncated:
            break
        if hop + 1 < config.depth:
            for s in new:
                heapq.heappush(heap, (*_cos_key(s), hop + 1))
    receipt.update(
        expansions=len(expanded), expanded_ids=expanded, expansion_budget=budget
    )
    return pool, truncated


def _expand_beam(adapter, config, seeds, receipt) -> tuple[dict, bool]:
    """Per-seed beam search, seeds processed in seed order, each to completion.

    Each hop expands every session of the seed's beam; all newly added sessions join the pool, and the top
    ``beam_width`` of them (query cosine desc, id asc) form the next beam. A session already in the pool
    (reached by an earlier seed or hop) is not re-added and does not enter this seed's beam."""
    pool = {s["id"]: s for s in seeds}
    expanded: list = []
    beams: list[dict] = []
    for seed in seeds:
        beam = [seed]
        for hop in range(config.depth):
            reached: list[dict] = []
            for node in beam:
                expanded.append(node["id"])
                new, truncated = _expand_node(adapter, config, receipt, pool, node, hop)
                reached.extend(new)
                if truncated:
                    receipt.update(
                        expansions=len(expanded), expanded_ids=expanded, beams=beams
                    )
                    return pool, True
            beam = sorted(reached, key=_cos_key)[: config.beam_width]
            beams.append(
                {
                    "seed": seed["id"],
                    "hop": hop + 1,
                    "beam_ids": [s["id"] for s in beam],
                }
            )
            if not beam:
                break
    receipt.update(expansions=len(expanded), expanded_ids=expanded, beams=beams)
    return pool, False


def _expand_mcts(adapter, config, seeds, receipt) -> tuple[dict, bool]:
    """UCT over a virtual root whose children are the seeds; ``seed_count * fanout`` iterations at most.

    Selection descends while the current node is expanded and has children, choosing unvisited children first
    (query cosine desc, id asc), else the child maximizing mean_value + sqrt(2) * sqrt(ln(N_parent) / N_child)
    (ties: cosine desc, id asc). An unexpanded leaf with hop < depth is expanded (its newly added sessions become
    its children at hop + 1) and valued by the best child cosine (its own cosine if no new child); any other
    leaf (hop == depth, or expanded with no children) is valued by its own cosine. The value is back-propagated
    to the root. Stops early when no expandable leaf remains."""
    pool = {s["id"]: s for s in seeds}
    root = {
        "id": None,
        "hop": -1,
        "children": [s["id"] for s in seeds],
        "expanded": True,
        "visits": 0,
        "value_sum": 0.0,
    }
    tree = {
        s["id"]: {
            "id": s["id"],
            "hop": 0,
            "children": [],
            "expanded": False,
            "visits": 0,
            "value_sum": 0.0,
        }
        for s in seeds
    }
    expandable = {s["id"] for s in seeds}  # depth >= 1, so every seed can be expanded
    budget = config.seed_count * config.fanout
    expanded: list = []
    leaves: list = []
    iterations = 0
    truncated = False

    def choose(parent: dict) -> dict:
        log_n = math.log(parent["visits"]) if parent["visits"] > 0 else 0.0

        def key(cid):
            c = tree[cid]
            if c["visits"] == 0:
                return (0, 0.0, *_cos_key(pool[cid]))
            uct = c["value_sum"] / c["visits"] + MCTS_C * math.sqrt(log_n / c["visits"])
            return (1, -uct, *_cos_key(pool[cid]))

        return tree[min(parent["children"], key=key)]

    while iterations < budget and expandable:
        iterations += 1
        path, cur = [root], root
        while cur["expanded"] and cur["children"]:
            cur = choose(cur)
            path.append(cur)
        leaves.append(cur["id"])
        session = pool[cur["id"]]
        if not cur["expanded"] and cur["hop"] < config.depth:
            expanded.append(cur["id"])
            new, truncated = _expand_node(
                adapter, config, receipt, pool, session, cur["hop"]
            )
            cur["expanded"] = True
            expandable.discard(cur["id"])
            for s in new:
                tree[s["id"]] = {
                    "id": s["id"],
                    "hop": cur["hop"] + 1,
                    "children": [],
                    "expanded": False,
                    "visits": 0,
                    "value_sum": 0.0,
                }
                cur["children"].append(s["id"])
                if cur["hop"] + 1 < config.depth:
                    expandable.add(s["id"])
            if truncated:
                break
            value = max(
                (float(s["cosine"]) for s in new), default=float(session["cosine"])
            )
        else:
            value = float(session["cosine"])
        for node in path:
            node["visits"] += 1
            node["value_sum"] += value
    receipt.update(
        expansions=len(expanded),
        expanded_ids=expanded,
        mcts_iterations=iterations,
        mcts_budget=budget,
        mcts_leaf_ids=leaves,
    )
    return pool, truncated


_GRAPH_EXPANDERS = {
    "best-first": _expand_best_first,
    "beam": _expand_beam,
    "MCTS": _expand_mcts,
}


def retrieve(
    adapter: SessionAdapter,
    config: SessionConfig,
    semfilter: Callable | None = None,
    semrank: Callable | None = None,
    pause: Callable[[str], AbstractContextManager] | None = None,
    prefilter_top_n: int | None = None,
    sem_fallback: str = "none",
) -> dict:
    """Every admitted seed expands before one global final top-k selection.

    P1 / P1+Rank filter seed sessions before expansion. P2 / P2+Rank filter the final session pool.
    Descendants of P1 / P1+Rank seeds need not independently pass the seed predicate.
    P1+Rank (plan-p1rank-v1): V -> SemFilter(seeds) -> G -> R / pool cap -> Rank -> top-k.
    Final selection takes the first top-k sessions, including different sessions
    belonging to the same source task.

    ``pause`` (retrieval-slot-v2, user 2026-10-21): a context-manager factory, called with ``"semfilter"`` or
    ``"semrank"``, entered around every ``semfilter`` / ``semrank`` call. The cards those calls see are hydrated BEFORE it is entered, so no adapter call happens
    inside it and a caller may close the adapter connection / free its host-wide retrieval slot for the LLM wait
    (``retrieval_lease.RetrievalLease.pause``). It never changes which adapter calls run or in what order, so
    the receipt is identical with or without it.

    ``prefilter_top_n`` (semfilter-v2, opt-in, default None = unchanged): every SemFilter call judges only the
    top-N of its candidates by query cosine (``semfilter_v2.prefilter``); the rest count as NOT passed and are
    listed in ``receipt["semfilter_prefilter"]`` (not as semantic decisions).

    ``sem_fallback`` (semfilter-v2b, WebShop from round 1, user 2026-10-23; default "none" = unchanged): with
    ``"fill-topk"`` a SemFilter stage that passes FEWER than ``top_k`` candidates is filled up with the candidates
    that did not pass (judged-and-rejected or beyond the prefilter top-N), in the plan's normal order:
    P2 / P2+Rank -> the passed candidates first, then the rest, each part in the plan's order (query cosine; "+Rank":
    ColBERT score, one ColBERT call over passed + rest), and ``select_top_k`` runs over that list (so the final
    selection is filled to ``top_k`` when the capped pool has enough candidates; the failed-card cap still
    applies). P1 / P1+Rank -> the SEEDS are filled the same way up to ``min(top_k, |seeds|)`` with the
    non-passed seeds in seed order (query cosine) before expansion. Recorded in ``receipt["semfilter_fallback"]``
    (mode, stage, candidates, passed, needed, filled_ids). The semantic decisions are unchanged.
    """
    if sem_fallback not in SEM_FALLBACKS:
        raise ValueError(f"unknown sem_fallback {sem_fallback!r}")
    pause = pause or nullcontext
    if config.plan in SEMFILTER_PLANS and semfilter is None:
        raise ValueError("Semantic plan requires an actual semantic evaluator")
    if config.plan in RANK_PLANS and semrank is None:
        raise ValueError("Ranking plan requires an actual semantic evaluator")
    receipt = {
        "selection": "session-full-pool-v1",
        "config": asdict(config),
        "paths": [],
        "excluded_ids": [],
        "semantic_decisions": [],
        "plan_order": list(PLAN_ORDER[config.plan]),
    }
    graph_impl = getattr(config, "graph_impl", "BFS")
    if (
        graph_impl == "BFS"
    ):  # keep the BFS receipt config byte-identical to the pre-extension receipts
        receipt["config"].pop("graph_impl", None)
        receipt["config"].pop("beam_width", None)
    v2 = isinstance(config, SessionConfigV2)
    seeds = adapter.seeds(config.seed_count)
    if len({s["id"] for s in seeds}) != len(seeds):
        raise ValueError("Adapter returned duplicate seeds")
    if any(not adapter.allowed(s) for s in seeds):
        raise ValueError("Adapter violated seed isolation")
    receipt["seed_ids"] = [s["id"] for s in seeds]
    hydrated = {}

    def card(s):
        if s["id"] not in hydrated:
            hydrated[s["id"]] = adapter.hydrate(s)
        return hydrated[s["id"]]

    def filtered(sessions):
        if prefilter_top_n is not None:
            from trimem.experiments.semfilter_v2 import prefilter

            judged, dropped = prefilter(sessions, prefilter_top_n)
            receipt["semfilter_prefilter"] = {
                "top_n": prefilter_top_n,
                "stage": "seeds" if config.plan in SEED_FILTER_PLANS else "pool",
                "candidates": len(sessions),
                "judged_ids": [s["id"] for s in judged],
                "dropped_ids": [s["id"] for s in dropped],
            }
            sessions = judged
        if not sessions:
            return []
        cards = [card(s) for s in sessions]  # DB phase (hydration) before the LLM wait
        with pause("semfilter"):
            decisions = semfilter(cards)
        ids = [d["id"] for d in decisions]
        if len(ids) != len(set(ids)) or set(ids) != {s["id"] for s in sessions}:
            raise ValueError("SemFilter must decide each candidate exactly once")
        if any(type(d["useful"]) is not bool for d in decisions):
            raise ValueError("SemFilter useful must be a JSON boolean")
        receipt["semantic_decisions"].extend(decisions)
        passed = {d["id"] for d in decisions if d["useful"]}
        return [s for s in sessions if s["id"] in passed]

    if config.plan in SEED_FILTER_PLANS:
        all_seeds = list(seeds)
        seeds = filtered(seeds)
        receipt["filtered_seed_ids"] = [s["id"] for s in seeds]
        if sem_fallback == "fill-topk":
            passed_ids = {s["id"] for s in seeds}
            needed = min(config.top_k, len(all_seeds))
            add = [s for s in all_seeds if s["id"] not in passed_ids][
                : max(0, needed - len(seeds))
            ]
            receipt["semfilter_fallback"] = {
                "mode": sem_fallback,
                "stage": "seeds",
                "candidates": len(all_seeds),
                "passed": len(seeds),
                "needed": needed,
                "filled_ids": [s["id"] for s in add],
            }
            keep = passed_ids | {s["id"] for s in add}
            # seed order (query cosine) is kept: passed and filled seeds in their original order
            seeds = [s for s in all_seeds if s["id"] in keep]
            receipt["filled_seed_ids"] = [s["id"] for s in seeds]
    if graph_impl == "BFS":
        pool = {s["id"]: s for s in seeds}
        frontier = seeds
        truncated = False
        for hop in range(config.depth):
            next_frontier = {}
            for source in frontier:
                if v2 and len(pool) + len(next_frontier) >= config.expand_limit:
                    truncated = True
                    break
                for edge in config.edges:
                    for neighbor in adapter.neighbors(source, edge, config.fanout):
                        if not adapter.allowed(neighbor):
                            receipt["excluded_ids"].append(neighbor["id"])
                            continue
                        receipt["paths"].append(
                            {
                                "src": source["id"],
                                "dst": neighbor["id"],
                                "edge": edge,
                                "hop": hop + 1,
                            }
                        )
                        if neighbor["id"] not in pool:
                            next_frontier.setdefault(neighbor["id"], neighbor)
            pool.update(next_frontier)
            frontier = list(next_frontier.values())
    else:  # best-first / beam / MCTS (SessionConfigV2 only); same seeds, same downstream stages
        receipt["graph_impl"] = graph_impl
        pool, truncated = _GRAPH_EXPANDERS[graph_impl](adapter, config, seeds, receipt)
    receipt["pool_ids"] = list(pool)
    candidates = list(pool.values())
    if v2:
        best = max((float(s["cosine"]) for s in candidates), default=None)
        receipt.update(
            pool_size=len(candidates),
            expansion_truncated=truncated,
            best_cosine=best,
            pool_cap=config.pool_cap,
            pool_cap_hit=len(candidates) > config.pool_cap,
            sim_gate=config.sim_gate,
            gated=config.sim_gate is not None
            and (best is None or best < config.sim_gate),
        )
        if receipt["gated"]:  # inject nothing; no SemFilter / Rank cost
            receipt.update(ranked_ids=[], selected_ids=[], cards=[], empty_result=True)
            return receipt
        if receipt["pool_cap_hit"]:
            candidates = sorted(candidates, key=lambda s: (-s["cosine"], s["id"]))[
                : config.pool_cap
            ]
        receipt["capped_pool_ids"] = [s["id"] for s in candidates]
    rest: list[dict] = []
    if config.plan in POOL_FILTER_PLANS:
        before = list(candidates)
        candidates = filtered(candidates)
        if sem_fallback == "fill-topk":
            if len(candidates) < config.top_k:
                passed_ids = {s["id"] for s in candidates}
                rest = [s for s in before if s["id"] not in passed_ids]
            receipt["semfilter_fallback"] = {
                "mode": sem_fallback,
                "stage": "pool",
                "candidates": len(before),
                "passed": len(candidates),
                "needed": config.top_k,
                "filled_ids": [],  # set after ordering / selection (the fill follows the plan's order)
            }
    # "+Rank" plans: a non-LLM ranker (ColBERT) passed as ``semrank`` orders the (filtered) pool.
    if rest:  # fill-topk with a short pool stage: passed first, then the rest, each part in the plan's order
        if config.plan in RANK_PLANS:
            both = candidates + rest
            cards = [card(s) for s in both]
            with pause("semrank"):
                scores = semrank(cards)
            if set(scores) != {s["id"] for s in both}:
                raise ValueError("SemRank must score the full pool")
            receipt["semantic_scores"] = scores

            def order(s):
                return (-scores[s["id"]], -s["cosine"], s["id"])
        else:

            def order(s):
                return (-s["cosine"], s["id"])

        candidates = sorted(candidates, key=order) + sorted(rest, key=order)
    elif config.plan in RANK_PLANS:
        scores = {}
        if candidates:
            cards = [card(s) for s in candidates]
            with pause("semrank"):
                scores = semrank(cards)
        if set(scores) != {s["id"] for s in candidates}:
            raise ValueError("SemRank must score the full pool")
        candidates.sort(key=lambda s: (-scores[s["id"]], -s["cosine"], s["id"]))
        receipt["semantic_scores"] = scores
    else:
        candidates.sort(key=lambda s: (-s["cosine"], s["id"]))
    receipt["ranked_ids"] = [s["id"] for s in candidates]
    selected = (
        select_top_k(candidates, config.top_k, config.max_failed)
        if v2
        else candidates[: config.top_k]
    )
    receipt["selected_ids"] = [s["id"] for s in selected]
    if rest:  # the selected cards that came from the fill (not passed by the judge)
        rest_ids = {s["id"] for s in rest}
        receipt["semfilter_fallback"]["filled_ids"] = [
            s["id"] for s in selected if s["id"] in rest_ids
        ]
    receipt["cards"] = [card(s) for s in selected]
    receipt["empty_result"] = not receipt["cards"]
    return receipt
