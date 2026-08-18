"""Derive additional STARK-PRIME E0 queries by inverting the official answer sets.

WHY THIS EXISTS
---------------
`configs/e0/stark_prime_pilot_annotations.jsonl` holds 10 HAND-WRITTEN annotations, each
naming the anchor entity, the typed edge set and the hop limit that reaches a query's
official answers. Ten queries cannot carry E0's P0 stop condition -- that condition is
"plan spread median < 2x => proposition O fails, re-scope immediately"
(haikaidocs/trevillisPlan.md §6), and a median over ten points is noise driving a major
decision. trevillisPlan §3 E0 asks for 30-50. Decision D3 froze the target at 40.

Hand-annotating 30 more is the bottleneck this module removes: it DERIVES the annotation
from the graph instead of from a reader, by solving the inverse problem the human solved.

    given: query text, official answer ids
    find : an anchor entity mentioned in the text, plus a typed edge set E and hop limit h,
           such that EVERY official answer lies in typed_reachable(anchor, E, h)

Candidate anchors come from the GRAPH first (nodes within h hops of every answer) and are
then filtered by "does this entity's name actually occur in the query text". Deriving from
the graph alone would happily invent an anchor the query never mentions; requiring the text
mention keeps the annotation faithful to what the query asks.

THE HONESTY BOUNDARY -- READ BEFORE USING THE OUTPUT
---------------------------------------------------
Queries emitted here are `annotation_status="auto_derived_v0.2"`. They are NOT
`manually_audited_v0.1`. What is machine-CHECKED is exactly one thing: every official
answer is reachable within the emitted typed-hop envelope (the same `path_audit` gate
`stark_prime_prepare._resolve_annotations` applies, reimplemented here over the normalized
parquet). What is NOT checked is whether the derived anchor/edge-set is the reading a human
would give the natural-language question -- a different anchor or edge set can satisfy the
same reachability constraint. The 10 audited pilot queries are carried through unchanged so
every downstream report can stratify by `annotation_status` and show whether the auto-derived
rows behave like the audited ones. If they do not, that is a finding, not a nuisance.

`typed_reachable` here is byte-for-byte the semantics of
`tools/e0/stark_prime_prepare._typed_reachable`: the edge-type set is applied at EVERY hop
(it is a set, not an ordered path), already-seen nodes are not re-collected, and the anchor
itself is never a result.

CLI:
    python -m tools.e0.query_expand --target-total 40
    python -m tools.e0.query_expand --target-total 40 --out data/e0/.../queries_v0.2.jsonl
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tools.e0.common import environment_record, write_json, write_jsonl

DATASET = "stark_prime"
AUTO_STATUS = "auto_derived_v0.2"
AUDITED_STATUS = "manually_audited_v0.1"
ORACLE_METHOD = "official_answer_ids_plus_exact_graph_path_audit"

# Hop limits E0 can actually use on STARK-PRIME. The pilot's audited distribution is
# {1: 5, 2: 5} and the dataset carries no deeper query semantics, so scanning beyond 2
# would time plans that answer no real question (docs/e0_plan_space_execution §1.1).
HOP_LIMITS = (1, 2)
# An anchor name shorter than this matches text by accident ("A", "cGMP" inside a word).
MIN_ANCHOR_NAME_CHARS = 4
# Queries whose answer set is huge are recall-saturated and carry no plan-quality signal.
MAX_ANSWERS = 20


class Graph:
    """CSR adjacency over the normalized edge parquet.

    STARK stores each PrimeKG relation as TWO arcs (verified: every sampled arc has its
    reverse with the same edge_type, zero self-loops), so a forward-only CSR already walks
    the graph in both directions -- no reverse index is built or needed.
    """

    def __init__(
        self,
        indptr: np.ndarray,
        nbr: np.ndarray,
        etype: np.ndarray,
        type_names: list[str],
    ) -> None:
        self.indptr = indptr
        self.nbr = nbr
        self.etype = etype
        self.type_names = type_names
        self.type_id = {name: i for i, name in enumerate(type_names)}

    @classmethod
    def load(cls, edges_path: Path) -> "Graph":
        import pyarrow.parquet as pq

        table = pq.read_table(edges_path)
        src = table["src_id"].to_numpy().astype(np.int64)
        dst = table["dst_id"].to_numpy().astype(np.int64)
        names = table["edge_type"].to_pylist()
        uniq = sorted(set(names))
        lookup = {name: i for i, name in enumerate(uniq)}
        etype = np.fromiter(
            (lookup[n] for n in names), dtype=np.int16, count=len(names)
        )

        order = np.argsort(src, kind="stable")
        src_sorted = src[order]
        n_nodes = int(max(src.max(), dst.max())) + 1
        indptr = np.searchsorted(src_sorted, np.arange(n_nodes + 1), side="left")
        return cls(indptr, dst[order], etype[order], uniq)

    def neighbors(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.indptr[node], self.indptr[node + 1]
        return self.nbr[lo:hi], self.etype[lo:hi]

    def typed_reachable(
        self, anchor: int, type_ids: Sequence[int], hops: int
    ) -> set[int]:
        """Identical semantics to stark_prime_prepare._typed_reachable, over CSR."""
        allowed = np.asarray(sorted(type_ids), dtype=np.int16)
        seen = {anchor}
        frontier = {anchor}
        reachable: set[int] = set()
        for _ in range(hops):
            following: set[int] = set()
            for node in frontier:
                nb, et = self.neighbors(node)
                if nb.size:
                    following.update(nb[np.isin(et, allowed)].tolist())
            frontier = following - seen
            seen.update(frontier)
            reachable.update(frontier)
        return reachable

    def within(self, node: int, hops: int) -> set[int]:
        """All nodes at distance 1..hops via ANY edge type (anchor-candidate generation)."""
        seen = {node}
        frontier = {node}
        out: set[int] = set()
        for _ in range(hops):
            following: set[int] = set()
            for cur in frontier:
                nb, _ = self.neighbors(cur)
                following.update(nb.tolist())
            frontier = following - seen
            seen.update(frontier)
            out.update(frontier)
        return out

    def incident_types(self, node: int) -> set[int]:
        _, et = self.neighbors(node)
        return set(et.tolist())


def load_nodes(
    nodes_path: Path,
) -> tuple[dict[int, str], dict[int, str], dict[str, list[int]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(nodes_path, columns=["node_id", "entity_type", "name"])
    ids = table["node_id"].to_pylist()
    types = table["entity_type"].to_pylist()
    names = table["name"].to_pylist()
    entity_type = dict(zip(ids, types))
    name_of = dict(zip(ids, names))
    by_name: dict[str, list[int]] = {}
    for node_id, name in zip(ids, names):
        by_name.setdefault(str(name).casefold(), []).append(node_id)
    return entity_type, name_of, by_name


def read_qa(qa_csv: Path, split_index: Path | None) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(qa_csv)
    if split_index is not None and split_index.exists():
        keep = {int(x) for x in split_index.read_text().split()}
        frame = frame[frame["id"].isin(keep)]
    rows = []
    for _, row in frame.iterrows():
        answers = ast.literal_eval(str(row["answer_ids"]))
        rows.append(
            {
                "source_query_id": int(row["id"]),
                "query_text": str(row["query"]),
                "answer_ids": [int(a) for a in answers],
            }
        )
    return rows


def _min_type_cover(
    graph: Graph, anchor: int, answers: Sequence[int], hops: int, budget: Iterable[int]
) -> list[int] | None:
    """Smallest edge-type set (size 1, then 2) whose typed reach covers every answer.

    Size is capped at 2 on purpose: every audited pilot annotation uses one or two edge
    types, and a larger set stops being a meaningful structured predicate -- it degenerates
    towards "follow anything", which is not the workload E0 is characterizing.
    """
    answer_set = set(answers)
    candidates = sorted(budget)
    for size in (1, 2):
        for combo in itertools.combinations(candidates, size):
            if answer_set <= graph.typed_reachable(anchor, combo, hops):
                return list(combo)
    return None


def _template(n_anchors: int, hops: int, intersection: bool) -> str:
    if n_anchors > 1:
        return "neighbor_intersection" if hops == 1 else "shared_neighbor_constraint"
    return "typed_neighbors" if hops == 1 else "typed_chain"


def derive_one(
    query: dict[str, Any],
    graph: Graph,
    entity_type: dict[int, str],
    name_of: dict[int, str],
    by_name: dict[str, list[int]],
    *,
    max_anchor_candidates: int,
) -> dict[str, Any] | None:
    answers = query["answer_ids"]
    if not answers or len(answers) > MAX_ANSWERS:
        return None
    if any(a not in entity_type for a in answers):
        return None
    target_types = {entity_type[a] for a in answers}
    if len(target_types) != 1:
        # The pilot resolver requires a single answer type; keep the same contract so the
        # derived rows and the audited rows mean the same thing.
        return None
    target_type = next(iter(target_types))
    text = query["query_text"].casefold()

    for hops in HOP_LIMITS:
        # Anchor candidates: within `hops` of EVERY answer. Graph-grounded by construction.
        common: set[int] | None = None
        for answer in answers:
            near = graph.within(answer, hops)
            common = near if common is None else (common & near)
            if not common:
                break
        if not common:
            continue
        common -= set(answers)

        # ...then keep only entities the query actually names, longest name first (a longer
        # surface form is a more specific mention and less likely to be incidental).
        mentioned = [
            node
            for node in common
            if len(str(name_of.get(node, ""))) >= MIN_ANCHOR_NAME_CHARS
            and str(name_of[node]).casefold() in text
            and len(by_name.get(str(name_of[node]).casefold(), [])) == 1
        ]
        mentioned.sort(key=lambda n: (-len(str(name_of[n])), n))
        if not mentioned:
            continue

        # Edge-type budget = types incident to the anchor UNION types incident to the
        # answers. Anchor-only is wrong for a chain: in `drug -[target]-> gene
        # -[interacts with]-> component`, the second hop's type touches the intermediate
        # and the answer, never the anchor. Restricting to the anchor silently biases the
        # derived 2-hop set towards queries whose every type happens to sit on the anchor.
        answer_types: set[int] = set()
        for answer in answers:
            answer_types |= graph.incident_types(answer)

        for anchor in mentioned[:max_anchor_candidates]:
            budget = graph.incident_types(anchor) | answer_types
            if not budget:
                continue
            cover = _min_type_cover(graph, anchor, answers, hops, budget)
            if cover is None:
                continue
            edge_types = [graph.type_names[t] for t in cover]
            predicate: dict[str, Any] = (
                {"edge_type": edge_types[0]}
                if len(edge_types) == 1
                else {"edge_types": edge_types}
            )
            predicate["hop_limit"] = hops
            predicate["target_type"] = target_type
            return {
                "source_query_id": query["source_query_id"],
                "anchor_names": [str(name_of[anchor])],
                "anchor_ids": [int(anchor)],
                "target_entity_type": target_type,
                "edge_types": edge_types,
                "hop_limit": hops,
                "template": _template(1, hops, False),
                "structured_predicate": predicate,
                "query_text": query["query_text"],
                "answer_ids": answers,
            }
    return None


def expand(
    normalized_dir: Path,
    raw_dir: Path,
    pilot_queries: Path,
    out_path: Path,
    *,
    target_total: int,
    max_scan: int,
    max_anchor_candidates: int,
    seed: int,
) -> dict[str, Any]:
    entity_type, name_of, by_name = load_nodes(normalized_dir / "nodes.parquet")
    graph = Graph.load(normalized_dir / "edges.parquet")

    audited = [
        json.loads(line) for line in pilot_queries.read_text().splitlines() if line
    ]
    taken = {int(row["source_query_id"]) for row in audited}

    qa = read_qa(
        raw_dir / "stark_qa" / "stark_qa.csv", raw_dir / "split" / "test.index"
    )
    qa = [row for row in qa if row["source_query_id"] not in taken]
    rng = np.random.default_rng(seed)
    rng.shuffle(qa)

    need = target_total - len(audited)
    derived: list[dict[str, Any]] = []
    # Keep the audited hop balance ({1: 5, 2: 5}) rather than letting the search drift to
    # whichever hop is easier to satisfy -- an unbalanced hop mix would confound the
    # per-hop plan-spread breakdown with a per-query-population difference.
    quota = {1: need // 2, 2: need - need // 2}
    scanned = 0
    started = time.time()
    for query in qa:
        if len(derived) >= need or scanned >= max_scan:
            break
        scanned += 1
        found = derive_one(
            query,
            graph,
            entity_type,
            name_of,
            by_name,
            max_anchor_candidates=max_anchor_candidates,
        )
        if found is None:
            continue
        if quota.get(found["hop_limit"], 0) <= 0:
            continue
        quota[found["hop_limit"]] -= 1
        derived.append(found)

    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(audited):
        record = dict(row)
        record["query_id"] = f"stark-prime-{ordinal:03d}"
        rows.append(record)
    for offset, found in enumerate(derived):
        ordinal = len(audited) + offset
        rows.append(
            {
                "query_id": f"stark-prime-{ordinal:03d}",
                "dataset": DATASET,
                "source_query_id": found["source_query_id"],
                "query_text": found["query_text"],
                "answer_ids": found["answer_ids"],
                "anchor_ids": found["anchor_ids"],
                "anchor_names": found["anchor_names"],
                "target_entity_type": found["target_entity_type"],
                "edge_types": found["edge_types"],
                "hop_limit": found["hop_limit"],
                "structured_predicate": found["structured_predicate"],
                "template": found["template"],
                "annotation_status": AUTO_STATUS,
                "oracle_method": ORACLE_METHOD,
                "path_audit": {
                    "all_official_answers_reachable": True,
                    "required_from_each_anchor": False,
                },
            }
        )

    write_jsonl(out_path, rows)
    manifest = {
        "schema_version": "e0-query-expansion-v0.2.0",
        "environment": environment_record(),
        "output": str(out_path),
        "target_total": target_total,
        "total": len(rows),
        "audited": len(audited),
        "auto_derived": len(derived),
        "scanned_candidates": scanned,
        "seconds": round(time.time() - started, 1),
        "hop_distribution": {
            str(h): sum(1 for r in rows if int(r["hop_limit"]) == h) for h in HOP_LIMITS
        },
        "status_distribution": {
            AUDITED_STATUS: len(audited),
            AUTO_STATUS: len(derived),
        },
        "reached_target": len(rows) >= target_total,
    }
    write_json(out_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--normalized-dir", type=Path, default=Path("data/e0/stark_prime/normalized")
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/e0/stark_prime/raw/prime")
    )
    parser.add_argument("--target-total", type=int, default=40)
    parser.add_argument("--max-scan", type=int, default=1500)
    parser.add_argument("--max-anchor-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or args.normalized_dir / "queries_v0.2.jsonl"
    manifest = expand(
        args.normalized_dir,
        args.raw_dir,
        args.normalized_dir / "queries_v0.1.jsonl",
        out,
        target_total=args.target_total,
        max_scan=args.max_scan,
        max_anchor_candidates=args.max_anchor_candidates,
        seed=args.seed,
    )
    print(
        f"[query_expand] {manifest['total']} queries "
        f"({manifest['audited']} audited + {manifest['auto_derived']} auto-derived), "
        f"hops={manifest['hop_distribution']}, {manifest['seconds']}s, "
        f"scanned={manifest['scanned_candidates']}"
    )
    if not manifest["reached_target"]:
        print(
            f"[query_expand] WARNING: only {manifest['total']} of {args.target_total} "
            f"-- raise --max-scan or relax the derivation constraints"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
