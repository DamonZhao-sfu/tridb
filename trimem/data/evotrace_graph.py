"""Native graph edges for the EvoTrace trajectory store.

Topology lives ONLY in the graph_store access method (golden rule 3). `edge_prop` mirrors each
physical edge's properties (label, ordinal, weight, provenance) and is the idempotency guard: a
graph edge is inserted only when its mirror row is new, so re-running the loader never stacks
duplicate adjacency entries. It is never used for traversal.

Edge direction (forward / `_up` reverse):
  lineage       parent -> child          (dataset lineage_edges, `has_child`)
  merge_parent  gepa Merge Parent B -> child (the 92 context edges labelled 'Merge Parent B')
  context       inspiration -> consumer  (other_context_ids; consumer's `context_up` lists them)
  similar       attempt -> k nearest attempts of the same task in OTHER sessions (derived kNN)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import psycopg

from trimem.data import evotrace_text as tx

MERGE_LABEL = "Merge Parent B"


class Edge(NamedTuple):
    src: int
    dst: int
    etype: str
    label: str | None
    ordinal: int
    weight: float | None
    provenance: str


def _with_reverse(
    edges: list[Edge], rank_up: dict[tuple[int, int], int] | None = None
) -> list[Edge]:
    """Forward edges plus their `_up` copies. `rank_up` gives the reverse edge's ordinal;
    by default reverse edges are ranked by (weight desc, dst) per new source."""
    up = [
        Edge(e.dst, e.src, e.etype + "_up", e.label, 0, e.weight, e.provenance)
        for e in edges
    ]
    if rank_up is not None:
        up = [e._replace(ordinal=rank_up.get((e.src, e.dst), 0)) for e in up]
    else:
        by: dict[int, list[Edge]] = defaultdict(list)
        for e in up:
            by[e.src].append(e)
        up = [
            e._replace(ordinal=i)
            for grp in by.values()
            for i, e in enumerate(
                sorted(grp, key=lambda e: (-(e.weight or 0.0), e.dst))
            )
        ]
    return edges + up


def _rank_children(
    pairs: list[tuple[int, int]], iteration: dict[int, int]
) -> list[int]:
    """Ordinal of each (src, dst) among src's out-edges, by (dst iteration, dst id)."""
    by: dict[int, list[int]] = defaultdict(list)
    for s, d in pairs:
        by[s].append(d)
    rank = {
        (s, d): i
        for s, ds in by.items()
        for i, d in enumerate(sorted(ds, key=lambda d: (iteration[d], d)))
    }
    return [rank[p] for p in pairs]


def lineage_edges(ds) -> list[Edge]:
    it = {ds.ids[n["node_uid"]]: int(n["iteration"]) for n in ds.nodes}
    pairs = [
        (ds.ids[e["src_node_uid"]], ds.ids[e["dst_node_uid"]])
        for e in tx.read_jsonl(ds.src / "lineage_edges.jsonl")
    ]
    fwd = [
        Edge(s, d, "lineage", None, r, None, "programs.jsonl:parent_id")
        for (s, d), r in zip(pairs, _rank_children(pairs, it), strict=True)
    ]
    return _with_reverse(fwd, {(d, s): 0 for s, d in pairs})


def _raw_context(ds, sessions) -> tuple[dict, dict]:
    """(dst_uid, src_uid) -> context_info label, and -> position in other_context_ids."""
    labels: dict[tuple[str, str], str] = {}
    position: dict[tuple[str, str], int] = {}
    for s in sessions:
        for pid, prog in ds.raw_programs(s).items():
            dst = f"{s}#{pid}"
            ctx = list(dict.fromkeys(prog.get("other_context_ids") or []))
            position.update({(dst, f"{s}#{c}"): i for i, c in enumerate(ctx)})
            for item in prog.get("context_info") or []:
                if isinstance(item, list) and len(item) == 2 and item[0]:
                    labels[(dst, f"{s}#{item[1]}")] = item[0]
    return labels, position


def merge_labels(ds) -> int:
    """Independent count for the receipt gate: raw 'Merge Parent B' labels between two nodes of
    the dataset, over EVERY session (not only those with context edges)."""
    labels, _ = _raw_context(ds, ds.run_rel)
    return sum(
        1
        for (d, s), lab in labels.items()
        if lab == MERGE_LABEL and d in ds.ids and s in ds.ids
    )


def context_edges(ds, *, merge_as_context: bool = False) -> list[Edge]:
    """Context edges split into `context` and gepa `merge_parent` by the raw context_info label.
    Ordinal of the reverse (consumer -> source) edge = position in the raw other_context_ids."""
    it = {ds.ids[n["node_uid"]]: int(n["iteration"]) for n in ds.nodes}
    rows = list(tx.read_jsonl(ds.src / "context_edges.jsonl"))
    labels, position = _raw_context(ds, {r["session_uid"] for r in rows})
    missing = [
        r for r in rows if (r["dst_node_uid"], r["src_node_uid"]) not in position
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} context edges absent from raw other_context_ids"
        )
    out: list[Edge] = []
    for etype in ("context",) if merge_as_context else ("context", "merge_parent"):
        sel = [
            r
            for r in rows
            if merge_as_context
            or (labels.get((r["dst_node_uid"], r["src_node_uid"])) == MERGE_LABEL)
            == (etype == "merge_parent")
        ]
        pairs = [(ds.ids[r["src_node_uid"]], ds.ids[r["dst_node_uid"]]) for r in sel]
        fwd = [
            Edge(
                s,
                d,
                etype,
                labels.get((r["dst_node_uid"], r["src_node_uid"])),
                rk,
                None,
                r["provenance"],
            )
            for (s, d), r, rk in zip(pairs, sel, _rank_children(pairs, it), strict=True)
        ]
        rank_up = {
            (d, s): position[(r["dst_node_uid"], r["src_node_uid"])]
            for (s, d), r in zip(pairs, sel, strict=True)
        }
        out += _with_reverse(fwd, rank_up)
    return out


def similar_edges(ds, k: int) -> list[Edge]:
    """Cross-session kNN per task over the node_artifact vectors (cosine, stable tie order)."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for n in ds.nodes:
        if ("node", n["node_uid"]) in ds.vectors:
            by_task[n["task_uid"]].append(n)
    fwd: list[Edge] = []
    for nodes in by_task.values():
        ids = np.array([ds.ids[n["node_uid"]] for n in nodes])
        sess = np.array([n["session_uid"] for n in nodes])
        m = np.stack([ds.vectors[("node", n["node_uid"])] for n in nodes]).astype(
            np.float32
        )
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        sim = m @ m.T
        sim[sess[:, None] == sess[None, :]] = -np.inf  # other sessions only
        top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        for i, js in enumerate(top):
            fwd += [
                Edge(
                    int(ids[i]),
                    int(ids[j]),
                    "similar",
                    None,
                    r,
                    float(sim[i, j]),
                    "derived:knn_node_artifact",
                )
                for r, j in enumerate(js)
                if np.isfinite(sim[i, j])
            ]
    return _with_reverse(fwd)


def insert_edges(conn: psycopg.Connection, edges: list[Edge]) -> dict[str, int]:
    """Mirror-then-insert in ONE transaction: only edges whose edge_prop row is new reach the
    native store, in (src, type, ordinal, dst) order so `expand(.., fanout)` sees rank order."""
    with conn.transaction():
        for t in sorted({e.etype for e in edges}):
            conn.execute("SELECT graph_store.register_edge_type(%s)", (t,))
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS e_in (src int, dst int, etype text, "
            "label text, ordinal int, weight float8, provenance text)"
        )
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS e_new "
            "(src int, dst int, edge_type int, ordinal int)"
        )
        conn.execute("TRUNCATE e_in, e_new")
        with conn.cursor().copy("COPY e_in FROM STDIN") as cp:
            for e in edges:
                cp.write_row(e)
        conn.execute(
            "WITH ins AS (INSERT INTO edge_prop SELECT e.src, e.dst, t.id, e.label, e.ordinal, "
            "e.weight, e.provenance FROM e_in e JOIN graph_store.edge_type t ON t.name = e.etype "
            "ON CONFLICT DO NOTHING RETURNING src, dst, edge_type, ordinal) "
            "INSERT INTO e_new SELECT * FROM ins"
        )
        # A plain aggregate over the ordered subquery evaluates the inserts in that order.
        inserted = conn.execute(
            "SELECT count(graph_store.gph_insert_edge(q.sv, q.dv, q.edge_type)) FROM "
            "(SELECT s.vid AS sv, d.vid AS dv, n.edge_type FROM e_new n "
            " JOIN attempt s ON s.id = n.src JOIN attempt d ON d.id = n.dst "
            " ORDER BY n.src, n.edge_type, n.ordinal, n.dst) q"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT t.name, count(*) FROM e_new n JOIN graph_store.edge_type t "
            "ON t.id = n.edge_type GROUP BY t.name"
        ).fetchall()
        if inserted != sum(c for _, c in rows):
            raise RuntimeError(
                f"edge insert mismatch: {inserted} native vs {rows} mirrored"
            )
    return dict(rows)


def all_edges(
    ds, similar_k: int, *, merge_as_context: bool = False
) -> Iterator[list[Edge]]:
    yield lineage_edges(ds)
    yield context_edges(ds, merge_as_context=merge_as_context)
    if similar_k > 0:
        yield similar_edges(ds, similar_k)
