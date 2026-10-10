"""EvoTrace as a replayable ingest/retrieve workload (Figure 1 experiment).

Each EvoTrace node is one finished attempt of an evolutionary coding agent (program,
fitness, judge outcome, edit description). The workload replays the corpus as if the
agents were running concurrently:

* sessions are interleaved round-robin on a global timeline;
* inside a session, nodes are ordered topologically along lineage (parent first, ties by
  iteration then source row), so a parent is always ingested before its children;
* before a node is ingested, its agent *retrieves* experience for the parent it is about
  to improve (Retrieve), then the node is written (Ingest).

No agent is re-run; the trace is the workload ("mock" agents). Node embeddings are the
pre-computed Qwen3-Embedding-0.6B vectors (``vectors.npz``, 1024-d, cosine).
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIR = Path("data/evotrace/normalized")


@dataclass(frozen=True)
class Node:
    uid: str
    session_uid: str
    task_uid: str
    parent_uid: str | None
    iteration: int
    row_offset: int
    is_valid: bool
    fitness: float | None
    status: str
    language: str | None
    error_signature: str | None
    judge: str | None
    changes: str
    artifact_uid: str | None

    @property
    def failed(self) -> bool:
        return (not self.is_valid) or (self.fitness is None) or self.fitness <= 0.0

    def outcome(self) -> dict[str, Any]:
        """The compact, LLM-facing description of this attempt."""
        return {
            "task": self.task_uid,
            "language": self.language,
            "status": self.status,
            "valid": self.is_valid,
            "fitness": self.fitness,
            "error": self.error_signature,
            "judge": self.judge,
            "change": (self.changes or "")[:400],
        }


@dataclass
class Corpus:
    nodes: dict[str, Node]
    vectors: dict[str, np.ndarray]
    timeline: list[str]  # node uids in replay order
    tasks: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timeline)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _topo_session(uids: list[str], nodes: dict[str, Node]) -> list[str]:
    """Kahn order along lineage; ready nodes by (iteration, row_offset)."""
    members = set(uids)
    children: dict[str, list[str]] = defaultdict(list)
    indeg = {u: 0 for u in uids}
    for u in uids:
        p = nodes[u].parent_uid
        if p is not None and p in members:
            children[p].append(u)
            indeg[u] += 1
    ready = [
        (nodes[u].iteration, nodes[u].row_offset, u) for u in uids if indeg[u] == 0
    ]
    heapq.heapify(ready)
    out: list[str] = []
    while ready:
        _, _, u = heapq.heappop(ready)
        out.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, (nodes[c].iteration, nodes[c].row_offset, c))
    if len(out) != len(uids):  # a lineage cycle would be a normalizer bug
        raise ValueError("lineage cycle inside a session")
    return out


def load_corpus(
    directory: Path = DEFAULT_DIR,
    *,
    tasks: list[str] | None = None,
    max_nodes_per_session: int | None = None,
    max_sessions_per_task: int | None = None,
) -> Corpus:
    parent: dict[str, str] = {}
    for edge in _read_jsonl(directory / "lineage_edges.jsonl"):
        if edge["relation"] == "has_child":
            parent[edge["dst_node_uid"]] = edge["src_node_uid"]

    nodes: dict[str, Node] = {}
    for row in _read_jsonl(directory / "nodes.jsonl"):
        if tasks is not None and row["task_uid"] not in tasks:
            continue
        metrics = row.get("metrics") or {}
        fitness = row.get("fitness")
        nodes[row["node_uid"]] = Node(
            uid=row["node_uid"],
            session_uid=row["session_uid"],
            task_uid=row["task_uid"],
            parent_uid=parent.get(row["node_uid"]),
            iteration=int(row.get("iteration") or 0),
            row_offset=int(row.get("row_offset") or 0),
            is_valid=bool(row.get("is_valid")),
            fitness=None if fitness is None else float(fitness),
            status=str(row.get("status") or ""),
            language=row.get("language"),
            error_signature=row.get("error_signature"),
            judge=metrics.get("judge_result") if isinstance(metrics, dict) else None,
            changes=str(row.get("changes") or ""),
            artifact_uid=row.get("artifact_uid"),
        )

    by_session: dict[str, list[str]] = defaultdict(list)
    for uid, node in nodes.items():
        by_session[node.session_uid].append(uid)

    if max_sessions_per_task is not None:
        per_task: dict[str, list[str]] = defaultdict(list)
        for s in sorted(by_session):
            per_task[nodes[by_session[s][0]].task_uid].append(s)
        keep = {s for ss in per_task.values() for s in ss[:max_sessions_per_task]}
        by_session = {s: u for s, u in by_session.items() if s in keep}

    orders: dict[str, list[str]] = {}
    for s, uids in by_session.items():
        order = _topo_session(uids, nodes)
        if max_nodes_per_session is not None:
            order = order[:max_nodes_per_session]
        orders[s] = order
    kept = {u for order in orders.values() for u in order}
    nodes = {
        u: (n if n.parent_uid is None or n.parent_uid in kept else _orphan(n))
        for u, n in nodes.items()
        if u in kept
    }

    timeline: list[str] = []
    sessions = sorted(orders)
    depth = max((len(o) for o in orders.values()), default=0)
    for step in range(depth):
        for s in sessions:
            if step < len(orders[s]):
                timeline.append(orders[s][step])

    z = np.load(directory / "vectors.npz", allow_pickle=False)
    index = {str(u): i for i, u in enumerate(z["uids"])}
    matrix = z["vectors"]
    vectors = {u: matrix[index[u]] for u in kept if u in index}
    missing = [u for u in kept if u not in vectors]
    if missing:
        raise ValueError(f"{len(missing)} node(s) have no embedding, e.g. {missing[0]}")
    return Corpus(
        nodes=nodes,
        vectors=vectors,
        timeline=timeline,
        tasks=sorted({n.task_uid for n in nodes.values()}),
    )


def _orphan(node: Node) -> Node:
    """A node whose parent was cut by a per-session cap becomes a root."""
    return Node(**{**node.__dict__, "parent_uid": None})
