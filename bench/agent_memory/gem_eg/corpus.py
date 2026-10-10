"""The Experience Graph held in memory, for the exact oracle and the ground truth.

Deliberately independent of the database. The oracle exists to say what the *right*
answer is, so it must not share code — or bugs — with the thing it is checking. Both
sides read the same normalized JSONL, and nothing else is shared.

10,672 nodes at 1024 dimensions is 44 MB of float32; adjacency is a few hundred
thousand small lists. The whole corpus fits comfortably in memory, so every structure
here is precomputed once and reused across all ~17.9k decision points.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def failure_class(error_signature: str | None) -> str:
    """Who rejected the attempt — see normalize.error_signature.

    ``execution`` is a repair target (it timed out / failed to compile / crashed);
    ``search_rejected`` is not (the program ran fine, the backend just discarded it).
    Measured on the pinned corpus: 1,056 execution vs 1,439 search_rejected.
    """
    if not error_signature:
        return "clean"
    return "execution" if error_signature.startswith(("judge:", "error:")) else "search_rejected"


@dataclass
class Corpus:
    """Nodes, tasks, sessions, typed adjacency and the vector matrix."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: lineage: parent -> children, child -> parent
    children: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    parent: dict[str, str] = field(default_factory=dict)
    #: context/inspiration, kept strictly apart from lineage
    context_out: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    #: hierarchy
    task_sessions: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    session_nodes: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    #: uid -> row in `vectors`; a uid absent here has no embedding, which is a real
    #: state (Sessions and Prompts are hops, not answers) and never an error.
    vector_row: dict[str, int] = field(default_factory=dict)
    vectors: np.ndarray | None = None
    vector_source: dict[str, str] = field(default_factory=dict)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(
        cls, normalized: Path = DEFAULT_NORMALIZED, *, vectors: Path | None = None
    ) -> Corpus:
        corpus = cls()
        for task in read_jsonl(normalized / "tasks.jsonl"):
            corpus.tasks[task["task_uid"]] = task
        for session in read_jsonl(normalized / "sessions.jsonl"):
            corpus.sessions[session["session_uid"]] = session
            corpus.task_sessions[session["task_uid"]].append(session["session_uid"])
        for node in read_jsonl(normalized / "nodes.jsonl"):
            uid = node["node_uid"]
            node["failure_class"] = failure_class(node.get("error_signature"))
            corpus.nodes[uid] = node
            corpus.session_nodes[node["session_uid"]].append(uid)
        for edge in read_jsonl(normalized / "lineage_edges.jsonl"):
            src, dst = edge["src_node_uid"], edge["dst_node_uid"]
            corpus.children[src].append(dst)
            corpus.parent[dst] = src
        context_path = normalized / "context_edges.jsonl"
        if context_path.is_file():
            for edge in read_jsonl(context_path):
                corpus.context_out[edge["src_node_uid"]].append(edge["dst_node_uid"])

        # Deterministic iteration order everywhere: a set would make the oracle's
        # tie-breaks depend on hash seeding, and then "exact" would not be exact.
        for mapping in (corpus.children, corpus.context_out, corpus.session_nodes):
            for key in mapping:
                mapping[key].sort()
        for key in corpus.task_sessions:
            corpus.task_sessions[key].sort()

        if vectors is not None:
            corpus.load_vectors(vectors)
        return corpus

    def load_vectors(self, path: Path) -> None:
        blob = np.load(path, allow_pickle=False)
        uids = [str(u) for u in blob["uids"]]
        matrix = blob["vectors"].astype(np.float32, copy=False)
        # Normalize once so cosine distance is 1 - dot, and every downstream ranking
        # is a single matmul rather than per-pair norm arithmetic.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.vectors = matrix / norms
        self.vector_row = {uid: i for i, uid in enumerate(uids)}
        if "sources" in blob:
            self.vector_source = dict(zip(uids, (str(s) for s in blob["sources"])))

    # -- vectors ---------------------------------------------------------

    def vector(self, uid: str) -> np.ndarray | None:
        row = self.vector_row.get(uid)
        return None if row is None or self.vectors is None else self.vectors[row]

    def cosine_distance(self, query: np.ndarray, uids: list[str]) -> np.ndarray:
        """1 - cosine similarity, matching the HNSW index's `vector_cosine_ops`.

        uids without a vector get +inf — the same outcome as tjs_open's
        `if (vnull) continue;`, so the oracle drops exactly what the engine drops.
        """
        out = np.full(len(uids), np.inf, dtype=np.float64)
        rows = [(i, self.vector_row[u]) for i, u in enumerate(uids) if u in self.vector_row]
        if not rows or self.vectors is None:
            return out
        idx = np.fromiter((i for i, _ in rows), dtype=np.int64, count=len(rows))
        vrows = np.fromiter((r for _, r in rows), dtype=np.int64, count=len(rows))
        out[idx] = 1.0 - (self.vectors[vrows] @ query)
        return out

    # -- traversal (mirrors the registered native edge types) -------------

    def traverse(self, seed: str, *, relation: str, hops: int) -> list[str]:
        """Bounded BFS, distinct vertices, seed excluded — gph_traverse_bounded's contract.

        `relation` names the same thing the native edge type does:
          hier        task -> session -> node   (the eg_hier rollup)
          lineage     parent -> child           (descendants)
          child_of    child -> parent           (ancestors; needs the derived inverse)
          context     inspiration -> consumer
        """
        if hops <= 0:
            return []
        frontier = [seed]
        seen = {seed}
        order: list[str] = []
        for _ in range(hops):
            nxt: list[str] = []
            for node in frontier:
                for neighbour in self._out(node, relation):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        order.append(neighbour)
                        nxt.append(neighbour)
            if not nxt:
                break
            frontier = nxt
        return order

    def _out(self, node: str, relation: str) -> list[str]:
        if relation == "hier":
            if node in self.tasks:
                return self.task_sessions.get(node, [])
            if node in self.sessions:
                return self.session_nodes.get(node, [])
            return []
        if relation == "lineage":
            return self.children.get(node, [])
        if relation == "child_of":
            up = self.parent.get(node)
            return [up] if up else []
        if relation == "context":
            return self.context_out.get(node, [])
        raise ValueError(f"unknown relation {relation!r}")

    # -- derived views used by the ground truth ---------------------------

    def ancestors(self, uid: str) -> list[str]:
        """Root-ward path, nearest first."""
        out: list[str] = []
        cur = self.parent.get(uid)
        while cur is not None and cur not in out:
            out.append(cur)
            cur = self.parent.get(cur)
        return out

    def siblings(self, uid: str) -> list[str]:
        up = self.parent.get(uid)
        return [] if up is None else [c for c in self.children.get(up, []) if c != uid]

    def best_so_far(self, session_uid: str) -> list[tuple[int, str, float]]:
        """(iteration, node_uid, fitness) each time this session set a new best.

        Ordering is by (iteration, node_uid) because EvoTrace has NO wall clock — the
        logical step is the only order that exists (see gem_eg_state_event).
        """
        rows = [
            self.nodes[u]
            for u in self.session_nodes.get(session_uid, [])
            if self.nodes[u]["is_valid"] and self.nodes[u]["fitness"] is not None
        ]
        rows.sort(key=lambda n: (n["iteration"] if n["iteration"] is not None else 0, n["node_uid"]))
        out: list[tuple[int, str, float]] = []
        best: float | None = None
        for node in rows:
            if best is None or node["fitness"] > best:
                best = node["fitness"]
                out.append((node["iteration"] or 0, node["node_uid"], node["fitness"]))
        return out

    def best_at(self, session_uid: str, iteration: int) -> float | None:
        """Best fitness this session had achieved STRICTLY BEFORE `iteration`."""
        best: float | None = None
        for it, _, fitness in self.best_so_far(session_uid):
            if it >= iteration:
                break
            best = fitness
        return best

    # -- summary ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "tasks": len(self.tasks),
            "sessions": len(self.sessions),
            "nodes": len(self.nodes),
            "lineage_edges": sum(len(v) for v in self.children.values()),
            "context_edges": sum(len(v) for v in self.context_out.values()),
            "roots": len(self.nodes) - len(self.parent),
            "vectors": len(self.vector_row),
        }
