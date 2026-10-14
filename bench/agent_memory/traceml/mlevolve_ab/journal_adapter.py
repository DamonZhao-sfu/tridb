"""Restore MLEvolve's parent links so the TraceML toolkit can read the tree.

Why this exists
---------------
MLEvolve strips parent pointers when it serializes a journal
(``mlevolve/utils/serialize.py:19-30``)::

    node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
    for n in obj.nodes:
        n.parent = None                       # cleared on the node
    obj_dict["node2parent"] = node2parent     # moved to the top level

The TraceML extractor only ever looks at the node
(``traceml/code/05_toolkit/extract.py:234``)::

    "parent_id": n.get("parent"),             # always None on an MLEvolve journal

So every node comes out of extraction as a root. Two things break, and they
break differently:

1. **Tree structure is gone.** ``parent_id`` is ``None`` everywhere, so depth,
   branching factor, and anything derived from lineage is unrecoverable.

2. **Action labels are computed, but against the wrong pairs.**
   ``label.py:89-112`` builds each action row from *consecutive versions in
   ctime order*, not from parent->child. For a linear agent those coincide; for
   a tree search they do not. Version k+1 is frequently a sibling in another
   subtree, so the labeler is handed a diff between two unrelated code states
   and dutifully labels it. The rows are not empty -- they are plausible and
   wrong, which is worse.

The released TraceML corpus already says how to handle this: its `paired` split
linearizes 13 MLEvolve tree searches into 189 root-to-leaf branches. Within a
branch, consecutive *is* parent->child, so the stock label pipeline becomes
correct without touching it.

This module therefore emits two artifacts and modifies nothing upstream:

``journal.parented.json``
    The whole tree with ``parent`` restored on each node. Use for structural
    metrics (depth, branching, revisit).

``branches/branch_NNN/journal.json``
    One root-to-leaf chain each, ctime-ordered. Feed these to the stock
    ``analyze_run.py`` / ``label.py`` and the action diffs are parent-relative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "JournalIntegrityError",
    "TreeStats",
    "load_journal",
    "restore_parents",
    "tree_stats",
    "check_integrity",
    "linearize_branches",
    "write_adapted",
]

# Node ids MLEvolve may emit without code; TraceML's extractor drops
# parentless+codeless nodes, and so do we when linearizing.
# Order matters: MLEvolve writes BOTH, and `filtered_journal.json` is
# `filter_journal()`'s best-path-only pruning (search_node.py). Grading that one
# silently scores a single branch and reports it as the whole run -- measured:
# 2 nodes instead of 10. Always prefer the complete journal.
_JOURNAL_NAMES = ("journal.json", "filtered_journal.json")


class JournalIntegrityError(RuntimeError):
    """The restored tree is not a forest of the shape MLEvolve should produce."""


@dataclass(frozen=True)
class TreeStats:
    """Shape of a restored journal. All counts are post-restoration."""

    n_nodes: int
    n_roots: int
    n_edges: int
    max_depth: int
    n_leaves: int
    branching_factor: float
    root_ids: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_roots": self.n_roots,
            "n_edges": self.n_edges,
            "max_depth": self.max_depth,
            "n_leaves": self.n_leaves,
            "branching_factor": round(self.branching_factor, 4),
            "root_ids": list(self.root_ids),
        }


def find_journal(run_dir: Path) -> Path:
    """Locate a journal under ``run_dir``, preferring the filtered one."""
    for name in _JOURNAL_NAMES:
        hits = sorted(run_dir.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no journal ({' or '.join(_JOURNAL_NAMES)}) under {run_dir}"
    )


def load_journal(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def restore_parents(journal: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``journal`` with ``node["parent"]`` filled from ``node2parent``.

    Raises if the mapping references ids that are not in ``nodes`` -- a silent
    drop there would reintroduce the exact bug this module exists to fix.
    """
    nodes = journal.get("nodes")
    if nodes is None:
        raise JournalIntegrityError("journal has no 'nodes' key")

    if "node2parent" not in journal:
        # A journal that already carries parents on the node (e.g. stock AIDE)
        # needs no adaptation; say so rather than silently returning it changed.
        if any(n.get("parent") is not None for n in nodes):
            return json.loads(json.dumps(journal))
        raise JournalIntegrityError(
            "journal has neither 'node2parent' nor any node-level 'parent'; "
            "nothing to restore -- is this an MLEvolve journal?"
        )

    node2parent: dict[str, str] = journal["node2parent"]
    known = {n["id"] for n in nodes}

    unknown_children = set(node2parent) - known
    unknown_parents = set(node2parent.values()) - known
    if unknown_children or unknown_parents:
        raise JournalIntegrityError(
            f"node2parent references unknown ids: "
            f"children={sorted(unknown_children)[:5]} parents={sorted(unknown_parents)[:5]}"
        )

    out = json.loads(json.dumps(journal))
    for node in out["nodes"]:
        node["parent"] = node2parent.get(node["id"])
    return out


def _children_map(
    nodes: Iterable[dict[str, Any]],
) -> dict[str | None, list[dict[str, Any]]]:
    kids: dict[str | None, list[dict[str, Any]]] = {}
    for n in nodes:
        kids.setdefault(n.get("parent"), []).append(n)
    for group in kids.values():
        group.sort(key=lambda n: (n.get("ctime") or 0, n["id"]))
    return kids


def tree_stats(journal: dict[str, Any]) -> TreeStats:
    """Compute the shape of a *parent-restored* journal."""
    nodes = journal["nodes"]
    by_id = {n["id"]: n for n in nodes}
    roots = [n for n in nodes if n.get("parent") is None]
    kids = _children_map(nodes)

    # Depth by walking up, with a visited set so a cycle raises instead of hanging.
    def depth_of(node: dict[str, Any]) -> int:
        seen = {node["id"]}
        d = 0
        cur = node
        while cur.get("parent") is not None:
            pid = cur["parent"]
            if pid in seen:
                raise JournalIntegrityError(f"cycle in journal tree at node {pid}")
            seen.add(pid)
            cur = by_id[pid]
            d += 1
        return d

    max_depth = max((depth_of(n) for n in nodes), default=0)
    n_leaves = sum(1 for n in nodes if not kids.get(n["id"]))
    n_internal = len(nodes) - n_leaves
    branching = (len(nodes) - len(roots)) / n_internal if n_internal else 0.0

    return TreeStats(
        n_nodes=len(nodes),
        n_roots=len(roots),
        n_edges=sum(1 for n in nodes if n.get("parent") is not None),
        max_depth=max_depth,
        n_leaves=n_leaves,
        branching_factor=branching,
        root_ids=tuple(sorted(n["id"] for n in roots)),
    )


def check_integrity(
    journal: dict[str, Any],
    *,
    expect_max_roots: int | None = None,
    require_branching: bool = True,
) -> TreeStats:
    """Assert the restored journal is a sane forest; return its stats.

    ``require_branching`` guards the failure mode that motivated this module in
    reverse: an adapter bug that produces a single straight chain looks fine to
    every downstream consumer but silently destroys the search structure. A real
    MLEvolve run has ``initial_drafts`` siblings under the root at minimum, so a
    tree with no branching at all is treated as a bug unless explicitly allowed.
    """
    stats = tree_stats(journal)  # raises on cycles

    if stats.n_nodes and stats.n_roots == 0:
        raise JournalIntegrityError("no root node: every node claims a parent")

    if stats.n_edges != stats.n_nodes - stats.n_roots:
        raise JournalIntegrityError(
            f"edge count {stats.n_edges} != nodes {stats.n_nodes} - roots {stats.n_roots}; "
            "the parent mapping is not a forest"
        )

    if expect_max_roots is not None and stats.n_roots > expect_max_roots:
        raise JournalIntegrityError(
            f"{stats.n_roots} roots, expected at most {expect_max_roots}. "
            "Un-restored parents make every node a root -- did restore_parents() run?"
        )

    if require_branching and stats.n_nodes > 2 and stats.max_depth == stats.n_nodes - 1:
        raise JournalIntegrityError(
            f"tree is a straight chain of {stats.n_nodes} nodes (depth {stats.max_depth}); "
            "a tree search should branch. Suspect a linearization bug."
        )

    return stats


def linearize_branches(
    journal: dict[str, Any], *, drop_codeless: bool = True
) -> list[list[dict[str, Any]]]:
    """Split a restored tree into root-to-leaf chains.

    This mirrors how the released TraceML `paired` split treats MLEvolve (13 tree
    searches -> 189 root-to-leaf branches). Within a returned chain, consecutive
    entries are genuinely parent->child, which is the property ``label.py``
    assumes when it diffs adjacent versions.
    """
    nodes = journal["nodes"]
    kids = _children_map(nodes)
    roots = kids.get(None, [])
    branches: list[list[dict[str, Any]]] = []

    def walk(node: dict[str, Any], path: list[dict[str, Any]]) -> None:
        path = path + [node]
        children = kids.get(node["id"], [])
        if not children:
            branches.append(path)
            return
        for child in children:
            walk(child, path)

    for root in roots:
        walk(root, [])

    if drop_codeless:
        # extract.py skips parentless+codeless nodes; a codeless root would
        # otherwise become an empty v001.py and shift every version number.
        branches = [[n for n in b if n.get("code")] for b in branches]
        branches = [b for b in branches if b]

    return branches


def write_adapted(
    journal_path: Path,
    out_dir: Path,
    *,
    expect_max_roots: int | None = None,
    require_branching: bool = True,
) -> dict[str, Any]:
    """Restore parents, verify, and write the parented tree plus per-branch journals.

    Returns a receipt dict (also written to ``out_dir/adapter_receipt.json``).
    """
    journal_path = Path(journal_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_journal(journal_path)
    restored = restore_parents(raw)
    stats = check_integrity(
        restored,
        expect_max_roots=expect_max_roots,
        require_branching=require_branching,
    )

    parented = out_dir / "journal.parented.json"
    parented.write_text(json.dumps(restored, separators=(",", ":")))

    branches = linearize_branches(restored)
    bdir = out_dir / "branches"
    bdir.mkdir(exist_ok=True)
    branch_receipts = []
    for i, chain in enumerate(branches, start=1):
        sub = bdir / f"branch_{i:03d}"
        sub.mkdir(exist_ok=True)
        # Keep the journal shape TraceML expects, minus the tree keys that no
        # longer apply to a linear chain.
        payload = {
            k: v for k, v in restored.items() if k not in ("nodes", "node2parent")
        }
        payload["nodes"] = chain
        (sub / "journal.json").write_text(json.dumps(payload, separators=(",", ":")))
        branch_receipts.append(
            {
                "branch": i,
                "n_versions": len(chain),
                "leaf_node_id": chain[-1]["id"],
                "path": str(sub / "journal.json"),
            }
        )

    receipt = {
        "source_journal": str(journal_path),
        "out_dir": str(out_dir),
        "tree": stats.as_dict(),
        "n_branches": len(branches),
        "branches": branch_receipts,
        "parented_journal": str(parented),
    }
    (out_dir / "adapter_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "run_dir", type=Path, help="MLEvolve run directory (or a journal.json)"
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--expect-max-roots",
        type=int,
        default=None,
        help="fail if the restored tree has more roots than this",
    )
    ap.add_argument(
        "--allow-chain",
        action="store_true",
        help="permit a non-branching tree (only for tiny smoke runs)",
    )
    args = ap.parse_args(argv)

    src = args.run_dir if args.run_dir.is_file() else find_journal(args.run_dir)
    receipt = write_adapted(
        src,
        args.out,
        expect_max_roots=args.expect_max_roots,
        require_branching=not args.allow_chain,
    )
    print(json.dumps(receipt["tree"], indent=2))
    print(f"{receipt['n_branches']} branches -> {args.out}/branches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
