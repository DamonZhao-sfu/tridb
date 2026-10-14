"""Append a read-only ALE10 native graph report to the preserved Math5 PDF.

Materialized layouts are offline reporting artifacts, not retrieval operators.
Run with bounded CPU affinity and one BLAS thread. Never invokes an LLM or GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np

from trimem.experiments.visualize_experience_graph import dump, export, prepare

TASKS = [
    "ale:ahc" + x
    for x in ("008", "011", "015", "016", "024", "025", "026", "027", "039", "046")
]
INK = "#193349"
MUTED = "#60768a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base(title, subtitle, page):
    fig = plt.figure(figsize=(14, 9.5), facecolor="#f7f9fc")
    fig.text(0.045, 0.95, title, fontsize=24, weight="bold", color=INK)
    fig.text(0.045, 0.915, subtitle, fontsize=10, color=MUTED)
    fig.text(
        0.045,
        0.025,
        "TriMEM / ALE-10 / observed native adjacency / read-only export",
        fontsize=9,
        color=MUTED,
    )
    fig.text(0.95, 0.025, str(page), ha="right", color=MUTED)
    return fig


def paragraph(fig, y, title, text, width=130):
    fig.text(0.055, y, title, fontsize=12, weight="bold", color=INK)
    lines = "\n".join(textwrap.fill(line, width) for line in text.splitlines())
    fig.text(
        0.055, y - 0.027, lines, fontsize=10.5, color=MUTED, va="top", linespacing=1.45
    )
    return y - 0.068 - 0.020 * len(lines.splitlines())


def render(data, out):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = [r for r in data["summary"] if r["scope"] == "corpus"]
    by_id = {n["id"]: n for n in data["nodes"]}
    with PdfPages(out / "ale10_appendix.pdf") as book:
        fig = base(
            "TriMEM | Ten ALE Experience Graphs",
            "ALE appendix to the original Math5 report (pages 1-7 preserved)",
            8,
        )
        fig.text(
            0.05,
            0.84,
            f"{len(by_id):,} attempts     {sum(r['sessions'] for r in rows)} sessions     {len(data['edges']):,} forward typed edges",
            fontsize=21,
            color=INK,
        )
        ax = fig.add_axes([0.045, 0.36, 0.91, 0.42])
        ax.axis("off")
        table = ax.table(
            cellText=[
                [
                    r["title"],
                    f"{r['nodes']:,}",
                    r["sessions"],
                    f"{r['lineage']:,}",
                    f"{r['similar']:,}",
                    f"{r['valid'] / r['nodes']:.1%}",
                ]
                for r in rows
            ],
            colLabels=["Task", "Nodes", "Sessions", "Lineage", "Similar", "Valid"],
            cellLoc="left",
            loc="center",
            colWidths=[0.23, 0.15, 0.14, 0.16, 0.17, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.9)
        for (i, _), cell in table.get_celld().items():
            cell.set_edgecolor("#dce5ed")
            cell.set_facecolor("#e8f0f9" if i == 0 else "white")
        lines = [
            "These are existing TriMEM graph vertices and edges, not simulated construction or a new experiment.",
            "Lineage: parent -> child in one session. Similar: 16 outgoing code-neighbor edges per node, across sessions.",
            "A trajectory is a path through a session's search tree; a node is one program attempt, not an input test case.",
            "All ALE nodes belong to the offline corpus (cell IS NULL). No local online-search results were added here.",
            f"Snapshot: {data['meta']['snapshot_at']} | Forward edges exclude matching *_up traversal copies.",
        ]
        for i, line in enumerate(lines):
            fig.text(0.055, 0.31 - i * 0.046, line, fontsize=10.5, color=INK)
        book.savefig(fig)
        fig.savefig(out / "ale10_overview.png", dpi=145)
        plt.close(fig)

        for page, (task, row) in enumerate(zip(data["tasks"], rows, strict=True), 9):
            ns = [n for n in data["nodes"] if n["task"] == task["id"]]
            ids = {n["id"] for n in ns}
            es = [e for e in data["edges"] if e[0] in ids]
            fig = base(
                task["title"] + " | Native experience graph",
                f"{row['nodes']:,} nodes | {row['sessions']} sessions | {row['lineage']:,} lineage + {row['similar']:,} similar edges | {row['valid'] / row['nodes']:.1%} valid",
                page,
            )
            ax = fig.add_axes([0.04, 0.19, 0.63, 0.65])
            ax.set_facecolor("white")
            lin = [
                [(by_id[a]["x"], by_id[a]["y"]), (by_id[b]["x"], by_id[b]["y"])]
                for a, b, k in es
                if k == "lineage"
            ]
            ax.add_collection(
                LineCollection(lin, colors="#8a9fb3", linewidths=0.65, alpha=0.6)
            )
            valid = [n for n in ns if n["is_valid"] and n["fitness"] is not None]
            invalid = [n for n in ns if not n["is_valid"] or n["fitness"] is None]
            vals = [n["fitness"] for n in valid]
            norm = Normalize(min(vals) if vals else 0, max(vals) if vals else 1)
            dots = ax.scatter(
                [n["x"] for n in valid],
                [n["y"] for n in valid],
                c=vals,
                cmap="viridis",
                norm=norm,
                s=13,
                edgecolors="none",
            )
            ax.scatter(
                [n["x"] for n in invalid],
                [n["y"] for n in invalid],
                s=13,
                c="#d3dbe3",
                edgecolors="#df6864",
                linewidths=0.65,
            )
            for i, session in enumerate(task["sessions"]):
                group = [n for n in ns if n["session"] == session]
                ax.text(
                    i % task["columns"] * 2.9,
                    min(n["y"] for n in group) - 0.15,
                    f"S{i + 1} / {len(group)} attempts",
                    fontsize=8,
                    ha="center",
                    color=MUTED,
                )
            ax.autoscale()
            ax.margins(0.08)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
            cax = fig.add_axes([0.11, 0.13, 0.46, 0.015])
            fig.colorbar(
                dots,
                cax=cax,
                orientation="horizontal",
                label="Stored fitness, valid attempts only (task-local scale)",
            )
            fig.text(
                0.10,
                0.075,
                "Grey fill / red outline: invalid or unavailable fitness. All nodes and lineage edges are shown.",
                fontsize=9,
                color=MUTED,
            )
            ax2 = fig.add_axes([0.73, 0.55, 0.23, 0.28])
            sessions = task["sessions"]
            idx = {s: i for i, s in enumerate(sessions)}
            mat = np.zeros((len(sessions), len(sessions)), dtype=int)
            for a, b, k in es:
                if k == "similar":
                    mat[idx[by_id[a]["session"]], idx[by_id[b]["session"]]] += 1
            im = ax2.imshow(mat, cmap="Blues")
            fig.colorbar(im, ax=ax2, shrink=0.8)
            ticks = list(range(0, len(sessions), max(1, len(sessions) // 6)))
            ax2.set_xticks(ticks, [f"S{i + 1}" for i in ticks], rotation=45)
            ax2.set_yticks(ticks, [f"S{i + 1}" for i in ticks])
            ax2.set_xlabel("Destination session")
            ax2.set_ylabel("Source session")
            ax2.set_title(
                "All similar edges, by session\n(zero diagonal = cross-session only)",
                fontsize=10,
            )
            judges = Counter(n["judge"] or n["status"] for n in ns)
            state = "\n".join(
                textwrap.fill(f"{k}: {v:,}", width=36) for k, v in judges.most_common(5)
            )
            info = (
                f"Native vertices: {len(ns):,}\nCode attached: {sum(n['code_chars'] is not None for n in ns):,}\n"
                f"Code embeddings: {row['with_embeddings']:,}\nNon-empty change text: {row['with_changes']:,}\n"
                f"Lineage roots: {row['roots']}\nWeak components (all edges): {row['components']}\n\n"
                f"Stored evaluation status\n{state}\n\n"
                "No new evaluator or semantic-op calls.\nFitness is stored corpus telemetry,\nnot newly measured private performance."
            )
            fig.text(
                0.715, 0.475, info, fontsize=10, linespacing=1.55, color=INK, va="top"
            )
            book.savefig(fig)
            fig.savefig(out / f"{task['id'].split(':')[1]}.png", dpi=145)
            plt.close(fig)

        fig = base(
            "How ALE maps to TriMEM",
            "Construction conditions, inputs / outputs, and the role of hybrid retrieval",
            19,
        )
        y = 0.85
        for title, body in [
            (
                "Experience unit and inputs / outputs",
                "Node input: task specification, parent program, agent change text and session/iteration identity. Node output: candidate program plus stored validity, fitness, judge result, time and memory telemetry. An ALE evaluator input case is not itself an experience node. The complete task specifications are included in graph_snapshot.json.",
            ),
            (
                "Three stores, one database",
                "Relational attributes filter task/session/validity/judge/fitness. Code embeddings support vector candidate search. Parent-child lineage and cross-session similar relations live in native graph-store adjacency. This report reads those adjacency edges; it does not replace topology with relational joins.",
            ),
            (
                "Observed construction rule",
                "The local loader defines directed same-task, different-session top-16 cosine neighbors over code embeddings. All 5,269 exported nodes have 16 outgoing similar edges; every such edge crosses sessions. No similarity threshold or valid-only filter is applied by this builder. Existing similarity is not re-computed in this report.",
            ),
            (
                "Embedding provenance and semantic operators",
                "The local manifest reports Qwen3-Embedding-0.6B, 1,024 dimensions, and a 2,000-character code input limit; its node-track receipt was reconstructed from loaded data. This export checks embedding availability, not vector-model identity. Change text is a node attribute; it is not proof that an LLM semantic operator judged a transfer successful.",
            ),
            (
                "Hybrid retrieval is possible, but usefulness remains an experiment",
                "A plan may filter invalid/task-mismatched attempts, obtain vector seeds, traverse lineage or similar edges, and apply semantic selection to candidate changes. Different plans may visit different candidates. End-to-end quality, token use, construction cost and generation latency must be measured separately; graph connectivity alone does not establish a benefit.",
            ),
        ]:
            y = paragraph(fig, y, title, body)
        book.savefig(fig)
        plt.close(fig)

        fig = base(
            "Snapshot evidence and interpretation",
            "Audit trail, counting convention, and reproducibility",
            20,
        )
        y = 0.85
        for title, body in [
            (
                "Native read-only evidence",
                f"Source: {data['meta']['database']}. Timestamp: {data['meta']['snapshot_at']}. Repeatable-read transaction with default_transaction_read_only=on and bounded statement timeout. Nodes come from attempt; edges from graph_store.gph_traverse_typed(vid, type_id, 0, -1). No graph writes, model calls, or evaluator runs.",
            ),
            (
                "Forward relations versus physical reverse copies",
                f"Observed {data['audit']['native_physical_edges']:,} typed directed arcs: 5,173 lineage + 84,304 similar, and exactly matching lineage_up / similar_up reverse copies. The figures count 89,477 forward typed relations; reverse traversal copies are kept in native_snapshot.json and excluded from graph_snapshot.json. Reciprocal top-k similar edges remain distinct directed source relations.",
            ),
            (
                "Integrity checks",
                f"{len(data['nodes']):,} unique nodes / native vertex IDs; all endpoints resolve within the same task. No duplicate typed edges, self loops or external destinations. All lineage parent_id values agree. Lineage is acyclic and stays within each session; similar crosses sessions and has out-degree 16 for every node. Reverse-copy matching has zero missing or extra edges.",
            ),
            (
                "Preserved history and missing evidence",
                "Pages 1-7 are the original Math5 PDF and retain their original snapshot date. Pages 8-20 are this new ALE snapshot. All ALE nodes have cell IS NULL; the report does not merge later online agent runs. Some attempts have no change text; no summaries or links were invented to fill gaps. Node coordinates illustrate lineage layout, not embedding distance.",
            ),
            (
                "Artifacts and reproduction",
                "ale10/native_snapshot.json preserves all exported native arc types. ale10/graph_snapshot.json contains forward graph data and layout; summary.json contains counts and audit. Per-task PNGs and source scripts are stored alongside the report. experience_graphs_math5_original.pdf preserves the old PDF; ale10/ale10_appendix.pdf contains the new section. No GPU or model service is required to render again from the snapshot.",
            ),
        ]:
            y = paragraph(fig, y, title, body)
        book.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--out", type=Path, default=Path("triMEM_optimizer/experience_graphs_20261012")
    )
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55441 user=hza214 dbname=trimem_evo connect_timeout=5",
    )
    args = ap.parse_args()
    os.nice(10)
    root = args.root.resolve()
    dest = args.out.resolve()
    out = dest / "ale10"
    out.mkdir(parents=True, exist_ok=True)
    original = dest / "experience_graphs_math5_original.pdf"
    if not original.exists():
        shutil.copy2(dest / "experience_graphs.pdf", original)
    native = (
        json.loads(args.snapshot.read_text())
        if args.snapshot
        else export(args.dsn, TASKS, [t.split(":")[1].upper() for t in TASKS])
    )
    assert {n["task"] for n in native["nodes"]} == set(TASKS)
    assert all(n["cell"] is None and n["vid"] is not None for n in native["nodes"])
    assert len({n["vid"] for n in native["nodes"]}) == len(native["nodes"])
    arcs = set(map(tuple, native["edges"]))
    assert len(arcs) == len(native["edges"])
    assert not native["meta"]["external_edges"]
    by = {n["id"]: n for n in native["nodes"]}
    for a, b, k in arcs:
        assert a != b and a in by and b in by and by[a]["task"] == by[b]["task"]
        assert k in {"lineage", "similar", "lineage_up", "similar_up"}
    for k in ["lineage", "similar"]:
        assert {(b, a, k + "_up") for a, b, t in arcs if t == k} == {
            e for e in arcs if e[2] == k + "_up"
        }
    dump(out / "native_snapshot.json", native)
    data = dict(native)
    data["meta"] = dict(native["meta"])
    data["edges"] = [e for e in native["edges"] if e[2] in {"lineage", "similar"}]
    graph = nx.DiGraph()
    graph.add_nodes_from(by)
    graph.add_edges_from((a, b) for a, b, k in data["edges"] if k == "lineage")
    assert nx.is_directed_acyclic_graph(graph)
    deg = Counter(a for a, b, k in data["edges"] if k == "similar")
    assert all(deg[i] == 16 for i in by)
    assert all(
        (by[a]["session"] == by[b]["session"]) == (k == "lineage")
        for a, b, k in data["edges"]
    )
    prepare(data)
    assert not data["audit"]["lineage_parent_id_mismatches"]
    for row in data["summary"]:
        row["roots"] = sum(
            graph.in_degree(n["id"]) == 0
            for n in data["nodes"]
            if n["task"] == row["task"]
        )
    data["audit"].update(
        native_physical_edges=len(native["edges"]),
        native_edge_types=dict(Counter(e[2] for e in native["edges"])),
        reverse_copy_mismatches=0,
        similar_outdegree=16,
        similar_cross_session=True,
        lineage_acyclic=True,
    )
    data["meta"].update(
        default_scope="All ALE nodes are offline corpus (cell IS NULL)",
        counting="Forward lineage/similar only; *_up physical reverse copies are preserved separately in native_snapshot.json.",
    )
    dump(out / "graph_snapshot.json", data)
    dump(
        out / "summary.json",
        dict(summary=data["summary"], audit=data["audit"], meta=data["meta"]),
    )
    render(data, out)
    tmp = dest / "experience_graphs.updated.pdf"
    subprocess.run(
        ["pdfunite", str(original), str(out / "ale10_appendix.pdf"), str(tmp)],
        check=True,
    )
    info = subprocess.check_output(["pdfinfo", str(tmp)], text=True)
    assert "Pages:           20" in info
    original_text = subprocess.check_output(["pdftotext", str(original), "-"])
    retained_text = subprocess.check_output(
        ["pdftotext", "-f", "1", "-l", "7", str(tmp), "-"]
    )
    assert original_text == retained_text
    full_text = subprocess.check_output(["pdftotext", str(tmp), "-"], text=True)
    assert all(
        t.split(":")[1].upper() + " | Native experience graph" in full_text
        for t in TASKS
    )
    tmp.replace(dest / "experience_graphs.pdf")
    dump(
        out / "pdf_validation.json",
        dict(
            pages=20,
            original_pages=7,
            ale_pages=13,
            all_10_task_pages=True,
            retained_math_text_identical=True,
            pdfinfo=info,
        ),
    )
    for p in [
        Path(__file__),
        Path(__file__).with_name("visualize_experience_graph.py"),
    ]:
        shutil.copy2(p, out / p.name)
    sources = [
        root / "data/evotrace/normalized/embedding_manifest.json",
        root / "trimem/data/evotrace.py",
        original,
    ]
    dump(
        out / "manifest.json",
        dict(
            created_at=datetime.now(timezone.utc).isoformat(),
            audit=data["audit"],
            sources={str(p.relative_to(root)): sha(p) for p in sources},
            sha256={
                p.name: sha(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        ),
    )
    old = json.loads((dest / "manifest.json").read_text())
    old.setdefault("original_math_pdf_sha256", sha(original))
    old["sha256"]["experience_graphs.pdf"] = sha(dest / "experience_graphs.pdf")
    old["sha256"][original.name] = sha(original)
    old["pdf_extension"] = dict(
        updated_at=datetime.now(timezone.utc).isoformat(),
        total_pages=20,
        math_pages="1-7",
        ale_pages="8-20",
        ale_audit="ale10/manifest.json",
    )
    dump(dest / "manifest.json", old)
    print(json.dumps(data["audit"], indent=2))


if __name__ == "__main__":
    main()
