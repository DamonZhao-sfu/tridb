"""Read native TriMEM adjacency into an auditable, offline HTML/PDF graph snapshot.

This is an export/visualization tool, not a measured retrieval operator. Never
reconstruct topology from parent_id, similarity matrices or relational edge tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

import networkx as nx
import numpy as np
import psycopg
from psycopg.rows import dict_row

TASKS = [
    "math:circle_packing",
    "math:heilbronn_convex_13",
    "math:heilbronn_triangle",
    "math:second_autocorr_ineq",
    "math:third_autocorr_ineq",
]
TITLES = [
    "Circle Packing",
    "Heilbronn Convex 13",
    "Heilbronn Triangle",
    "Second Autocorrelation Inequality",
    "Third Autocorrelation Inequality",
]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def export(dsn, tasks=None, titles=None):
    tasks = TASKS if tasks is None else tasks
    titles = TITLES if titles is None else titles
    with psycopg.connect(
        dsn,
        row_factory=dict_row,
        options=(
            "-c default_transaction_read_only=on -c statement_timeout=60000 "
            "-c application_name=trimem_graph_visualization -c max_parallel_workers_per_gather=0"
        ),
    ) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        stamp = conn.execute(
            "SELECT current_timestamp AS time, txid_current_snapshot()::text AS snapshot"
        ).fetchone()
        nodes = conn.execute(
            """SELECT id, uid, vid, task, session, iteration, is_valid,
            fitness, status, judge, exec_time, memory_mib, language, error, changes,
            cell, parent_id, tokens, length(code) AS code_chars, emb IS NOT NULL AS has_embedding
            FROM attempt WHERE task = ANY(%s) ORDER BY id""",
            (tasks,),
        ).fetchall()
        for n in nodes:
            for k, v in n.items():
                if isinstance(v, float) and not math.isfinite(v):
                    n[k] = None
        types = conn.execute(
            "SELECT id, name FROM graph_store.edge_type ORDER BY id"
        ).fetchall()
        # Native typed adjacency is the only edge source. Stream each source's adjacency;
        # no global sort, no materialized intermediate or substitute relational topology.
        edge_sql = """SELECT a.vid AS src, e.dst FROM attempt a
            CROSS JOIN LATERAL graph_store.gph_traverse_typed(a.vid, %s, 0, -1) e
            WHERE a.task = ANY(%s)"""
        edges, external = [], []
        vids = {n["vid"]: n for n in nodes}
        for kind in types:
            with conn.cursor(name=f"graph_export_{kind['id']}") as cur:
                cur.itersize = 1000
                cur.execute(edge_sql, (kind["id"], tasks))
                for e in cur:
                    if e["dst"] in vids:
                        edges.append(
                            [vids[e["src"]]["id"], vids[e["dst"]]["id"], kind["name"]]
                        )
                    else:
                        external.append(dict(e, type=kind["name"]))
        specs = conn.execute(
            "SELECT task, specification FROM task WHERE task=ANY(%s)", (tasks,)
        ).fetchall()
    return dict(
        meta=dict(
            snapshot_at=stamp["time"].isoformat(),
            snapshot=stamp["snapshot"],
            database="127.0.0.1:55441/trimem_evo",
            read_only=True,
            source="attempt attributes + native graph_store.gph_traverse_typed adjacency",
            edge_sql=edge_sql,
            types=types,
            external_edges=external,
            default_scope="cell IS NULL: shared offline corpus used by OpenEvolve",
            other_scope="cell IS NOT NULL: historical replay writes, excluded by current OpenEvolve retrieval",
            similar_definition="Directed top-16 cosine neighbors of code embeddings, same task, different sessions; not LLM judgments.",
            lineage_definition="Directed parent-to-child program evolution; not retrieval-use attribution.",
            position_definition="Deterministic layout per session, using native lineage only. Distances are not embedding similarity.",
            references=[
                "trimem/data/evotrace.py",
                "trimem/data/build_experience_graph.py",
                "bench/agent_memory/gem_oe/trimem_retriever.py",
                "trimem/replay/memory.py",
            ],
        ),
        tasks=[
            dict(
                id=t,
                title=title,
                specification=next(
                    (r["specification"] for r in specs if r["task"] == t), ""
                ),
            )
            for t, title in zip(tasks, titles, strict=True)
        ],
        nodes=nodes,
        edges=edges,
    )


def prepare(data):
    # Partial historical ingests can leave relational rows without a native vertex.
    # Preserve these as audit evidence, but never visualize invented graph vertices.
    data.setdefault("unmapped_attempts", [n for n in data["nodes"] if n["vid"] is None])
    data["nodes"] = [n for n in data["nodes"] if n["vid"] is not None]
    by_id = {n["id"]: n for n in data["nodes"]}
    assert len(by_id) == len(data["nodes"])
    assert len({n["vid"] for n in data["nodes"]}) == len(by_id)
    mismatches = []
    for src, dst, kind in data["edges"]:
        assert src in by_id and dst in by_id
        assert by_id[src]["task"] == by_id[dst]["task"]
        if kind == "lineage" and by_id[dst]["parent_id"] != src:
            mismatches.append([src, dst])
    rows = []
    for task in data["tasks"]:
        ns = [n for n in data["nodes"] if n["task"] == task["id"]]
        ids = {n["id"] for n in ns}
        es = [e for e in data["edges"] if e[0] in ids]
        for scope in ("corpus", "all"):
            selected = [n for n in ns if scope == "all" or n["cell"] is None]
            selected_ids = {n["id"] for n in selected}
            se = [e for e in es if e[0] in selected_ids and e[1] in selected_ids]
            graph = nx.Graph()
            graph.add_nodes_from(selected_ids)
            graph.add_edges_from((e[0], e[1]) for e in se)
            counts = Counter(e[2] for e in se)
            row = dict(
                task=task["id"],
                title=task["title"],
                scope=scope,
                nodes=len(selected),
                sessions=len({n["session"] for n in selected}),
                lineage=counts["lineage"],
                similar=counts["similar"],
                edges=len(se),
                valid=sum(n["is_valid"] for n in selected),
                with_changes=sum(bool(n["changes"]) for n in selected),
                with_embeddings=sum(n["has_embedding"] for n in selected),
                components=nx.number_connected_components(graph),
                isolated=len(list(nx.isolates(graph))),
            )
            rows.append(row)
        sessions = sorted(
            {n["session"] for n in ns},
            key=lambda s: (
                next(n["cell"] is not None for n in ns if n["session"] == s),
                s,
            ),
        )
        task["sessions"] = sessions
        # All corpus sessions occupy the first rows. Replay sessions follow below.
        cols = math.ceil(
            math.sqrt(len({n["session"] for n in ns if n["cell"] is None}))
        )
        task["columns"] = cols
        for i, session in enumerate(sessions):
            group = [n for n in ns if n["session"] == session]
            member = {n["id"] for n in group}
            g = nx.Graph()
            g.add_nodes_from(sorted(member))
            g.add_edges_from(
                (a, b)
                for a, b, k in es
                if k == "lineage" and a in member and b in member
            )
            pos = nx.spring_layout(g, seed=42, iterations=55)
            for n in group:
                n["x"] = round((i % cols) * 2.9 + float(pos[n["id"]][0]), 6)
                n["y"] = round((i // cols) * 2.9 + float(pos[n["id"]][1]), 6)
                n["session_index"] = i + 1
    data["summary"] = rows
    data["audit"] = dict(
        nodes=len(by_id),
        unmapped_attempt_rows=len(data["unmapped_attempts"]),
        edges=len(data["edges"]),
        duplicate_edges=len(data["edges"]) - len({tuple(e) for e in data["edges"]}),
        lineage_parent_id_mismatches=mismatches,
        external_edges=len(data["meta"]["external_edges"]),
        verified="All exported edge endpoints resolve to actual attempt rows of the same task",
    )


def pdf(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    ink = "#183047"

    def base(title, sub, page):
        fig = plt.figure(figsize=(14, 9.5), facecolor="#f7f9fc")
        fig.text(0.05, 0.946, title, fontsize=24, fontweight="bold", color=ink)
        fig.text(0.05, 0.908, sub, fontsize=10, color="#506275")
        fig.text(
            0.05,
            0.025,
            "TriMEM  /  native experience graph  /  read-only snapshot",
            fontsize=9,
            color="#66788a",
        )
        fig.text(0.94, 0.025, str(page), ha="right", color="#66788a")
        return fig

    rows = [r for r in data["summary"] if r["scope"] == "corpus"]
    with PdfPages(out / "experience_graphs.pdf") as book:
        fig = base(
            "TriMEM | Five Math Experience Graphs",
            f"Snapshot: {data['meta']['snapshot_at']}  |  Scope: shared offline corpus (cell IS NULL)",
            1,
        )
        total = sum(r["nodes"] for r in rows)
        edges = sum(r["edges"] for r in rows)
        fig.text(
            0.05,
            0.83,
            f"{total:,} attempts       {edges:,} directed edges       {sum(r['sessions'] for r in rows)} sessions",
            fontsize=21,
            color=ink,
        )
        ax = fig.add_axes([0.045, 0.50, 0.91, 0.27])
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
            loc="center",
            cellLoc="left",
            colWidths=[0.4, 0.11, 0.11, 0.13, 0.13, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.1)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("#e0e6ed")
            cell.set_facecolor("#e5edf6" if i == 0 else "white")
        notes = [
            "A node is a program attempt: code artifact, fitness/validity, iteration, session and change summary.",
            "Lineage: parent -> child evolution. Similar: directed top-16 code-embedding neighbors in other sessions.",
            "Current OpenEvolve retrieval reads the offline corpus; its generated programs stay in OpenEvolve's database.",
            "Historical replay adds another {:,} attempts in this DB; these are available through the HTML scope selector.".format(
                len(data["nodes"]) - total
            ),
            "The shared graph is not a single task trajectory. Sessions provide the individual evolution trajectories.",
            "Edges come directly from native adjacency. Layout distances are illustrative, not embedding distances.",
            "Default HTML view: lineage edges. Enable similarity edges, or click a node to inspect its neighborhood.",
        ]
        for i, line in enumerate(notes):
            fig.text(0.055, 0.455 - i * 0.042, line, color=ink, fontsize=11)
        book.savefig(fig)
        fig.savefig(out / "overview.png", dpi=145)
        plt.close(fig)
        by_id = {n["id"]: n for n in data["nodes"]}
        for page, (task, row) in enumerate(zip(data["tasks"], rows, strict=True), 2):
            ns = [
                n
                for n in data["nodes"]
                if n["task"] == task["id"] and n["cell"] is None
            ]
            ids = {n["id"] for n in ns}
            es = [e for e in data["edges"] if e[0] in ids and e[1] in ids]
            fig = base(
                task["title"],
                f"{row['nodes']:,} nodes  |  {row['sessions']} sessions  |  {row['lineage']:,} lineage + {row['similar']:,} similarity edges  |  {row['valid'] / row['nodes']:.1%} valid",
                page,
            )
            ax = fig.add_axes([0.045, 0.18, 0.63, 0.67])
            ax.set_facecolor("white")
            fit = [n["fitness"] for n in ns if n["fitness"] is not None]
            norm = Normalize(min(fit), max(fit))
            lin = [
                [(by_id[a]["x"], by_id[a]["y"]), (by_id[b]["x"], by_id[b]["y"])]
                for a, b, k in es
                if k == "lineage"
            ]
            ax.add_collection(
                LineCollection(lin, colors="#94a6b8", linewidths=0.65, alpha=0.55)
            )
            dots = ax.scatter(
                [n["x"] for n in ns],
                [n["y"] for n in ns],
                c=[n["fitness"] if n["fitness"] is not None else np.nan for n in ns],
                cmap="viridis",
                norm=norm,
                s=11,
                edgecolors=["none" if n["is_valid"] else "#e35962" for n in ns],
                linewidths=0.6,
                plotnonfinite=True,
            )
            for i, s in enumerate(task["sessions"][: row["sessions"]]):
                group = [n for n in ns if n["session"] == s]
                ax.text(
                    (i % task["columns"]) * 2.9,
                    min(n["y"] for n in group) - 0.19,
                    f"S{i + 1} · {len(group)}",
                    ha="center",
                    fontsize=8,
                    color="#53677a",
                )
            ax.autoscale()
            ax.margins(0.07)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
            cax = fig.add_axes([0.1, 0.125, 0.49, 0.017])
            fig.colorbar(
                dots,
                cax=cax,
                orientation="horizontal",
                label="Raw fitness (scale local to this task; red outline = invalid)",
            )
            ax2 = fig.add_axes([0.735, 0.53, 0.22, 0.30])
            sessions = task["sessions"][: row["sessions"]]
            sidx = {s: i for i, s in enumerate(sessions)}
            mat = np.zeros((len(sessions), len(sessions)), dtype=int)
            for a, b, k in es:
                if k == "similar":
                    mat[sidx[by_id[a]["session"]], sidx[by_id[b]["session"]]] += 1
            im = ax2.imshow(mat, cmap="Blues")
            fig.colorbar(im, ax=ax2, shrink=0.7)
            ax2.set_title("All similarity edges\naggregated by session", fontsize=11)
            ticks = list(range(0, len(sessions), max(1, len(sessions) // 6)))
            ax2.set_xticks(ticks, [f"S{i + 1}" for i in ticks], rotation=45)
            ax2.set_yticks(ticks, [f"S{i + 1}" for i in ticks])
            ax2.set_xlabel("Destination session")
            ax2.set_ylabel("Source session")
            text = (
                f"Semantic change text: {row['with_changes']:,}/{row['nodes']:,}\n"
                f"Code embeddings: {row['with_embeddings']:,}/{row['nodes']:,}\n"
                f"Weak components (all edges): {row['components']}\nIsolated nodes (all edges): {row['isolated']}\n\n"
                "Left: all corpus nodes and lineage edges.\nSimilarity edges appear in the matrix\nto avoid obscuring the trajectories.\n\n"
                "Session layout uses lineage structure.\nPosition is not a semantic distance.\nUse HTML to inspect IDs and summaries."
            )
            fig.text(
                0.725, 0.445, text, va="top", linespacing=1.6, fontsize=10, color=ink
            )
            book.savefig(fig)
            fig.savefig(out / f"{task['id'].split(':')[1]}.png", dpi=145)
            plt.close(fig)
        fig = base(
            "How to read & reproduce this snapshot",
            "Sources, interpretation and export integrity",
            7,
        )
        paragraphs = [
            (
                "Database snapshot",
                f"{data['meta']['database']} | {data['meta']['snapshot_at']}\nRepeatable-read, read-only transaction. No model inference or graph mutations.",
            ),
            (
                "Node payload",
                "Each exported node includes attempt/vertex IDs, session, iteration, parent_id, fitness, validity, status, cell, full change summary, code length and embedding availability. Code bodies and embedding arrays are not copied into the viewer.",
            ),
            (
                "Native topology",
                "Typed edges are read using graph_store.gph_traverse_typed(vid, type_id, 0, -1). The parent_id field is checked against lineage, but is never used to manufacture edges. Similarity is not recomputed.",
            ),
            (
                "Scope boundaries",
                "The corpus is cell IS NULL. Other cells belong to historical replay runs and are selectable in HTML. The current OpenEvolve adapter keeps generated programs in its own database; its optimizer decision/outcome rows are not experience-graph vertices.",
            ),
            (
                "What an edge does not establish",
                "Lineage means parent-to-child evolution, not that a retrieval supplied that experience. Similarity means a code-embedding neighbor, not successful transfer. There are no separate lesson/skill vertices in this graph; change summaries are node attributes.",
            ),
            (
                "Audit",
                f"Exported {len(data['nodes']):,} nodes and {len(data['edges']):,} typed edges across both scopes. Duplicate edges: {data['audit']['duplicate_edges']}. Lineage/parent_id mismatches: {len(data['audit']['lineage_parent_id_mismatches'])}. Out-of-scope destination edges: {data['audit']['external_edges']}. Excluded {data['audit']['unmapped_attempt_rows']} historical attempt rows with no native vertex; preserved under unmapped_attempts in JSON.",
            ),
            (
                "Artifacts",
                "experience_graphs.html: self-contained interactive viewer. graph_snapshot.json: complete exported payload. summary.json: per-task counts. manifest.json: file hashes. Source: trimem/experiments/visualize_experience_graph.py.",
            ),
        ]
        y = 0.84
        for title, body in paragraphs:
            fig.text(0.055, y, title, fontsize=12, fontweight="bold", color=ink)
            lines = "\n".join(textwrap.fill(line, 135) for line in body.splitlines())
            fig.text(
                0.055,
                y - 0.026,
                lines,
                va="top",
                fontsize=10,
                color="#4a5e71",
                linespacing=1.5,
            )
            y -= 0.057 + 0.019 * len(lines.splitlines())
        book.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dsn", default="host=127.0.0.1 port=55441 user=hza214 dbname=trimem_evo"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--snapshot", type=Path, help="Render a previous export without DB access"
    )
    args = ap.parse_args()
    os.nice(10)
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.snapshot.read_text()) if args.snapshot else export(args.dsn)
    prepare(data)
    dump(args.out / "graph_snapshot.json", data)
    dump(
        args.out / "summary.json",
        dict(snapshot=data["meta"], summary=data["summary"], audit=data["audit"]),
    )
    template = Path(__file__).with_name("experience_graph_viewer.html").read_text()
    payload = json.dumps(
        data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    (args.out / "experience_graphs.html").write_text(
        template.replace("__GRAPH_DATA__", payload)
    )
    pdf(data, args.out)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    dump(
        args.out / "manifest.json",
        dict(
            created_at=datetime.now(timezone.utc).isoformat(),
            sha256=files,
            audit=data["audit"],
        ),
    )
    print(json.dumps(data["audit"], indent=2))


if __name__ == "__main__":
    main()
