"""Offline, provenance-preserving TriMEM mapping preview of local MLEBench data.

No DB writes, embedding inference, LLM calls, or newly inferred similarity edges.
Layout/materialization here is an offline export, not a retrieval operator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import networkx as nx

from bench.agent_memory.traceml.mlevolve_ab.journal_adapter import (
    check_integrity,
    restore_parents,
)

TASKS = [
    (
        "aes2",
        "AES2",
        "learning-agency-lab-automated-essay-scoring-2",
        "QWK ↑",
        "aes2_s41_20260904",
        "作文文本 → 1–6 分作文评分",
    ),
    (
        "ranzcr",
        "RANZCR",
        "ranzcr-clip-catheter-line-classification",
        "macro AUC ↑",
        "ranzcr_3arm_s41_20260904",
        "胸片影像 → 导管／管线位置多标签预测",
    ),
    (
        "gquest",
        "Google QUEST",
        "google-quest-challenge",
        "Spearman ρ ↑",
        "gquest_3arm_s41_20260905",
        "问题与回答文本 → 多目标质量评分",
    ),
    (
        "uspatent",
        "US Patent",
        "us-patent-phrase-to-phrase-matching",
        "Pearson r ↑",
        "uspatent_3arm_s41_20260908",
        "专利短语对与上下文 → 语义匹配分数",
    ),
    (
        "ventilator",
        "Ventilator",
        "ventilator-pressure-prediction",
        "MAE ↓",
        "ventilator_3arm_s41_20260908",
        "呼吸机控制与时间序列 → 压力预测",
    ),
]


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("triMEM_optimizer/mlebench_experience_graphs_20261012"),
    )
    args = ap.parse_args()
    os.nice(10)
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    corpus = root / "results/traceml_gem_v0.2.0/corpus/release_full_02"
    sources = []

    def record(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1048576), b""):
                h.update(chunk)
        rel = str(path.relative_to(root))
        sources.append(dict(path=rel, bytes=path.stat().st_size, sha256=h.hexdigest()))
        return rel

    manifest = json.loads((corpus / "manifest.json").read_text())
    for name in ("rows.jsonl", "edges.jsonl", "manifest.json"):
        record(corpus / name)
    record(root / "results/traceml_gem_ab/cross_task_5comp_s41/README.md")
    record(root / "bench/agent_memory/traceml/vsgt_mle/release_v2.py")
    record(root / "bench/agent_memory/traceml/mlevolve_ab/journal_adapter.py")
    datasets = [
        dict(id=k, title=t, slug=s, metric=m, task_io=io, nodes=[], edges=[], runs=[])
        for k, t, s, m, _, io in TASKS
    ]
    by_slug = {d["slug"]: d for d in datasets}
    uid_map = {}
    total = 0
    # Full original fields used for the preview are archived separately from the HTML.
    # Source prompt_input / raw execution logs are excluded from local run payloads.
    with gzip.open(
        out / "node_payloads.jsonl.gz", "wt", encoding="utf-8", compresslevel=3
    ) as archive:
        for line in (corpus / "rows.jsonl").open():
            row = json.loads(line)
            d = by_slug[row["task_uid"].removeprefix("mle:")]
            p = row["payload"]
            total += 1
            n = dict(
                id=total,
                uid=row["uid"],
                scope="corpus",
                session=row["session_uid"],
                origin="source_agent" if p["is_agent"] else "human",
                stage=p.get("stage"),
                status="unknown",
                summary=(p.get("summary") or p.get("plan") or "")[:1800],
                changes=(p.get("changes_vs_parent") or "")[:2000],
                code_preview=(p.get("code") or "")[:1800],
                code_chars=len(p.get("code") or ""),
                artifact=p.get("artifact_status"),
                ctime=p.get("ctime"),
                cv=None,
                heldout=None,
                exec_time=None,
                source="results/traceml_gem_v0.2.0/corpus/release_full_02/rows.jsonl",
                transitions=p.get("transitions", [])[:3],
            )
            assert row["uid"] not in uid_map
            uid_map[row["uid"]] = (d, n)
            d["nodes"].append(n)
            archive.write(encode(dict(id=total, uid=n["uid"], source_row=row)) + "\n")
        edge_seen = set()
        for line in (corpus / "edges.jsonl").open():
            a, b, kind = json.loads(line)
            da, na = uid_map[a]
            db, nb = uid_map[b]
            assert da is db and a != b and (a, b, kind) not in edge_seen
            assert kind in {"version", "fork", "code_sim"}
            edge_seen.add((a, b, kind))
            da["edges"].append([na["id"], nb["id"], kind])
        assert total == manifest["unique_nodes"] and len(edge_seen) == manifest["edges"]

        for (key, _, slug, _, folder, _), d in zip(TASKS, datasets, strict=True):
            run_root = root / "results/traceml_gem_ab" / folder
            for path in sorted((run_root / "raw").glob("*/journal.json")):
                source = record(path)
                journal = restore_parents(json.loads(path.read_text()))
                stat = check_integrity(journal, expect_max_roots=1).as_dict()
                arm = path.parent.name
                run = f"{key}/{arm}/s41"
                grades_path = (
                    run_root / "metrics" / f"{arm}_grading" / "node_grades.csv"
                )
                grades = {}
                if grades_path.exists():
                    record(grades_path)
                    with grades_path.open() as f:
                        grades = {r["node_id"]: r for r in csv.DictReader(f)}
                local = {}
                for p in journal["nodes"]:
                    total += 1
                    g = grades.get(p["id"], {})
                    value = finite((p.get("metric") or {}).get("value"))
                    status = (
                        "buggy"
                        if p.get("is_buggy") is True
                        else (
                            "scored"
                            if p.get("is_buggy") is False and value is not None
                            else "unknown"
                        )
                    )
                    n = dict(
                        id=total,
                        uid=f"local:{run}:{p['id']}",
                        scope="local",
                        session=run,
                        origin="local_agent",
                        stage=p.get("stage"),
                        status=status,
                        step=p.get("step"),
                        summary=p.get("plan") or p.get("code_summary") or "",
                        changes=p.get("analysis") or "",
                        code_preview=(p.get("code") or "")[:4000],
                        code_chars=len(p.get("code") or ""),
                        artifact="local_journal",
                        ctime=p.get("ctime"),
                        cv=value,
                        heldout=finite(g.get("heldout_score")),
                        valid_submission=g.get("valid_submission") or None,
                        exec_time=finite(p.get("exec_time")),
                        source=source,
                        error=p.get("exc_type"),
                        parent_source_id=p.get("parent"),
                    )
                    local[p["id"]] = n
                    d["nodes"].append(n)
                    archive.write(
                        encode(
                            dict(
                                id=total,
                                uid=n["uid"],
                                source=source,
                                node={
                                    k: p.get(k)
                                    for k in (
                                        "id",
                                        "parent",
                                        "stage",
                                        "step",
                                        "ctime",
                                        "code",
                                        "plan",
                                        "analysis",
                                        "metric",
                                        "is_buggy",
                                        "exec_time",
                                        "exc_type",
                                        "code_summary",
                                    )
                                },
                                grading=g,
                            )
                        )
                        + "\n"
                    )
                for p in journal["nodes"]:
                    if p.get("parent"):
                        assert p["parent"] in local
                        d["edges"].append(
                            [local[p["parent"]]["id"], local[p["id"]]["id"], "lineage"]
                        )
                d["runs"].append(
                    dict(
                        id=run,
                        arm=arm,
                        source=source,
                        tree=stat,
                        scope="complete saved journal; no budget cutoff",
                    )
                )

    for d in datasets:
        by_id = {n["id"]: n for n in d["nodes"]}
        d["summary"] = {}
        for scope in ("local", "corpus"):
            ns = [n for n in d["nodes"] if n["scope"] == scope]
            ids = {n["id"] for n in ns}
            es = [e for e in d["edges"] if e[0] in ids]
            assert all(e[1] in ids for e in es)
            g = nx.DiGraph()
            g.add_nodes_from(ids)
            g.add_edges_from((a, b) for a, b, kind in es if kind != "code_sim")
            assert nx.is_directed_acyclic_graph(g)
            components = sorted(
                nx.weakly_connected_components(g), key=lambda c: (-len(c), min(c))
            )
            full = g.to_undirected()
            full.add_edges_from((a, b) for a, b, _ in es)
            counts = Counter(e[2] for e in es)
            d["summary"][scope] = dict(
                nodes=len(ns),
                edges=len(es),
                edge_types=dict(counts),
                sessions=len({n["session"] for n in ns}),
                lineage_components=len(components),
                full_components=nx.number_connected_components(full),
                origins=dict(Counter(n["origin"] for n in ns)),
                statuses=dict(Counter(n["status"] for n in ns)),
                unresolved=sum(n["artifact"] == "unresolved" for n in ns),
            )
            cols = math.ceil(math.sqrt(len(components) * 1.45))
            for ci, ids in enumerate(components):
                sub = g.subgraph(sorted(ids))
                if scope == "local":
                    # Actual parent tree: depth vertically, ordered leaves horizontally.
                    pos, cursor = {}, [0]

                    def visit(u, depth):
                        children = sorted(
                            sub.successors(u),
                            key=lambda v: (by_id[v].get("step") or 0, v),
                        )
                        xs = [visit(v, depth + 1) for v in children]
                        if xs:
                            x = sum(xs) / len(xs)
                        else:
                            x = cursor[0]
                            cursor[0] += 1
                        pos[u] = (x, depth)
                        return x

                    for u in sub:
                        if sub.in_degree(u) == 0:
                            visit(u, 0)
                elif len(ids) > 1:
                    pos = nx.spring_layout(sub.to_undirected(), seed=41, iterations=45)
                else:
                    pos = {next(iter(ids)): (0, 0)}
                xmin = min(v[0] for v in pos.values())
                ymin = min(v[1] for v in pos.values())
                xr = max(v[0] for v in pos.values()) - xmin or 1
                yr = max(v[1] for v in pos.values()) - ymin or 1
                for nid, (x, y) in pos.items():
                    by_id[nid]["xy"] = [
                        round(ci % cols * 360 + (x - xmin) / xr * 280, 4),
                        round(ci // cols * 300 + (y - ymin) / yr * 220, 4),
                    ]
                    by_id[nid]["component"] = ci
        print(d["id"], encode(d["summary"]), flush=True)

    data = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        datasets=datasets,
        meta=dict(
            mode="offline TriMEM mapping preview; not a live TriMEM adjacency export",
            corpus_revision=manifest["revision"],
            source_manifest=manifest,
            journal_selection="12 complete A0/A1/GEM journals listed by cross_task_5comp_s41/README.md; not an exhaustive inventory of later cohorts",
            new_inference=False,
            database_writes=False,
            gpu_used=False,
            edge_count_semantics="One original source record per edge; no reverse-edge duplication. code_sim orientation is preserved in JSON but displayed without causal arrows.",
            validation="Unique UIDs/typed edges, endpoints present, same competition/scope, no self loops, acyclic lineage; restored local parent maps checked as trees.",
            placeholders="Corpus fitness=0 and is_valid=true are indexing sentinels; execution outcomes are unknown.",
            session_semantics="Historical human session_uid is inherited from source tree_id (or key_id), not necessarily one notebook/run. Agent branch suffixes were collapsed by the source importer. See release_v2.identity().",
            layout="Source lineage components laid out separately; positions are not embedding distances.",
        ),
        sources=sources,
    )
    raw = encode(data)
    (out / "graph_data.json").write_text(raw, encoding="utf-8")
    template = Path(__file__).with_name("mlebench_graph_viewer.html").read_text()
    (out / "experience_graphs.html").write_text(
        template.replace("__GRAPH_DATA__", raw.replace("<", "\\u003c")),
        encoding="utf-8",
    )
    (out / "source_manifest.json").write_text(
        json.dumps(
            dict(created_at=data["created_at"], sources=sources, meta=data["meta"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "Output:",
        out,
        "nodes:",
        total,
        "edges:",
        sum(len(d["edges"]) for d in datasets),
        flush=True,
    )


if __name__ == "__main__":
    main()
