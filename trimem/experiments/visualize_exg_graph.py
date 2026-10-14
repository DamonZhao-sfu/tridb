"""Visualize existing EXG commit receipts without starting DB/model services."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

from exg.collection.common import case_hash_matches

REPO = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def collect(source, out, dataset):
    folder = source / dataset
    nodes, edges, evidence, source_hashes = {}, {}, [], []
    completed = {}
    for path in sorted((folder / "tasks").glob("*/*/completed.json")):
        r = read(path)
        assert r["sequence"] not in completed
        completed[r["sequence"]] = r
        source_hashes.append(dict(path=str(path.relative_to(REPO)), sha256=sha(path)))
    planned = set(
        read(
            REPO
            / "results/trimem_exg_qwen38_v1.0.3/main/data"
            / dataset
            / "manifest.json"
        )["online"]
    )
    task_ids = set()
    last_commit = 0
    with gzip.open(
        out / f"{dataset}_case_payloads.jsonl.gz", "wt", encoding="utf-8"
    ) as full:
        for index, path in enumerate(
            sorted((folder / "tasks").glob("*/*/attempt_*/commit.json"))
        ):
            receipt = read(path)
            casepath = path.with_name("case.json")
            case = read(casepath)
            assert case_hash_matches(case, receipt["case_hash"]), str(casepath)
            assert case["uid"] == receipt["uid"] and case["dataset"] == dataset
            vid, anchor = receipt["case_id"], receipt["anchor_id"]
            assert vid not in nodes
            task_ids.add(case["task_id"])
            last_commit = max(last_commit, receipt["committed_at"])
            prediction = case.get("prediction") or {}
            answer = (
                prediction.get("answer") if isinstance(prediction, dict) else prediction
            )
            node = dict(
                id=vid,
                uid=case["uid"],
                kind="Case",
                task_id=case["task_id"],
                sequence=case["sequence"],
                attempt=case["attempt"],
                status=case["status"],
                question=case["input"].get("question", ""),
                answer=answer,
                score=case["score"],
                failure_type=case.get("failure_type"),
                feedback=(case.get("feedback") or "")[:1500],
                feedback_preview_truncated=len(case.get("feedback") or "") > 1500,
                retry_feedback=(case.get("retry_feedback") or "")[:1500],
                retry_feedback_preview_truncated=len(case.get("retry_feedback") or "")
                > 1500,
                needs_review=prediction.get("needs_review")
                if isinstance(prediction, dict)
                else None,
                fact_count=len(case["input"].get("facts", [])),
                source_case=str(casepath.relative_to(REPO)),
                committed_at=receipt["committed_at"],
                in_planned_500=case["task_id"] in planned,
                hints=case.get("memory", {}).get("actual_hint_count"),
                model=case.get("model", {}).get("agent_model"),
            )
            nodes[vid] = node
            if anchor not in nodes:
                nodes[anchor] = dict(
                    id=anchor,
                    uid="anchor:" + case["task_id"],
                    kind="TaskAnchor",
                    task_id=case["task_id"],
                    sequence=case["sequence"],
                    attempt=0,
                    status="anchor",
                    question=node["question"],
                    in_planned_500=node["in_planned_500"],
                )
            assert nodes[anchor]["task_id"] == case["task_id"]
            for e in receipt["edges"]:
                key = (e["src"], e["dst"], e["relation"])
                assert key not in edges
                edges[key] = e
            # Full output, feedback and score retained separately; omit large vectors,
            # passages and retrieval internals, whose original paths/hashes are recorded.
            full.write(
                json.dumps(
                    dict(
                        id=vid,
                        **{
                            k: v
                            for k, v in case.items()
                            if k
                            not in (
                                "query_vector",
                                "failure_vector",
                                "memory",
                                "embedding_receipt",
                                "input",
                            )
                        },
                        question=node["question"],
                        original_case=node["source_case"],
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            source_hashes.extend(
                dict(path=str(p.relative_to(REPO)), sha256=sha(p))
                for p in (path, casepath)
            )
            if index % 4000 == 0:
                print(dataset, "verified cases", index + 1, flush=True)
    contain = Counter()
    similar = set()
    fixed = []
    for (a, b, k), e in edges.items():
        assert a in nodes and b in nodes
        na, nb = nodes[a], nodes[b]
        if k == "contain":
            assert (
                na["kind"] == "TaskAnchor"
                and nb["kind"] == "Case"
                and na["task_id"] == nb["task_id"]
            )
            contain[b] += 1
        elif k == "similar_to":
            assert na["kind"] == nb["kind"] == "Case" and a != b
            assert (b, a, k) in edges
            assert e["evidence"]["cosine"] >= e["evidence"]["threshold"] - 1e-6
            similar.add(tuple(sorted((a, b))))
        elif k == "fixed_by":
            assert na["task_id"] == nb["task_id"] and na["attempt"] < nb["attempt"]
            assert na["status"] == "warning" and nb["status"] == "golden"
            assert (
                na["score"]["valid"]
                and not na["score"]["passed"]
                and nb["score"]["passed"]
            )
            assert (
                na["score"]["scorer"] == nb["score"]["scorer"]
                and e["evidence"]["feedback"]
            )
            fixed.append((a, b))
        else:
            raise AssertionError(k)
        evidence.append(e)
    cases = [n for n in nodes.values() if n["kind"] == "Case"]
    assert all(contain[n["id"]] == 1 for n in cases)
    for seq, r in completed.items():
        for attempt in r["attempts"]:
            n = nodes[attempt["case_id"]]
            assert (n["uid"], n["sequence"], n["status"]) == (
                attempt["uid"],
                seq,
                attempt["status"],
            )
    # Compare the saved pilot graph to the corresponding receipt-derived subgraph.
    with gzip.open(folder / "report/nodes.jsonl.gz", "rt") as f:
        pilot_nodes = [json.loads(line) for line in f]
    with gzip.open(folder / "report/edges.jsonl.gz", "rt") as f:
        pilot_edges = [json.loads(line) for line in f]
    for n in pilot_nodes:
        assert nodes[n["id"]]["uid"] == n["uid"]
    pilot_ids = {n["id"] for n in pilot_nodes}
    assert {(e["src"], e["dst"], e["relation"]) for e in pilot_edges} == {
        k for k in edges if k[0] in pilot_ids and k[1] in pilot_ids
    }
    logical = [
        [a, b, k, edges[(a, b, k)]["evidence"].get("cosine")]
        for a, b, k in sorted(edges)
        if k != "similar_to" or a < b
    ]
    for n in nodes.values():
        n["degree"] = 0
    for a, b, _, _ in logical:
        nodes[a]["degree"] += 1
        nodes[b]["degree"] += 1
    status = Counter(n["status"] for n in cases)
    summary = dict(
        dataset=dataset,
        completed_tasks=len(completed),
        tasks_with_cases=len(task_ids),
        target=19938 if dataset == "musique" else 90447,
        nodes=len(nodes),
        anchors=len(task_ids),
        cases=len(cases),
        contain=sum(contain.values()),
        similar_to=len(similar),
        fixed_by=len(fixed),
        physical_arcs=len(edges),
        logical_edges=len(logical),
        **status,
        first_commit=min(n["committed_at"] for n in cases),
        last_commit=last_commit,
        historical_tasks_matching_planned_500=len(task_ids & planned),
        collection_complete=False,
        current_500_experiment="paused, no formal results",
        source_status=read(folder / "status.json"),
        topology_source="Persisted commit.json receipts from native-graph insertion transactions; live DB not checked",
        checks=dict(
            case_hashes=len(cases),
            edge_endpoints=True,
            similar_symmetric=True,
            contain_one_per_case=True,
            fixed_by_status_direction=True,
            completed_receipts=len(completed),
            pilot_export_edges=len(pilot_edges),
        ),
    )
    with gzip.open(
        out / f"{dataset}_physical_edges.jsonl.gz", "wt", encoding="utf-8"
    ) as f:
        for e in evidence:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with gzip.open(
        out / f"{dataset}_source_hashes.jsonl.gz", "wt", encoding="utf-8"
    ) as f:
        for r in source_hashes:
            f.write(json.dumps(r) + "\n")
    return dict(
        id=dataset,
        title="MuSiQue" if dataset == "musique" else "HotpotQA",
        summary=summary,
        nodes=sorted(nodes.values(), key=lambda n: n["id"]),
        edges=logical,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--source", type=Path, default=REPO / "exg/results/experience_graph_v0.1.0"
    )
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    os.nice(10)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / "graph_snapshot.json"
    if args.render_only:
        data = read(snapshot)
    else:
        data = dict(
            created_at=datetime.now(timezone.utc).isoformat(),
            source=str(args.source),
            scope="Historical partial EXG collection, separate from the paused 500-task experiment",
            live_native_graph_verified=False,
            protocol=dict(
                similarity="Question embedding, prior ANN top10, cosine >= 0.75; symmetric arcs",
                retry="Historical QA: needs_review or execution/parse error; gold used only for offline scoring",
                current_protocol_difference="Planned 500-task run uses correct/incorrect feedback; keep results separate",
            ),
            datasets=[
                collect(args.source, args.out, d) for d in ("musique", "hotpotqa")
            ],
        )
        write(snapshot, data)
        write(
            args.out / "summary.json", {d["id"]: d["summary"] for d in data["datasets"]}
        )
        shutil.copy2(
            args.source / "protocol.json", args.out / "historical_protocol.json"
        )
    template = Path(__file__).with_name("exg_graph_viewer.html").read_text()
    payload = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    (args.out / "experience_graphs.html").write_text(
        template.replace("__GRAPH_DATA__", payload)
    )
    target = args.out / "source"
    target.mkdir(exist_ok=True)
    for p in (Path(__file__), Path(__file__).with_name("exg_graph_viewer.html")):
        shutil.copy2(p, target / p.name)
    write(
        args.out / "manifest.json",
        dict(
            sha256={
                str(p.relative_to(args.out)): sha(p)
                for p in args.out.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            }
        ),
    )
    print(json.dumps({d["id"]: d["summary"] for d in data["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
