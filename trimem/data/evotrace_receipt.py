"""Load receipt and completeness gates of the EvoTrace trajectory store. `ok` is true only when
every gate passes; the CLI exits non-zero otherwise."""

from __future__ import annotations

from pathlib import Path

import psycopg

from trimem.data.evotrace_ds import Dataset

# Full EvoTrace revision 349117b0 (docs/trimem_trajectory_storage_v0.2.0.md §1). Applied only
# when the source directory carries that integrity_report and the full node set.
FULL = {
    "attempt": 10672,
    "session": 121,
    "task": 18,
    "lineage": 10479,
    "context_file": 30086,
    "merge_parent": 92,
    "llm_call": 18400,
    "prompt_blob": 6536,
}
FORWARD = ("lineage", "context", "merge_parent", "similar")


def _lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def _per_backend_loaded(conn: psycopg.Connection) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        "SELECT a.backend, count(*), count(a.code), count(a.emb), "
        "count(p.attempt_id) FILTER (WHERE p.source = 'prompt_blob'), "
        "count(p.attempt_id) FILTER (WHERE p.source = 'shinka_llm_result') "
        "FROM attempt a LEFT JOIN prompt p ON p.attempt_id = a.id GROUP BY 1"
    ).fetchall()
    keys = ("nodes", "code", "emb", "prompt_blob", "prompt_llm_result")
    return {b: dict(zip(keys, vals, strict=True)) for b, *vals in rows}


def _counts(conn: psycopg.Connection) -> dict:
    def q(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def grouped(sql: str) -> dict:
        return dict(conn.execute(sql).fetchall())

    return {
        "attempt": q("SELECT count(*) FROM attempt"),
        "attempt_with_vid": q("SELECT count(vid) FROM attempt"),
        "intent": q("SELECT count(intent) FROM attempt"),
        "intent_emb": q("SELECT count(intent_emb) FROM attempt"),
        "task": q("SELECT count(*) FROM task"),
        "task_emb": q("SELECT count(emb) FROM task"),
        "session": q("SELECT count(*) FROM session"),
        "prompt": q("SELECT count(*) FROM prompt"),
        "prompt_system": q("SELECT count(*) FROM prompt_system"),
        "llm_call": q("SELECT count(*) FROM llm_call"),
        "llm_call_linked": q("SELECT count(attempt_id) FROM llm_call"),
        "llm_call_ambiguous": q(
            "SELECT count(*) FROM llm_call WHERE link_method = 'response_sha1_ambiguous'"
        ),
        "llm_call_candidate": q("SELECT count(*) FROM llm_call_candidate"),
        "llm_call_purpose": grouped("SELECT purpose, count(*) FROM llm_call GROUP BY 1"),
        "search_membership": q("SELECT count(*) FROM search_membership"),
        "search_scalar": q("SELECT count(*) FROM search_scalar"),
        "search_checkpoint": q("SELECT count(*) FROM search_checkpoint"),
        "edges_mirrored": grouped(
            "SELECT t.name, count(*) FROM edge_prop e JOIN graph_store.edge_type t "
            "ON t.id = e.edge_type GROUP BY 1"
        ),
        "edges_native": grouped("SELECT edge_type, n_edges FROM trimem.graph_stats"),
    }


def receipt(conn: psycopg.Connection, ds: Dataset, merge_labels: int, **extra) -> dict:
    r = {**_counts(conn), **extra}
    edges, native = r["edges_mirrored"], r["edges_native"]
    fwd = {t: edges.get(t, 0) for t in FORWARD}
    expected, loaded = ds.expectations(), _per_backend_loaded(conn)
    r["per_backend"] = {"expected": expected, "loaded": loaded}
    gates = {
        "attempt == nodes, all with vid": r["attempt"] == len(ds.nodes) == r["attempt_with_vid"],
        "task_emb == tasks": r["task_emb"] == r["task"] == len(ds.tasks),
        "session == sessions": r["session"] == len(ds.sessions),
        "lineage == file": fwd["lineage"] == _lines(ds.src / "lineage_edges.jsonl"),
        "context + merge_parent == file": fwd["context"] + fwd["merge_parent"]
        == _lines(ds.src / "context_edges.jsonl"),
        "merge_parent == raw Merge Parent B labels": fwd["merge_parent"] == merge_labels,
        "llm_call == file": r["llm_call"] == _lines(ds.src / "llm_calls.jsonl"),
        "_up mirrors forward": all(edges.get(t + "_up", 0) == n for t, n in fwd.items()),
        "native == mirrored": all(native.get(t, 0) == n for t, n in edges.items()),
        "intent_emb == intent or 0": r["intent_emb"] in (0, r["intent"]),
        "membership nodes resolve": r.get("membership_unresolved_intervals", 0) == 0,
    }
    for b, exp in expected.items():
        got = loaded.get(b, {})
        for k in ("nodes", "code", "emb", "prompt_blob"):
            gates[f"{b}: {k} {got.get(k, 0)} == expected {exp.get(k, 0)}"] = got.get(
                k, 0
            ) == exp.get(k, 0)
        gates[f"{b}: every node has code and a vector"] = exp["code"] == exp["emb"] == exp[
            "nodes"
        ]
        if "prompt_llm_result" in exp:
            gates[f"{b}: prompt_llm_result == expected"] = (
                got.get("prompt_llm_result", 0) == exp["prompt_llm_result"]
            )
    r["full_dataset"] = bool(
        ds.integrity and ds.integrity["observed"]["nodes"] == len(ds.nodes)
    )
    if r["full_dataset"]:
        full_got = {
            "attempt": r["attempt"],
            "session": r["session"],
            "task": r["task"],
            "lineage": fwd["lineage"],
            "context_file": fwd["context"] + fwd["merge_parent"],
            "merge_parent": fwd["merge_parent"],
            "llm_call": r["llm_call"],
            "prompt_blob": sum(v.get("prompt_blob", 0) for v in loaded.values()),
        }
        for k, want in FULL.items():
            gates[f"full: {k} {full_got[k]} == {want}"] = full_got[k] == want
    r["gates"] = gates
    r["ok"] = all(gates.values())
    return r
