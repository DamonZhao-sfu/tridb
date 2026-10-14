"""Relational loaders of the EvoTrace trajectory store. Each streams into COPY and runs inside
the caller's load transaction (trimem/data/evotrace.py `load`)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import psycopg

from trimem.data import evotrace_text as tx
from trimem.data.evotrace_ds import Dataset, as_float, clean, jsonb, vec

NodeSha = dict[str, dict[str, list[int]]]  # session -> response sha1 -> attempt ids


def load_tasks_sessions(conn: psycopg.Connection, ds: Dataset) -> None:
    with conn.cursor().copy(
        "COPY task (task, domain, family, task_key, specification, emb) FROM STDIN"
    ) as cp:
        for t in ds.tasks:
            cp.write_row(
                (
                    t["task_uid"],
                    t.get("domain"),
                    t.get("task_family"),
                    t.get("task_key"),
                    clean(t.get("specification")),
                    vec(ds.vectors.get(("task", t["task_uid"]))),
                )
            )
    with conn.cursor().copy(
        "COPY session (session, task, backend, domain, model, mode, temperature, "
        "search_algorithm, configured_iterations, run_rel, prompt_complete_ratio, "
        "has_evaluator, best_attempt) FROM STDIN"
    ) as cp:
        for s in ds.sessions:
            files = s.get("files_present") or {}
            cp.write_row(
                (
                    s["session_uid"],
                    s["task_uid"],
                    s["backend"],
                    s.get("domain"),
                    s.get("model"),
                    s.get("mode"),
                    as_float(s.get("temperature")),
                    s.get("search_algorithm"),
                    s.get("configured_iterations"),
                    s["run_rel"],
                    as_float(s.get("prompt_complete_ratio")),
                    bool(files.get("evaluate.py")),
                    ds.ids.get(s.get("best_node_uid") or ""),
                )
            )


def load_attempts(
    conn: psycopg.Connection, ds: Dataset, count_tokens: Callable | None
) -> tuple[NodeSha, dict[str, str]]:
    """Pass 1: one row per node. Returns per-session {response sha1: [attempt ids]} for the
    llm_call link and the distinct system prompts."""
    parent = {
        e["dst_node_uid"]: ds.ids[e["src_node_uid"]]
        for e in tx.read_jsonl(ds.src / "lineage_edges.jsonl")
    }
    node_sha: NodeSha = defaultdict(lambda: defaultdict(list))
    systems: dict[str, str] = {}
    cols = (
        "id, uid, task, session, backend, program_id, iteration, generation, is_valid, "
        "fitness, status, judge, exec_time, memory_mib, language, error, "
        "framework_changes, intent, intent_source, artifact_sha, code, parent_id, tokens, emb"
    )
    with conn.cursor().copy(f"COPY attempt ({cols}) FROM STDIN") as cp:
        for n, t in ds.texts():
            aid, s = ds.ids[n["node_uid"]], n["session_uid"]
            if t and t.response:
                node_sha[s][tx.sha1(clean(t.response))].append(aid)
            if t and t.system:
                systems[tx.sha1(clean(t.system))] = clean(t.system)
            code = clean(ds.code(n))
            m = n.get("metrics") or {}
            cp.write_row(
                (
                    aid,
                    n["node_uid"],
                    n["task_uid"],
                    s,
                    ds.backend[s],
                    n["program_id"],
                    int(n["iteration"]),
                    n.get("generation"),
                    bool(n["is_valid"]),
                    as_float(n.get("fitness")),
                    n["status"],
                    m.get("judge_result"),
                    as_float(m.get("max_execution_time_sec")),
                    as_float(m.get("max_memory_usage_mib")),
                    n.get("language"),
                    clean(n.get("error_signature")),
                    clean(n.get("changes")),
                    clean(t.intent) if t else None,
                    t.intent_source if t else None,
                    n.get("solution_sha256"),
                    code,
                    parent.get(n["node_uid"]),
                    count_tokens(code) if count_tokens and code else None,
                    vec(ds.vectors.get(("node", n["node_uid"]))),
                )
            )
    # Native vids: allocated in id order; gph_upsert_vertex reuses gph_vid_map on a re-run.
    # Readers join through attempt.vid (trimem.expand, tjs_open id_col='vid'); id need not = vid.
    conn.execute(
        "UPDATE attempt a SET vid = m.vid FROM (SELECT id, graph_store.gph_upsert_vertex(id)"
        " AS vid FROM (SELECT id FROM attempt ORDER BY id) o) m WHERE a.id = m.id"
    )
    for aids in (x for d in node_sha.values() for x in d.values()):
        aids.sort(key=lambda a: (int(ds.nodes[a - 1]["iteration"]), a))
    return node_sha, systems


def load_prompts(conn: psycopg.Connection, ds: Dataset, systems: dict[str, str]) -> None:
    """Pass 2 (separate so attempt rows exist for the FK): prompt text per attempt."""
    with conn.cursor().copy("COPY prompt_system (system_sha, text) FROM STDIN") as cp:
        for sha, text in systems.items():
            cp.write_row((sha, text))
    with conn.cursor().copy(
        "COPY prompt (attempt_id, template, system_sha, user_text, response_text, "
        "response_sha1, prompts_sha256, source) FROM STDIN"
    ) as cp:
        for n, t in ds.texts():
            if t is None or (t.response is None and t.user is None):
                continue  # shinka nodes with only a patch_description: intent, no prompt
            resp = clean(t.response)
            cp.write_row(
                (
                    ds.ids[n["node_uid"]],
                    t.template,
                    tx.sha1(clean(t.system)) if t.system else None,
                    clean(t.user),
                    resp,
                    tx.sha1(resp) if resp else None,
                    n.get("prompts_sha256"),
                    t.source,
                )
            )


def _llm_row(s: str, r: dict, x: dict, h: str | None, link: tx.Link, backend: str):
    system = clean(x.get("system_msg"))
    purpose = (
        "mutation_ambiguous"
        if link.method == "response_sha1_ambiguous" and not x.get("error")
        else tx.classify_call(x, link.attempt_id is not None, backend)
    )
    return system, (
        s,
        r["seq"],
        link.attempt_id,
        link.method,
        purpose,
        r.get("model"),
        as_float(r.get("temperature")),
        r.get("finish_reason"),
        r.get("prompt_tokens"),
        r.get("completion_tokens"),
        r.get("reasoning_tokens"),
        clean(r.get("error")),
        clean(x.get("content")),
        clean(x.get("reasoning_content")),
        h,
        tx.sha1(system) if system else None,
        clean(x.get("user_msg")),
    )


def load_llm_calls(conn: psycopg.Connection, ds: Dataset, node_sha: NodeSha) -> None:
    """Stream each session's raw LLM log beside its normalized rows (same seq order). Only a
    1:1 exact match sets attempt_id; ambiguous groups go to llm_call_candidate."""
    norm: dict[str, list[dict]] = defaultdict(list)
    for r in tx.read_jsonl(ds.src / "llm_calls.jsonl"):
        norm[r["session_uid"]].append(r)
    systems: dict[str, str] = {}
    candidates: list[tuple[str, int, int]] = []
    cols = (
        "session, seq, attempt_id, link_method, purpose, model, temperature, finish_reason, "
        "prompt_tokens, completion_tokens, reasoning_tokens, error, content, reasoning, "
        "content_sha1, system_sha, user_text"
    )
    with conn.cursor().copy(f"COPY llm_call ({cols}) FROM STDIN") as cp:
        for s in ds.run_rel:
            rows = sorted(norm.get(s, []), key=lambda r: r["seq"])
            if not rows:
                continue
            raw = list(tx.read_jsonl(ds.run_dir(s) / "logs/llm_calls.jsonl"))
            if len(raw) != len(rows):
                raise RuntimeError(f"{s}: {len(raw)} raw vs {len(rows)} normalized calls")
            shas = [tx.sha1(clean(x["content"])) if x.get("content") else None for x in raw]
            links = tx.pair_links(shas, node_sha.get(s, {}))
            for r, x, h, link in zip(rows, raw, shas, links, strict=True):
                system, row = _llm_row(s, r, x, h, link, ds.backend[s])
                if system:
                    systems[tx.sha1(system)] = system
                if link.method == "response_sha1_ambiguous":
                    candidates += [(s, r["seq"], a) for a in link.candidates]
                cp.write_row(row)
    with conn.cursor().copy(
        "COPY llm_call_candidate (session, seq, attempt_id) FROM STDIN"
    ) as cp:
        for row in candidates:
            cp.write_row(row)
    known = {r[0] for r in conn.execute("SELECT system_sha FROM prompt_system")}
    with conn.cursor().copy("COPY prompt_system (system_sha, text) FROM STDIN") as cp:
        for sha, text in systems.items():
            if sha not in known:
                cp.write_row((sha, text))


def load_state(conn: psycopg.Connection, ds: Dataset) -> dict[str, int]:
    """Stream state_events: scalars straight to COPY, membership run-length encoded."""
    rle = tx.MembershipRLE()
    scalars = 0
    with conn.cursor().copy(
        "COPY search_scalar (session, iteration, key, value, attempt_id) FROM STDIN"
    ) as cp:
        for e in tx.read_jsonl(ds.src / "state_events.jsonl"):
            if e["entity"] == "membership":
                rle.add(e)
            else:
                scalars += 1
                cp.write_row(
                    (
                        e["session_uid"],
                        e["iteration"],
                        e["role"],
                        jsonb(e["value"]),
                        ds.ids.get(e.get("node_uid") or ""),
                    )
                )
    with conn.cursor().copy("COPY search_checkpoint (session, iteration) FROM STDIN") as cp:
        for row in rle.checkpoint_rows():
            cp.write_row(row)
    with conn.cursor().copy(
        "COPY search_membership (session, role, slot_key, attempt_id, from_iter, to_iter, "
        "value) FROM STDIN"
    ) as cp:
        unresolved = 0
        for s, role, slot, node, lo, hi, value in rle.intervals():
            unresolved += bool(node) and node not in ds.ids
            cp.write_row((s, role, slot, ds.ids.get(node or ""), lo, hi, value))
    return {
        "membership_unresolved_intervals": unresolved,
        "membership_events": rle.events,
        "membership_duplicate_events": rle.duplicates,
        "scalar_events": scalars,
    }
