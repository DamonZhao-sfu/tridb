"""P2 — load the normalized EvoTrace corpus into the Experience Graph.

    python3 -m bench.agent_memory.gem_eg.load --dsn ... --scope evotrace:349117b0
    python3 -m bench.agent_memory.gem_eg.load --counts-only     # re-print the receipt

One transaction per session, plus one for the corpus-global Tasks and Artifacts. Every
vertex, its typed detail row, its edges, its inverse edges and its state events commit
together — a half-loaded session never becomes visible, so a query can never see a Node
whose Session is missing.

Idempotence is structural, not hopeful: vertices are keyed ``(scope_id, uid)`` and
edges ``(src, dst, edge_type)``, and the adjacency-method insert is gated on the
relational mirror actually having inserted. Re-running adds zero of either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.store import (
    DEFAULT_DIM,
    DEFAULT_DSN,
    EgStore,
    LoadReceipt,
    read_jsonl,
)

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_RAW = Path("data/evotrace/raw")

#: Artifacts larger than this keep their reference and checksum but not their bytes.
#: TriDB rule 2 forbids a sidecar object store, so oversized payloads are reported as
#: not-inlined rather than quietly shipped somewhere else.
ARTIFACT_INLINE_LIMIT = 1 << 20


def _vertex_row(
    *,
    vid: int,
    scope_id: str,
    kind: str,
    uid: str,
    label: str,
    **columns: Any,
) -> tuple[Any, ...]:
    return (
        vid,
        scope_id,
        kind,
        uid,
        label[:200],
        columns.get("task_uid"),
        columns.get("session_uid"),
        columns.get("backend"),
        columns.get("domain"),
        columns.get("language"),
        columns.get("model"),
        columns.get("status"),
        columns.get("is_valid"),
        columns.get("fitness"),
        columns.get("iteration"),
        columns.get("generation"),
        columns.get("visibility_group"),
        json.dumps(columns.get("provenance") or {}),
        json.dumps(columns.get("metadata") or {}),
    )


_VERTEX_SQL = (
    "INSERT INTO gem_eg_vertex"
    " (id, scope_id, kind, uid, label, task_uid, session_uid, backend, domain,"
    "  language, model, status, is_valid, fitness, iteration, generation,"
    "  visibility_group, provenance, metadata)"
    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    " ON CONFLICT (scope_id, uid) DO NOTHING"
)


class Loader:
    def __init__(
        self,
        store: EgStore,
        *,
        scope_id: str,
        normalized: Path,
        raw: Path,
        revision: str,
    ) -> None:
        self.store = store
        self.conn = store.conn
        self.scope_id = scope_id
        self.normalized = normalized
        self.raw = raw
        self.revision = revision
        self.receipt = LoadReceipt(scope_id=scope_id, dataset_revision=revision)
        self.vids: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------

    def _alloc(self, uids: list[str]) -> dict[str, int]:
        """Allocate vids for uids not already present. Returns uid -> vid for ALL."""
        missing = [u for u in uids if u not in self.vids]
        if missing:
            for uid, vid in zip(missing, self.store.allocate_vertices(len(missing))):
                self.vids[uid] = vid
        return {u: self.vids[u] for u in uids}

    def _hier_link(
        self, src: int, dst: int, *, relation: str, specific: str, provenance: str
    ) -> None:
        """A hierarchy edge: specific type, rollup type, and both inverses.

        Four physical adjacency entries for one logical parent-child containment. The
        rollup exists so ONE bounded BFS can cross Task -> Session -> Node
        (gph_traverse_bounded takes a single type id); the inverses exist because the
        access method traverses outgoing edges only.
        """
        if self.store.link(
            src, dst, relation=relation, edge_type=specific, provenance=provenance,
            dataset_revision=self.revision,
        ):
            self.receipt.add(f"edge:{relation}")
        if self.store.link(
            src, dst, relation=relation, edge_type="eg_hier", rollup_of=specific,
            provenance=provenance, dataset_revision=self.revision,
        ):
            self.receipt.add("edge:rollup")
        if self.store.link(
            dst, src, relation=f"{relation}_inv", edge_type="eg_hier_inv",
            derived_inverse=True, rollup_of=specific, provenance=provenance,
            dataset_revision=self.revision,
        ):
            self.receipt.add("edge:inverse")

    # -- phases ----------------------------------------------------------

    def load_tasks(self) -> None:
        tasks = list(read_jsonl(self.normalized / "tasks.jsonl"))
        mapping = self._alloc([t["task_uid"] for t in tasks])
        for task in tasks:
            vid = mapping[task["task_uid"]]
            self.conn.execute(
                _VERTEX_SQL,
                _vertex_row(
                    vid=vid,
                    scope_id=self.scope_id,
                    kind="task",
                    uid=task["task_uid"],
                    label=task["task_key"],
                    task_uid=task["task_uid"],
                    domain=task["domain"],
                    metadata={
                        "backends": task.get("backends", []),
                        "models": task.get("models", []),
                        "languages": task.get("languages", []),
                        "session_count": task.get("session_count"),
                    },
                ),
            )
            self.conn.execute(
                "INSERT INTO gem_eg_task"
                " (id, task_uid, domain, task_key, task_family, specification,"
                "  specification_complete, specification_source, success_metric,"
                "  target_environment)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (task_uid) DO NOTHING",
                (
                    vid,
                    task["task_uid"],
                    task["domain"],
                    task["task_key"],
                    task.get("task_family"),
                    # No canonical problem statement ships in the core files. This is a
                    # stand-in, and `specification_complete=false` is what forbids
                    # calling the ANN over it a task-DESCRIPTION ANN.
                    task.get("specification") or f"{task['domain']} task {task['task_key']}",
                    bool(task.get("specification_complete", False)),
                    task.get("specification_source", "run_name_only"),
                    "combined_score",
                    json.dumps({"languages": task.get("languages", [])}),
                ),
            )
            self.receipt.add("task_processed")
        self.conn.commit()

    def load_artifacts(self) -> None:
        rows = list(read_jsonl(self.normalized / "artifacts.jsonl"))
        for art in rows:
            payload, byte_len = self._artifact_payload(art)
            self.conn.execute(
                "INSERT INTO gem_eg_artifact"
                " (artifact_uid, sha256, kind, language, blob_rel, byte_len, payload,"
                "  reference_count, session_count)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (artifact_uid) DO UPDATE SET"
                "   reference_count = EXCLUDED.reference_count,"
                "   session_count   = EXCLUDED.session_count",
                (
                    art["artifact_uid"],
                    art["sha256"],
                    art.get("kind", "solution"),
                    art.get("language"),
                    art.get("blob_rel"),
                    byte_len,
                    payload,
                    art.get("reference_count", 0),
                    len(art.get("sessions", [])),
                ),
            )
            self.receipt.add("artifact_processed")
        self.conn.commit()

    def _artifact_payload(self, art: dict[str, Any]) -> tuple[str | None, int | None]:
        """Inline the code bytes when we hold them and they fit.

        A missing blob yields (None, None) and is COUNTED. "we have the checksum but
        not the bytes" is a real state of this corpus and must stay visible.
        """
        sessions = art.get("sessions") or []
        blob_rel = art.get("blob_rel")
        if not blob_rel:
            self.receipt.reject("artifact_without_blob_ref")
            return None, None
        for session_uid in sessions:
            run_rel = session_uid.split(":", 1)[-1]
            path = self.raw / run_rel / blob_rel
            if path.is_file():
                size = path.stat().st_size
                if size > ARTIFACT_INLINE_LIMIT:
                    self.receipt.add("artifact_payload_too_large")
                    return None, size
                self.receipt.add("artifact_payload_inlined")
                return path.read_text(encoding="utf-8", errors="replace"), size
        self.receipt.reject("artifact_blob_missing")
        return None, None

    def load_sessions(self) -> None:
        sessions = {s["session_uid"]: s for s in read_jsonl(self.normalized / "sessions.jsonl")}
        nodes_by_session = _group(read_jsonl(self.normalized / "nodes.jsonl"), "session_uid")
        lineage = _group(read_jsonl(self.normalized / "lineage_edges.jsonl"), "session_uid")
        context = _group(read_jsonl(self.normalized / "context_edges.jsonl"), "session_uid")
        prompts = _group(read_jsonl(self.normalized / "prompts.jsonl"), "session_uid")
        events = _group(read_jsonl(self.normalized / "state_events.jsonl"), "session_uid")
        calls = _group(read_jsonl(self.normalized / "llm_calls.jsonl"), "session_uid")

        for uid, session in sessions.items():
            self._load_one_session(
                session,
                nodes=nodes_by_session.get(uid, []),
                lineage=lineage.get(uid, []),
                context=context.get(uid, []),
                prompts=prompts.get(uid, []),
                events=events.get(uid, []),
                calls=calls.get(uid, []),
            )

    def _load_one_session(
        self,
        session: dict[str, Any],
        *,
        nodes: list[dict[str, Any]],
        lineage: list[dict[str, Any]],
        context: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        events: list[dict[str, Any]],
        calls: list[dict[str, Any]],
    ) -> None:
        uid = session["session_uid"]
        # State events and LLM calls are append-shaped log rows with no natural key, so
        # ON CONFLICT cannot make them idempotent. Replacing the session's rows inside
        # the same transaction can: a re-load ends with exactly one copy, and a reader
        # never sees zero.
        self.conn.execute(
            "DELETE FROM gem_eg_state_event WHERE scope_id=%s AND session_uid=%s",
            (self.scope_id, uid),
        )
        self.conn.execute(
            "DELETE FROM gem_eg_llm_call WHERE scope_id=%s AND session_uid=%s",
            (self.scope_id, uid),
        )
        wanted = [uid] + [n["node_uid"] for n in nodes] + [p["prompt_uid"] for p in prompts]
        mapping = self._alloc(wanted)
        session_vid = mapping[uid]

        self.conn.execute(
            _VERTEX_SQL,
            _vertex_row(
                vid=session_vid,
                scope_id=self.scope_id,
                kind="session",
                uid=uid,
                label=session["run_rel"],
                task_uid=session["task_uid"],
                session_uid=uid,
                backend=session["backend"],
                domain=session["domain"],
                model=session.get("model"),
                # Sessions are hops, never answers: no embedding, by design.
                metadata={"group": session.get("group"), "mode": session.get("mode")},
            ),
        )
        self.conn.execute(
            "INSERT INTO gem_eg_session"
            " (id, session_uid, task_uid, run_rel, backend, search_algorithm,"
            "  experiment_group, model, edit_mode, temperature, configured_iterations,"
            "  node_count, lineage_edge_count, best_node_uid, best_fitness,"
            "  status_counts, prompt_complete_ratio, wall_clock_available, files_present)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (session_uid) DO NOTHING",
            (
                session_vid,
                uid,
                session["task_uid"],
                session["run_rel"],
                session["backend"],
                session["search_algorithm"],
                session.get("group"),
                session.get("model"),
                session.get("mode"),
                session.get("temperature"),
                session.get("configured_iterations"),
                session.get("node_count", len(nodes)),
                session.get("lineage_edge_count", len(lineage)),
                session.get("best_node_uid"),
                session.get("best_fitness"),
                json.dumps(session.get("status_counts", {})),
                session.get("prompt_complete_ratio", 0.0),
                bool(session.get("wall_clock_available", False)),
                json.dumps(session.get("files_present", {})),
            ),
        )
        self.receipt.add("session_processed")

        task_vid = self.vids.get(session["task_uid"])
        if task_vid is None:
            raise SystemExit(
                f"session {uid} references unknown task {session['task_uid']}; "
                "load_tasks must run first"
            )
        self._hier_link(
            task_vid,
            session_vid,
            relation="has_session",
            specific="eg_has_session",
            provenance="sessions.jsonl",
        )

        parent_of = {e["dst_node_uid"]: e["src_node_uid"] for e in lineage}
        for node in nodes:
            self._load_node(
                node,
                mapping,
                session_vid,
                parent_of.get(node["node_uid"]),
                backend=session["backend"],
                domain=session["domain"],
            )

        for edge in lineage:
            src, dst = mapping.get(edge["src_node_uid"]), mapping.get(edge["dst_node_uid"])
            if src is None or dst is None:
                self.receipt.reject("lineage_endpoint_missing")
                continue
            if self.store.link(
                src, dst, relation="has_child", edge_type="eg_lineage",
                provenance=edge["provenance"], source_row=edge.get("row_offset"),
                dataset_revision=self.revision,
            ):
                self.receipt.add("edge:lineage")
            # Ancestors are only reachable through this: the AM traverses out-edges only.
            if self.store.link(
                dst, src, relation="child_of", edge_type="eg_child_of",
                derived_inverse=True, provenance=edge["provenance"],
                dataset_revision=self.revision,
            ):
                self.receipt.add("edge:lineage_inverse")

        for edge in context:
            src, dst = mapping.get(edge["src_node_uid"]), mapping.get(edge["dst_node_uid"])
            if src is None or dst is None:
                self.receipt.reject("context_endpoint_missing")
                continue
            # Its own type, never folded into lineage: an inspiration is not a parent.
            if self.store.link(
                src, dst, relation="context_for", edge_type="eg_context",
                provenance=edge["provenance"], source_row=edge.get("row_offset"),
                dataset_revision=self.revision,
            ):
                self.receipt.add("edge:context")

        for prompt in prompts:
            self._load_prompt(prompt, mapping)

        for event in events:
            self.conn.execute(
                "INSERT INTO gem_eg_state_event"
                " (scope_id, session_uid, iteration, entity, role, slot_key, node_uid,"
                "  value, provenance, source_row)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    self.scope_id,
                    uid,
                    event.get("iteration") or 0,
                    event["entity"],
                    event.get("role"),
                    event.get("slot_key"),
                    event.get("node_uid"),
                    json.dumps(event.get("value")),
                    event["provenance"],
                    event.get("row_offset"),
                ),
            )
            self.receipt.add("state_event_processed")

        for call in calls:
            self.conn.execute(
                "INSERT INTO gem_eg_llm_call"
                " (scope_id, session_uid, seq, model, api_base, temperature,"
                "  prompt_tokens, completion_tokens, reasoning_tokens, error)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    self.scope_id,
                    uid,
                    call["seq"],
                    call.get("model"),
                    call.get("api_base"),
                    call.get("temperature"),
                    call.get("prompt_tokens"),
                    call.get("completion_tokens"),
                    call.get("reasoning_tokens"),
                    call.get("error"),
                ),
            )
            self.receipt.add("llm_call_processed")

        # One session, one commit: a partially visible session is never a valid state.
        self.conn.commit()

    def _load_node(
        self,
        node: dict[str, Any],
        mapping: dict[str, int],
        session_vid: int,
        parent_uid: str | None,
        *,
        backend: str | None = None,
        domain: str | None = None,
    ) -> None:
        vid = mapping[node["node_uid"]]
        self.conn.execute(
            _VERTEX_SQL,
            _vertex_row(
                vid=vid,
                scope_id=self.scope_id,
                kind="node",
                uid=node["node_uid"],
                label=f"{node['status']}@{node.get('iteration')}",
                task_uid=node["task_uid"],
                session_uid=node["session_uid"],
                # Denormalised from the owning session on purpose: `tjs_open`'s filter
                # is raw SQL evaluated once per graph-reached vertex, so a predicate on
                # backend or domain must not cost a join. Leaving these NULL silently
                # made every `backend=...` pushdown match nothing.
                backend=backend,
                domain=domain,
                language=node.get("language"),
                status=node["status"],
                is_valid=node["is_valid"],
                fitness=node.get("fitness"),
                iteration=node.get("iteration"),
                generation=node.get("generation"),
            ),
        )
        self.conn.execute(
            "INSERT INTO gem_eg_node"
            " (id, node_uid, session_uid, task_uid, program_id, parent_node_uid,"
            "  iteration, generation, status, is_valid, language, fitness, metrics,"
            "  error_signature, changes, artifact_uid, prompts_sha256, source_file,"
            "  source_row)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (node_uid) DO NOTHING",
            (
                vid,
                node["node_uid"],
                node["session_uid"],
                node["task_uid"],
                node["program_id"],
                parent_uid,
                node.get("iteration"),
                node.get("generation"),
                node["status"],
                node["is_valid"],
                node.get("language"),
                node.get("fitness"),
                json.dumps(node.get("metrics", {})),
                node.get("error_signature"),
                node.get("changes"),
                node.get("artifact_uid"),
                node.get("prompts_sha256"),
                "programs.jsonl",
                node.get("row_offset"),
            ),
        )
        self.receipt.add("node_processed")
        self._hier_link(
            session_vid,
            vid,
            relation="has_node",
            specific="eg_has_node",
            provenance="programs.jsonl",
        )

    def _load_prompt(self, prompt: dict[str, Any], mapping: dict[str, int]) -> None:
        vid = mapping[prompt["prompt_uid"]]
        node_vid = mapping.get(prompt["node_uid"])
        self.conn.execute(
            _VERTEX_SQL,
            _vertex_row(
                vid=vid,
                scope_id=self.scope_id,
                kind="prompt",
                uid=prompt["prompt_uid"],
                label=prompt["prompts_sha256"][:16],
                session_uid=prompt["session_uid"],
            ),
        )
        payload = None
        if prompt.get("blob_present"):
            run_rel = prompt["session_uid"].split(":", 1)[-1]
            path = self.raw / run_rel / prompt["blob_rel"]
            if path.is_file() and path.stat().st_size <= ARTIFACT_INLINE_LIMIT:
                try:
                    payload = json.dumps(json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.receipt.reject("prompt_blob_unparseable")
        self.conn.execute(
            "INSERT INTO gem_eg_prompt"
            " (id, prompt_uid, node_uid, session_uid, prompts_sha256, blob_rel,"
            "  blob_present, payload)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (prompt_uid) DO NOTHING",
            (
                vid,
                prompt["prompt_uid"],
                prompt["node_uid"],
                prompt["session_uid"],
                prompt["prompts_sha256"],
                prompt.get("blob_rel"),
                bool(prompt.get("blob_present")),
                payload,
            ),
        )
        self.receipt.add("prompt_processed")
        if node_vid is not None and self.store.link(
            node_vid, vid, relation="has_prompt", edge_type="eg_has_prompt",
            provenance="programs.jsonl:prompts_sha256", dataset_revision=self.revision,
        ):
            self.receipt.add("edge:has_prompt")

    def run(self) -> LoadReceipt:
        self.vids = self.store.vertex_ids(self.scope_id)
        before = len(self.vids)
        self.load_tasks()
        self.load_artifacts()
        self.load_sessions()
        self.receipt.add("vertices_added", len(self.vids) - before)
        self.store.record_load(self.receipt)
        self.conn.commit()
        return self.receipt


def _group(rows: Any, key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--scope", default=None, help="default: evotrace:<revision[:8]>")
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args(argv)

    report_path = args.normalized / "integrity_report.json"
    if not report_path.is_file():
        print(f"no integrity report at {report_path}; run the normalizer first", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    revision = report["revision"]
    scope = args.scope or f"evotrace:{revision[:8]}"

    store = EgStore.connect(args.dsn, dim=args.dim)
    try:
        store.init_schema()
        if args.counts_only:
            print(json.dumps(store.counts(scope), indent=2))
            return 0
        receipt = Loader(
            store,
            scope_id=scope,
            normalized=args.normalized,
            raw=args.raw,
            revision=revision,
        ).run()
        print(f"scope   : {scope}")
        print(f"load id : {receipt.load_id}")
        print("counts  :")
        for key in sorted(receipt.counts):
            print(f"  {key:28} {receipt.counts[key]:>9,}")
        if receipt.rejects:
            print("rejects :")
            for key in sorted(receipt.rejects):
                print(f"  {key:28} {receipt.rejects[key]:>9,}")
        print("db      :")
        for key, value in store.counts(scope).items():
            print(f"  {key:28} {value:>9,}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
