"""Cognee as a third memory arm: build the Experience Graph, then serve retrieval.

Runs in Cognee's OWN virtualenv (`agent-memory-table5/venv/cognee`), because
Cognee and OpenEvolve cannot share one -- their dependency sets are why those
environments were split in the first place. The OpenEvolve side talks to this
over HTTP, exactly as it already does for the Milvus/Neo4j/pgvector arm, so the
latency comparison stays like-for-like instead of privileging an in-process arm.

Two honest asymmetries, both recorded rather than papered over:

* **Cognee does not compute the same answer.** GEM and the polyglot stack were
  verified to return byte-identical result sets against an exhaustive oracle;
  Cognee's retrieval is its own semantics (chunk retrieval over an LLM-built
  graph). It is a "what an off-the-shelf agent-memory system does with the same
  question" baseline and belongs on its own row, never in a parity claim.
* **The reward predicate is applied outside Cognee.** `fitness >= parent_fitness`
  has no pushdown here, so it is a post-filter on Cognee's ranked chunks. That is
  architecturally the same place the polyglot stack applies it (pgvector, stage
  3), and the count that survives the filter is reported per query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

#: Provenance marker. Cognee stores and returns text, so the only way back from a
#: retrieved chunk to a corpus node is a token embedded in the document itself.
UID_RE = re.compile(r"\[EVOTRACE_NODE:([^\]]+)\]")
DSN = "postgresql://127.0.0.1:55432/evotrace_eg"


def documents(
    dsn: str,
    domains: tuple[str, ...],
    max_code: int,
    tasks: tuple[str, ...] = (),
    marker_every: int = 0,
    metadata: bool = True,
) -> list[dict[str, Any]]:
    """One document per historical program, plus one per task specification.

    Task specs are included because the paper's reuse query enters at the Task
    level; without them Cognee would be answering a different question from the
    other two arms for a reason that has nothing to do with its retrieval.
    """
    import psycopg

    conn = psycopg.connect(dsn)
    # An explicit task list wins over the domain prefix: only the tasks actually
    # being re-run need to be in the graph, and cognify cost is per document.
    like = tuple(tasks) if tasks else tuple(f"{d}:%" for d in domains)
    match = "= ANY(%s)" if tasks else "LIKE ANY(%s)"
    out: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT task_uid, specification FROM gem_eg_task"
            f" WHERE task_uid {match} AND specification IS NOT NULL",
            (list(like),),
        )
        for task_uid, spec in cur.fetchall():
            out.append(
                {
                    "uid": task_uid,
                    "kind": "task",
                    "task_uid": task_uid,
                    "text": f"[EVOTRACE_TASK:{task_uid}] Task specification.\n{spec}",
                }
            )
        cur.execute(
            "SELECT n.node_uid, n.task_uid, n.session_uid, n.fitness, n.iteration,"
            "       n.changes, a.payload"
            "  FROM gem_eg_node n"
            "  JOIN gem_eg_artifact a ON a.artifact_uid = n.artifact_uid"
            f" WHERE n.task_uid {match} AND n.is_valid AND n.fitness IS NOT NULL"
            "   AND a.payload IS NOT NULL",
            (list(like),),
        )
        for uid, task, session, fitness, iteration, changes, code in cur.fetchall():
            out.append(
                {
                    "uid": uid,
                    "kind": "node",
                    "task_uid": task,
                    "session_uid": session,
                    "fitness": float(fitness),
                    "text": _with_markers(
                        uid,
                        _node_text(
                            uid,
                            task,
                            session,
                            iteration,
                            fitness,
                            changes,
                            code,
                            max_code,
                            metadata,
                        ),
                        marker_every,
                    ),
                }
            )
    conn.close()
    return out


def _node_text(
    uid: str,
    task: str,
    session: str,
    iteration: int,
    fitness: float,
    changes: str | None,
    code: str | None,
    max_code: int,
    metadata: bool,
) -> str:
    """The document body, with or without the metadata header.

    GEM and the polyglot stack receive `fitness` as a COLUMN they filter on; they
    never embed it. Putting it in Cognee's document text is therefore an
    asymmetry, whatever its effect. Measured on the metadata build, the effect on
    ranking is nil -- Spearman rho between returned rank and fitness is +0.139,
    and the top ten of a 40-row answer average 0.7495 against 0.7293 for the
    bottom ten -- so the earlier claim that the number biased retrieval toward
    high-reward programs does not survive measurement. `metadata=False` removes
    the asymmetry anyway, so the comparison does not rest on that one check.
    """
    body = (code or "")[:max_code]
    if not metadata:
        return f"[EVOTRACE_NODE:{uid}]\n{body}"
    return (
        f"[EVOTRACE_NODE:{uid}] task={task} session={session} "
        f"iteration={iteration} fitness={fitness:.6f}\n"
        f"Change: {(changes or '').strip()[:500]}\n"
        f"Program:\n{body}"
    )


def _with_markers(uid: str, text: str, every: int) -> str:
    """Repeat the provenance marker so every chunk stays traceable.

    The marker sits on line 1 only, and Cognee splits a document into several
    chunks: of 4,000 sampled chunks from the first build, only 15% carried one
    (0 were truncated -- the marker survives, it simply is not there). A chunk
    retrieved from the middle of a program is unmappable and gets dropped, which
    reads as "Cognee retrieved nothing useful" when the real cause is a document
    format that was not written for chunk-level retrieval. `every=0` reproduces
    that original single-marker build.
    """
    if every <= 0 or len(text) <= every:
        return text
    tag = f"\n[EVOTRACE_NODE:{uid}]\n"
    return tag.join(text[i : i + every] for i in range(0, len(text), every))


def adapter(args: argparse.Namespace) -> Any:
    from bench.agent_memory.table5_track_c.adapters.cognee import (
        CogneeAdapter,
        CogneeConfig,
    )

    return CogneeAdapter(
        CogneeConfig(
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=os.environ["COGNEE_DB_PASSWORD"],
            db_name=args.db_name,
            dataset_prefix="evotrace_eg",
            answer_base_url=args.llm_base,
            # The adapter asserts the endpoint serves EXACTLY the configured model
            # name. That is a good gate -- a silently different answer model would
            # make this arm incomparable -- but it means the SERVED name has to be
            # passed in rather than left at the Track C default of Qwen3-32B.
            answer_model=args.llm_model,
            embedding_base_url=args.embedding_base,
            embedding_model=args.embedding_model,
        )
    )


def export(args: argparse.Namespace) -> int:
    """Write the documents to a file, from an environment that has psycopg.

    Cognee's virtualenv deliberately does not carry this project's database
    driver -- keeping their dependency sets apart is why the environments were
    split. So the corpus is read here and the ingest side reads only a file,
    which also makes exactly what was cognified an artifact rather than a
    re-query that could drift between the two steps.
    """
    docs = documents(
        args.dsn,
        tuple(args.domains),
        args.max_code,
        tuple(args.tasks or ()),
        args.marker_every,
        not args.no_metadata,
    )
    args.documents.parent.mkdir(parents=True, exist_ok=True)
    with args.documents.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    chars = sum(len(d["text"]) for d in docs)
    print(
        f"{len(docs)} documents -> {args.documents}  "
        f"({chars:,} chars, ~{chars // 4:,} tokens to cognify)"
    )
    return 0


def ingest(args: argparse.Namespace) -> int:
    if not args.documents.exists():
        raise SystemExit(
            f"no document file at {args.documents}; run `export` from an "
            "environment with psycopg first"
        )
    docs = [
        json.loads(line) for line in args.documents.read_text().splitlines() if line
    ]
    print(
        f"{len(docs)} documents "
        f"({sum(1 for d in docs if d['kind'] == 'task')} task specs, "
        f"{sum(1 for d in docs if d['kind'] == 'node')} programs)"
    )
    if args.limit:
        docs = docs[: args.limit]
        print(f"  limited to {len(docs)} for this run")
    index = {d["uid"]: {k: v for k, v in d.items() if k != "text"} for d in docs}
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(index, indent=1), encoding="utf-8")

    if args.dry_run:
        chars = sum(len(d["text"]) for d in docs)
        print(f"dry-run: {chars:,} chars (~{chars // 4:,} tokens) would be cognified")
        return 0

    ad = adapter(args)
    ad.init_schema()
    import cognee

    started = time.perf_counter()

    async def build() -> Any:
        await cognee.add([d["text"] for d in docs], dataset_name=args.dataset)
        # `data_per_batch` is the semaphore width over documents
        # (cognee's run_tasks.py:109) and defaults to 20. Measured at the default,
        # the process used 0.6 of 40 cores while both GPUs sat at 0% -- the
        # pipeline was the bottleneck, not the model. This is the one knob that
        # turns idle CPU and idle GPU into throughput.
        return await cognee.cognify(
            datasets=[args.dataset],
            data_per_batch=args.data_per_batch,
            chunks_per_batch=args.chunks_per_batch,
        )

    result = ad.bridge.run(build())
    receipt = {
        "documents": len(docs),
        "dataset": args.dataset,
        "cognify_seconds": round(time.perf_counter() - started, 1),
        "data_per_batch": args.data_per_batch,
        "chunks_per_batch": args.chunks_per_batch,
        "domains": list(args.domains),
        "max_code": args.max_code,
        "result": str(result)[:2000],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"cognify done in {receipt['cognify_seconds']}s -> {args.receipt}")
    return 0


def check(args: argparse.Namespace) -> int:
    """Verify a built graph can be served WITHOUT rebuilding it.

    Cognify is the expensive step (measured: >1.5h and ~12k LLM calls for four
    tasks), so reproducing the experiment must never re-run it by accident. Three
    things have to survive, and two of them are ordinary files that a clean
    checkout would not carry:

    * the Postgres database -- graph, vectors and relations all live there
      (`GRAPH_DATABASE_PROVIDER=postgres`, `VECTOR_DB_PROVIDER=pgvector`), so
      there is no separate store to lose;
    * `index.json` -- uid -> task/fitness, needed for the post-filter;
    * `documents.jsonl` -- what was actually cognified.

    The database and the `vector` extension have to be created by a superuser:
    the Cognee role deliberately cannot do either, which is how the first run
    failed twice.
    """
    import psycopg

    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(
            f"  [{'ok ' if good else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}"
        )

    for path in (args.index, args.documents):
        report(
            str(path),
            path.exists(),
            "" if path.exists() else "missing; run `export`, then `ingest`",
        )

    dsn = (
        f"postgresql://{args.db_user}:{os.environ.get('COGNEE_DB_PASSWORD', '')}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )
    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:  # noqa: BLE001
        report(f"database {args.db_name}", False, f"{type(exc).__name__}: {exc}"[:120])
        print(
            "\n  superuser prerequisites:\n"
            f"    CREATE DATABASE {args.db_name} OWNER {args.db_user};\n"
            f"    \\c {args.db_name}\n"
            "    CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"    GRANT ALL ON SCHEMA public TO {args.db_user};"
        )
        return 1
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        report(
            "pgvector extension",
            cur.fetchone() is not None,
            "" if cur.rowcount else "CREATE EXTENSION vector (needs superuser)",
        )
        cur.execute("SELECT nspname FROM pg_namespace WHERE nspname LIKE 'ds%%'")
        schemas = [r[0] for r in cur.fetchall()]
        report(
            "dataset schema",
            bool(schemas),
            ", ".join(schemas) or "none; cognify never ran",
        )
        for schema in schemas:
            for table in ("graph_node", "graph_edge", "Entity_name"):
                try:
                    cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
                    count = cur.fetchone()[0]
                except Exception:  # noqa: BLE001
                    conn.rollback()
                    report(f"{schema}.{table}", False, "absent")
                    continue
                report(f"{schema}.{table}", count > 0, f"{count:,} rows")
    conn.close()
    print(
        "\n  "
        + (
            "ready to serve; cognify is NOT needed"
            if ok
            else "NOT ready -- see failures above"
        )
    )
    return 0 if ok else 1


def serve(args: argparse.Namespace) -> int:
    """Minimal HTTP retrieval endpoint, mirroring the polyglot backend's contract."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import cognee
    from cognee.modules.search.types import SearchType

    ad = adapter(args)
    ad.init_schema()
    index = json.loads(args.index.read_text())
    # document_id -> uid, built from the chunks that DO carry a marker. The
    # marker sits on line 1, so every document has exactly one marked chunk;
    # 2,406 of 2,410 documents map (the other four are task specifications,
    # which have no node uid). Without this, a chunk retrieved from the middle
    # of a program is unmappable -- measured at 18% marker coverage over 13,653
    # chunks, so four out of five hits were being discarded.
    doc_map: dict[str, str] = {}
    if args.doc_map.exists():
        doc_map = json.loads(args.doc_map.read_text())
    print(f"document_id -> uid map: {len(doc_map)} entries")

    def retrieve(payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        k = int(payload.get("k", 10))
        min_fitness = payload.get("min_fitness")
        task_uid = payload.get("task_uid")
        started = time.perf_counter()
        rows = ad.bridge.run(
            cognee.search(
                query_text=query,
                query_type=SearchType.CHUNKS,
                datasets=[args.dataset],
                top_k=max(k * args.overfetch, k),
                # `node_type` defaults to NodeSet and filters the result set;
                # the chunks themselves are wanted here.
                node_type=None,
            )
        )
        # `search` returns ONE dict per dataset, with the chunks nested under
        # `search_result`. Treating the outer list as the chunk list reported
        # `chunks_returned: 1` and zero uids while 20 chunks had come back
        # perfectly well -- the arm then injected nothing, silently.
        chunks: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict) and "search_result" in row:
                chunks.extend(row["search_result"])
            else:
                chunks.append(row)
        search_ms = (time.perf_counter() - started) * 1000
        uids: list[str] = []
        seen: set[str] = set()
        unmapped = 0
        for chunk in chunks:
            text = chunk if isinstance(chunk, str) else json.dumps(chunk, default=str)
            found = UID_RE.findall(text)
            if not found and isinstance(chunk, dict):
                mapped = doc_map.get(str(chunk.get("document_id")))
                if mapped:
                    found = [mapped]
            if not found:
                unmapped += 1
            for uid in found:
                if uid in seen:
                    continue
                seen.add(uid)
                meta = index.get(uid)
                if meta is None or meta.get("kind") != "node":
                    continue
                if task_uid and meta.get("task_uid") != task_uid:
                    continue
                if min_fitness is not None and meta.get("fitness", 0.0) < min_fitness:
                    continue
                uids.append(uid)
        return {
            "uids": uids[:k],
            "telemetry": {
                "search_ms": round(search_ms, 3),
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                # No streaming: cognee returns the whole ranked list at once, so
                # the first row and the last arrive together -- same as polyglot.
                "first_row_ms": round((time.perf_counter() - started) * 1000, 3),
                "chunks_returned": len(chunks),
                "uids_seen": len(seen),
                "uids_after_filter": len(uids),
                # Chunks that matched the query but carry no marker, so they
                # cannot be traced back to a corpus node and are unusable.
                # Reported per query: without it, a document-format problem is
                # indistinguishable from Cognee retrieving nothing relevant.
                "chunks_without_uid": unmapped,
            },
        }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                body = json.dumps(retrieve(payload)).encode()
                code = 200
            except Exception as exc:  # noqa: BLE001 - the reason must reach the arm
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:  # keep the run log readable
            return

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"cognee retrieval serving on 127.0.0.1:{args.port} "
        f"(dataset={args.dataset}, {len(index)} indexed)"
    )
    server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["export", "ingest", "serve", "check"])
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--domains", nargs="*", default=["math"])
    ap.add_argument("--dataset", default="evotrace_eg_math")
    ap.add_argument("--index", type=Path, default=Path("bench/out/cognee/index.json"))
    ap.add_argument(
        "--receipt", type=Path, default=Path("bench/out/cognee/receipt.json")
    )
    ap.add_argument(
        "--max-code",
        type=int,
        default=8000,
        help="chars of program source per document; cognify cost scales "
        "with total text, and the corpus median math program is 6,098",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="explicit task_uids; overrides --domains",
    )
    ap.add_argument(
        "--documents", type=Path, default=Path("bench/out/cognee/documents.jsonl")
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument(
        "--no-metadata",
        action="store_true",
        help="document text carries only the uid marker and the code, so Cognee "
        "embeds exactly what the other arms retrieve over",
    )
    ap.add_argument(
        "--marker-every",
        type=int,
        default=0,
        help="repeat the [EVOTRACE_NODE:uid] marker every N characters so every "
        "chunk stays mappable; 0 reproduces the single-marker build",
    )
    ap.add_argument(
        "--data-per-batch",
        type=int,
        default=64,
        help="documents cognified concurrently (cognee's default is 20)",
    )
    ap.add_argument(
        "--chunks-per-batch",
        type=int,
        default=2000,
    )
    ap.add_argument(
        "--doc-map",
        type=Path,
        default=Path("bench/out/cognee/doc_to_uid.json"),
        help="document_id -> uid, so chunks without an inline marker still map",
    )
    ap.add_argument(
        "--overfetch",
        type=int,
        default=6,
        help="chunks fetched per requested program; the predicate is a "
        "post-filter here, so under-fetching silently shrinks k",
    )
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=55432)
    ap.add_argument("--db-user", default="table5c_cognee_user")
    ap.add_argument("--db-name", default="evotrace_cognee")
    ap.add_argument("--llm-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--llm-model", default="qwen3.8")
    ap.add_argument("--embedding-base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    args = ap.parse_args(argv)
    if args.mode == "export":
        return export(args)
    if args.mode == "check":
        return check(args)
    return ingest(args) if args.mode == "ingest" else serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
