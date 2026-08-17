"""Prepare and run the TriDB/GEM analogue of [AM] Figure 9.

The ``prepare`` command builds nested 64K..1M-token prefixes from complete
LongMemEval sessions.  The ``run`` command performs construction plus fixed
retrieval probes only: answer generation and judging are deliberately absent
because they are not Figure 9 variables.

Two footprint notions are recorded. ``logical_total_bytes`` is attributable to
one GEM scope. ``physical_delta_bytes`` is the before/after relation-size delta
and is scientifically usable only when ``physical_isolated`` is true (normally
one fresh database per budget/repeat via ``--dsn-template``).
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bench.agent_memory.energy import GpuEnergySampler
from bench.agent_memory.gem.types import Query
from bench.agent_memory.gem_bench import points as pointsmod
from bench.agent_memory.gem_bench.runner import connect_memory, construct_history
from bench.agent_memory.serving import (
    CallLedger,
    LongMemEvalQuestion,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
    _atomic_write_json,
    _git_state,
    _sha256,
    _validate_single_model,
    extract_retrieval_query,
    latency_summary,
    load_workloads,
)

DEFAULT_BUDGETS = (64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024)
SCALE_SOURCE = "longmemeval_sstar_scale"
PREPARE_SCHEMA = "tridb_gem_figure9_inputs_v0.1.0"
RESULT_SCHEMA = "tridb_gem_figure9_results_v0.1.0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _encoding(model: str) -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required; install requirements-agent-memory.txt"
        ) from exc
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.encoding_for_model("gpt-4o-mini")


def _token_count(encoding: Any, text: str) -> int:
    return len(encoding.encode(text, allowed_special={"<|endoftext|>"}))


def _source_rows(path: Path, source: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise ValueError("LongMemEval input must contain a data array")
    rows = [
        row
        for row in payload
        if isinstance(row, dict)
        and str((row.get("metadata") or {}).get("source", row.get("source", "")))
        == source
    ]
    if not rows:
        raise ValueError(f"no input rows found for source={source!r}")
    return rows


def _sessions(rows: Iterable[Mapping[str, Any]]) -> list[Any]:
    sessions: list[Any] = []
    for row_index, row in enumerate(rows):
        context = row.get("context")
        if not isinstance(context, str):
            raise ValueError(f"row {row_index} has no string context")
        try:
            parsed = ast.literal_eval(context)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"row {row_index} context is not a literal session list"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"row {row_index} context is not a session list")
        sessions.extend(parsed)
    return sessions


def build_prefixes(
    sessions: Sequence[Any], budgets: Sequence[int], encoding: Any
) -> list[dict[str, Any]]:
    """Return nested complete-session prefixes at or below each token budget."""
    if not sessions:
        raise ValueError("no sessions available for scaling")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")
    ordered_budgets = sorted(set(int(value) for value in budgets))
    rendered = [repr(session) for session in sessions]
    full_context = "[" + ", ".join(rendered) + "]"
    prefix_ends: list[int] = []
    offset = 1
    for index, text in enumerate(rendered):
        offset += len(text)
        prefix_ends.append(offset)
        if index + 1 < len(rendered):
            offset += 2

    def context_for(count: int) -> str:
        if count == 0:
            return "[]"
        if count == len(sessions):
            return full_context
        return full_context[: prefix_ends[count - 1]] + "]"

    token_cache: dict[int, int] = {}

    def tokens_for(count: int) -> int:
        if count not in token_cache:
            token_cache[count] = _token_count(encoding, context_for(count))
        return token_cache[count]

    cursor = 0
    prefixes: list[dict[str, Any]] = []
    for budget in ordered_budgets:
        low, high = cursor, len(sessions) + 1
        while low + 1 < high:
            middle = (low + high) // 2
            if tokens_for(middle) <= budget:
                low = middle
            else:
                high = middle
        cursor = low
        context = context_for(cursor)
        actual_tokens = tokens_for(cursor)
        prefixes.append(
            {
                "requested_tokens": budget,
                "actual_tokens": actual_tokens,
                "sessions": cursor,
                "context": context,
            }
        )
    if prefixes[-1]["actual_tokens"] < ordered_budgets[-1] * 0.9:
        raise ValueError(
            "source corpus cannot reach 90% of the largest token budget without "
            "splitting a session; refusing to repeat or synthesize text"
        )
    return prefixes


def _probe_fields(row: Mapping[str, Any], limit: int) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    questions = list(row.get("questions") or [])[:limit]
    answers = list(row.get("answers") or [])[:limit]
    if not questions or len(questions) != len(answers):
        raise ValueError("the source row has no aligned retrieval probes")
    return {
        "questions": questions,
        "answers": answers,
        "question_ids": list(metadata.get("question_ids") or [])[:limit],
        "question_types": list(metadata.get("question_types") or [])[:limit],
        "qa_pair_ids": list(metadata.get("qa_pair_ids") or [])[:limit],
    }


def prepare_inputs(
    *,
    input_path: Path,
    output_dir: Path,
    budgets: Sequence[int],
    source: str,
    tokenizer_model: str,
    probe_limit: int,
) -> dict[str, Any]:
    rows = _source_rows(input_path, source)
    encoding = _encoding(tokenizer_model)
    prefixes = build_prefixes(_sessions(rows), budgets, encoding)
    probes = _probe_fields(rows[0], probe_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for prefix in prefixes:
        budget = int(prefix["requested_tokens"])
        budget_k = budget // 1024
        filename = f"scale_{budget_k:04d}k.json"
        metadata = {
            "source": SCALE_SOURCE,
            "question_ids": probes["question_ids"],
            "question_types": probes["question_types"],
            "qa_pair_ids": probes["qa_pair_ids"],
            "scaling": {
                "requested_tokens": budget,
                "actual_tokens": prefix["actual_tokens"],
                "sessions": prefix["sessions"],
                "complete_session_prefix": True,
            },
        }
        payload = {
            "dataset": "MemoryAgentBench LongMemEval_S* deterministic prefix",
            "source_input": str(input_path.resolve()),
            "data": [
                {
                    "context": prefix["context"],
                    "questions": probes["questions"],
                    "answers": probes["answers"],
                    "metadata": metadata,
                }
            ],
        }
        path = output_dir / filename
        _atomic_write_json(path, payload)
        points.append(
            {
                "requested_tokens": budget,
                "actual_tokens": prefix["actual_tokens"],
                "sessions": prefix["sessions"],
                "path": filename,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": PREPARE_SCHEMA,
        "created_at": _utc_now(),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "source": source,
        },
        "tokenizer_model": tokenizer_model,
        "prefix_policy": "complete sessions, nested prefixes, no repeated text",
        "probe_policy": (
            "fixed first-source-row questions; used only for retrieval latency, "
            "not accuracy"
        ),
        "probe_count": len(probes["questions"]),
        "scale_source": SCALE_SOURCE,
        "points": points,
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _logical_footprint(conn: Any, scope_id: str) -> dict[str, int]:
    queries = {
        "unit_bytes": (
            "SELECT COALESCE(sum(pg_column_size(row(u.*))), 0)::bigint "
            "FROM gem_unit u WHERE u.scope_id = %s"
        ),
        "field_value_bytes": (
            "SELECT COALESCE(sum(pg_column_size(row(fv.*))), 0)::bigint "
            "FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id "
            "WHERE u.scope_id = %s"
        ),
        "edge_metadata_bytes": (
            "SELECT COALESCE(sum(pg_column_size(row(e.*))), 0)::bigint "
            "FROM gem_edge e JOIN gem_unit u ON u.id = e.src "
            "WHERE u.scope_id = %s"
        ),
        "transition_bytes": (
            "SELECT COALESCE(sum(pg_column_size(row(t.*))), 0)::bigint "
            "FROM gem_transition t WHERE t.scope_id = %s"
        ),
        "policy_bytes": (
            "SELECT COALESCE(sum(pg_column_size(row(p.*))), 0)::bigint "
            "FROM gem_policy p WHERE p.scope_id = %s"
        ),
    }
    result = {
        key: int(conn.execute(sql, (scope_id,)).fetchone()[0])
        for key, sql in queries.items()
    }
    result["logical_total_bytes"] = sum(result.values())
    return result


def _physical_relations(conn: Any) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        "SELECT n.nspname, c.relname, pg_relation_size(c.oid)::bigint, "
        "pg_indexes_size(c.oid)::bigint, pg_total_relation_size(c.oid)::bigint "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relkind IN ('r', 'm') AND "
        "((n.nspname = 'public' AND c.relname LIKE 'gem\\_%' ESCAPE '\\') "
        "OR n.nspname = 'graph_store') ORDER BY n.nspname, c.relname"
    ).fetchall()
    return {
        f"{schema}.{relation}": {
            "heap_bytes": int(heap),
            "index_bytes": int(indexes),
            "total_bytes": int(total),
        }
        for schema, relation, heap, indexes, total in rows
    }


def _physical_delta(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    relations: dict[str, dict[str, int]] = {}
    for relation in sorted(set(before) | set(after)):
        relations[relation] = {
            key: int(after.get(relation, {}).get(key, 0))
            - int(before.get(relation, {}).get(key, 0))
            for key in ("heap_bytes", "index_bytes", "total_bytes")
        }
    return {
        "physical_delta_bytes": sum(row["total_bytes"] for row in relations.values()),
        "relations": relations,
    }


def _existing_rows(conn: Any) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in ("gem_unit", "gem_transition")
    }


def _probe_retrieval(
    *,
    memory: Any,
    point: pointsmod.OperatingPoint,
    questions: Sequence[LongMemEvalQuestion],
    scope_id: str,
    embedder: PhasedEmbedder,
    top_k: int,
    term_cond: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for question in questions:
        text = extract_retrieval_query(question.question)
        started = time.perf_counter()
        embed_started = time.perf_counter()
        with embedder.phase("query"):
            vector = embedder.encode([text])[0]
        embed_ended = time.perf_counter()
        retrieve_started = time.perf_counter()
        result = memory.retrieve(
            Query(
                scope_id=scope_id,
                text=text,
                embedding=vector,
                k=top_k,
                mode=point.mode,
                route=point.route,
                reinforce=point.reinforce,
                term_cond=term_cond,
            )
        )
        ended = time.perf_counter()
        if not result.committed:
            raise RuntimeError(
                f"{point.key}/{question.question_id} retrieve aborted: "
                f"{result.aborted_reason}"
            )
        records.append(
            {
                "question_id": question.question_id,
                "query_embedding_seconds": embed_ended - embed_started,
                "gem_retrieval_seconds": ended - retrieve_started,
                "end_to_end_seconds": ended - started,
                "hits": len(result.hits),
                "cost": asdict(result.cost),
                "probes": dict(result.probes),
            }
        )
    summary = {
        key: latency_summary(record[key] for record in records)
        for key in (
            "query_embedding_seconds",
            "gem_retrieval_seconds",
            "end_to_end_seconds",
        )
    }
    summary["queries"] = len(records)
    return records, summary


def _dsn_for(template: str, budget: int, repeat: int) -> str:
    return template.format(
        budget=budget,
        budget_k=budget // 1024,
        repeat=repeat,
    )


def _redact_dsn(dsn: str) -> str:
    return re.sub(r"://[^@]+@", "://***@", dsn)


def run_scaling(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.scale_dir / "manifest.json"
    prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {int(row["requested_tokens"]): row for row in prepared["points"]}
    budgets = sorted(available) if args.budgets is None else sorted(set(args.budgets))
    missing = sorted(set(budgets) - set(available))
    if missing:
        raise ValueError(f"budgets not prepared: {missing}")
    point = pointsmod.POINTS_BY_KEY[args.point]
    if point.uses_llm_construction:
        raise ValueError(
            "Figure 9 runner currently supports deterministic GEM points only; "
            "LLM construction needs the answer endpoint and capability gate"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = []
    execution_order = []
    for repeat in range(1, args.repeats + 1):
        repeat_budgets = list(budgets)
        random.Random(args.seed + repeat).shuffle(repeat_budgets)
        for budget in repeat_budgets:
            execution_order.append({"repeat": repeat, "requested_tokens": budget})
            prepared_point = available[budget]
            input_path = args.scale_dir / prepared_point["path"]
            workload = load_workloads(
                input_path, source=SCALE_SOURCE, strict_shape=False
            )[0]
            dsn = _dsn_for(args.dsn_template, budget, repeat)
            args.dsn = dsn
            ledger = CallLedger()
            client = OpenAIEmbeddingClient(
                args.embedding_base_url,
                args.embedding_api_key,
                args.embedding_model,
                batch_size=args.embedding_batch_size,
                timeout=args.request_timeout,
                ledger=ledger,
            )
            advertised = client.discover_models()
            _validate_single_model(
                advertised, args.embedding_model, endpoint_name="embedding endpoint"
            )
            embedder = PhasedEmbedder(client)
            sampler = GpuEnergySampler(enabled=False)
            memory = connect_memory(args, point, embedder)
            scope_id = f"figure9_{point.key}_{budget // 1024}k_r{repeat}"
            try:
                from bench.agent_memory.demo.scenario import reset_scope

                reset_scope(memory, scope_id)
                existing = _existing_rows(memory.store.conn)
                physical_before = _physical_relations(memory.store.conn)
                lifecycle_started = time.perf_counter()
                construction = construct_history(
                    memory=memory,
                    point=point,
                    workload=workload,
                    scope_id=scope_id,
                    history_index=0,
                    args=args,
                    embedder=embedder,
                    extractor=None,
                    sampler=sampler,
                    lifecycle_started=lifecycle_started,
                )
                footprint = _logical_footprint(memory.store.conn, scope_id)
                physical_after = _physical_relations(memory.store.conn)
                footprint.update(_physical_delta(physical_before, physical_after))
                footprint["physical_isolated"] = not any(existing.values())
                footprint["preexisting_rows"] = existing
                probe_records, retrieval = _probe_retrieval(
                    memory=memory,
                    point=point,
                    questions=workload.questions[: args.probe_limit],
                    scope_id=scope_id,
                    embedder=embedder,
                    top_k=args.top_k,
                    term_cond=args.term_cond,
                )
            finally:
                memory.close()
            result = {
                "requested_tokens": budget,
                "actual_input_tokens": int(prepared_point["actual_tokens"]),
                "sessions": int(prepared_point["sessions"]),
                "repeat": repeat,
                "operating_point": point.key,
                "scope_id": scope_id,
                "dsn_redacted": _redact_dsn(dsn),
                "construction": {
                    "seconds": construction["construction_seconds"],
                    "ingest_seconds": construction["ingest_seconds"],
                    "revise_seconds": construction["revise_seconds"],
                    "units_created": construction["units_created"],
                    "fields_appended": construction["fields_appended"],
                    "edges_created": construction["edges_created"],
                    "cost": construction["ingest_cost"],
                },
                "tokens": ledger.tokens(),
                "calls": ledger.summary(),
                "footprint": footprint,
                "retrieval": retrieval,
                "retrieval_records": probe_records,
            }
            output_path = (
                args.output_dir / f"scale_{budget // 1024:04d}k_r{repeat}.json"
            )
            _atomic_write_json(output_path, result)
            points.append(result)
            print(
                f"[gem-scale] {budget // 1024}K repeat={repeat} "
                f"construct={construction['construction_seconds']:.3f}s "
                f"logical={footprint['logical_total_bytes']} bytes",
                file=sys.stderr,
                flush=True,
            )
    results = {
        "schema_version": RESULT_SCHEMA,
        "completed_at": _utc_now(),
        "paper": {
            "reference": "arXiv:2606.06448 Figure 9 analogue",
            "claim": "same-class TriDB/GEM metrics, not paper-number replication",
        },
        "input_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        },
        "operating_point": point.to_dict(),
        "repeats": args.repeats,
        "seed": args.seed,
        "execution_order": execution_order,
        "probe_count": args.probe_limit,
        "footprint_semantics": {
            "primary": "scope-attributable logical row bytes",
            "physical": (
                "relation-size before/after delta; use only when each point's "
                "physical_isolated flag is true"
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
        "points": points,
    }
    _atomic_write_json(args.output_dir / "scale_results.json", results)
    return results


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build nested scale inputs")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument(
        "--budgets", nargs="+", type=_positive_int, default=DEFAULT_BUDGETS
    )
    prepare.add_argument("--source", default="longmemeval_s*")
    prepare.add_argument("--tokenizer-model", default="gpt-4o-mini")
    prepare.add_argument("--probe-limit", type=_positive_int, default=20)

    run = subparsers.add_parser("run", help="construct and probe every scale point")
    run.add_argument("--scale-dir", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument(
        "--dsn-template",
        required=True,
        help=(
            "Postgres DSN; placeholders {budget}, {budget_k}, {repeat} allow a "
            "fresh database per point"
        ),
    )
    run.add_argument("--budgets", nargs="+", type=_positive_int)
    run.add_argument("--repeats", type=_positive_int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--point", choices=("II_embedrag", "gem_conformant"), default="gem_conformant"
    )
    run.add_argument("--probe-limit", type=_positive_int, default=20)
    run.add_argument("--top-k", type=_positive_int, default=10)
    run.add_argument("--term-cond", type=_positive_int, default=32)
    run.add_argument("--chunk-size", type=_positive_int, default=4096)
    run.add_argument("--embedding-batch-size", type=_positive_int, default=64)
    run.add_argument("--embedding-dim", type=_positive_int, default=1024)
    run.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:8001/v1",
    )
    run.add_argument("--embedding-api-key", default="EMPTY")
    run.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    run.add_argument("--request-timeout", type=float, default=600.0)
    run.add_argument("--answer-model", default="Qwen/Qwen3-32B")
    run.add_argument("--max-rejection-rate", type=float)
    run.add_argument("--revise-max-hops", type=_positive_int, default=3)
    run.add_argument("--revise-max-evidence", type=int, default=0)
    run.add_argument("--agentic-max-rounds", type=_positive_int, default=8)
    run.add_argument("--agentic-max-tool-calls", type=_positive_int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_inputs(
            input_path=args.input,
            output_dir=args.output_dir,
            budgets=args.budgets,
            source=args.source,
            tokenizer_model=args.tokenizer_model,
            probe_limit=args.probe_limit,
        )
    else:
        if args.revise_max_evidence < 0:
            raise SystemExit("--revise-max-evidence must be non-negative")
        result = run_scaling(args)
        result = {
            "schema_version": result["schema_version"],
            "output": str((args.output_dir / "scale_results.json").resolve()),
            "points": [
                {
                    "requested_tokens": point["requested_tokens"],
                    "actual_input_tokens": point["actual_input_tokens"],
                    "repeat": point["repeat"],
                    "construction_seconds": point["construction"]["seconds"],
                    "logical_total_bytes": point["footprint"]["logical_total_bytes"],
                    "retrieval_p50_seconds": point["retrieval"]["end_to_end_seconds"][
                        "p50"
                    ],
                    "physical_isolated": point["footprint"]["physical_isolated"],
                }
                for point in result["points"]
            ],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
