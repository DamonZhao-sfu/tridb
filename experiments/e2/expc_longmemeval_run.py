"""Build + search one memory system over LongMemEval-S. Quality only.

    python3 -m experiments.e2.expc_longmemeval_run \
        --system tridb_gem --run-dir bench/out/expc_lme_tridb_gem_2026_08_22 \
        --system-config configs/e2/lme_tridb_gem.json

WHY THIS DOES NOT USE ``table5_track_c.protocol``
-------------------------------------------------
``run_search`` builds its warmup set from ``corpus.queries_by_sample[
WARMUP_SAMPLE_ID]`` -- a hard-coded LoCoMo conversation id -- and then drives
an open-loop QPS scheduler to produce latency evidence. Neither applies here:
LongMemEval has 500 independent scopes and no warmup conversation, and this
driver makes **no latency claim at all**. Reusing that protocol would mean
either faking a warmup scope or reporting latencies from a run that was never
scheduled for them.

So this driver does the minimum that ``expc_longmemeval_table4`` needs:
construct the store, issue exactly one search per question, and emit a
``formal.jsonl`` in the shape ``quality.generate_answers`` already consumes.
Timing fields are recorded for progress reporting and are explicitly NOT a
Table 5 measurement.

SHARDING
--------
A full 500-question build is 246,930 turns and, for the LLM-construction
systems, tens of GPU-hours. ``--shard-index/--shard-count`` splits the question
set deterministically so a failure costs one shard rather than the whole run,
and so shards can be spread across GPUs. Shards are disjoint by construction;
each writes its own run directory and the driver refuses to overwrite one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bench.agent_memory.table5_track_c.longmemeval import (
    LongMemEvalCorpus,
    QUESTION_TYPE_TO_COLUMN,
    load_longmemeval,
    stratified_sample_ids,
)

#: The CLEANED split, which is what Mandol's own REPRODUCE.md pins:
#:   huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
#: It is not cosmetic. Against the raw ``longmemeval_s.json`` it rewrites
#: ``haystack_session_ids`` and ``haystack_dates`` for 454 of the 500
#: questions and drops 180 turns. Those dates are exactly what this harness
#: renders into each turn's text, so scoring the raw file would measure a
#: different temporal signal than the paper did.
DEFAULT_DATASET = (
    "/local-scratch/localhome/hza214/tridb/data/longmemeval/longmemeval_s_cleaned.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ── adapter registry ────────────────────────────────────────────────────────
# Each factory takes the parsed --system-config blob and returns a live
# adapter. Config keys mirror the adapter's own dataclass fields, so the JSON
# is the adapter's documented surface rather than a second vocabulary.


def _tridb_gem(cfg: dict[str, Any]) -> Any:
    from bench.agent_memory.table5_track_c.adapters import (
        TriDBGEMAdapter,
        TriDBGEMConfig,
    )

    return TriDBGEMAdapter(TriDBGEMConfig(**cfg))


def _mem0(cfg: dict[str, Any]) -> Any:
    from bench.agent_memory.table5_track_c.adapters import Mem0Adapter, Mem0Config

    return Mem0Adapter(Mem0Config(**cfg))


def _memos(cfg: dict[str, Any]) -> Any:
    from bench.agent_memory.table5_track_c.adapters import MemosAdapter, MemosConfig

    return MemosAdapter(MemosConfig(**cfg))


def _cognee(cfg: dict[str, Any]) -> Any:
    from bench.agent_memory.table5_track_c.adapters import CogneeAdapter, CogneeConfig

    return CogneeAdapter(CogneeConfig(**cfg))


def _graphiti(cfg: dict[str, Any]) -> Any:
    from experiments.graphiti_track_c.adapter import (
        GraphitiTrackCAdapter,
        GraphitiTrackCConfig,
    )

    return GraphitiTrackCAdapter(GraphitiTrackCConfig(**cfg))


def _evermemos(cfg: dict[str, Any]) -> Any:
    from experiments.evermemos_track_c.adapter import (
        EverMemOSTrackCAdapter,
        EverMemOSTrackCConfig,
    )

    return EverMemOSTrackCAdapter(EverMemOSTrackCConfig(**cfg))


#: Each system is installed in its own virtualenv -- mem0ai 2.0.18, MemoryOS
#: 2.0.30, cognee 1.5.0 and everos 1.2.3 have incompatible dependency pins, so
#: there is no single interpreter that can import all of them. Run this module
#: with the interpreter named here for the system being built; the repo venv
#: only carries TriDB/GEM.
SYSTEM_INTERPRETER = {
    "tridb_gem": "/local-scratch/localhome/hza214/tridb/.venv/bin/python",
    "mem0": "/localhome/hza214/agent-memory-table5/venv/mem0/bin/python",
    "memos": "/localhome/hza214/agent-memory-table5/venv/memos/bin/python",
    "cognee": "/localhome/hza214/agent-memory-table5/venv/cognee/bin/python",
    "evermemos": "/localhome/hza214/agent-memory-table5/venv/evermemos/bin/python",
    "graphiti": "/localhome/hza214/agent-memory-table5/venv/graphiti/bin/python",
}

FACTORIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "tridb_gem": _tridb_gem,
    "mem0": _mem0,
    "memos": _memos,
    "cognee": _cognee,
    "graphiti": _graphiti,
    "evermemos": _evermemos,
}


def shard_corpus(
    corpus: LongMemEvalCorpus, index: int, count: int
) -> LongMemEvalCorpus:
    """Deterministic contiguous split over question ids in file order."""
    if count < 1 or not (0 <= index < count):
        raise SystemExit(f"bad shard {index}/{count}")
    ids = list(corpus.sample_ids)
    size, extra = divmod(len(ids), count)
    start = index * size + min(index, extra)
    stop = start + size + (1 if index < extra else 0)
    keep = set(ids[start:stop])
    if not keep:
        raise SystemExit(f"shard {index}/{count} is empty for {len(ids)} questions")
    return LongMemEvalCorpus(
        path=corpus.path,
        sha256=corpus.sha256,
        events_by_sample={k: v for k, v in corpus.events_by_sample.items() if k in keep},
        queries_by_sample={
            k: v for k, v in corpus.queries_by_sample.items() if k in keep
        },
        warmup_sample_ids=frozenset(),
    )


#: Config values of the form ``"env:VAR_NAME"`` are replaced by that
#: environment variable. Credentials therefore stay in
#: snapshots/sw15v2/credentials.env (mode 0600) instead of being copied into a
#: config file that lands in the run receipt -- the receipt records the literal
#: ``env:...`` marker, not the secret.
_SECRET_PREFIX = "env:"


def _resolve_secrets(cfg: dict[str, Any]) -> dict[str, Any]:
    import os

    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith(_SECRET_PREFIX):
            var = value[len(_SECRET_PREFIX) :]
            secret = os.environ.get(var)
            if not secret:
                raise SystemExit(
                    f"config key {key!r} needs environment variable {var}, "
                    "which is unset"
                )
            resolved[key] = secret
        else:
            resolved[key] = value
    return resolved


def run(args: argparse.Namespace) -> int:
    corpus = load_longmemeval(args.dataset, limit=args.limit)
    if args.per_type:
        keep = set(stratified_sample_ids(corpus, json.loads(args.per_type)))
        corpus = LongMemEvalCorpus(
            path=corpus.path,
            sha256=corpus.sha256,
            events_by_sample={
                k: v for k, v in corpus.events_by_sample.items() if k in keep
            },
            queries_by_sample={
                k: v for k, v in corpus.queries_by_sample.items() if k in keep
            },
        )
    if args.shard_count > 1:
        corpus = shard_corpus(corpus, args.shard_index, args.shard_count)

    run_dir = Path(args.run_dir)
    search_dir = run_dir / "search" / f"qps_{args.qps_label}"
    if search_dir.exists() and any(search_dir.iterdir()):
        raise SystemExit(f"refusing non-empty run dir: {search_dir}")
    search_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "run_receipt.json"

    expected = SYSTEM_INTERPRETER.get(args.system)
    if expected and Path(sys.executable).resolve() != Path(expected).resolve():
        raise SystemExit(
            f"{args.system} must run under {expected}, not {sys.executable}. "
            "Its package is installed only in that virtualenv."
        )

    raw_cfg = json.loads(Path(args.system_config).read_text(encoding="utf-8"))
    cfg = _resolve_secrets(raw_cfg)
    stats = corpus.stats()
    receipt: dict[str, Any] = {
        "schema_version": "experiment_c_longmemeval_run_v0.1.0",
        "system": args.system,
        "build_id": args.build_id,
        "status": "running",
        "started_at": _now(),
        "dataset": {
            "path": str(corpus.path),
            "sha256": corpus.sha256,
            **stats,
        },
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "top_k": args.top_k,
        "latency_claim": (
            "NONE. This driver issues searches back-to-back with no scheduler; "
            "timings here are progress instrumentation, not Table 5 evidence."
        ),
        # The pre-resolution config: any "env:VAR" markers stay as markers, so
        # a receipt can be shared without leaking the credential it named.
        "config": raw_cfg,
    }
    _write_json(receipt_path, receipt)

    adapter = FACTORIES[args.system](cfg)
    formal_path = search_dir / "formal.jsonl"
    try:
        receipt["schema_gate"] = adapter.init_schema()
        _write_json(receipt_path, receipt)

        started = time.perf_counter()
        receipt["build"] = adapter.ingest_history(corpus.all_events)
        receipt["build_finalize"] = adapter.finalize_build()
        receipt["build_wall_seconds"] = time.perf_counter() - started
        receipt["build_completed_at"] = _now()
        _write_json(receipt_path, receipt)

        queries = list(corpus.formal_queries)
        failures = 0
        with formal_path.open("w", encoding="utf-8") as sink:
            for index, query in enumerate(queries):
                record: dict[str, Any] = {
                    "schema_version": "experiment_c_longmemeval_formal_v0.1.0",
                    "request_index": index,
                    "system": args.system,
                    "build_id": args.build_id,
                    "sample_id": query.sample_id,
                    "question_id": query.question_id,
                    "question": query.question,
                    "answer": query.answer,
                    "category": query.category,
                    "column": QUESTION_TYPE_TO_COLUMN[str(query.category)],
                    "evidence_ids": list(query.evidence_ids),
                    "success": False,
                    "error": None,
                }
                began = time.perf_counter_ns()
                try:
                    record["receipt"] = adapter.search(query, top_k=args.top_k)
                    record["success"] = True
                except Exception as exc:  # noqa: BLE001 - a failure is data
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["receipt"] = {"contexts": [], "empty": True}
                    failures += 1
                record["wall_ms"] = (time.perf_counter_ns() - began) / 1_000_000
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if (index + 1) % 25 == 0:
                    print(
                        f"[{args.system}] {index + 1}/{len(queries)} "
                        f"failures={failures}",
                        flush=True,
                    )

        receipt["search"] = {
            "questions": len(queries),
            "failures": failures,
            "formal_path": str(formal_path),
        }
        receipt["final_stats"] = adapter.stats()
        receipt["status"] = "complete"
        receipt["completed_at"] = _now()
        _write_json(receipt_path, receipt)
        print(f"[{args.system}] complete: {len(queries)} questions, {failures} failed")
        return 0
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        _write_json(receipt_path, receipt)
        raise
    finally:
        adapter.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", required=True, choices=sorted(FACTORIES))
    p.add_argument("--system-config", required=True, help="JSON of adapter config")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--build-id", default=None)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--top-k", type=int, default=35)
    p.add_argument("--qps-label", default="1", help="subdirectory label only")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument(
        "--per-type",
        default=None,
        help='JSON map of question_type -> count for a stratified subset, e.g. '
        '\'{"temporal-reasoning": 20, "multi-session": 20}\'',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="truncate to first N questions in FILE ORDER; smoke tests only, "
        "not a valid reporting sample",
    )
    args = p.parse_args()
    if args.build_id is None:
        args.build_id = f"lme_{args.system}_s{args.shard_index}of{args.shard_count}"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
