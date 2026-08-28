"""Command-line entry point for one isolated Track C build/phase."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from .conformance import run_conformance
from .dataset import load_locomo
from .protocol import run_add, run_build_search, run_search

DEFAULT_DATASET = (
    "/localhome/hza214/Mandol/experimental/self_host_benchmarks/locomo/"
    "data/locomo10.json"
)
DEFAULT_EXTERNAL_ROOT = Path("/localhome/hza214/agent-memory-table5")
_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _safe(value: str) -> str:
    return _SAFE.sub("_", value).strip("_")


def _secret(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"required secret environment variable is unset: {name}")
    return value


def _adapter(args: argparse.Namespace) -> Any:
    common = {
        "answer_base_url": args.answer_base_url,
        "answer_model": args.answer_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
    }
    stem = _safe(args.build_id)
    if args.system == "tridb_gem":
        from .adapters.tridb_gem import TriDBGEMAdapter, TriDBGEMConfig

        return TriDBGEMAdapter(
            TriDBGEMConfig(
                dsn=args.dsn,
                embedding_base_url=args.embedding_base_url,
                embedding_model=args.embedding_model,
                embedding_dim=args.embedding_dim,
            )
        )
    if args.system == "mem0":
        from .adapters.mem0 import Mem0Adapter, Mem0Config

        history_path = args.history_path or str(
            DEFAULT_EXTERNAL_ROOT / "volumes" / "mem0" / stem / "history.sqlite3"
        )
        return Mem0Adapter(
            Mem0Config(
                database_dsn=args.database_dsn,
                database_name=args.database_name,
                collection_name=args.collection_name or f"trackc_{stem}",
                history_path=history_path,
                **common,
            )
        )
    if args.system == "cognee":
        from .adapters.cognee import CogneeAdapter, CogneeConfig

        return CogneeAdapter(
            CogneeConfig(
                db_host=args.db_host,
                db_port=args.db_port,
                db_user=args.db_user,
                db_password=_secret(args.db_password_env),
                db_name=args.database_name,
                dataset_prefix=args.dataset_prefix or f"trackc_{stem}",
                **common,
            )
        )
    if args.system == "memos":
        from .adapters.memos import MemosAdapter, MemosConfig

        user_db_dir = args.user_db_dir or str(
            DEFAULT_EXTERNAL_ROOT / "volumes" / "memos" / stem / "users"
        )
        return MemosAdapter(
            MemosConfig(
                namespace=args.namespace or stem,
                user_db_dir=user_db_dir,
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=_secret(args.neo4j_password_env),
                neo4j_database=args.neo4j_database,
                neo4j_state_dir=args.neo4j_state_dir,
                require_answer_endpoint=args.command != "add",
                **common,
            )
        )
    raise ValueError(f"unsupported Track C system: {args.system}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(
        target: argparse.ArgumentParser, *, require_run_dir: bool = True
    ) -> None:
        target.add_argument(
            "--system", required=True, choices=("tridb_gem", "mem0", "memos", "cognee")
        )
        target.add_argument("--build-id", required=True)
        target.add_argument("--run-dir", required=require_run_dir)
        target.add_argument("--dataset", default=DEFAULT_DATASET)
        target.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
        target.add_argument("--answer-model", default="Qwen/Qwen3-32B")
        target.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
        target.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
        target.add_argument("--embedding-dim", type=int, default=1024)
        target.add_argument(
            "--timeout-seconds",
            type=float,
            default=60.0,
            help="per-request harness timeout; frozen Table 5 default is 60 seconds",
        )
        target.add_argument(
            "--qps",
            type=float,
            default=10.0,
            help="formal open-loop admission rate; sweep points are 1, 5, and 10",
        )
        target.add_argument(
            "--profile-stages",
            action="store_true",
            help="write per-request spans.jsonl and receipt time_breakdown",
        )

        # TriDB/GEM
        target.add_argument(
            "--dsn", default="postgresql://hza214@127.0.0.1:55432/postgres"
        )

        # Mem0 and Cognee
        target.add_argument(
            "--database-dsn", default="postgresql://hza214@127.0.0.1:55432/postgres"
        )
        target.add_argument("--database-name", default="postgres")
        target.add_argument("--collection-name")
        target.add_argument("--history-path")

        # Cognee
        target.add_argument("--db-host", default="127.0.0.1")
        target.add_argument("--db-port", type=int, default=55432)
        target.add_argument("--db-user", default="table5c_cognee_user")
        target.add_argument("--db-password-env", default="COGNEE_DB_PASSWORD")
        target.add_argument("--dataset-prefix")

        # MemOS
        target.add_argument("--namespace")
        target.add_argument("--user-db-dir")
        target.add_argument("--neo4j-uri", default="bolt://127.0.0.1:17687")
        target.add_argument("--neo4j-user", default="neo4j")
        target.add_argument("--neo4j-password-env", default="MEMOS_NEO4J_PASSWORD")
        target.add_argument("--neo4j-database", default="neo4j")
        target.add_argument("--neo4j-state-dir")

    search = subparsers.add_parser(
        "search", help="build corpus, warm up, run 1,787 Search"
    )
    add_common(search)
    search.add_argument(
        "--reuse-build-receipt",
        help="skip ingest and verify an already-cloned canonical Search build",
    )
    build_search = subparsers.add_parser(
        "build-search", help="build and fingerprint one immutable Search snapshot"
    )
    add_common(build_search)
    conformance = subparsers.add_parser(
        "conformance", help="run the common 3-session adapter gate"
    )
    add_common(conformance, require_run_dir=False)
    conformance.add_argument("--receipt", required=True)
    add = subparsers.add_parser("add", help="run 10 warmup + 2,000 formal Add")
    add_common(add)
    add.add_argument(
        "--definition", required=True, choices=("native", "source_to_searchable")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_locomo(args.dataset, verify=True)
    if args.command == "search":
        adapter = _adapter(args)
        receipt = run_search(
            adapter=adapter,
            corpus=corpus,
            build_id=args.build_id,
            run_dir=args.run_dir,
            timeout_seconds=args.timeout_seconds,
            qps=args.qps,
            profile_stages=args.profile_stages,
            reuse_build_receipt=args.reuse_build_receipt,
        )
    elif args.command == "build-search":
        adapter = _adapter(args)
        receipt = run_build_search(
            adapter=adapter,
            corpus=corpus,
            build_id=args.build_id,
            run_dir=args.run_dir,
        )
    elif args.command == "add":
        adapter = _adapter(args)
        receipt = run_add(
            adapter=adapter,
            corpus=corpus,
            build_id=args.build_id,
            definition=args.definition,
            run_dir=args.run_dir,
            timeout_seconds=args.timeout_seconds,
            qps=args.qps,
            profile_stages=args.profile_stages,
        )
    else:
        receipt = run_conformance(
            adapter_factory=lambda: _adapter(args),
            corpus=corpus,
            build_id=args.build_id,
            output=args.receipt,
        )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "system": receipt["system"],
                "build_id": receipt["build_id"],
                "phase": receipt["phase"],
                "output": str(
                    Path(
                        args.receipt if args.command == "conformance" else args.run_dir
                    ).resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
