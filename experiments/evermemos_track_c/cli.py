"""Run EverMemOS (EverOS OSS) with the shared Track C protocol."""

from __future__ import annotations

import argparse

from bench.agent_memory.table5_track_c.dataset import load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    run_add,
    run_build_search,
    run_search,
)

from .adapter import EverMemOSTrackCAdapter, EverMemOSTrackCConfig

DEFAULT_DATASET = (
    "/localhome/hza214/Mandol/experimental/self_host_benchmarks/locomo/"
    "data/locomo10.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build-search", "search", "add"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", default=DEFAULT_DATASET)
        child.add_argument("--run-dir", required=True)
        child.add_argument("--build-id", required=True)
        child.add_argument("--base-url", default="http://127.0.0.1:8020")
        child.add_argument("--api-version", default="v2", choices=("v1", "v2"))
        child.add_argument("--app-id", default="tridb_trackc")
        child.add_argument("--project-id", default="locomo")
        child.add_argument(
            "--search-method",
            default="hybrid",
            choices=("keyword", "vector", "hybrid", "agentic"),
        )
        child.add_argument("--include-profile", action="store_true")
        child.add_argument("--enable-llm-rerank", action="store_true")
        child.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
        child.add_argument("--answer-model", default="Qwen/Qwen3-32B")
        child.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
        child.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
        child.add_argument("--embedding-dim", type=int, default=1024)
        child.add_argument(
            "--evermemos-source-root",
            default="/localhome/hza214/agent-memory-table5/src/evermemos",
        )
        child.add_argument("--build-scope-workers", type=int, default=5)
        child.add_argument("--request-timeout", type=float, default=300.0)
        child.add_argument("--drain-timeout", type=float, default=7200.0)
        if command in {"search", "add"}:
            child.add_argument("--qps", type=float, required=True)
            child.add_argument("--timeout-seconds", type=float, default=60.0)
            child.add_argument("--profile-stages", action="store_true")
        if command == "search":
            child.add_argument(
                "--reuse-build-receipt",
                help="optional immutable build receipt; omit for a fresh build",
            )
        if command == "add":
            child.add_argument(
                "--definition",
                choices=("native", "source_to_searchable"),
                default="native",
            )
    return parser


def _config(args: argparse.Namespace) -> EverMemOSTrackCConfig:
    return EverMemOSTrackCConfig(
        dataset_path=args.dataset,
        base_url=args.base_url,
        api_version=args.api_version,
        app_id=args.app_id,
        project_id=args.project_id,
        search_method=args.search_method,
        include_profile=args.include_profile,
        enable_llm_rerank=args.enable_llm_rerank,
        answer_base_url=args.answer_base_url,
        answer_model=args.answer_model,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        evermemos_source_root=args.evermemos_source_root,
        build_scope_workers=args.build_scope_workers,
        request_timeout=args.request_timeout,
        drain_timeout=args.drain_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_locomo(args.dataset)
    adapter = EverMemOSTrackCAdapter(_config(args))
    if args.command == "build-search":
        receipt = run_build_search(
            adapter=adapter,
            corpus=corpus,
            build_id=args.build_id,
            run_dir=args.run_dir,
        )
    elif args.command == "search":
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
    else:
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
    print(
        {
            "system": receipt["system"],
            "phase": receipt["phase"],
            "status": receipt["status"],
            "build_id": receipt["build_id"],
            "run_dir": args.run_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
