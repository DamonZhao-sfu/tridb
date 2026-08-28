"""Run the formally guarded Graphiti adapter with the shared Track C protocol."""

from __future__ import annotations

import argparse

from bench.agent_memory.table5_track_c.dataset import load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    run_add,
    run_build_search,
    run_search,
)
from experiments.graphiti_track_c.adapter import GraphitiTrackCConfig

from .adapter import FormalGraphitiTrackCAdapter

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
        child.add_argument("--neo4j-state-dir", required=True)
        child.add_argument("--neo4j-uri", default="bolt://127.0.0.1:27687")
        child.add_argument("--neo4j-user", default="neo4j")
        child.add_argument("--neo4j-password", default="trackc_graphiti_local_20260821")
        child.add_argument("--neo4j-database", default="neo4j")
        child.add_argument("--run-dir", required=True)
        child.add_argument("--build-id", required=True)
        child.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
        child.add_argument("--answer-model", default="Qwen/Qwen3-32B")
        child.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
        child.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
        child.add_argument("--embedding-dim", type=int, default=1024)
        child.add_argument(
            "--graphiti-source-root",
            default="/localhome/hza214/agent-memory-table5/src/graphiti",
        )
        child.add_argument("--build-scope-workers", type=int, default=2)
        child.add_argument("--max-coroutines", type=int, default=16)
        child.add_argument("--llm-timeout-seconds", type=float, default=60.0)
        child.add_argument("--resume-partial-build", action="store_true")
        child.add_argument(
            "--require-answer-endpoint",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "require the generation-model endpoint during schema init; use "
                "--no-require-answer-endpoint for fail-closed LLM-free Search/Add"
            ),
        )
        if command in {"search", "add"}:
            child.add_argument("--qps", type=float, required=True)
            child.add_argument("--timeout-seconds", type=float, default=60.0)
            child.add_argument("--profile-stages", action="store_true")
        if command == "search":
            child.add_argument("--reuse-build-receipt")
        if command == "add":
            child.add_argument(
                "--definition",
                choices=("native", "source_to_searchable"),
                default="native",
            )
    return parser


def main() -> int:
    args = _parser().parse_args()
    corpus = load_locomo(args.dataset)
    adapter = FormalGraphitiTrackCAdapter(
        GraphitiTrackCConfig(
            dataset_path=args.dataset,
            neo4j_state_dir=args.neo4j_state_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            answer_base_url=args.answer_base_url,
            answer_model=args.answer_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            graphiti_source_root=args.graphiti_source_root,
            build_scope_workers=args.build_scope_workers,
            max_coroutines=args.max_coroutines,
            llm_timeout_seconds=args.llm_timeout_seconds,
            resume_partial_build=args.resume_partial_build,
            require_answer_endpoint=args.require_answer_endpoint,
        )
    )
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
