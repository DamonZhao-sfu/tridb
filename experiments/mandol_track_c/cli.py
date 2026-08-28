"""Build once and run Mandol Search with the shared Track C protocol."""

from __future__ import annotations

import argparse

from bench.agent_memory.table5_track_c.dataset import load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    run_add,
    run_build_search,
    run_search,
)

from .adapter import MandolAdapter, MandolConfig

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
        child.add_argument("--snapshot-dir", required=True)
        child.add_argument("--run-dir", required=True)
        child.add_argument("--build-id", required=True)
        child.add_argument("--answer-base-url", default="http://127.0.0.1:8010/v1")
        child.add_argument("--answer-model", default="Qwen/Qwen3-32B")
        child.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
        child.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
        child.add_argument("--embedding-dim", type=int, default=1024)
        child.add_argument("--reranker-base-url", default="http://127.0.0.1:8012")
        child.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
        child.add_argument("--request-timeout-seconds", type=int, default=60)
        child.add_argument("--mandol-source-root", default="/localhome/hza214/Mandol")
        if command == "build-search":
            child.add_argument("--resume-snapshot-source")
            child.add_argument(
                "--resume-sample",
                action="append",
                default=[],
                help="exact completed sample expected in the partial snapshot",
            )
        if command in {"search", "add"}:
            child.add_argument("--qps", type=float, required=True)
            child.add_argument("--timeout-seconds", type=float, default=60.0)
            child.add_argument("--profile-stages", action="store_true")
        if command == "search":
            child.add_argument("--reuse-build-receipt", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    corpus = load_locomo(args.dataset)
    adapter = MandolAdapter(
        MandolConfig(
            snapshot_dir=args.snapshot_dir,
            dataset_path=args.dataset,
            answer_base_url=args.answer_base_url,
            answer_model=args.answer_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            reranker_base_url=args.reranker_base_url,
            reranker_model=args.reranker_model,
            request_timeout_seconds=args.request_timeout_seconds,
            mandol_source_root=args.mandol_source_root,
            resume_snapshot_source=getattr(args, "resume_snapshot_source", None),
            resume_expected_samples=tuple(getattr(args, "resume_sample", [])),
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
            definition="native",
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
