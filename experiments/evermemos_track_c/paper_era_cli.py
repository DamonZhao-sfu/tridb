"""Run paper-era EverMemOS with the shared Track C protocol."""

from __future__ import annotations

import argparse

from bench.agent_memory.table5_track_c.dataset import load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    run_add,
    run_build_search,
    run_search,
)

from .paper_era_adapter import EverMemOSPaperAdapter, EverMemOSPaperConfig

DEFAULT_DATASET = (
    "/localhome/hza214/Mandol/experimental/self_host_benchmarks/locomo/"
    "data/locomo10.json"
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for command in ("build-search", "search", "add"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", default=DEFAULT_DATASET)
        child.add_argument("--run-dir", required=True)
        child.add_argument("--build-id", required=True)
        child.add_argument("--base-url", default="http://127.0.0.1:8195")
        child.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
        child.add_argument("--answer-model", default="Qwen/Qwen3-32B")
        child.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
        child.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
        child.add_argument("--build-scope-workers", type=int, default=5)
        child.add_argument("--request-timeout", type=float, default=600.0)
        if command in {"search", "add"}:
            child.add_argument("--qps", type=float, required=True)
            child.add_argument("--timeout-seconds", type=float, default=300.0)
            child.add_argument("--profile-stages", action="store_true")
        if command == "search":
            child.add_argument("--reuse-build-receipt")
        if command == "add":
            child.add_argument("--definition", choices=("native",), default="native")
    return root


def main() -> int:
    args = parser().parse_args()
    corpus = load_locomo(args.dataset)
    adapter = EverMemOSPaperAdapter(
        EverMemOSPaperConfig(
            dataset_path=args.dataset,
            base_url=args.base_url,
            answer_base_url=args.answer_base_url,
            answer_model=args.answer_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            build_scope_workers=args.build_scope_workers,
            request_timeout=args.request_timeout,
            require_answer_endpoint=args.command != "add",
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
            "run_dir": args.run_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
