"""Command-line entry point for the GEM Wikipedia demo."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from bench.agent_memory.demo import report, scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase",
        choices=("all", "ingest", "retrieve", "revise", "forget"),
        default="all",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("TRIDB_GEM_DSN", scenario.DEFAULT_DSN),
    )
    parser.add_argument("--slice", type=Path, default=scenario.DEFAULT_SLICE)
    parser.add_argument("--output", type=Path, default=scenario.DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope",
        default=None,
        help="fresh by default; pass a stable name with --reset for a rerunnable demo",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="remove only --scope before running (native edges are tombstoned)",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="limit the fully-resolved question set (default: all)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--term-cond", type=int, default=32)
    parser.add_argument("--model", default=scenario.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.questions is not None and args.questions <= 0:
        raise SystemExit("--questions must be positive")
    if args.k <= 0 or args.term_cond <= 0:
        raise SystemExit("--k and --term-cond must be positive")
    scope = args.scope or ("gem-wiki-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))

    memory = scenario.connect(
        args.dsn, model_name=args.model, batch_size=args.batch_size
    )
    try:
        results, manifest = scenario.run(
            memory,
            slice_dir=args.slice,
            scope_id=scope,
            phase=args.phase,
            reset=args.reset,
            question_limit=args.questions,
            k=args.k,
            term_cond=args.term_cond,
            model_name=args.model,
        )
        paths = report.write(results, manifest, args.output)
    finally:
        memory.close()

    print(f"scope: {scope}")
    for name in ("report", "results", "manifest"):
        print(f"{name}: {paths[name]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
