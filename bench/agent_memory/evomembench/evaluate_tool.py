"""Run the pinned official CrossEp-Tool evaluator after generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

UPSTREAM_RELATIVE_ROOT = "Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL"


def evaluate(source_root: Path, run_dir: Path, *, workers: int) -> int:
    upstream = source_root / UPSTREAM_RELATIVE_ROOT
    results = sorted(run_dir.glob("phase*/**/result.jsonl"))
    if not results:
        raise FileNotFoundError(f"no CrossEp-Tool result.jsonl files under {run_dir}")
    for result in results:
        summary = result.parent / "summary.csv"
        if summary.exists():
            raise FileExistsError(f"refusing existing official evaluation: {summary}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "bfcl_eval.scripts.cross_episode.run_batch_evaluate",
                "--input",
                str(result),
                "--output-dir",
                str(result.parent),
                "--num-workers",
                str(workers),
            ],
            cwd=upstream,
            check=True,
        )
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        {
            "evaluated_settings": evaluate(
                args.source_root, args.run_dir, workers=args.workers
            )
        }
    )


if __name__ == "__main__":
    main()
