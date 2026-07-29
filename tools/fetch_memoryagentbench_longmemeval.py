"""Fetch the five-history MemoryAgentBench LongMemEval workload as JSON.

This is a network-gated helper.  It requires the optional ``datasets`` package
and records the Hugging Face source/revision in the exported payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

DATASET = "ai-hyz/MemoryAgentBench"
SPLIT = "Accurate_Retrieval"
SOURCE = "longmemeval_s*"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/longmemeval/memoryagentbench_longmemeval_sstar.json"),
    )
    parser.add_argument("--revision", default="main")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "install the fetch dependency first: pip install 'datasets>=4.5,<5'"
        ) from exc

    dataset = load_dataset(DATASET, split=SPLIT, revision=args.revision)
    rows = [row for row in dataset if row["metadata"]["source"] == SOURCE]
    counts = [len(row["questions"]) for row in rows]
    if len(rows) != 5 or counts != [60] * 5:
        raise RuntimeError(
            "unexpected MemoryAgentBench workload shape: "
            f"{len(rows)} histories, question counts {counts}"
        )
    payload = {
        "dataset": DATASET,
        "split": SPLIT,
        "source": SOURCE,
        "revision": args.revision,
        "data": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} histories and {sum(counts)} questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
