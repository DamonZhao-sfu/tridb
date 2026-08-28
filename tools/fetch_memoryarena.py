"""Fetch and content-address the pinned MemoryArena benchmark configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.memoryarena.dataset import canonical_json_sha256

DATASET = "ZexueHe/memoryarena"
PINNED_REVISION = "da1a37c8b19280e18627ca01cf368195a5e1d92e"
CONFIGS = (
    "progressive_search",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)
EXPECTED_RELEASED_ROWS = {
    "progressive_search": 221,
    "formal_reasoning_math": 40,
    "formal_reasoning_phys": 20,
}
PAPER_REPORTED_ROWS = {
    "progressive_search": 256,
    "formal_reasoning_math": 40,
    "formal_reasoning_phys": 20,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/agent_memory/memoryarena"),
    )
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _encode_rows(rows: Sequence[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "install the fetch dependencies first: pip install datasets huggingface_hub"
        ) from exc

    resolved_revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    if not resolved_revision:
        raise RuntimeError("Hugging Face did not return a resolved dataset revision")

    for config in args.configs:
        dataset = load_dataset(
            DATASET, config, split="test", revision=resolved_revision
        )
        rows = [dict(row) for row in dataset]
        expected = EXPECTED_RELEASED_ROWS[config]
        if len(rows) != expected:
            raise RuntimeError(
                f"unexpected {config} row count at {resolved_revision}: "
                f"{len(rows)} != {expected}"
            )
        payload = _encode_rows(rows)
        data_sha = hashlib.sha256(payload).hexdigest()
        session_counts = [len(row["questions"]) for row in rows]
        directory = args.output_dir / config
        directory.mkdir(parents=True, exist_ok=True)
        data_file = "data.jsonl"
        (directory / data_file).write_bytes(payload)
        manifest = {
            "schema_version": "memoryarena_fetch_v0.1.0",
            "dataset": DATASET,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "config": config,
            "split": "test",
            "data_file": data_file,
            "data_sha256": data_sha,
            "rows": len(rows),
            "sessions": sum(session_counts),
            "session_count_min": min(session_counts),
            "session_count_max": max(session_counts),
            "paper_reported_rows": PAPER_REPORTED_ROWS[config],
            "release_matches_paper_count": (len(rows) == PAPER_REPORTED_ROWS[config]),
            "license": "CC-BY-4.0",
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{config}: {len(rows)} tasks, {sum(session_counts)} sessions, "
            f"revision={resolved_revision}, sha256={data_sha}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
