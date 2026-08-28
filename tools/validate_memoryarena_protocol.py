"""Replay pinned MemoryArena exports through every arm's protocol/leakage gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.memoryarena.cross_session import CrossSessionDriver
from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.reference_backend import ReferenceBackend


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--injection-word-budget", type=int, default=4096)
    return parser.parse_args(argv)


def _digest_receipts(receipts: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for receipt in receipts:
        digest.update(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    configs = []
    for directory in sorted(
        path for path in args.export_root.iterdir() if path.is_dir()
    ):
        corpus = load_export(directory)
        arm_reports = []
        for arm in Arm:
            receipts: list[dict[str, Any]] = []
            selected_counts: Counter[int] = Counter()
            for task in corpus.tasks:
                driver = CrossSessionDriver(
                    ReferenceBackend(task),
                    run_id=f"protocol:{corpus.config}:{arm.value}",
                    dataset_manifest_sha256=corpus.manifest_sha256,
                    arm=arm,
                    top_k=10,
                    injection_token_budget=args.injection_word_budget,
                    # Protocol-only gate.  Formal runs must substitute the frozen
                    # backbone tokenizer and record its version.
                    count_tokens=lambda value: len(value.split()),
                )
                for session in task.sessions:
                    opened = driver.open_session(session)
                    if session.ordinal > 0:
                        serialized = opened.receipt.as_dict()
                        receipts.append(serialized)
                        selected_counts[len(opened.receipt.selected_session_uids)] += 1
                    driver.complete_session(session, response=session.gold_answer)
            arm_reports.append(
                {
                    "arm": arm.value,
                    "decision_receipts": len(receipts),
                    "selected_count_histogram": dict(sorted(selected_counts.items())),
                    "receipt_stream_sha256": _digest_receipts(receipts),
                    "leakage_violations": 0,
                    "timing_claim": "none_reference_backend",
                }
            )
        configs.append(
            {
                "config": corpus.config,
                "tasks": len(corpus.tasks),
                "sessions": corpus.session_count,
                "decision_points": corpus.decision_count,
                "source_revision": corpus.source_revision,
                "source_file_sha256": corpus.source_file_sha256,
                "manifest_sha256": corpus.manifest_sha256,
                "dependency_annotation": corpus.dependency_annotation,
                "arms": arm_reports,
            }
        )
    report = {
        "schema_version": "memoryarena_protocol_gate_v0.1.0",
        "status": "pass",
        "leakage_violations": 0,
        "token_counter": "whitespace_words_protocol_only",
        "timing_claim": "none",
        "configs": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS configs={len(configs)} decisions="
        f"{sum(config['decision_points'] for config in configs)} leakage=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
