"""Freeze the five-history MemoryAgentBench workload for all adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bench.agent_memory.serving import extract_retrieval_query, load_workloads
from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker


EXPECTED_SHA256 = "edf000491118e7bde37dcf13723025a2c18536955e7d518e52072f71daa16a54"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, ensure_ascii=False, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--tokenizer-model", default="gpt-4o-mini")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing workload artifact: {args.output}")
    checksum = sha256(args.input)
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"dataset checksum {checksum} != {EXPECTED_SHA256}")
    workloads = load_workloads(args.input, strict_shape=True)
    chunker = TiktokenSentenceChunker(
        chunk_size=args.chunk_tokens,
        tokenizer_model=args.tokenizer_model,
    )
    histories = []
    total_chunks = 0
    for history_index, workload in enumerate(workloads):
        chunks = chunker.chunk(workload.context)
        if not chunks:
            raise RuntimeError(f"history {history_index} produced no chunks")
        total_chunks += len(chunks)
        histories.append(
            {
                "history_index": history_index,
                "scope_id": workload.scope_id,
                "source": workload.source,
                "chunks": [
                    {
                        "event_id": f"{workload.scope_id}_chunk_{index:04d}",
                        "session_id": workload.scope_id,
                        "role": "memory_history",
                        "text": text,
                        "ordinal": index,
                        "metadata": {
                            "source": workload.source,
                            "history_index": history_index,
                            "chunk_index": index,
                            "chunk_tokens": chunker.count(text),
                        },
                    }
                    for index, text in enumerate(chunks)
                ],
                "questions": [
                    {
                        "question_index": question_index,
                        "question_id": question.question_id,
                        "question": question.question,
                        "retrieval_query": extract_retrieval_query(question.question),
                        "answer": question.answer,
                        "question_type": question.question_type,
                        "qa_pair_id": question.qa_pair_id,
                    }
                    for question_index, question in enumerate(workload.questions)
                ],
            }
        )
    atomic_json(
        args.output,
        {
            "schema_version": "figure10_frozen_workload_v0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(args.input.resolve()),
                "sha256": checksum,
            },
            "chunking": {
                "chunk_tokens": args.chunk_tokens,
                "tokenizer_model": args.tokenizer_model,
                "implementation": (
                    "bench.agent_memory.tridbBackend.chunking.TiktokenSentenceChunker"
                ),
            },
            "shape": {
                "histories": len(histories),
                "questions": sum(len(row["questions"]) for row in histories),
                "chunks": total_chunks,
                "questions_per_history": [len(row["questions"]) for row in histories],
                "chunks_per_history": [len(row["chunks"]) for row in histories],
            },
            "histories": histories,
        },
    )


if __name__ == "__main__":
    main()
