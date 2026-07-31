"""Run LoCoMo retrieval with TriDB as the persistent conversation backend.

Each LoCoMo conversation is ingested once and reused across all of its
questions. The output preserves the official JSON structure and augments each
QA item with ranked dialogue ids, retrieved text, and a ready-to-send prompt.
It does not call an answer model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from bench.agent_memory.tridbBackend.backend import (
    DEFAULT_DIM,
    DEFAULT_DSN,
    DEFAULT_MODEL,
    FastEmbedder,
    MemoryUnit,
    SearchHit,
    TriDBMemoryBackend,
)

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def iter_sessions(
    conversation: dict[str, Any],
) -> Iterable[tuple[int, str, list[dict[str, Any]]]]:
    """Yield every numbered LoCoMo session without a hard-coded upper bound."""
    numbered = []
    for key, value in conversation.items():
        match = _SESSION_KEY.fullmatch(key)
        if match is not None and isinstance(value, list):
            numbered.append((int(match.group(1)), value))
    for session_number, turns in sorted(numbered):
        timestamp = conversation.get(f"session_{session_number}_date_time", "")
        yield session_number, str(timestamp), turns


def _turn_text(turn: dict[str, Any], timestamp: str) -> str:
    speaker = str(turn.get("speaker", "Unknown"))
    text = str(
        turn.get(
            "compressed_text",
            turn.get("clean_text", turn.get("text", "")),
        )
    )
    rendered = f'({timestamp}) {speaker} said, "{text}"'
    caption = turn.get("blip_caption")
    if caption:
        rendered += f"\n[{speaker} shares {caption}]"
    return rendered


def build_units(sample: dict[str, Any]) -> list[MemoryUnit]:
    """Build dialogue-turn units; QA evidence is intentionally never read."""
    scope_id = str(sample["sample_id"])
    units = []
    for session_number, timestamp, turns in iter_sessions(sample["conversation"]):
        for turn_number, turn in enumerate(turns, start=1):
            dialogue_id = turn.get("dia_id")
            if not dialogue_id:
                raise ValueError(
                    f"{scope_id} session {session_number} turn {turn_number} "
                    "has no dia_id"
                )
            units.append(
                MemoryUnit(
                    scope_id=scope_id,
                    external_id=str(dialogue_id),
                    session_id=f"S{session_number}",
                    kind="turn",
                    role=turn.get("speaker"),
                    content=_turn_text(turn, timestamp),
                    event_time=timestamp,
                    event_order=session_number * 1_000_000 + turn_number,
                    metadata={
                        "source": "locomo",
                        "session_number": session_number,
                        "turn_number": turn_number,
                    },
                )
            )
    return units


def build_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    context = "\n\n".join(hit.content for hit in hits)
    return (
        "Answer the question using only the relevant conversation memory below. "
        "If the memory does not contain the answer, say that the information is "
        "not available.\n\n"
        f"Conversation memory:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _serialized_hits(hits: Sequence[SearchHit]) -> list[dict[str, Any]]:
    return [
        {
            "dia_id": hit.external_id,
            "session_id": hit.session_id,
            "text": hit.content,
            "timestamp": hit.event_time,
            "score": hit.score,
        }
        for hit in hits
    ]


def adapt_sample(
    sample: dict[str, Any],
    backend: TriDBMemoryBackend,
    *,
    top_k: int = 10,
    prediction_key: str = "tridb_prediction",
    isolated: bool = True,
) -> dict[str, Any]:
    """Augment one LoCoMo sample with TriDB retrieval results."""
    scope_id = str(sample["sample_id"])
    units = build_units(sample)
    backend.replace_scope(scope_id, units, isolated=isolated)

    output = dict(sample)
    output_qas = []
    for qa in sample["qa"]:
        hits = backend.search(
            scope_id,
            query_text=str(qa["question"]),
            k=top_k,
        )
        output_qa = dict(qa)
        # The official evaluator reads <prediction_key>_context when a
        # downstream answer model adds <prediction_key>.
        output_qa[f"{prediction_key}_context"] = [hit.external_id for hit in hits]
        output_qa[f"{prediction_key}_retrieval"] = _serialized_hits(hits)
        output_qa[f"{prediction_key}_prompt"] = build_prompt(
            str(qa["question"]),
            hits,
        )
        output_qas.append(output_qa)
    output["qa"] = output_qas
    output["tridb_backend"] = {
        "name": "tridb",
        "mode": "vector",
        "table": backend.table,
        "scope_id": scope_id,
        "units": len(units),
        "top_k": top_k,
        "prediction_key": prediction_key,
    }
    return output


def adapt_samples(
    samples: Iterable[dict[str, Any]],
    backend: TriDBMemoryBackend,
    *,
    top_k: int = 10,
    prediction_key: str = "tridb_prediction",
    isolated: bool = True,
) -> Iterable[dict[str, Any]]:
    for sample in samples:
        yield adapt_sample(
            sample,
            backend,
            top_k=top_k,
            prediction_key=prediction_key,
            isolated=isolated,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prediction-key", default="tridb_prediction")
    parser.add_argument(
        "--shared-index",
        action="store_true",
        help="retain other conversation scopes instead of isolating each corpus",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("TRIDB_DSN", DEFAULT_DSN),
    )
    parser.add_argument("--table", default="locomo_units")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = json.loads(args.input.read_text())
    if not isinstance(samples, list):
        raise ValueError("LoCoMo input must be a JSON array")
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]

    embedder = FastEmbedder(args.model, batch_size=args.batch_size)
    backend = TriDBMemoryBackend.connect(
        args.dsn,
        dim=args.dim,
        table=args.table,
        embedder=embedder,
    )
    try:
        backend.init_schema()
        output = []
        for index, result in enumerate(
            adapt_samples(
                samples,
                backend,
                top_k=args.top_k,
                prediction_key=args.prediction_key,
                isolated=not args.shared_index,
            ),
            start=1,
        ):
            output.append(result)
            print(
                f"[locomo] {index}/{len(samples)} {result['sample_id']}",
                file=sys.stderr,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
