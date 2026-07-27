"""Run LongMemEval retrieval with TriDB as the scoped vector backend.

The output is JSONL-compatible with LongMemEval's generation scripts. This
module intentionally mirrors the official flat index's corpus-id conventions
so its retrieval metrics and ``flat-session``/``flat-turn`` generation modes
remain applicable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from bench.agent_memory.backend import (
    DEFAULT_DIM,
    DEFAULT_DSN,
    DEFAULT_MODEL,
    FastEmbedder,
    MemoryUnit,
    TriDBMemoryBackend,
)

METRIC_KS = (1, 3, 5, 10, 30, 50)


def _included_turns(
    session: Sequence[dict[str, Any]],
    *,
    include_assistant: bool,
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, turn)
        for index, turn in enumerate(session, start=1)
        if include_assistant or turn.get("role") == "user"
    ]


def _session_corpus_id(
    session_id: str,
    turns: Sequence[dict[str, Any]],
) -> str:
    if "answer" in session_id and all(
        not bool(turn.get("has_answer", False)) for turn in turns
    ):
        return session_id.replace("answer", "noans")
    return session_id


def build_units(
    entry: dict[str, Any],
    *,
    granularity: str,
    include_assistant: bool = False,
) -> list[MemoryUnit]:
    """Convert one LongMemEval question's haystack without label leakage."""
    if granularity not in {"session", "turn"}:
        raise ValueError("granularity must be 'session' or 'turn'")
    session_ids = entry["haystack_session_ids"]
    sessions = entry["haystack_sessions"]
    dates = entry["haystack_dates"]
    if not (len(session_ids) == len(sessions) == len(dates)):
        raise ValueError("LongMemEval haystack arrays have different lengths")

    scope_id = str(entry["question_id"])
    units: list[MemoryUnit] = []
    for session_position, (session_id, session, timestamp) in enumerate(
        zip(session_ids, sessions, dates, strict=True)
    ):
        included = _included_turns(
            session,
            include_assistant=include_assistant,
        )
        if granularity == "session":
            turns = [turn for _, turn in included]
            corpus_id = _session_corpus_id(session_id, turns)
            if include_assistant:
                content = "\n".join(
                    f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
                    for turn in turns
                )
            else:
                content = " ".join(str(turn.get("content", "")) for turn in turns)
            units.append(
                MemoryUnit(
                    scope_id=scope_id,
                    external_id=corpus_id,
                    session_id=session_id,
                    kind="session",
                    content=content,
                    event_time=str(timestamp),
                    event_order=session_position,
                    metadata={
                        "source": "longmemeval",
                        "source_session_id": session_id,
                        "granularity": "session",
                    },
                )
            )
            continue

        for turn_index, turn in included:
            corpus_id = f"{session_id}_{turn_index}"
            if "answer" in session_id and not bool(turn.get("has_answer", False)):
                corpus_id = corpus_id.replace("answer", "noans")
            content = str(turn.get("content", ""))
            if include_assistant:
                content = f"{turn.get('role', 'unknown')}: {content}"
            units.append(
                MemoryUnit(
                    scope_id=scope_id,
                    external_id=corpus_id,
                    session_id=session_id,
                    kind="turn",
                    role=turn.get("role"),
                    content=content,
                    event_time=str(timestamp),
                    event_order=session_position * 1_000_000 + turn_index,
                    metadata={
                        "source": "longmemeval",
                        "source_session_id": session_id,
                        "turn_index": turn_index,
                        "granularity": "turn",
                    },
                )
            )
    return units


def _dcg(relevances: Sequence[int]) -> float:
    if not relevances:
        return 0.0
    total = float(relevances[0])
    for offset, relevance in enumerate(relevances[1:], start=2):
        total += float(relevance) / math.log2(offset)
    return total


def _evaluate_ids(
    ranked_ids: Sequence[str],
    correct_ids: set[str],
    *,
    k: int,
    corpus_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    top_ids = list(ranked_ids[:k])
    recalled = set(top_ids)
    recall_any = float(any(doc_id in recalled for doc_id in correct_ids))
    recall_all = float(all(doc_id in recalled for doc_id in correct_ids))
    actual = _dcg([int(doc_id in correct_ids) for doc_id in top_ids])
    ideal_source = ranked_ids if corpus_ids is None else corpus_ids
    ideal_count = min(
        k,
        sum(doc_id in correct_ids for doc_id in ideal_source),
    )
    ideal = _dcg([1] * ideal_count)
    ndcg_any = 0.0 if ideal == 0.0 else actual / ideal
    return {
        f"recall_any@{k}": recall_any,
        f"recall_all@{k}": recall_all,
        f"ndcg_any@{k}": ndcg_any,
    }


def _session_id_from_turn(corpus_id: str) -> str:
    return "_".join(corpus_id.split("_")[:-1])


def retrieval_metrics(
    ranked_ids: Sequence[str],
    corpus_ids: Sequence[str],
    *,
    granularity: str,
) -> dict[str, dict[str, float]]:
    """Compute the official LongMemEval flat-retrieval metric shapes."""
    correct_ids = {corpus_id for corpus_id in corpus_ids if "answer" in corpus_id}
    metrics: dict[str, dict[str, float]] = {"session": {}, "turn": {}}
    for k in METRIC_KS:
        metrics[granularity].update(
            _evaluate_ids(
                ranked_ids,
                correct_ids,
                k=k,
                corpus_ids=corpus_ids,
            )
        )
        if granularity != "turn":
            continue

        session_ranked = [_session_id_from_turn(doc_id) for doc_id in ranked_ids]
        correct_sessions = {_session_id_from_turn(doc_id) for doc_id in correct_ids}
        effective_k = min(k, len(session_ranked))
        while (
            effective_k < len(session_ranked)
            and len(set(session_ranked[:effective_k])) < k
        ):
            effective_k += 1
        session_values = _evaluate_ids(
            session_ranked,
            correct_sessions,
            k=effective_k,
            corpus_ids=[_session_id_from_turn(doc_id) for doc_id in corpus_ids],
        )
        metrics["session"].update(
            {
                f"recall_any@{k}": session_values[f"recall_any@{effective_k}"],
                f"recall_all@{k}": session_values[f"recall_all@{effective_k}"],
                f"ndcg_any@{k}": session_values[f"ndcg_any@{effective_k}"],
            }
        )
    return metrics


def adapt_entry(
    entry: dict[str, Any],
    backend: TriDBMemoryBackend,
    *,
    granularity: str,
    retrieve_k: int = 50,
    include_assistant: bool = False,
    isolated: bool = True,
) -> dict[str, Any]:
    units = build_units(
        entry,
        granularity=granularity,
        include_assistant=include_assistant,
    )
    scope_id = str(entry["question_id"])
    backend.replace_scope(scope_id, units, isolated=isolated)
    hits = backend.search(
        scope_id,
        query_text=str(entry["question"]),
        k=retrieve_k,
    )
    ranked_ids = [hit.external_id for hit in hits]
    corpus_ids = [unit.external_id for unit in units]

    output = dict(entry)
    output["retrieval_results"] = {
        "query": entry["question"],
        "ranked_items": [
            {
                "corpus_id": hit.external_id,
                "text": hit.content,
                "timestamp": hit.event_time,
                "score": hit.score,
            }
            for hit in hits
        ],
        "metrics": retrieval_metrics(
            ranked_ids,
            corpus_ids,
            granularity=granularity,
        ),
        "backend": {
            "name": "tridb",
            "mode": "vector",
            "table": backend.table,
            "scope_id": scope_id,
            "include_assistant": include_assistant,
        },
    }
    return output


def adapt_entries(
    entries: Iterable[dict[str, Any]],
    backend: TriDBMemoryBackend,
    *,
    granularity: str,
    retrieve_k: int = 50,
    include_assistant: bool = False,
    isolated: bool = True,
) -> Iterable[dict[str, Any]]:
    for entry in entries:
        yield adapt_entry(
            entry,
            backend,
            granularity=granularity,
            retrieve_k=retrieve_k,
            include_assistant=include_assistant,
            isolated=isolated,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--granularity", choices=("session", "turn"), required=True)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="index both roles; default mirrors the official user-only flat index",
    )
    parser.add_argument(
        "--shared-index",
        action="store_true",
        help="retain other question scopes instead of isolating each corpus",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("TRIDB_DSN", DEFAULT_DSN),
    )
    parser.add_argument("--table", default="longmemeval_units")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    entries = json.loads(args.input.read_text())
    if not isinstance(entries, list):
        raise ValueError("LongMemEval input must be a JSON array")
    if args.limit is not None:
        entries = entries[: args.limit]

    embedder = FastEmbedder(args.model, batch_size=args.batch_size)
    backend = TriDBMemoryBackend.connect(
        args.dsn,
        dim=args.dim,
        table=args.table,
        embedder=embedder,
    )
    try:
        backend.init_schema()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as output:
            for index, result in enumerate(
                adapt_entries(
                    entries,
                    backend,
                    granularity=args.granularity,
                    retrieve_k=args.retrieve_k,
                    include_assistant=args.include_assistant,
                    isolated=not args.shared_index,
                ),
                start=1,
            ):
                print(json.dumps(result, ensure_ascii=False), file=output)
                print(
                    f"[longmemeval] {index}/{len(entries)} {result['question_id']}",
                    file=sys.stderr,
                )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
