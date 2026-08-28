"""Serial, leakage-safe CrossEp-Know pilot over GEM's five primary arms."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import requests

from bench.agent_memory.evomembench.dataset import (
    EvoContext,
    EvoEpisode,
    load_crossep_know,
)
from bench.agent_memory.evomembench.manifest import (
    load_manifest,
    manifest_sha256,
    verify_assets,
)
from bench.agent_memory.evomembench.injection import fit_injection
from bench.agent_memory.evomembench.modeling import (
    PreparedExperienceStrategy,
    experience_query,
    knowledge_experience,
)
from bench.agent_memory.evomembench.receipts import build_experience_receipt
from bench.agent_memory.evomembench.task_signature import knowledge_task_signature
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import InteractionEvent, RetrievalMode
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)

ARMS = ("memory_off", "recent_fifo", "vector_only", "graph_relational", "gem_fused")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()


def select_pilot_contexts(
    contexts: Sequence[EvoContext], *, count: int, seed: int
) -> tuple[EvoContext, ...]:
    """Deterministic category-stratified selection, pinned by ids in the receipt."""
    if count < 1 or count > len(contexts):
        raise ValueError(f"context count must be within 1..{len(contexts)}")
    rng = random.Random(seed)
    groups: dict[str, list[EvoContext]] = {}
    for context in contexts:
        groups.setdefault(context.category, []).append(context)
    for items in groups.values():
        rng.shuffle(items)
    selected: list[EvoContext] = []
    categories = sorted(groups)
    while len(selected) < count:
        progressed = False
        for category in categories:
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
                progressed = True
        if not progressed:
            break
    return tuple(selected)


class TokenCounter:
    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - runtime dependency gate
            raise RuntimeError(
                "pilot requires tiktoken for a fixed injection budget"
            ) from exc
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def __call__(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.encoding.decode(list(tokens))


class VLLMTokenCounter:
    """Use the exact frozen answer-model tokenizer exposed by vLLM."""

    def __init__(self, base_url: str, model: str, *, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self._encode_cache: dict[str, tuple[int, ...]] = {}
        self._chat_count_cache: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        cached = self._encode_cache.get(text)
        if cached is None:
            response = self.session.post(
                f"{self.base_url}/tokenize",
                json={
                    "model": self.model,
                    "prompt": text,
                    "add_special_tokens": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            cached = tuple(int(token) for token in response.json()["tokens"])
            self._encode_cache[text] = cached
        return list(cached)

    def decode(self, tokens: Sequence[int]) -> str:
        response = self.session.post(
            f"{self.base_url}/detokenize",
            json={"model": self.model, "tokens": [int(token) for token in tokens]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["prompt"])

    def __call__(self, text: str) -> int:
        return len(self.encode(text))

    def count_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Count the exact prompt produced by the live model's chat template."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
        if tools is not None:
            payload["tools"] = list(tools)
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        cache_key = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        cached = self._chat_count_cache.get(cache_key)
        if cached is None:
            response = self.session.post(
                f"{self.base_url}/tokenize",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            cached = int(body.get("count", len(body["tokens"])))
            self._chat_count_cache[cache_key] = cached
        return cached


def _replacement_capacity(
    episode: EvoEpisode, count_tokens: Any, *, min_system_tokens: int = 128
) -> int:
    system = next(
        (item.content for item in episode.messages if item.role == "system"), None
    )
    if system is None:
        return 0
    return max(0, count_tokens(system) - min_system_tokens)


def _messages(
    episode: EvoEpisode, injection: str, count_tokens: Any
) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    messages = [
        {"role": item.role, "content": item.content} for item in episode.messages
    ]
    if not injection:
        return messages, {
            "injection_policy": "replace_equal_total_input_tokens",
            "system_tokens_replaced": 0,
        }
    block_tokens = count_tokens.encode("\n\n") + count_tokens.encode(injection)
    for message in messages:
        if message["role"] == "system":
            original_tokens = count_tokens.encode(message["content"])
            capacity = max(0, len(original_tokens) - 128)
            if len(block_tokens) > capacity:
                raise ValueError(
                    f"memory injection needs {len(block_tokens)} tokens but replacement slot has {capacity}"
                )
            message["content"] = count_tokens.decode(
                original_tokens[: len(original_tokens) - len(block_tokens)]
                + block_tokens
            )
            final_tokens = count_tokens(message["content"])
            if final_tokens != len(original_tokens):
                raise RuntimeError(
                    "token replacement changed the frozen system token count"
                )
            break
    else:
        raise ValueError(
            "token-matched memory injection requires an existing system message"
        )
    return messages, {
        "injection_policy": "replace_equal_total_input_tokens",
        "system_tokens_original": len(original_tokens),
        "system_tokens_final": final_tokens,
        "system_tokens_replaced": len(block_tokens),
    }


def _event_time(ordinal: int) -> str:
    return (
        datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ordinal)
    ).isoformat()


def _load_selected(
    memory: TriDBGovernedMemory, unit_ids: Sequence[int]
) -> list[dict[str, Any]]:
    if not unit_ids:
        return []
    rows = memory.store.conn.execute(
        "SELECT u.id, u.title, (u.metadata->>'experience_ordinal')::integer,"
        " fv.value FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id=u.id"
        " WHERE u.id=ANY(%s) AND fv.field='memory_payload' AND fv.valid_to IS NULL"
        " ORDER BY array_position(%s::bigint[], u.id)",
        (list(unit_ids), list(unit_ids)),
    ).fetchall()
    return [
        {
            "unit_id": int(row[0]),
            "episode_uid": row[1],
            "ordinal": int(row[2]),
            "text": row[3],
        }
        for row in rows
    ]


def _retrieve(
    memory: TriDBGovernedMemory,
    *,
    arm: str,
    scope_id: str,
    episode: EvoEpisode,
    top_k: int,
    hops: int,
    m_seeds: int,
    term_cond: int,
    query_embedding: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    started = time.perf_counter()
    if episode.ordinal == 0 or arm == "memory_off":
        return [], {"mode": "off_or_empty", "termination_reason": "empty_snapshot"}, 0.0
    cutoff = episode.ordinal
    if arm == "recent_fifo":
        rows = memory.store.conn.execute(
            "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
            " AND metadata->>'node_kind'='experience'"
            " AND (metadata->>'experience_ordinal')::integer < %s"
            " ORDER BY (metadata->>'experience_ordinal')::integer DESC, id DESC LIMIT %s",
            (scope_id, cutoff, top_k),
        ).fetchall()
        ids = [int(row[0]) for row in rows]
        probes: dict[str, Any] = {"mode": "relational", "termination_reason": "limit"}
    elif arm == "graph_relational":
        anchor_row = memory.store.conn.execute(
            "SELECT id FROM gem_unit WHERE scope_id=%s"
            " AND metadata->>'node_kind'='experience'"
            " AND (metadata->>'experience_ordinal')::integer < %s"
            " ORDER BY (metadata->>'experience_ordinal')::integer DESC LIMIT 1",
            (scope_id, cutoff),
        ).fetchone()
        if anchor_row is None:
            return [], {"mode": "graph", "termination_reason": "empty_snapshot"}, 0.0
        query = experience_query(
            scope_id=scope_id,
            task_signature=knowledge_task_signature(episode),
            cutoff_ordinal=cutoff,
            mode=RetrievalMode.GRAPH,
            top_k=top_k,
            hops=hops,
            m_seeds=m_seeds,
            term_cond=term_cond,
        )
        result = memory.retrieve(
            replace(query, text=None, anchor_id=int(anchor_row[0]))
        )
        if not result.committed:
            raise RuntimeError(result.aborted_reason)
        ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
        anchor_id = int(anchor_row[0])
        if anchor_id not in ids:
            ids.insert(0, anchor_id)
        ids = ids[:top_k]
        probes = dict(result.probes)
        probes["relational_anchor_injected"] = anchor_id not in {
            hit.unit_id for hit in result.hits
        }
    else:
        mode = RetrievalMode.VECTOR if arm == "vector_only" else RetrievalMode.FUSED
        query = experience_query(
            scope_id=scope_id,
            task_signature=knowledge_task_signature(episode),
            cutoff_ordinal=cutoff,
            mode=mode,
            top_k=top_k,
            hops=hops,
            m_seeds=m_seeds,
            term_cond=term_cond,
        )
        if query_embedding is not None:
            query = replace(query, text=None, embedding=tuple(query_embedding))
        result = memory.retrieve(query)
        if not result.committed:
            raise RuntimeError(result.aborted_reason)
        ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
        probes = dict(result.probes)
    items = _load_selected(memory, ids)
    raw_ms = (time.perf_counter() - started) * 1000
    telemetry_ms = float(probes.get("instrumentation_probe_read_ms") or 0.0)
    probes["raw_retrieval_including_instrumentation_ms"] = raw_ms
    return items, probes, max(0.0, raw_ms - telemetry_ms)


def _operations(arm: str) -> dict[str, bool]:
    return {
        "vector_search": arm in {"vector_only", "gem_fused"},
        "relation_scan_or_filter": arm != "memory_off",
        "graph_traversal": arm in {"graph_relational", "gem_fused"},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    manifest = load_manifest(args.manifest)
    verified = verify_assets(args.source_root, manifest)
    manifest_hash = manifest_sha256(manifest)
    source_path = Path(args.source_root) / manifest["assets"][0]["path"]
    corpus = load_crossep_know(source_path)
    contexts = select_pilot_contexts(
        corpus.contexts, count=args.contexts, seed=args.seed
    )
    tokenizer_base_url = args.tokenizer_base_url or args.answer_base_url.removesuffix(
        "/v1"
    )
    count_tokens = VLLMTokenCounter(
        tokenizer_base_url, args.answer_model, timeout=args.timeout
    )
    ledger = CallLedger()
    embedding = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.timeout,
        ledger=ledger,
    )
    embedder = PhasedEmbedder(embedding)
    chat = OpenAIChatClient(
        args.answer_base_url, args.api_key, timeout=args.timeout, ledger=ledger
    )
    memory = TriDBGovernedMemory.connect(
        args.dsn, dim=args.embedding_dim, embedder=embedder
    )
    memory.init_schema()
    predictions = 0
    decisions = 0
    began = time.time()
    try:
        for arm in args.arms:
            for context_index, context in enumerate(contexts, 1):
                scope_id = f"{args.run_id}:{arm}:{context.context_uid}"
                for episode in context.episodes:
                    signature = knowledge_task_signature(episode)
                    phase = (
                        embedder.phase("query")
                        if arm not in {"memory_off", "recent_fifo", "graph_relational"}
                        else nullcontext()
                    )
                    with phase:
                        items, probes, retrieval_ms = _retrieve(
                            memory,
                            arm=arm,
                            scope_id=scope_id,
                            episode=episode,
                            top_k=args.top_k,
                            hops=args.hops,
                            m_seeds=args.m_seeds,
                            term_cond=args.term_cond,
                        )
                    accepted, injection, injection_tokens = fit_injection(
                        items,
                        max_items=args.top_k,
                        token_budget=max(
                            0,
                            min(
                                args.injection_token_budget,
                                _replacement_capacity(episode, count_tokens)
                                - len(count_tokens.encode("\n\n")),
                            ),
                        ),
                        count_tokens=count_tokens,
                    )
                    if episode.ordinal > 0 and arm != "memory_off":
                        if not items:
                            raise RuntimeError(
                                f"{arm} retrieved no eligible history for "
                                f"{episode.episode_uid} at ordinal {episode.ordinal}"
                            )
                        if not accepted or injection_tokens <= 0:
                            raise RuntimeError(
                                f"{arm} dropped all eligible history while formatting "
                                f"{episode.episode_uid} at ordinal {episode.ordinal}"
                            )
                    generation_messages, replacement_stats = _messages(
                        episode, injection, count_tokens
                    )
                    receipt = build_experience_receipt(
                        run_id=args.run_id,
                        manifest_sha256=manifest_hash,
                        track="CrossEp-Know",
                        arm=arm,
                        target_episode_uid=episode.episode_uid,
                        cutoff_ordinal=episode.ordinal,
                        task_signature=signature,
                        selected_unit_ids=[item["unit_id"] for item in accepted],
                        selected_episode_uids=[
                            item["episode_uid"] for item in accepted
                        ],
                        selected_ordinals=[item["ordinal"] for item in accepted],
                        injection_text=injection,
                        injection_tokens=injection_tokens,
                        latency_ms=retrieval_ms,
                        probes={
                            **probes,
                            "operations": _operations(arm),
                            "experience_overflow_policy": "truncate_head_tail",
                            "truncated_experiences": sum(
                                bool(item.get("injection_truncated"))
                                for item in accepted
                            ),
                            **replacement_stats,
                        },
                    )
                    generated = chat.stream_chat(
                        model=args.answer_model,
                        messages=generation_messages,
                        max_tokens=args.answer_max_tokens,
                        temperature=0.0,
                        seed=args.seed,
                    )
                    generation_seconds = (
                        generated.completed_at - generated.request_started
                    )
                    model_ttft_seconds = (
                        generated.first_token_at - generated.request_started
                    )
                    prediction = {
                        "idx": episode.source_task_id,
                        "messages": [asdict(item) for item in episode.messages],
                        "model_output": generated.text,
                        # Evaluation labels are emitted only after generation and
                        # are never passed to retrieval or experience ingestion.
                        "rubrics": list(episode.rubrics),
                        "metadata": {
                            "task_id": episode.source_task_id,
                            "context_id": context.source_context_id,
                            "context_category": episode.category,
                            "sub_category": episode.subcategory,
                            "ordinal": episode.ordinal,
                        },
                        "memory_type": arm,
                        "memory_retrieved": injection,
                        "stats": {
                            "retrieval_ms": retrieval_ms,
                            "injection_tokens": injection_tokens,
                            "generation_usage": generated.usage,
                            "generation_seconds": generation_seconds,
                            "model_ttft_seconds": model_ttft_seconds,
                            "operations": _operations(arm),
                            "experience_overflow_policy": "truncate_head_tail",
                            "truncated_experiences": sum(
                                bool(item.get("injection_truncated"))
                                for item in accepted
                            ),
                            **replacement_stats,
                        },
                    }
                    _append_jsonl(output / "predictions" / f"{arm}.jsonl", prediction)
                    _append_jsonl(
                        output / "receipts" / f"{arm}.jsonl", receipt.as_dict()
                    )
                    predictions += 1
                    decisions += int(episode.ordinal > 0)
                    if arm != "memory_off":
                        unit = knowledge_experience(
                            episode, response=generated.text, scope_id=scope_id
                        )
                        event_time = _event_time(episode.ordinal)
                        wal_before = memory.store.conn.execute(
                            "SELECT pg_current_wal_lsn()"
                        ).fetchone()[0]
                        with embedder.phase("construction"):
                            update = memory.ingest(
                                [
                                    InteractionEvent(
                                        scope_id=scope_id,
                                        external_id=episode.source_task_id,
                                        content=unit.memory_payload,
                                        session_id=episode.episode_uid,
                                        role="agent_trajectory",
                                        kind="experience",
                                        event_time=event_time,
                                        event_order=episode.ordinal,
                                    )
                                ],
                                strategy=PreparedExperienceStrategy(
                                    unit, valid_from=event_time
                                ),
                                scope_id=scope_id,
                            )
                        if not update.committed:
                            raise RuntimeError(update.aborted_reason)
                        wal_after = memory.store.conn.execute(
                            "SELECT pg_current_wal_lsn()"
                        ).fetchone()[0]
                        wal_bytes = memory.store.conn.execute(
                            "SELECT pg_wal_lsn_diff(%s, %s)", (wal_after, wal_before)
                        ).fetchone()[0]
                        _append_jsonl(
                            output / "updates" / f"{arm}.jsonl",
                            {
                                "episode_uid": episode.episode_uid,
                                "delta": asdict(update.delta),
                                "cost": asdict(update.cost),
                                "wal_bytes_interval": int(wal_bytes),
                                "wal_scope_note": (
                                    "global WAL delta over this update interval; may include "
                                    "concurrent database activity"
                                ),
                            },
                        )
                print(
                    f"[evomembench/{arm}] context {context_index}/{len(contexts)} "
                    f"{context.source_context_id}",
                    flush=True,
                )
    finally:
        memory.close()
    return {
        "schema_version": "evomembench_gem_run_receipt_v0.1.0",
        "status": "complete",
        "run_id": args.run_id,
        "elapsed_seconds": time.time() - began,
        "manifest_sha256": manifest_hash,
        "verified_assets": verified,
        "source_revision": corpus.source_revision,
        "contexts": [item.source_context_id for item in contexts],
        "arms": list(args.arms),
        "injection_policy": "replace_equal_total_input_tokens",
        "experience_overflow_policy": "truncate_head_tail",
        "token_counter": f"vllm:{args.answer_model}",
        "answer_max_tokens": args.answer_max_tokens,
        "predictions": predictions,
        "cross_episode_decisions": decisions,
        "model_calls": ledger.summary(),
        "unavailable_metrics": manifest["unavailable_metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("protocol_manifest.json")),
    )
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="qwen3.8")
    parser.add_argument("--tokenizer-base-url", default=None)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--contexts", type=int, default=12)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--term-cond", type=int, default=32)
    parser.add_argument("--injection-token-budget", type=int, default=4096)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output_preexisted = output.exists()
    try:
        receipt = run(args)
    except BaseException as exc:
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_gem_run_receipt_v0.1.0",
                        "status": "failed",
                        "run_id": args.run_id,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                )
                + "\n"
            )
        raise
    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
