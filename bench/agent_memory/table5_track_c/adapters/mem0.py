"""Mem0 OSS adapter for the controlled-model Table 5 protocol."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..dataset import EventItem, QueryItem
from ..instrumentation import instrument_method
from ..tracing import stage_span

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PREPARED_ITEM_BOUNDARY = "prepared_memory_item_insertion_v1"


@dataclass(frozen=True)
class Mem0Config:
    database_dsn: str
    database_name: str
    collection_name: str
    history_path: str
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    max_connections: int = 64
    build_scope_workers: int = 10
    #: ``turn`` sends one native add() per LoCoMo/LongMemEval turn. ``session``
    #: sends one add() per source session, which is the granularity Mandol's
    #: own LongMemEval pipeline uses (dataset_maker/.../step1_build_batch_requests.py
    #: defaults ``sessions_per_group = 1``). On LongMemEval that is 23,854 calls
    #: instead of 246,738 -- a 10.3x reduction -- and it is also this API's
    #: canonical usage, since add() takes a message LIST. It is NOT free: the
    #: extractor sees a whole session per call, so the memories it writes differ
    #: from the per-turn ones. Whichever is chosen is recorded in the receipt.
    ingest_granularity: str = "turn"
    #: Abort the build once this fraction of batches has been dropped. The
    #: per-batch try/except below exists for a genuinely malformed extraction,
    #: which is rare. An endpoint that goes down produces the SAME exception on
    #: every subsequent call, and without this ceiling the build "succeeds"
    #: while silently discarding everything -- one run lost 13,093 of ~49,000
    #: batches (27%) to `LLMError: Connection error` after its vLLM replica
    #: exited, and still reported status=complete.
    max_dropped_fraction: float = 0.02
    #: Output budget for Mem0's own extraction LLM. 512 sufficed for one turn
    #: per call, but under `session` granularity the extractor must emit a JSON
    #: array covering a whole session and 12.8% of calls were truncated
    #: mid-array ("Expecting ',' delimiter" at lines 32-44). A truncated
    #: extraction is silently dropped by Mem0, so those sessions contributed no
    #: memory at all -- the cost lands on recall, not on an error count.
    extraction_max_tokens: int = 512

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.collection_name):
            raise ValueError(f"unsafe Mem0 collection name: {self.collection_name!r}")


class Mem0Adapter:
    name = "mem0"

    def __init__(self, config: Mem0Config) -> None:
        self.config = config
        # The official package reads this flag at import time.
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        from mem0 import Memory

        Path(config.history_path).parent.mkdir(parents=True, exist_ok=True)
        self.memory = Memory.from_config(self._native_config())
        self._stage_instrumentation: dict[str, bool] = {}
        self._llm_guard_lock = threading.Lock()
        self._llm_guard_installed = False
        self._forbidden_llm_calls = 0

    def _install_add_llm_guard(self) -> None:
        """Make a completed prepared-item Add proof that no LLM was invoked."""
        with self._llm_guard_lock:
            if self._llm_guard_installed:
                return

            def forbidden_llm_call(*_args: Any, **_kwargs: Any) -> Any:
                with self._llm_guard_lock:
                    self._forbidden_llm_calls += 1
                raise RuntimeError(
                    "Mem0 prepared-item Add attempted a forbidden LLM call"
                )

            self.memory.llm.generate_response = forbidden_llm_call
            self._llm_guard_installed = True

    @staticmethod
    def add_semantics() -> dict[str, Any]:
        return {
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "input_unit": "one frozen LoCoMo event rendered as one memory item",
            "construction_llm_policy": "forbidden_fail_closed",
            "included_stages": ["item_embedding", "vector_index", "persistence_commit"],
            "excluded_stages": [
                "conversation_memory_extraction",
                "deduplication_reasoning",
                "memory_update_reasoning",
            ],
            "native_call": "Memory.add(..., infer=False)",
        }

    def enable_stage_tracing(self) -> None:
        """Attach client-observed spans to stable Mem0 v2 component APIs."""
        targets = (
            (
                self.memory.embedding_model,
                "embed",
                "embedding",
                "mem0.embed",
                "http_client",
            ),
            (
                self.memory.vector_store,
                "search",
                "vector",
                "mem0.pgvector.search",
                "database_client",
            ),
            (
                self.memory.llm,
                "generate_response",
                "llm",
                "mem0.llm.generate",
                "http_client",
            ),
            (
                self.memory.vector_store,
                "insert",
                "persistence",
                "mem0.pgvector.insert",
                "database_client",
            ),
            (
                self.memory.vector_store,
                "update",
                "persistence",
                "mem0.pgvector.update",
                "database_client",
            ),
            (
                self.memory.vector_store,
                "delete",
                "persistence",
                "mem0.pgvector.delete",
                "database_client",
            ),
            (
                self.memory.db,
                "add_history",
                "persistence",
                "mem0.sqlite.history",
                "database_client",
            ),
        )
        self._stage_instrumentation = {
            operation: instrument_method(
                target,
                method,
                category,
                operation,
                backend="mem0_2.0.18",
                attributes={"observable_call_kind": call_kind},
            )
            for target, method, category, operation, call_kind in targets
        }

    def _native_config(self) -> dict[str, Any]:
        # Deliberately omit embedder.embedding_dims. Mem0 otherwise sends an
        # OpenAI `dimensions` request, which Qwen3-Embedding (fixed 1024-d,
        # non-Matryoshka) correctly rejects. The PGVector column and live gate
        # below still enforce exactly vector(1024).
        return {
            "version": "v1.1",
            "history_db_path": self.config.history_path,
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "collection_name": self.config.collection_name,
                    "embedding_model_dims": self.config.embedding_dim,
                    "connection_string": self.config.database_dsn,
                    "hnsw": True,
                    "minconn": 1,
                    "maxconn": self.config.max_connections,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.config.answer_model,
                    "api_key": "EMPTY",
                    "openai_base_url": self.config.answer_base_url,
                    "temperature": 0.0,
                    "max_tokens": self.config.extraction_max_tokens,
                    "is_reasoning_model": False,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.config.embedding_model,
                    "api_key": "EMPTY",
                    "openai_base_url": self.config.embedding_base_url,
                },
            },
        }

    def init_schema(self) -> dict[str, Any]:
        # PGVector creates its table lazily. A direct endpoint probe also
        # prevents a silent wrong-model or wrong-dimension run.
        vector = self.memory.embedding_model.embed("dimension probe", "search")
        if len(vector) != self.config.embedding_dim:
            raise RuntimeError(
                f"Mem0 embedding dimension {len(vector)} != {self.config.embedding_dim}"
            )
        self.memory.vector_store._ensure_collection()
        return {
            "ok": True,
            "collection": self.config.collection_name,
            "embedding_dim": len(vector),
        }

    @staticmethod
    def _message(item: EventItem) -> list[dict[str, str]]:
        # LoCoMo speaker identity remains in the rendered turn. Both speakers
        # are user-memory evidence rather than Mem0 procedural agent memory.
        return [{"role": "user", "content": item.text}]

    @staticmethod
    def _metadata(item: EventItem) -> dict[str, Any]:
        return {
            "source": "locomo",
            "event_id": item.event_id,
            "session_id": item.session_id,
            "speaker": item.role,
            "event_time": item.timestamp,
            "event_order": item.ordinal,
        }

    def _ingest_batches(
        self, scope_events: Sequence[EventItem]
    ) -> list[list[EventItem]]:
        """Group a scope's turns into the configured native-add unit."""
        if self.config.ingest_granularity == "turn":
            return [[item] for item in scope_events]
        if self.config.ingest_granularity != "session":
            raise ValueError(
                f"unknown ingest_granularity: {self.config.ingest_granularity!r}"
            )
        batches: dict[str, list[EventItem]] = {}
        for item in scope_events:
            batches.setdefault(item.session_id, []).append(item)
        return list(batches.values())

    def _scopes_with_memories(self) -> set[str]:
        """Scope ids that already have at least one row in the collection.

        Read straight from PGVector rather than a progress file: the store is
        the only thing that can say what actually landed, and a progress file
        could disagree with it after a crash.
        """
        import psycopg
        from psycopg import sql

        try:
            with psycopg.connect(self.config.database_dsn) as conn:
                rows = conn.execute(
                    sql.SQL(
                        "SELECT DISTINCT payload->>'user_id' FROM {} "
                        "WHERE payload->>'user_id' IS NOT NULL"
                    ).format(sql.Identifier(self.config.collection_name))
                ).fetchall()
        except psycopg.errors.UndefinedTable:
            return set()
        return {str(row[0]) for row in rows if row[0]}

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        started = time.perf_counter()
        by_scope: dict[str, list[EventItem]] = {}
        for item in events:
            by_scope.setdefault(item.sample_id, []).append(item)

        # Resume: a LongMemEval build is hours long and every scope is an
        # independent store, so a scope that already holds memories is done and
        # re-ingesting it would duplicate. Skipping them turns a crash from
        # "lose the whole build" into "lose the scope that was in flight".
        resumed = sorted(self._scopes_with_memories() & set(by_scope))
        for sample_id in resumed:
            by_scope.pop(sample_id, None)
        if not by_scope:
            return {
                "wall_seconds": time.perf_counter() - started,
                "events": len(events),
                "affected_memories": 0,
                "empty_extractions": 0,
                "dropped_batches": 0,
                "dropped_samples": [],
                "scopes_resumed": len(resumed),
                "scope_workers": 0,
            }

        total_batches = sum(
            len(self._ingest_batches(v)) for v in by_scope.values()
        )
        drop_ceiling = max(10, int(total_batches * self.config.max_dropped_fraction))
        drop_total = [0]
        drop_lock = threading.Lock()

        def ingest_scope(scope_events: Sequence[EventItem]) -> tuple[int, int, list[str]]:
            affected = 0
            empty = 0
            dropped: list[str] = []
            # A conversation remains strictly ordered.  Independent LoCoMo
            # conversations may build concurrently so vLLM can batch their
            # extraction calls; no query can observe another scope.
            for batch in self._ingest_batches(scope_events):
                try:
                    result = self.memory.add(
                        [self._message(entry)[0] for entry in batch],
                        user_id=batch[0].sample_id,
                        metadata=self._metadata(batch[0]),
                        infer=True,
                    )
                except Exception as exc:  # noqa: BLE001 - see below
                    # Mem0 assumes its extraction LLM always returns
                    # [{"text": ...}, ...] and indexes into it without a guard
                    # (mem0/memory/main.py: `m.get("text")`). A local model
                    # that answers with a bare list of strings therefore raises
                    # AttributeError from inside the library and, before this
                    # try, killed the entire build -- one malformed reply cost
                    # 6.5 hours of a 100-question shard.
                    #
                    # Degrade to skipping that batch. The turns in it contribute
                    # no memory, which lowers Mem0's recall, so the count is
                    # surfaced in the receipt rather than swallowed: a run with
                    # a high drop rate is not a clean measurement of Mem0.
                    dropped.append(
                        f"{batch[0].sample_id}/{batch[0].event_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    with drop_lock:
                        drop_total[0] += 1
                        if drop_total[0] > drop_ceiling:
                            raise RuntimeError(
                                f"Mem0 dropped {drop_total[0]} batches "
                                f"(ceiling {drop_ceiling}, "
                                f"{self.config.max_dropped_fraction:.1%} of "
                                f"{total_batches}); the extraction endpoint is "
                                f"most likely down. Last error: {exc}"
                            ) from exc
                    continue
                rows = list(result.get("results") or [])
                affected += len(rows)
                empty += int(not rows)
            return affected, empty, dropped

        workers = min(self.config.build_scope_workers, len(by_scope))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            counts = list(executor.map(ingest_scope, by_scope.values()))
        affected = sum(count[0] for count in counts)
        empty = sum(count[1] for count in counts)
        dropped_all = [msg for count in counts for msg in count[2]]
        return {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "affected_memories": affected,
            "empty_extractions": empty,
            "dropped_batches": len(dropped_all),
            "dropped_samples": dropped_all[:20],
            "scopes_resumed": len(resumed),
            "scope_workers": workers,
        }

    def finalize_build(self) -> dict[str, Any]:
        import psycopg
        from psycopg import sql

        with psycopg.connect(self.config.database_dsn) as conn:
            conn.execute(
                sql.SQL("ANALYZE {}").format(
                    sql.Identifier(self.config.collection_name)
                )
            )
            row = conn.execute(
                sql.SQL(
                    "SELECT count(*), pg_total_relation_size({}), "
                    "pg_database_size(current_database()) FROM {}"
                ).format(
                    sql.Literal(self.config.collection_name),
                    sql.Identifier(self.config.collection_name),
                )
            ).fetchone()
        return {
            "memories": int(row[0]),
            "collection_bytes": int(row[1]),
            "database_bytes": int(row[2]),
        }

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        with stage_span(
            "framework_other",
            "mem0.search.orchestration",
            backend="mem0_2.0.18",
            attributes={"observable_call_kind": "opaque_native_api"},
        ):
            result = self.memory.search(
                item.question,
                top_k=top_k,
                filters={"user_id": item.sample_id},
                # Mem0 has no native reranker in this operating point.
                rerank=False,
            )
        rows = list(result.get("results") or [])
        hit_ids = []
        contexts = []
        for row in rows:
            metadata = row.get("metadata") or {}
            if metadata.get("event_id") is not None:
                hit_ids.append(str(metadata["event_id"]))
            contexts.append(str(row.get("memory") or ""))
        return {
            "result_count": len(rows),
            "hit_ids": hit_ids,
            "memory_ids": [str(row.get("id")) for row in rows],
            "contexts": contexts,
            "context_tokens": None,
            "empty": not rows,
            "rerank": False,
        }

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._install_add_llm_guard()
        with stage_span(
            "framework_other",
            "mem0.add.prepared_item",
            backend="mem0_2.0.18",
            attributes={
                "includes": ["embedding", "vector_index", "persistence"],
                "excludes": ["llm", "memory_extraction", "deduplication_reasoning"],
                "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
                "observable_call_kind": "opaque_native_api",
            },
        ):
            result = self.memory.add(
                self._message(item),
                user_id=item.sample_id,
                metadata=self._metadata(item),
                infer=False,
            )
        committed_at_ns = time.perf_counter_ns()
        if progress is not None:
            progress({"commit_observed": True, "committed_at_ns": committed_at_ns})
        rows = list(result.get("results") or [])
        created = sum(str(row.get("event") or "").upper() == "ADD" for row in rows)
        visible = None
        visibility_error = None
        searchable_at_ns = None
        if visibility:
            if progress is not None:
                progress({"visibility_probe_started": True})
            try:
                visible = self.visibility_probe(item)
            except Exception as exc:  # noqa: BLE001 - retained as measured evidence
                visible = False
                visibility_error = f"{type(exc).__name__}: {exc}"
            searchable_at_ns = time.perf_counter_ns() if visible else None
        receipt = {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": searchable_at_ns,
            "visibility_probe": visible,
            "visibility_error": visibility_error,
            "affected_memories": len(rows),
            "created_memory_count": created,
            "created_node_count": None,
            "created_edge_count": None,
            "creation_counts_available": True,
            "creation_count_source": "mem0_add_results_event_ADD",
            "events": rows,
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "construction_llm_policy": "forbidden_fail_closed",
            "llm_call_count": 0,
            "updated_stores": ["pgvector_hnsw", "sqlite_history"],
            "prepared_item_id": item.event_id,
        }
        if progress is not None:
            progress(receipt)
        return receipt

    def visibility_probe(self, item: EventItem) -> bool:
        probe = QueryItem(
            sample_id=item.sample_id,
            question_id=f"probe:{item.event_id}",
            question=item.text,
            answer="",
            category="probe",
            evidence_ids=(item.event_id,),
            ordinal=0,
        )
        return item.event_id in self.search(probe, top_k=35)["hit_ids"]

    def stats(self) -> dict[str, Any]:
        result = self.finalize_build()
        config = asdict(self.config)
        config["database_dsn"] = re.sub(
            r"(://[^:/@]+:)[^@]+(@)", r"\1<redacted>\2", config["database_dsn"]
        )
        result["config"] = config
        result["native_shape"] = (
            "prepared item embedding + PGVector HNSW + SQLite history; no LLM"
        )
        result["stage_instrumentation"] = self._stage_instrumentation
        result["add_semantics"] = self.add_semantics()
        result["forbidden_llm_calls"] = self._forbidden_llm_calls
        return result

    def snapshot_fingerprint(self) -> dict[str, Any]:
        result = self.finalize_build()
        return {
            "memories": result["memories"],
            "collection": self.config.collection_name,
            "embedding_dim": self.config.embedding_dim,
        }

    def close(self) -> None:
        pool = getattr(self.memory.vector_store, "connection_pool", None)
        if pool is not None and hasattr(pool, "close"):
            pool.close()
        connection = getattr(self.memory.db, "connection", None)
        if connection is not None:
            connection.close()
