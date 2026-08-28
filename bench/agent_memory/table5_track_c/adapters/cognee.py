"""Cognee 1.5 adapter for the controlled-model Table 5 protocol.

Cognee is asynchronous while the common load generator deliberately exposes a
synchronous adapter contract.  A single private event loop owns Cognee's async
engines; load-generator worker threads submit coroutines to it, preserving real
concurrency without creating one incompatible event loop per request.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Coroutine, Sequence

from ..dataset import EventItem, LoCoMoCorpus, QueryItem
from ..tracing import stage_span

_EVENT_ID = re.compile(r"\[event_id=([^\]]+)\]")


class _AsyncBridge:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


_SHARED_BRIDGE: _AsyncBridge | None = None
_SHARED_BRIDGE_LOCK = threading.Lock()


def _shared_async_bridge() -> _AsyncBridge:
    """Keep Cognee's cached async engines on one process-lifetime event loop.

    Cognee caches SQLAlchemy/asyncpg engines globally.  A conformance restart
    constructs a second adapter in the same Python process; closing the first
    adapter's loop leaves those official caches bound to a dead loop.  A shared
    process-lifetime bridge models an application restart without moving the
    database engines across loops.  The bridge thread is a daemon and exits
    with the benchmark process.
    """
    global _SHARED_BRIDGE
    with _SHARED_BRIDGE_LOCK:
        if _SHARED_BRIDGE is None:
            _SHARED_BRIDGE = _AsyncBridge()
        return _SHARED_BRIDGE


@dataclass(frozen=True)
class CogneeConfig:
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    dataset_prefix: str
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    build_scope_workers: int = 10
    #: Cognee's unit is a *document*, not an add() call: everything is queued
    #: with add() and then processed by one cognify() pass, whose cost scales
    #: with document count. ``turn`` makes one document per turn (the LoCoMo-era
    #: default); ``session`` makes one per source session, matching Mandol's
    #: LongMemEval granularity and cutting 246,738 documents to 23,854. The
    #: event_id provenance markers are preserved either way, one per turn, so
    #: the evidence gate still works. Recorded in the receipt.
    ingest_granularity: str = "turn"


class CogneeAdapter:
    name = "cognee"

    def __init__(self, config: CogneeConfig) -> None:
        self.config = config
        self._clone_rebind: dict[str, Any] | None = None
        # Cognee loads a repository .env with override=True during import.
        # The benchmark therefore imports first, applies Track C values second,
        # and clears every cached factory that may have observed old values.
        import cognee

        self.cognee = cognee
        self._configure_after_import()
        self.bridge = _shared_async_bridge()

    def _configure_after_import(self) -> None:
        config = self.config
        os.environ.update(
            {
                "USE_UNIFIED_PROVIDER": "",
                # With access control disabled Cognee accepts a dataset name
                # but searches the shared graph/vector namespace.  Track C
                # requires strict per-LoCoMo-sample scope isolation, so use
                # Cognee's native shared-Postgres dataset handlers.  Each
                # dataset receives its own schema in the same PostgreSQL DB.
                "ENABLE_BACKEND_ACCESS_CONTROL": "true",
                "CACHING": "false",
                "DB_PROVIDER": "postgres",
                "DB_HOST": config.db_host,
                "DB_PORT": str(config.db_port),
                "DB_USERNAME": config.db_user,
                "DB_PASSWORD": config.db_password,
                "DB_NAME": config.db_name,
                "GRAPH_DATABASE_PROVIDER": "postgres",
                "GRAPH_DATASET_DATABASE_HANDLER": "postgres_graph_shared",
                "VECTOR_DB_PROVIDER": "pgvector",
                "VECTOR_DATASET_DATABASE_HANDLER": "pgvector_shared",
                # Cognee's multi-dataset cognify path occasionally resolves a
                # vector engine outside a dataset context (for example while
                # calculating the next scope's chunk limit).  With backend
                # access control enabled it does not fall back to DB_* there,
                # so bind the shared PostgreSQL target explicitly.
                "VECTOR_DB_HOST": config.db_host,
                "VECTOR_DB_PORT": str(config.db_port),
                "VECTOR_DB_USERNAME": config.db_user,
                "VECTOR_DB_PASSWORD": config.db_password,
                "VECTOR_DB_NAME": config.db_name,
                # The Postgres graph engine has the same global-context
                # credential contract; configure it now rather than allowing
                # a later graph operation to fail for the analogous reason.
                "GRAPH_DATABASE_HOST": config.db_host,
                "GRAPH_DATABASE_PORT": str(config.db_port),
                "GRAPH_DATABASE_USERNAME": config.db_user,
                "GRAPH_DATABASE_PASSWORD": config.db_password,
                "GRAPH_DATABASE_NAME": config.db_name,
                "CACHE_BACKEND": "postgres",
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": f"openai/{config.answer_model}",
                "LLM_ENDPOINT": config.answer_base_url,
                "LLM_API_KEY": "EMPTY",
                "OPENAI_API_KEY": "EMPTY",
                # Cognee only forwards its declared OpenAI default Instructor
                # mode for GPT-5 model names.  Qwen OpenAI-compatible servers
                # otherwise fall back to tool calls, where Qwen may emit more
                # than one call and Instructor rejects the response.  Use
                # Cognee's public explicit-mode knob so structured extraction
                # is constrained to one JSON object instead.
                "LLM_INSTRUCTOR_MODE": "json_schema_mode",
                "LLM_TEMPERATURE": "0.0",
                "LLM_MAX_COMPLETION_TOKENS": "512",
                "COGNEE_SKIP_CONNECTION_TEST": "true",
                "LLM_RATE_LIMIT_ENABLED": "false",
                "AUTO_RATE_LIMIT": "false",
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": config.embedding_model,
                "EMBEDDING_ENDPOINT": config.embedding_base_url,
                "EMBEDDING_API_KEY": "EMPTY",
                "EMBEDDING_DIMENSIONS": str(config.embedding_dim),
                "EMBEDDING_BATCH_SIZE": "64",
                "HUGGINGFACE_TOKENIZER": config.embedding_model,
                "LOG_LEVEL": "ERROR",
                "TELEMETRY_DISABLED": "true",
                "COGNEE_TELEMETRY_DISABLED": "true",
            }
        )
        from cognee.infrastructure.databases.graph.config import get_graph_config
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            _create_graph_engine,
        )
        from cognee.infrastructure.databases.relational.config import (
            get_relational_config,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.vector.config import get_vectordb_config
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )
        from cognee.infrastructure.llm.config import get_llm_config
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            _get_llm_client_cached,
        )

        for cached in (
            get_llm_config,
            _get_llm_client_cached,
            get_embedding_config,
            get_relational_config,
            create_relational_engine,
            get_graph_config,
            get_vectordb_config,
            _create_graph_engine,
            _create_vector_engine,
        ):
            cached.cache_clear()

    def _dataset(self, sample_id: str) -> str:
        return f"{self.config.dataset_prefix}__{sample_id}"

    @staticmethod
    def _document(item: EventItem) -> str:
        # Cognee's public add API does not expose per-passage metadata in the
        # returned hybrid context.  A non-semantic provenance marker allows the
        # same evidence-id quality gate used for systems with metadata columns.
        return f"[event_id={item.event_id}] {item.text}"

    @staticmethod
    def _endpoint_models(base_url: str) -> list[str]:
        import json

        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as response:
            payload = json.load(response)
        return [str(item["id"]) for item in payload["data"]]

    async def _setup(self) -> None:
        from cognee.modules.engine.operations.setup import setup

        await setup()

    def init_schema(self) -> dict[str, Any]:
        answer_models = self._endpoint_models(self.config.answer_base_url)
        embedding_models = self._endpoint_models(self.config.embedding_base_url)
        if answer_models != [self.config.answer_model]:
            raise RuntimeError(f"answer endpoint mismatch: {answer_models!r}")
        if embedding_models != [self.config.embedding_model]:
            raise RuntimeError(f"embedding endpoint mismatch: {embedding_models!r}")
        self.bridge.run(self._setup())
        return {
            "ok": True,
            "answer_models": answer_models,
            "embedding_models": embedding_models,
            "embedding_dim": self.config.embedding_dim,
        }

    def _scope_documents(self, events: Sequence[EventItem]) -> dict[str, list[str]]:
        """Render each scope's turns into the configured document unit."""
        granularity = self.config.ingest_granularity
        if granularity not in {"turn", "session"}:
            raise ValueError(f"unknown ingest_granularity: {granularity!r}")
        if granularity == "turn":
            by_scope: dict[str, list[str]] = {}
            for item in events:
                by_scope.setdefault(item.sample_id, []).append(self._document(item))
            return by_scope
        grouped: dict[str, dict[str, list[EventItem]]] = {}
        for item in events:
            grouped.setdefault(item.sample_id, {}).setdefault(
                item.session_id, []
            ).append(item)
        return {
            sample_id: [
                # One document per session, but every turn keeps its own
                # [event_id=...] marker so evidence attribution is unchanged.
                "\n".join(self._document(turn) for turn in turns)
                for turns in sessions.values()
            ]
            for sample_id, sessions in grouped.items()
        }

    async def _build(self, events: Sequence[EventItem]) -> dict[str, Any]:
        by_scope = self._scope_documents(events)
        add_started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.config.build_scope_workers)

        async def add_scope(sample_id: str, documents: list[str]) -> Any:
            async with semaphore:
                return await self.cognee.add(
                    documents, dataset_name=self._dataset(sample_id), data_per_batch=20
                )

        await asyncio.gather(
            *(
                add_scope(sample_id, documents)
                for sample_id, documents in by_scope.items()
            )
        )
        add_seconds = time.perf_counter() - add_started
        cognify_started = time.perf_counter()
        result = await self.cognee.cognify(
            datasets=[self._dataset(sample_id) for sample_id in by_scope]
        )
        return {
            "events": len(events),
            "documents": sum(len(v) for v in by_scope.values()),
            "ingest_granularity": self.config.ingest_granularity,
            "scopes": len(by_scope),
            "add_seconds": add_seconds,
            "cognify_seconds": time.perf_counter() - cognify_started,
            "pipeline_result": repr(result),
        }

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.bridge.run(self._build(events))
        result["wall_seconds"] = time.perf_counter() - started
        return result

    def finalize_build(self) -> dict[str, Any]:
        return self.bridge.run(self._database_stats())

    async def _database_stats(self) -> dict[str, Any]:
        import asyncpg

        connection = await asyncpg.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
        )
        try:
            table_count = await connection.fetchval(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public'"
            )
            database_bytes = await connection.fetchval(
                "SELECT pg_database_size(current_database())"
            )
        finally:
            await connection.close()
        return {
            "public_tables": int(table_count),
            "database_bytes": int(database_bytes),
        }

    async def _search(self, item: QueryItem, top_k: int) -> dict[str, Any]:
        from cognee.modules.search.types import SearchType

        result = await self.cognee.search(
            query_text=item.question,
            query_type=SearchType.HYBRID_COMPLETION,
            only_context=True,
            top_k=top_k,
            datasets=[self._dataset(item.sample_id)],
        )
        contexts = [str(value) for value in (result or [])]
        hit_ids = list(dict.fromkeys(_EVENT_ID.findall("\n".join(contexts))))
        return {
            "result_count": len(hit_ids),
            "hit_ids": hit_ids,
            "contexts": contexts,
            "context_tokens": None,
            "empty": not contexts,
            "query_type": "HYBRID_COMPLETION",
            "only_context": True,
        }

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        with stage_span(
            "fusion",
            "cognee.hybrid_completion",
            backend="cognee_1.5.0_postgresql",
            attributes={
                "includes": ["embedding", "vector", "graph", "relational"],
                "only_context": True,
                "attribution": "compound_native_api",
                "observable_call_kind": "opaque_native_api",
            },
        ):
            return self.bridge.run(self._search(item, top_k))

    async def _add(
        self,
        item: EventItem,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        dataset = self._dataset(item.sample_id)
        result = await self.cognee.add(self._document(item), dataset_name=dataset)
        committed_at_ns = time.perf_counter_ns()
        if progress is not None:
            progress({"commit_observed": True, "committed_at_ns": committed_at_ns})
        visible = None
        visibility_error = None
        searchable_at_ns = None
        if visibility:
            if progress is not None:
                progress({"visibility_probe_started": True})
            try:
                await self.cognee.cognify(datasets=[dataset])
                visible = await self._visibility_probe(item)
            except Exception as exc:  # noqa: BLE001 - retained as measured evidence
                visible = False
                visibility_error = f"{type(exc).__name__}: {exc}"
            searchable_at_ns = time.perf_counter_ns() if visible else None
        receipt = {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": searchable_at_ns,
            "visibility_probe": visible,
            "visibility_error": visibility_error,
            "created_memory_count": None,
            "created_node_count": None,
            "created_edge_count": None,
            "creation_counts_available": False,
            "creation_count_source": "cognee_add_has_no_stable_count_contract",
            "pipeline_result": repr(result),
        }
        if progress is not None:
            progress(receipt)
        return receipt

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        with stage_span(
            "persistence",
            "cognee.source_add",
            backend="cognee_1.5.0_postgresql",
            attributes={
                "cognify_included": bool(visibility),
                "observable_call_kind": "opaque_native_api",
            },
        ):
            return self.bridge.run(self._add(item, visibility, progress))

    async def _visibility_probe(self, item: EventItem) -> bool:
        probe = QueryItem(
            sample_id=item.sample_id,
            question_id=f"probe:{item.event_id}",
            question=item.text,
            answer="",
            category="probe",
            evidence_ids=(item.event_id,),
            ordinal=0,
        )
        return item.event_id in (await self._search(probe, 35))["hit_ids"]

    def visibility_probe(self, item: EventItem) -> bool:
        return self.bridge.run(self._visibility_probe(item))

    def stats(self) -> dict[str, Any]:
        config = asdict(self.config)
        config["db_password"] = "<redacted>"
        stats = {
            **self.finalize_build(),
            "config": config,
            "native_shape": (
                "Cognee PostgreSQL relational + pgvector + PostgreSQL graph; "
                "HYBRID_COMPLETION retrieval with only_context=True"
            ),
            "pghybrid_enabled": False,
        }
        if self._clone_rebind is not None:
            stats["clone_rebind"] = dict(self._clone_rebind)
        return stats

    async def _snapshot_fingerprint(self) -> dict[str, Any]:
        import asyncpg

        connection = await asyncpg.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
        )
        try:
            counts = {}
            for table in ("datasets", "data", "nodes", "edges"):
                counts[table] = int(
                    await connection.fetchval(f'SELECT count(*) FROM "{table}"')
                )
        finally:
            await connection.close()
        return {
            "semantic_table_rows": counts,
            "dataset_prefix": self.config.dataset_prefix,
            "embedding_dim": self.config.embedding_dim,
        }

    def snapshot_fingerprint(self) -> dict[str, Any]:
        return self.bridge.run(self._snapshot_fingerprint())

    async def _rebind_cloned_dataset_databases(self) -> dict[str, Any]:
        """Point cloned dataset handlers at the clone, never at the template.

        Cognee persists graph/vector database names in ``dataset_database``.
        PostgreSQL template cloning copies those rows verbatim, so they must be
        rebound before the first retrieval from a cloned Search snapshot.
        """
        import asyncpg

        connection = await asyncpg.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
        )
        try:
            async with connection.transaction():
                command = await connection.execute(
                    "UPDATE dataset_database "
                    "SET graph_database_name = $1, vector_database_name = $1 "
                    "WHERE graph_database_name IS DISTINCT FROM $1 "
                    "OR vector_database_name IS DISTINCT FROM $1",
                    self.config.db_name,
                )
                row = await connection.fetchrow(
                    "SELECT count(*) AS total, "
                    "count(*) FILTER (WHERE graph_database_name = $1 "
                    "AND vector_database_name = $1) AS rebound "
                    "FROM dataset_database",
                    self.config.db_name,
                )
        finally:
            await connection.close()
        total = int(row["total"])
        rebound = int(row["rebound"])
        if total == 0 or rebound != total:
            raise RuntimeError(
                "Cognee clone dataset database rebind failed: "
                f"database={self.config.db_name!r}, total={total}, rebound={rebound}"
            )
        return {
            "database_name": self.config.db_name,
            "dataset_database_rows": total,
            "rows_rebound": int(command.rsplit(" ", 1)[-1]),
            "all_handlers_target_clone": True,
        }

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        self._clone_rebind = self.bridge.run(self._rebind_cloned_dataset_databases())
        if self._clone_rebind["dataset_database_rows"] != len(corpus.sample_ids):
            raise RuntimeError(
                "Cognee clone dataset scope count mismatch: "
                f"{self._clone_rebind['dataset_database_rows']} != "
                f"{len(corpus.sample_ids)}"
            )

    def close(self) -> None:
        # Cognee's async engine factories are process-global.  The shared bridge
        # deliberately stays alive across adapter reconstruction so a restart
        # visibility check cannot inherit engines bound to a closed event loop.
        return None
