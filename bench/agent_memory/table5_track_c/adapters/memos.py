"""MemOS tree-text adapter for the controlled-model Table 5 protocol.

The selected OSS operating point is the public ``MOS`` API backed by MemOS'
tree-text memory and an isolated Neo4j Community instance. A LoCoMo
conversation maps to one MemOS user/cube so every query is scope-isolated.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..dataset import EventItem, LoCoMoCorpus, QueryItem
from ..instrumentation import instrument_method
from ..tracing import stage_span

_SAFE = re.compile(r"[^A-Za-z0-9_]+")
_EVENT_ID = re.compile(r"\[event_id=([^\]\s]+)\]")
_PREPARED_ITEM_BOUNDARY = "prepared_memory_item_insertion_v1"


@dataclass(frozen=True)
class MemosConfig:
    namespace: str
    user_db_dir: str
    neo4j_uri: str = "bolt://127.0.0.1:17687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "trackc_memos_local_20260819"
    neo4j_database: str = "neo4j"
    neo4j_state_dir: str | None = None
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    build_scope_workers: int = 10
    require_answer_endpoint: bool = True
    #: ``turn`` = one native add() per turn (the LoCoMo-era default).
    #: ``session`` = one add() per source session, matching Mandol's own
    #: LongMemEval pipeline (``sessions_per_group = 1``). On LongMemEval this is
    #: 23,854 calls instead of 246,738. The extractor sees a whole session per
    #: call, so the stored memories differ; the choice is recorded in the receipt.
    ingest_granularity: str = "turn"


@dataclass
class _Scope:
    sample_id: str
    user_id: str
    cube_id: str
    mos: Any
    user_manager: Any
    llm_guarded: bool = False


def _safe(value: str) -> str:
    rendered = _SAFE.sub("_", value).strip("_")
    return rendered or "scope"


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _strings(value: Any) -> list[str]:
    """Collect context strings, including serialized MemOS source records."""
    value = _plain(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if key in {"embedding", "vector"}:
                continue
            result.extend(_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_strings(item))
        return result
    return []


class MemosAdapter:
    name = "memos"

    def __init__(self, config: MemosConfig) -> None:
        self.config = config
        self._scopes: dict[str, _Scope] = {}
        self._scopes_lock = threading.Lock()
        self._llm_guard_lock = threading.Lock()
        self._forbidden_llm_calls = 0
        self._stage_tracing = False
        self._stage_instrumentation: dict[str, bool] = {}
        Path(config.user_db_dir).mkdir(parents=True, exist_ok=True)

    def enable_stage_tracing(self) -> None:
        self._stage_tracing = True

    def _instrument_scope(self, scope: _Scope) -> None:
        """Attach observation-only spans to the concrete per-user clients."""
        if not self._stage_tracing:
            return
        cube = scope.mos.mem_cubes[scope.cube_id]
        text_mem = cube.text_mem
        prefix = _safe(scope.sample_id)
        targets = (
            (
                text_mem.embedder,
                "embed",
                "embedding",
                f"memos.{prefix}.embed",
                "http_client",
            ),
            (
                text_mem.graph_store,
                "search_by_embedding",
                "vector",
                f"memos.{prefix}.neo4j.search_by_embedding",
                "database_client",
            ),
            (
                text_mem.graph_store,
                "get_nodes",
                "graph",
                f"memos.{prefix}.neo4j.get_nodes",
                "database_client",
            ),
            (
                text_mem.graph_store,
                "add_nodes_batch",
                "persistence",
                f"memos.{prefix}.neo4j.add_nodes_batch",
                "database_client",
            ),
        )
        for target, method, category, operation, call_kind in targets:
            self._stage_instrumentation[operation] = instrument_method(
                target,
                method,
                category,
                operation,
                backend="memos_2.0.30_tree_text_neo4j",
                attributes={"observable_call_kind": call_kind},
            )

    @staticmethod
    def add_semantics() -> dict[str, Any]:
        return {
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "input_unit": "one frozen LoCoMo event rendered as one TextualMemoryItem",
            "construction_llm_policy": "forbidden_fail_closed",
            "included_stages": [
                "item_embedding",
                "neo4j_node_and_index_update",
                "commit",
            ],
            "excluded_stages": [
                "MemReader extraction",
                "memory reorganization",
                "preference extraction",
            ],
            "native_call": "GeneralMemCube.text_mem.add([TextualMemoryItem])",
        }

    def _forbid_llm(self, *_args: Any, **_kwargs: Any) -> Any:
        with self._llm_guard_lock:
            self._forbidden_llm_calls += 1
        raise RuntimeError("MemOS prepared-item Add attempted a forbidden LLM call")

    def _prepare_add_scope(self, scope: _Scope) -> None:
        """Instrument storage stages and make every reachable LLM fail closed."""
        with self._llm_guard_lock:
            if scope.llm_guarded:
                return
            cube = scope.mos.mem_cubes[scope.cube_id]
            text_mem = cube.text_mem
            llm_objects = [
                getattr(scope.mos.mem_reader, "llm", None),
                getattr(scope.mos.mem_reader, "general_llm", None),
                getattr(scope.mos.mem_reader, "preference_extractor_llm", None),
                getattr(text_mem, "extractor_llm", None),
                getattr(text_mem, "dispatcher_llm", None),
                getattr(getattr(text_mem, "memory_manager", None), "reorganizer", None),
            ]
            for target in llm_objects:
                if target is None:
                    continue
                for method in ("generate", "generate_response"):
                    if hasattr(target, method):
                        setattr(target, method, self._forbid_llm)
            scope.llm_guarded = True

            self._instrument_scope(scope)

    @staticmethod
    def _endpoint_models(base_url: str) -> list[str]:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as response:
            payload = json.load(response)
        return [str(item["id"]) for item in payload["data"]]

    def init_schema(self) -> dict[str, Any]:
        answer_models = (
            self._endpoint_models(self.config.answer_base_url)
            if self.config.require_answer_endpoint
            else []
        )
        embedding_models = self._endpoint_models(self.config.embedding_base_url)
        if self.config.require_answer_endpoint and answer_models != [
            self.config.answer_model
        ]:
            raise RuntimeError(f"answer endpoint mismatch: {answer_models!r}")
        if embedding_models != [self.config.embedding_model]:
            raise RuntimeError(f"embedding endpoint mismatch: {embedding_models!r}")

        request = urllib.request.Request(
            f"{self.config.embedding_base_url}/embeddings",
            data=json.dumps(
                {"model": self.config.embedding_model, "input": ["dimension probe"]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        dimension = len(payload["data"][0]["embedding"])
        if dimension != self.config.embedding_dim:
            raise RuntimeError(
                f"MemOS embedding dimension {dimension} != {self.config.embedding_dim}"
            )
        return {
            "ok": True,
            "answer_models": answer_models,
            "answer_endpoint_required": self.config.require_answer_endpoint,
            "embedding_models": embedding_models,
            "embedding_dim": dimension,
            "backend": "tree_text+neo4j",
        }

    def _scope_names(self, sample_id: str) -> tuple[str, str]:
        stem = _safe(f"{self.config.namespace}_{sample_id}")
        return f"trackc_{stem}", f"trackc_{stem}_cube"

    def _new_scope(self, sample_id: str) -> _Scope:
        from memos.mem_cube.general import GeneralMemCube
        from memos.mem_os.core import MOSCore
        from memos.mem_os.utils.default_config import (
            get_default_config,
            get_default_cube_config,
        )
        from memos.mem_user.user_manager import UserManager

        user_id, cube_id = self._scope_names(sample_id)
        common = {
            "openai_api_key": "EMPTY",
            "openai_api_base": self.config.answer_base_url,
            "text_mem_type": "tree_text",
            "user_id": user_id,
            "model_name": self.config.answer_model,
            "embedder_model": self.config.embedding_model,
            "embedding_dimension": self.config.embedding_dim,
            "neo4j_uri": self.config.neo4j_uri,
            "neo4j_user": self.config.neo4j_user,
            "neo4j_password": self.config.neo4j_password,
            "neo4j_db_name": self.config.neo4j_database,
            "neo4j_auto_create": False,
            "use_multi_db": False,
            "enable_reorganize": False,
            "temperature": 0.0,
            "max_tokens": 512,
            "top_k": 35,
            "cube_id": cube_id,
        }
        mos_config = get_default_config(**common)
        cube_config = get_default_cube_config(**common)

        # The convenience helper applies one base URL to both model classes.
        # Correct it before any factory is instantiated. embedding_dims stays
        # None to avoid an unsupported OpenAI `dimensions` request; the live
        # dimension gate above and Neo4j schema both enforce 1024 dimensions.
        mos_config.mem_reader.config.embedder.config.base_url = (
            self.config.embedding_base_url
        )
        mos_config.mem_reader.config.embedder.config.model_name_or_path = (
            self.config.embedding_model
        )
        mos_config.mem_reader.config.embedder.config.embedding_dims = None
        cube_config.text_mem.config.embedder.config.base_url = (
            self.config.embedding_base_url
        )
        cube_config.text_mem.config.embedder.config.model_name_or_path = (
            self.config.embedding_model
        )
        cube_config.text_mem.config.embedder.config.embedding_dims = None

        db_path = Path(self.config.user_db_dir) / f"{_safe(sample_id)}.sqlite3"
        user_manager = UserManager(db_path=str(db_path), user_id=user_id)
        # MOSConfig exposes a user_manager field, but MemOS 2.0.30's public
        # MOS.__init__ neither consumes it nor accepts an injected manager.
        # MOSCore is the official implementation behind MOS and exposes the
        # required injection point, keeping fresh-build metadata isolated.
        mos = MOSCore(mos_config, user_manager=user_manager)
        cube = GeneralMemCube(cube_config)
        mos.register_mem_cube(cube, mem_cube_id=cube_id, user_id=user_id)
        scope = _Scope(sample_id, user_id, cube_id, mos, user_manager)
        self._instrument_scope(scope)
        return scope

    def _scope(self, sample_id: str) -> _Scope:
        # Search builds create every scope before load.  Add-only runs discover
        # a new LoCoMo sample while requests are already arriving at 10 QPS;
        # serialize only first construction so two worker threads cannot
        # register the same MemOS user/cube concurrently.
        with self._scopes_lock:
            scope = self._scopes.get(sample_id)
            if scope is None:
                scope = self._new_scope(sample_id)
                self._scopes[sample_id] = scope
            return scope

    @staticmethod
    def _messages(item: EventItem) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"[event_id={item.event_id}] {item.text}",
            }
        ]

    def _native_add(self, scope: _Scope, item: EventItem) -> None:
        self._native_add_batch(scope, [item])

    def _native_add_batch(self, scope: _Scope, batch: Sequence[EventItem]) -> None:
        messages: list[Any] = []
        for item in batch:
            messages.extend(self._messages(item))
        scope.mos.add(
            messages=messages,
            mem_cube_id=scope.cube_id,
            user_id=scope.user_id,
            # One session per call under `session` granularity, so the id is
            # unambiguous; under `turn` the batch holds exactly one item.
            session_id=batch[0].session_id,
        )

    def _ingest_batches(
        self, scope_events: Sequence[EventItem]
    ) -> list[list[EventItem]]:
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

    def _prepared_item_add(self, scope: _Scope, item: EventItem) -> list[str]:
        from memos.memories.textual.item import (
            SourceMessage,
            TextualMemoryItem,
            TreeNodeTextualMemoryMetadata,
        )

        self._prepare_add_scope(scope)
        text_mem = scope.mos.mem_cubes[scope.cube_id].text_mem
        rendered = f"[event_id={item.event_id}] {item.text}"
        embedding = text_mem.embedder.embed([rendered])[0]
        prepared = TextualMemoryItem(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tridb-track-c:{item.event_id}")),
            memory=rendered,
            metadata=TreeNodeTextualMemoryMetadata(
                user_id=scope.user_id,
                session_id=item.session_id,
                memory_type="LongTermMemory",
                status="activated",
                is_fast=True,
                source="conversation",
                tags=["mode:fast", "track-c:prepared-item"],
                embedding=embedding,
                sources=[
                    SourceMessage(
                        type="chat",
                        role="user",
                        chat_time=item.timestamp,
                        message_id=item.event_id,
                        content=item.text,
                    )
                ],
                confidence=1.0,
                type="fact",
                info={
                    "source": "locomo",
                    "sample_id": item.sample_id,
                    "speaker": item.role,
                    "event_order": item.ordinal,
                },
            ),
        )
        return list(text_mem.add([prepared]))

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        started = time.perf_counter()
        by_scope: dict[str, list[EventItem]] = {}
        for item in events:
            by_scope.setdefault(item.sample_id, []).append(item)

        scopes = {sample_id: self._scope(sample_id) for sample_id in by_scope}

        def ingest_scope(sample_id: str) -> int:
            scope = scopes[sample_id]
            count = 0
            for batch in self._ingest_batches(by_scope[sample_id]):
                self._native_add_batch(scope, batch)
                count += len(batch)
            return count

        workers = min(self.config.build_scope_workers, len(by_scope))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            counts = list(executor.map(ingest_scope, by_scope))
        return {
            "wall_seconds": time.perf_counter() - started,
            "events": sum(counts),
            "scopes": len(by_scope),
            "scope_workers": workers,
        }

    def _memories(self, item: QueryItem, top_k: int) -> list[Any]:
        scope = self._scope(item.sample_id)
        with stage_span(
            "fusion",
            "memos.fast_search",
            backend="memos_2.0.30_tree_text_neo4j",
            attributes={
                "includes": ["embedding", "vector", "graph"],
                "attribution": "compound_native_api",
                "observable_call_kind": "opaque_native_api",
            },
        ):
            result = scope.mos.search(
                item.question,
                user_id=scope.user_id,
                install_cube_ids=[scope.cube_id],
                top_k=top_k,
                mode="fast",
            )
        memories: list[Any] = []
        for cube_result in result.get("text_mem", []):
            memories.extend(cube_result.get("memories", []))
        return memories

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        memories = self._memories(item, top_k)
        contexts: list[str] = []
        hit_ids: list[str] = []
        memory_ids: list[str] = []
        for memory in memories:
            plain = _plain(memory)
            memory_ids.append(str(plain.get("id", "")))
            strings = _strings(plain)
            contexts.append(str(plain.get("memory", "")))
            hit_ids.extend(_EVENT_ID.findall("\n".join(strings)))
        return {
            "result_count": len(memories),
            "hit_ids": list(dict.fromkeys(hit_ids)),
            "memory_ids": memory_ids,
            "contexts": contexts,
            "context_tokens": None,
            "empty": not memories,
            "mode": "fast",
        }

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        scope = self._scope(item.sample_id)
        with stage_span(
            "framework_other",
            "memos.add.prepared_item",
            backend="memos_2.0.30_tree_text_neo4j",
            attributes={
                "includes": ["embedding", "graph", "vector_index", "persistence"],
                "excludes": ["llm", "MemReader", "memory_reorganization"],
                "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
                "attribution": "compound_native_api",
                "observable_call_kind": "opaque_native_api",
            },
        ):
            created_ids = self._prepared_item_add(scope, item)
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
            "native_add_is_intrinsically_searchable": True,
            "created_memory_count": len(created_ids),
            "created_node_count": len(created_ids),
            "created_edge_count": 0,
            "creation_counts_available": True,
            "creation_count_source": "memos_text_mem_add_returned_ids",
            "created_ids": created_ids,
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "construction_llm_policy": "forbidden_fail_closed",
            "llm_call_count": 0,
            "updated_stores": ["neo4j_memory_node", "neo4j_vector_index"],
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

    def finalize_build(self) -> dict[str, Any]:
        from neo4j import GraphDatabase

        users = [scope.user_id.replace("_", "") for scope in self._scopes.values()]
        graph_users = [f"memos{user}" for user in users]
        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        try:
            with driver.session(database=self.config.neo4j_database) as session:
                nodes = session.run(
                    "MATCH (n:Memory) WHERE n.user_name IN $users RETURN count(n) AS n",
                    users=graph_users,
                ).single()["n"]
                edges = session.run(
                    "MATCH (a:Memory)-[r]->(b:Memory) "
                    "WHERE a.user_name IN $users AND b.user_name IN $users "
                    "RETURN count(r) AS n",
                    users=graph_users,
                ).single()["n"]
        finally:
            driver.close()
        sqlite_bytes = sum(
            path.stat().st_size
            for path in Path(self.config.user_db_dir).glob("*.sqlite3")
        )
        neo4j_store_bytes = (
            sum(
                path.stat().st_size
                for path in Path(self.config.neo4j_state_dir).rglob("*")
                if path.is_file()
            )
            if self.config.neo4j_state_dir
            else None
        )
        return {
            "memory_nodes": int(nodes),
            "memory_edges": int(edges),
            "sqlite_user_metadata_bytes": sqlite_bytes,
            "neo4j_store_bytes": neo4j_store_bytes,
            "scopes": len(self._scopes),
        }

    def stats(self) -> dict[str, Any]:
        config = asdict(self.config)
        config["neo4j_password"] = "<redacted>"
        return {
            **self.finalize_build(),
            "config": config,
            "native_shape": (
                "prepared TextualMemoryItem embedding + shared-DB tenant-isolated "
                "Neo4j graph/vector; no MemReader or LLM"
            ),
            "native_add_is_intrinsically_searchable": True,
            "add_semantics": self.add_semantics(),
            "forbidden_llm_calls": self._forbidden_llm_calls,
            "stage_instrumentation": self._stage_instrumentation,
        }

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        """Recreate process-local MOS handles for every persisted LoCoMo scope."""
        for sample_id in sorted({item.sample_id for item in corpus.all_events}):
            self._scope(sample_id)

    def snapshot_fingerprint(self) -> dict[str, Any]:
        result = self.finalize_build()
        return {
            "memory_nodes": result["memory_nodes"],
            "memory_edges": result["memory_edges"],
            "scopes": result["scopes"],
        }

    def close(self) -> None:
        for scope in self._scopes.values():
            cube = scope.mos.mem_cubes.get(scope.cube_id)
            graph_store = getattr(getattr(cube, "text_mem", None), "graph_store", None)
            driver = getattr(graph_store, "driver", None)
            if driver is not None:
                driver.close()
            engine = getattr(scope.user_manager, "engine", None)
            if engine is not None:
                engine.dispose()
        self._scopes.clear()
