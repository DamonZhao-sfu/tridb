"""Official Graphiti adapter for the controlled-model Track C comparison.

The system label is deliberately ``graphiti_zep_oss_proxy``.  Graphiti is the
official Zep open-source temporal graph framework, but it is not the managed
production Zep system measured by the source paper.

Graphiti is async while the shared Track C protocol exposes a synchronous
adapter contract.  A single dedicated event-loop thread owns the Neo4j driver
and OpenAI-compatible clients; concurrent harness workers submit coroutines to
that loop.  This avoids sharing one async Neo4j driver across multiple event
loops and preserves Graphiti's native concurrency behavior.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Sequence, TypeVar

from bench.agent_memory.table5_track_c.dataset import (
    EventItem,
    LoCoMoCorpus,
    QueryItem,
)
from bench.agent_memory.table5_track_c.tracing import stage_span

_T = TypeVar("_T")
_EVENT_NAMESPACE = uuid.UUID("d433d65c-c6ff-4fe8-ab2f-3b5b81beeb42")
_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"
_MANIFEST = "graphiti_track_c_snapshot_manifest.json"
_SNAPSHOT_SCHEMA = "graphiti_track_c_snapshot_v0.1.0"
_EXPECTED_COMMIT = "993e081a6d7948a0d8851c12a5fbdbeb49fed862"
_SAFE_GROUP = re.compile(r"^[A-Za-z0-9_-]+$")
_PREPARED_ITEM_BOUNDARY = "prepared_memory_item_insertion_v1"
_SEARCH_BOUNDARY = "retrieval_only_no_generation_llm_v1"
_PREPARED_NAMESPACE = uuid.UUID("b88d5f96-a6a1-4fb9-8c03-dc34ed033b69")


def snapshot_adapter_compatibility(
    snapshot_sha256: str | None, runtime_sha256: str
) -> dict[str, Any]:
    """Gate an explicit, launcher-pinned adapter upgrade over a frozen build.

    Escape hatches are NOT interchangeable -- the mode written
    into the receipt is the claim a reader will rely on:

    ``explicit_instrumentation_only``
        The edit cannot change any measured value (tracing, logging).

    ``explicit_search_render_only``
        The edit changes how retrieved material is rendered for the answer
        model, so it DOES move accuracy, but it touches no build-side code
        and the stored graph stays byte-identical. Rebuilding to satisfy a
        file hash would re-run non-deterministic LLM extraction and yield a
        *different* graph, which is strictly worse evidence than reusing the
        frozen one. Anything that would change what gets written into the
        graph must rebuild instead -- this hatch does not cover it.

    ``explicit_llm_free_operation_boundary``
        The frozen Search graph remains byte-identical, while measured Search
        installs a fail-closed LLM guard and measured Add uses a fresh database
        plus Graphiti's public low-level namespace APIs. This mode may never be
        used to claim that the high-level ``add_episode`` API is LLM-free.
    """
    if snapshot_sha256 == runtime_sha256:
        return {
            "compatible": True,
            "mode": "exact",
            "snapshot_sha256": snapshot_sha256,
            "runtime_sha256": runtime_sha256,
        }
    expected_snapshot = os.environ.get("GRAPHITI_COMPAT_SNAPSHOT_SHA256")
    expected_runtime = os.environ.get("GRAPHITI_COMPAT_RUNTIME_SHA256")
    declared = os.environ.get("GRAPHITI_COMPAT_MODE", "instrumentation_only")
    pinned = (
        bool(expected_snapshot)
        and bool(expected_runtime)
        and snapshot_sha256 == expected_snapshot
        and runtime_sha256 == expected_runtime
    )
    allowed = {
        "instrumentation_only",
        "search_render_only",
        "llm_free_operation_boundary",
    }
    compatible = pinned and declared in allowed
    return {
        "compatible": compatible,
        "mode": f"explicit_{declared}" if compatible else "rejected",
        "declared_scope": declared,
        "affects_measured_accuracy": declared == "search_render_only",
        "snapshot_sha256": snapshot_sha256,
        "runtime_sha256": runtime_sha256,
        "expected_snapshot_sha256": expected_snapshot,
        "expected_runtime_sha256": expected_runtime,
    }


@dataclass(frozen=True)
class GraphitiTrackCConfig:
    dataset_path: str
    neo4j_state_dir: str
    neo4j_uri: str = "bolt://127.0.0.1:27687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "trackc_graphiti_local_20260821"
    neo4j_database: str = "neo4j"
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    graphiti_source_root: str = "/localhome/hza214/agent-memory-table5/src/graphiti"
    build_scope_workers: int = 2
    max_coroutines: int = 16
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 4_096
    resume_partial_build: bool = False
    require_answer_endpoint: bool = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def event_uuid(item: EventItem) -> str:
    """Return a stable UUID while preserving repeated LoCoMo dialogue ids."""
    return str(uuid.uuid5(_EVENT_NAMESPACE, f"{item.sample_id}\0{item.event_id}"))


def reference_time(item: EventItem) -> datetime:
    """Parse LoCoMo's frozen timestamp and order turns within one session."""
    base = datetime.strptime(item.timestamp, _TIMESTAMP_FORMAT).replace(
        tzinfo=timezone.utc
    )
    turn_number = int(item.metadata.get("turn_number") or item.ordinal % 1_000_000)
    return base + timedelta(microseconds=turn_number)


def cypher_stage(query: str) -> tuple[str, str, dict[str, str]]:
    """Classify a native Neo4j request for client-observed time breakdown."""
    normalized = " ".join(query.upper().split())
    if "DB.INDEX.VECTOR" in normalized or "VECTOR.SIMILARITY" in normalized:
        return "vector", "graphiti.neo4j.vector_search", {"index_kind": "vector"}
    if "DB.INDEX.FULLTEXT" in normalized:
        return "vector", "graphiti.neo4j.fulltext_search", {"index_kind": "fulltext"}
    write_tokens = (" MERGE ", " CREATE ", " SET ", " DELETE ", " REMOVE ", " DROP ")
    padded = f" {normalized} "
    if any(token in padded for token in write_tokens):
        return "persistence", "graphiti.neo4j.write", {"query_kind": "write"}
    return "graph", "graphiti.neo4j.read", {"query_kind": "read"}


class _AsyncRuntime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._serve,
            name="graphiti-track-c-event-loop",
            daemon=True,
        )
        self.thread.start()
        if not self.ready.wait(timeout=10):
            raise RuntimeError("Graphiti event loop did not start")

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Graphiti event loop did not stop")
        self.loop.close()


class GraphitiTrackCAdapter:
    """Graphiti 0.29.3 OSS proxy using native add and hybrid edge search."""

    name = "graphiti_zep_oss_proxy"

    def __init__(self, config: GraphitiTrackCConfig) -> None:
        self.config = config
        # Graphiti reads telemetry configuration during lazy imports below.
        # Formal benchmark runs must not emit PostHog traffic or add WAN work.
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        self.dataset_path = Path(config.dataset_path).resolve()
        self.state_dir = Path(config.neo4j_state_dir).resolve()
        self.manifest_path = self.state_dir / _MANIFEST
        self._runtime = _AsyncRuntime()
        self._graphiti = self._runtime.run(self._create_client())
        self._episode_to_event: dict[str, str] = {}
        self._episode_map_path = self.state_dir / "graphiti_episode_event_map.json"
        self._build_stats: dict[str, Any] = {}
        self._stage_tracing = False
        self._instrumented = False
        self._snapshot_adapter_compatibility: dict[str, Any] | None = None
        self._llm_guard_lock = threading.Lock()
        self._llm_guard_installed = False
        self._forbidden_llm_calls = 0
        self._forbidden_cross_encoder_calls = 0

    async def _create_client(self) -> Any:
        from openai import AsyncOpenAI

        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.embedder.openai import (
            OpenAIEmbedder,
            OpenAIEmbedderConfig,
        )
        from graphiti_core.graphiti import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        llm_config = LLMConfig(
            api_key="EMPTY",
            model=self.config.answer_model,
            small_model=self.config.answer_model,
            base_url=self.config.answer_base_url,
            temperature=0.0,
            max_tokens=self.config.llm_max_tokens,
        )
        llm_http = AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.config.answer_base_url,
            timeout=self.config.llm_timeout_seconds,
        )
        llm_client = OpenAIGenericClient(
            config=llm_config,
            client=llm_http,
            max_tokens=self.config.llm_max_tokens,
            structured_output_mode="json_schema",
        )
        # Graphiti's public ``generate_response`` signature defaults to 16K
        # output tokens. Call sites that omit the argument therefore bypass
        # the instance-level LLMConfig value. Enforce the frozen construction
        # cap at the actual call boundary so input + output always fits the
        # 32K Qwen3.8 service context. This changes only the maximum permitted
        # construction response length; prompts, retries, and parsed outputs
        # remain Graphiti-native.
        native_generate_response = llm_client.generate_response

        async def generate_response_with_frozen_cap(
            *args: Any, **kwargs: Any
        ) -> Any:
            requested = kwargs.get("max_tokens")
            if requested is None or int(requested) > self.config.llm_max_tokens:
                kwargs["max_tokens"] = self.config.llm_max_tokens
            return await native_generate_response(*args, **kwargs)

        llm_client.generate_response = generate_response_with_frozen_cap
        embedder_config = OpenAIEmbedderConfig(
            api_key="EMPTY",
            base_url=self.config.embedding_base_url,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
        )
        embedding_http = AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.config.embedding_base_url,
            timeout=60.0,
        )
        embedder = OpenAIEmbedder(config=embedder_config, client=embedding_http)
        # Graphiti's basic ``search`` recipe uses RRF, not this cross encoder.
        # Supplying the same local Qwen endpoint prevents an accidental external
        # provider if a future conformance probe exercises a reranked recipe.
        cross_encoder = OpenAIRerankerClient(config=llm_config, client=llm_http)
        return Graphiti(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=True,
            max_coroutines=self.config.max_coroutines,
        )

    def _install_no_llm_guard(self) -> None:
        """Fail closed if a measured Search/Add reaches a model reranker or LLM."""
        with self._llm_guard_lock:
            if self._llm_guard_installed:
                return

            async def forbidden_llm_call(*_args: Any, **_kwargs: Any) -> Any:
                with self._llm_guard_lock:
                    self._forbidden_llm_calls += 1
                raise RuntimeError(
                    "Graphiti LLM-free operation attempted a forbidden generation call"
                )

            async def forbidden_cross_encoder_call(
                *_args: Any, **_kwargs: Any
            ) -> Any:
                with self._llm_guard_lock:
                    self._forbidden_cross_encoder_calls += 1
                raise RuntimeError(
                    "Graphiti LLM-free operation attempted a forbidden model reranker call"
                )

            self._graphiti.llm_client.generate_response = forbidden_llm_call
            self._graphiti.cross_encoder.rank = forbidden_cross_encoder_call
            self._llm_guard_installed = True

    @staticmethod
    def search_semantics() -> dict[str, Any]:
        return {
            "semantic_boundary": _SEARCH_BOUNDARY,
            "generation_llm_policy": "forbidden_fail_closed",
            "included_stages": [
                "query_embedding",
                "edge_vector_search",
                "edge_fulltext_search",
                "rrf_fusion",
                "graph_read",
            ],
            "excluded_stages": ["answer_generation", "cross_encoder_reranking"],
            "native_call": "Graphiti.search(... EDGE_HYBRID_SEARCH_RRF)",
        }

    @staticmethod
    def add_semantics() -> dict[str, Any]:
        return {
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "input_unit": "one frozen LoCoMo event rendered as one prepared fact edge",
            "construction_llm_policy": "forbidden_fail_closed",
            "included_stages": [
                "entity_name_embedding",
                "fact_embedding",
                "neo4j_node_edge_and_vector_index_update",
                "commit",
            ],
            "excluded_stages": [
                "add_episode entity extraction",
                "LLM node deduplication",
                "LLM edge resolution",
                "community summarization",
            ],
            "native_call": (
                "Graphiti public low-level namespaces: nodes.entity/episode.save "
                "+ edges.entity/episodic.save"
            ),
            "not_equivalent_to": "Graphiti.add_episode high-level construction API",
        }

    def _adapter_sha256(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _source_identity(self) -> dict[str, Any]:
        root = Path(self.config.graphiti_source_root).resolve()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {
            "root": str(root),
            "commit": commit,
            "expected_commit": _EXPECTED_COMMIT,
            "tracked_dirty": tracked_dirty,
            "version": "0.29.3",
            "repository": "https://github.com/getzep/graphiti.git",
        }

    @staticmethod
    def _endpoint_models(base_url: str) -> list[str]:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as response:
            payload = json.load(response)
        return [str(item["id"]) for item in payload["data"]]

    @staticmethod
    def _embedding_dimension(base_url: str, model: str) -> int:
        request = urllib.request.Request(
            f"{base_url}/embeddings",
            data=json.dumps({"model": model, "input": ["dimension probe"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        return len(payload["data"][0]["embedding"])

    def enable_stage_tracing(self) -> None:
        self._stage_tracing = True
        if self._instrumented:
            return

        llm = self._graphiti.llm_client
        embedder = self._graphiti.embedder
        driver = self._graphiti.driver

        original_llm = llm.generate_response
        original_create = embedder.create
        original_create_batch = embedder.create_batch
        original_query = driver.execute_query

        async def traced_llm(*args: Any, **kwargs: Any) -> Any:
            with stage_span(
                "llm",
                "graphiti.llm.generate",
                backend="Qwen3-32B-FP8",
                attributes={"observable_call_kind": "http_client"},
            ):
                return await original_llm(*args, **kwargs)

        async def traced_create(*args: Any, **kwargs: Any) -> Any:
            with stage_span(
                "embedding",
                "graphiti.embedding.create",
                backend="Qwen3-Embedding-0.6B",
                attributes={"observable_call_kind": "http_client"},
            ):
                return await original_create(*args, **kwargs)

        async def traced_create_batch(*args: Any, **kwargs: Any) -> Any:
            with stage_span(
                "embedding",
                "graphiti.embedding.create_batch",
                backend="Qwen3-Embedding-0.6B",
                attributes={"observable_call_kind": "http_client"},
            ):
                return await original_create_batch(*args, **kwargs)

        # ``query`` is also a legitimate Cypher parameter name in Graphiti's
        # full-text search calls (``execute_query(cypher, query=...)``).  Keep
        # the wrapper's first argument distinct so that profiling does not
        # collide with that parameter.
        async def traced_query(cypher_query: str, *args: Any, **kwargs: Any) -> Any:
            category, operation, attributes = cypher_stage(cypher_query)
            attributes = {
                **attributes,
                "observable_call_kind": "database_client",
            }
            with stage_span(
                category,
                operation,
                backend="neo4j-community-5.26.6",
                attributes=attributes,
            ):
                return await original_query(cypher_query, *args, **kwargs)

        llm.generate_response = traced_llm
        embedder.create = traced_create
        embedder.create_batch = traced_create_batch
        driver.execute_query = traced_query
        self._instrumented = True

    async def _initialize_schema(self) -> dict[str, Any]:
        await self._graphiti.driver.health_check()
        await self._graphiti.build_indices_and_constraints()
        counts = await self._counts_async()
        return counts

    def init_schema(self) -> dict[str, Any]:
        source = self._source_identity()
        if source["commit"] != _EXPECTED_COMMIT or source["tracked_dirty"]:
            raise RuntimeError(f"Graphiti source identity rejected: {source!r}")
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
        dimension = self._embedding_dimension(
            self.config.embedding_base_url, self.config.embedding_model
        )
        if dimension != self.config.embedding_dim:
            raise RuntimeError(
                f"Graphiti embedding dimension {dimension} != {self.config.embedding_dim}"
            )
        if not self.state_dir.is_dir():
            raise RuntimeError(f"Neo4j state directory is absent: {self.state_dir}")
        mode = "load_persisted" if self.manifest_path.is_file() else "fresh_build"
        if self.config.resume_partial_build and not self.manifest_path.is_file():
            mode = "resume_partial_build"
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != _SNAPSHOT_SCHEMA:
                raise RuntimeError("Graphiti snapshot schema mismatch")
            if manifest.get("dataset_sha256") != _sha256_file(self.dataset_path):
                raise RuntimeError("Graphiti snapshot dataset checksum mismatch")
            self._snapshot_adapter_compatibility = snapshot_adapter_compatibility(
                manifest.get("adapter_sha256"), self._adapter_sha256()
            )
            if not self._snapshot_adapter_compatibility["compatible"]:
                raise RuntimeError("Graphiti adapter changed after canonical build")
            if manifest.get("graphiti_source", {}).get("commit") != _EXPECTED_COMMIT:
                raise RuntimeError("Graphiti source changed after canonical build")
        counts = self._runtime.run(self._initialize_schema())
        return {
            "ok": True,
            "mode": mode,
            "backend": "neo4j-community-5.26.6",
            "adapter_sha256": self._adapter_sha256(),
            "answer_models": answer_models,
            "answer_endpoint_required": self.config.require_answer_endpoint,
            "embedding_models": embedding_models,
            "embedding_dim": dimension,
            "graphiti_source": source,
            "live_counts": counts,
            "snapshot_adapter_compatibility": self._snapshot_adapter_compatibility,
        }

    @staticmethod
    def _raw_episode(item: EventItem) -> Any:
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.utils.bulk_utils import RawEpisode

        # No uuid on purpose.  In graphiti-core 0.29.3 `add_episode_bulk` treats
        # a supplied uuid as a reference to an ALREADY-STORED episode and calls
        # `EpisodicNode.get_by_uuid`, which raises NodeNotFoundError for a new
        # one (graphiti.py:1320).  Traceability does not depend on the uuid:
        # `name` carries event_id and the body is prefixed with
        # `[event_id=...]`, which is what search-side attribution reads.
        return RawEpisode(
            name=item.event_id,
            content=f"[event_id={item.event_id}] {item.text}",
            source_description=(
                f"LoCoMo conversation={item.sample_id} session={item.session_id}"
            ),
            source=EpisodeType.message,
            reference_time=reference_time(item),
        )

    async def _ingest_scope(self, events: Sequence[EventItem]) -> dict[str, Any]:
        sample_id = events[0].sample_id
        if not _SAFE_GROUP.fullmatch(sample_id):
            raise ValueError(f"unsafe Graphiti group id: {sample_id!r}")
        sessions: dict[str, list[EventItem]] = defaultdict(list)
        for item in events:
            if item.sample_id != sample_id:
                raise ValueError("mixed LoCoMo samples reached one Graphiti scope")
            sessions[item.session_id].append(item)
        episodes = 0
        nodes = 0
        edges = 0
        for session_id in sorted(
            sessions, key=lambda value: int(value.removeprefix("S"))
        ):
            items = sorted(sessions[session_id], key=lambda item: item.ordinal)
            result = await self._graphiti.add_episode_bulk(
                [self._raw_episode(item) for item in items],
                group_id=sample_id,
            )
            # `add_episode_bulk` assigns its own uuids (we cannot supply ours --
            # see `_raw_episode`).  Recover the mapping search-side attribution
            # needs by matching on `name`, which we set to event_id.
            by_name = {item.event_id: item for item in items}
            for node in result.episodes:
                event = by_name.get(str(getattr(node, "name", "")))
                if event is not None:
                    self._episode_to_event[str(node.uuid)] = event.event_id
            episodes += len(result.episodes)
            nodes += len(result.nodes)
            edges += len(result.edges)
        return {
            "sample_id": sample_id,
            "sessions": len(sessions),
            "episodes": episodes,
            "resolved_nodes": nodes,
            "resolved_edges": edges,
        }

    async def _ingest_all(
        self, grouped: dict[str, list[EventItem]]
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.config.build_scope_workers)

        async def bounded(events: list[EventItem]) -> dict[str, Any]:
            async with semaphore:
                return await self._ingest_scope(events)

        return await asyncio.gather(
            *(bounded(grouped[sample_id]) for sample_id in sorted(grouped))
        )

    async def _resume_episode_inventory(self) -> list[dict[str, str]]:
        records, _, _ = await self._graphiti.driver.execute_query(
            """
            MATCH (n:Episodic)
            WHERE n.group_id IS NOT NULL
            RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id
            ORDER BY group_id, name
            """,
            routing_="r",
        )
        return [
            {
                "uuid": str(record["uuid"]),
                "name": str(record["name"]),
                "group_id": str(record["group_id"]),
            }
            for record in records
        ]

    async def _delete_partial_groups(self, group_ids: list[str]) -> None:
        if not group_ids:
            return
        await self._graphiti.driver.execute_query(
            """
            MATCH ()-[r]->()
            WHERE r.group_id IN $group_ids
            DELETE r
            """,
            group_ids=group_ids,
        )
        await self._graphiti.driver.execute_query(
            """
            MATCH (n)
            WHERE n.group_id IN $group_ids
            DETACH DELETE n
            """,
            group_ids=group_ids,
        )

    def _prepare_partial_resume(
        self, grouped: dict[str, list[EventItem]]
    ) -> tuple[set[str], list[str]]:
        inventory = self._runtime.run(self._resume_episode_inventory())
        expected = {
            group_id: {item.event_id for item in items}
            for group_id, items in grouped.items()
        }
        observed: dict[str, set[str]] = defaultdict(set)
        for episode in inventory:
            group_id = episode["group_id"]
            if group_id not in expected:
                raise RuntimeError(
                    f"partial Graphiti state has an unexpected group: {group_id}"
                )
            observed[group_id].add(episode["name"])
        complete = {
            group_id
            for group_id, names in observed.items()
            if names == expected[group_id]
        }
        partial = sorted(set(observed) - complete)
        # Resume always runs on a writable clone.  Drop only incomplete groups
        # from that clone so add_episode_bulk can safely replay them; the
        # original paused state remains immutable evidence.
        self._runtime.run(self._delete_partial_groups(partial))
        for episode in inventory:
            if episode["group_id"] in complete:
                self._episode_to_event[episode["uuid"]] = episode["name"]
        return complete, partial

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        if self.manifest_path.exists():
            raise RuntimeError("refusing to rebuild an existing Graphiti snapshot")
        grouped: dict[str, list[EventItem]] = defaultdict(list)
        for item in events:
            grouped[item.sample_id].append(item)
            self._episode_to_event[event_uuid(item)] = item.event_id
        resumed_groups: set[str] = set()
        repaired_groups: list[str] = []
        if self.config.resume_partial_build:
            resumed_groups, repaired_groups = self._prepare_partial_resume(grouped)
        pending = {
            group_id: items
            for group_id, items in grouped.items()
            if group_id not in resumed_groups
        }
        started = time.perf_counter()
        reports = self._runtime.run(self._ingest_all(pending))
        # Persist the episode-uuid -> event_id map: `search` runs in a separate
        # process via --reuse-build-receipt, and the uuids are Graphiti's, so
        # they cannot be recomputed from the corpus.
        self._episode_map_path.write_text(
            json.dumps(self._episode_to_event, sort_keys=True), encoding="utf-8"
        )
        self._build_stats = {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "samples": len(grouped),
            "scope_workers": self.config.build_scope_workers,
            "sample_reports": reports,
            "episode_map_entries": len(self._episode_to_event),
            "resume": {
                "enabled": self.config.resume_partial_build,
                "reused_complete_groups": sorted(resumed_groups),
                "repaired_partial_groups": repaired_groups,
                "rebuilt_groups": sorted(pending),
            },
        }
        return self._build_stats

    async def _counts_async(self) -> dict[str, int]:
        records, _, _ = await self._graphiti.driver.execute_query(
            """
            MATCH (n)
            WITH count(n) AS nodes, count(DISTINCT n.group_id) AS groups
            OPTIONAL MATCH ()-[r]->()
            RETURN nodes, groups, count(r) AS relationships
            """,
            routing_="r",
        )
        if not records:
            return {"nodes": 0, "relationships": 0, "groups": 0}
        row = records[0]
        return {
            "nodes": int(row["nodes"]),
            "relationships": int(row["relationships"]),
            "groups": int(row["groups"]),
        }

    def _counts(self) -> dict[str, int]:
        return self._runtime.run(self._counts_async())

    async def _group_counts_async(self) -> dict[str, dict[str, int]]:
        node_records, _, _ = await self._graphiti.driver.execute_query(
            """
            MATCH (n)
            WHERE n.group_id IS NOT NULL
            RETURN n.group_id AS group_id, count(n) AS count
            ORDER BY group_id
            """,
            routing_="r",
        )
        edge_records, _, _ = await self._graphiti.driver.execute_query(
            """
            MATCH ()-[r]->()
            WHERE r.group_id IS NOT NULL
            RETURN r.group_id AS group_id, count(r) AS count
            ORDER BY group_id
            """,
            routing_="r",
        )
        result: dict[str, dict[str, int]] = {}
        for record in node_records:
            result[str(record["group_id"])] = {
                "nodes": int(record["count"]),
                "relationships": 0,
            }
        for record in edge_records:
            result.setdefault(
                str(record["group_id"]), {"nodes": 0, "relationships": 0}
            )["relationships"] = int(record["count"])
        return result

    def group_counts(self) -> dict[str, dict[str, int]]:
        """Expose per-group counts for isolation and persistence conformance."""
        return self._runtime.run(self._group_counts_async())

    def finalize_build(self) -> dict[str, Any]:
        counts = self._counts()
        manifest = {
            "schema": _SNAPSHOT_SCHEMA,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": _sha256_file(self.dataset_path),
            "adapter_sha256": self._adapter_sha256(),
            "graphiti_source": self._source_identity(),
            "models": {
                "answer": self.config.answer_model,
                "answer_quantization": "FP8 endpoint receipt required",
                "embedding": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "reranker": "Graphiti EDGE_HYBRID_SEARCH_RRF (no cross encoder)",
            },
            "backend": {
                "kind": "neo4j-community",
                "version": "5.26.6",
                "database": self.config.neo4j_database,
            },
            "build_operating_point": {
                "api": "Graphiti.add_episode_bulk",
                "batch_boundary": "one LoCoMo session",
                "scope_workers": self.config.build_scope_workers,
                "max_coroutines": self.config.max_coroutines,
                "resumed_partial_build": self.config.resume_partial_build,
            },
            "counts": counts,
        }
        _atomic_json(self.manifest_path, manifest)
        return counts

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        if self._episode_map_path.is_file():
            self._episode_to_event = json.loads(
                self._episode_map_path.read_text(encoding="utf-8")
            )
            return
        # Fallback for a build made by the single-episode `add_episode` path,
        # which does supply our deterministic uuid.
        self._episode_to_event = {
            event_uuid(item): item.event_id for item in corpus.all_events
        }

    def _combined_config(self, top_k: int) -> Any:
        """Zep's retrieval surface: semantic edges + entity nodes + communities.

        ``Graphiti.search()`` is a convenience wrapper pinned to
        ``EDGE_HYBRID_SEARCH_RRF`` that returns ``.edges`` and nothing else.
        The Zep paper (arXiv:2501.13956 §3) defines retrieval as
        ``phi: query -> (E_sn, N_sn, N_cn)`` and its constructor emits a FACTS
        block *and* an ENTITIES block. Returning only edges therefore drops
        half the specified context -- measured on the LoCoMo graph, 610 of 718
        entities carry a non-empty summary that never reached the answer model.

        RRF is kept rather than the library default cross-encoder: no reranker
        is served here, and holding the reranker fixed keeps this change a test
        of "add the entity block", not of two variables at once. Episodes are
        searched by the recipe but deliberately NOT rendered -- the paper's
        constructor takes edges, entity nodes and community nodes only.
        """
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_RRF,
        )

        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = top_k
        return config

    async def _search_async(self, item: QueryItem, top_k: int) -> Any:
        return await self._graphiti.search_(
            item.question,
            config=self._combined_config(top_k),
            group_ids=[item.sample_id],
        )

    @staticmethod
    def _edge_context(edge: Any) -> str:
        """Render one edge as `fact` plus the validity interval it carries.

        Graphiti stores `valid_at` / `invalid_at` on every RELATES_TO edge --
        that temporal envelope is the whole point of a "temporal context
        graph". An earlier version rendered `str(edge.fact)` alone, which
        threw the timestamps away before the answer model ever saw them.
        Every date-shaped LoCoMo question then came back "Not mentioned in
        memories": Temp. scored 36.27% and the mean context was 697 tokens
        against the 1.4k the paper reports for Zep.
        """
        parts = [str(edge.fact)]
        valid_at = getattr(edge, "valid_at", None)
        invalid_at = getattr(edge, "invalid_at", None)
        if valid_at is not None:
            # Date-only: LoCoMo gold answers are calendar dates, and the
            # sub-second component is intra-session ordering, not evidence.
            window = f"from {valid_at:%d %B, %Y}"
            if invalid_at is not None:
                window += f" until {invalid_at:%d %B, %Y}"
            parts.append(f"({window})")
        elif invalid_at is not None:
            parts.append(f"(until {invalid_at:%d %B, %Y})")
        return " ".join(parts)

    @staticmethod
    def _node_context(node: Any) -> str:
        """``ENTITY_NAME: entity summary`` -- the paper's ENTITIES line."""
        name = str(getattr(node, "name", "") or "").strip()
        summary = str(getattr(node, "summary", "") or "").strip()
        return f"{name}: {summary}" if summary else name

    def _render_results(self, results: Any) -> dict[str, Any]:
        """Build the context per Zep's constructor chi.

        chi emits, in order: each edge's fact plus its validity interval, each
        entity node's name and summary, and each community node's summary.
        Episodes are searched by the recipe but are not part of chi, so they
        are counted for provenance and left out of the context.
        """
        edges = list(getattr(results, "edges", []) or [])
        nodes = list(getattr(results, "nodes", []) or [])
        communities = list(getattr(results, "communities", []) or [])
        episodes = list(getattr(results, "episodes", []) or [])

        hit_ids: list[str] = []
        for edge in edges:
            for episode_id in list(getattr(edge, "episodes", []) or []):
                event_id = self._episode_to_event.get(str(episode_id))
                if event_id is not None and event_id not in hit_ids:
                    hit_ids.append(event_id)

        facts = [self._edge_context(edge) for edge in edges]
        entities = [self._node_context(node) for node in nodes]
        entities = [line for line in entities if line]
        summaries = [
            str(getattr(c, "summary", "") or "").strip()
            for c in communities
        ]
        summaries = [line for line in summaries if line]
        contexts = facts + entities + summaries

        return {
            "result_count": len(contexts),
            "hit_ids": hit_ids,
            "memory_ids": [str(edge.uuid) for edge in edges]
            + [str(getattr(n, "uuid", "")) for n in nodes],
            "contexts": contexts,
            "context_tokens": None,
            "empty": not contexts,
            "mode": "Graphiti.search_ COMBINED_HYBRID_SEARCH_RRF",
            "reranker": "rrf",
            # Per-layer counts, so a reader can see which part of Zep's
            # (edges, entity nodes, community nodes) tuple actually contributed.
            "layer_counts": {
                "edges": len(edges),
                "entity_nodes": len(nodes),
                "communities": len(communities),
                "episodes_searched_not_rendered": len(episodes),
            },
        }

    def _render_edges(self, edges: Sequence[Any]) -> dict[str, Any]:
        """Kept for the Add path, which still yields bare edges."""
        hit_ids: list[str] = []
        for edge in edges:
            for episode_id in list(getattr(edge, "episodes", []) or []):
                event_id = self._episode_to_event.get(str(episode_id))
                if event_id is not None and event_id not in hit_ids:
                    hit_ids.append(event_id)
        return {
            "result_count": len(edges),
            "hit_ids": hit_ids,
            "memory_ids": [str(edge.uuid) for edge in edges],
            "contexts": [self._edge_context(edge) for edge in edges],
            "context_tokens": None,
            "empty": not edges,
            "mode": "Graphiti.search EDGE_HYBRID_SEARCH_RRF",
            "reranker": "rrf",
        }

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        self._install_no_llm_guard()
        with stage_span(
            "fusion",
            "graphiti.search.combined_hybrid_rrf",
            backend="graphiti-core-0.29.3+neo4j-5.26.6",
            attributes={
                "includes": ["embedding", "vector", "fulltext", "graph", "rrf"],
                "layers": ["edges", "entity_nodes", "communities"],
                "group_id": item.sample_id,
                "observable_call_kind": "opaque_native_api",
            },
        ):
            results = self._runtime.run(self._search_async(item, top_k))
        result = self._render_results(results)
        result.update(
            {
                "semantic_boundary": _SEARCH_BOUNDARY,
                "generation_llm_policy": "forbidden_fail_closed",
                "llm_call_count": 0,
                "cross_encoder_call_count": 0,
            }
        )
        return result

    async def _add_async(self, item: EventItem) -> Any:
        """Insert one already-prepared searchable Graphiti fact without an LLM."""
        from graphiti_core.edges import EntityEdge, EpisodicEdge
        from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType

        created_at = datetime.now(timezone.utc)
        episode_id = event_uuid(item)
        source_id = str(
            uuid.uuid5(
                _PREPARED_NAMESPACE,
                f"{item.sample_id}\0speaker\0{item.role}",
            )
        )
        target_id = str(
            uuid.uuid5(
                _PREPARED_NAMESPACE,
                f"{item.sample_id}\0memory\0{item.event_id}",
            )
        )
        edge_id = str(
            uuid.uuid5(
                _PREPARED_NAMESPACE,
                f"{item.sample_id}\0fact\0{item.event_id}",
            )
        )
        rendered = f"[event_id={item.event_id}] {item.text}"
        source = EntityNode(
            uuid=source_id,
            name=item.role,
            group_id=item.sample_id,
            labels=["TrackCSpeaker"],
            summary=f"LoCoMo speaker {item.role}",
            created_at=created_at,
            attributes={"track_c_kind": "speaker"},
        )
        target = EntityNode(
            uuid=target_id,
            name=rendered,
            group_id=item.sample_id,
            labels=["TrackCPreparedMemory"],
            summary=rendered,
            created_at=created_at,
            attributes={
                "track_c_kind": "prepared_memory",
                "event_id": item.event_id,
                "session_id": item.session_id,
            },
        )
        fact = EntityEdge(
            uuid=edge_id,
            group_id=item.sample_id,
            source_node_uuid=source_id,
            target_node_uuid=target_id,
            created_at=created_at,
            name="HAS_MEMORY",
            fact=rendered,
            episodes=[episode_id],
            valid_at=reference_time(item),
            reference_time=reference_time(item),
            attributes={
                "track_c_prepared_item": True,
                "event_id": item.event_id,
                "session_id": item.session_id,
                "speaker": item.role,
            },
        )
        episode = EpisodicNode(
            uuid=episode_id,
            name=item.event_id,
            group_id=item.sample_id,
            source=EpisodeType.message,
            source_description=(
                f"LoCoMo Track C prepared Add conversation={item.sample_id} "
                f"session={item.session_id}"
            ),
            content=rendered,
            valid_at=reference_time(item),
            entity_edges=[edge_id],
            created_at=created_at,
        )
        mentions = [
            EpisodicEdge(
                uuid=str(
                    uuid.uuid5(
                        _PREPARED_NAMESPACE,
                        f"{item.sample_id}\0mention\0{item.event_id}\0{node_id}",
                    )
                ),
                group_id=item.sample_id,
                source_node_uuid=episode_id,
                target_node_uuid=node_id,
                created_at=created_at,
            )
            for node_id in (source_id, target_id)
        ]

        # These are Graphiti's public low-level namespace APIs. Entity saves
        # generate embeddings before writing Neo4j vector properties; no LLM
        # extraction, deduplication, or edge-resolution API is reachable.
        await asyncio.gather(
            self._graphiti.nodes.entity.save(source),
            self._graphiti.nodes.entity.save(target),
            self._graphiti.nodes.episode.save(episode),
        )
        await self._graphiti.edges.entity.save(fact)
        await asyncio.gather(
            *(self._graphiti.edges.episodic.save(edge) for edge in mentions)
        )
        return {
            "episode_id": episode_id,
            "entity_node_ids": [source_id, target_id],
            "entity_edge_id": edge_id,
            "episodic_edge_ids": [edge.uuid for edge in mentions],
        }

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._install_no_llm_guard()
        with stage_span(
            "framework_other",
            "graphiti.add.prepared_item",
            backend="graphiti-core-0.29.3+neo4j-5.26.6",
            attributes={
                "includes": ["embedding", "graph", "vector_index", "persistence"],
                "excludes": ["llm", "extraction", "deduplication_reasoning"],
                "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
                "observable_call_kind": "opaque_native_api",
            },
        ):
            result = self._runtime.run(self._add_async(item))
        episode_id = str(result["episode_id"])
        self._episode_to_event[episode_id] = item.event_id
        committed_at_ns = time.perf_counter_ns()
        if progress is not None:
            progress({"commit_observed": True, "committed_at_ns": committed_at_ns})
        visible: bool | None = None
        visibility_error: str | None = None
        searchable_at_ns: int | None = None
        if visibility:
            if progress is not None:
                progress({"visibility_probe_started": True})
            try:
                probe = QueryItem(
                    sample_id=item.sample_id,
                    question_id=f"probe:{item.event_id}",
                    question=item.text,
                    answer=None,
                    category=None,
                    evidence_ids=(),
                    ordinal=0,
                )
                visible = item.event_id in self.search(probe, top_k=35)["hit_ids"]
            except Exception as exc:  # noqa: BLE001 - recorded visibility evidence
                visible = False
                visibility_error = f"{type(exc).__name__}: {exc}"
            searchable_at_ns = time.perf_counter_ns() if visible else None
        receipt = {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": searchable_at_ns,
            "visibility_probe": visible,
            "visibility_error": visibility_error,
            "native_add_is_intrinsically_searchable": True,
            "created_memory_count": 1,
            # The event-specific episode and memory node are guaranteed new.
            # The speaker node is a deterministic MERGE and may be an upsert.
            "created_node_count": 2,
            "created_edge_count": 3,
            "creation_counts_available": True,
            "creation_count_source": "prepared Graphiti namespace objects",
            "created_entity_node_ids": result["entity_node_ids"],
            "upserted_shared_entity_node_count": 1,
            "created_entity_edge_id": result["entity_edge_id"],
            "created_episode_id": episode_id,
            "created_episodic_edge_ids": result["episodic_edge_ids"],
            "foreground_boundary": (
                "prepared fact embedding + Graphiti namespace persistence"
            ),
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "construction_llm_policy": "forbidden_fail_closed",
            "llm_call_count": 0,
            "cross_encoder_call_count": 0,
            "updated_stores": [
                "neo4j_entity_nodes",
                "neo4j_entity_edge",
                "neo4j_vector_index",
                "neo4j_episode_provenance",
            ],
            "prepared_item_id": item.event_id,
        }
        if progress is not None:
            progress(receipt)
        return receipt

    def stats(self) -> dict[str, Any]:
        manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.is_file()
            else None
        )
        config = asdict(self.config)
        config["neo4j_password"] = "<redacted>"
        return {
            "config": config,
            "display_label": "Graphiti (Zep OSS proxy)",
            "interpretation_boundary": "not production Zep",
            "adapter_sha256": self._adapter_sha256(),
            "graphiti_source": self._source_identity(),
            "live_counts": self._counts(),
            "group_counts": self.group_counts(),
            "snapshot_manifest": manifest,
            "snapshot_adapter_compatibility": self._snapshot_adapter_compatibility,
            "stage_tracing": {
                "enabled": self._stage_tracing,
                "instrumented": self._instrumented,
                "boundaries": [
                    "llm.generate_response",
                    "embedder.create/create_batch",
                    "neo4j.execute_query",
                    "compound Graphiti search/add APIs",
                ],
            },
            "search_semantics": self.search_semantics(),
            "add_semantics": self.add_semantics(),
            "llm_free_operation_guard": {
                "installed": self._llm_guard_installed,
                "forbidden_generation_calls": self._forbidden_llm_calls,
                "forbidden_cross_encoder_calls": self._forbidden_cross_encoder_calls,
                "passed": (
                    self._forbidden_llm_calls == 0
                    and self._forbidden_cross_encoder_calls == 0
                ),
            },
        }

    def snapshot_fingerprint(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise RuntimeError("Graphiti snapshot manifest does not exist")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        observed_counts = self._counts()
        if observed_counts != manifest["counts"]:
            raise RuntimeError(
                "Graphiti live counts differ from snapshot manifest: "
                f"expected={manifest['counts']!r}, observed={observed_counts!r}"
            )
        return {
            "schema": manifest["schema"],
            "dataset_sha256": manifest["dataset_sha256"],
            "adapter_sha256": manifest["adapter_sha256"],
            "graphiti_commit": manifest["graphiti_source"]["commit"],
            "counts": observed_counts,
        }

    def close(self) -> None:
        if self._runtime.loop.is_closed():
            return
        try:
            self._runtime.run(self._graphiti.close())
        finally:
            self._runtime.close()
