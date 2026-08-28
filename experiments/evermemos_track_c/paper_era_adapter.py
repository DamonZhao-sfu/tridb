"""Track C adapter for the paper-era EverMemOS HTTP API.

This intentionally targets the pre-EverOS server pinned in the execution log:
``POST /api/v1/memories`` and ``GET /api/v1/memories/search``.  The current
EverOS v2 adapter lives in :mod:`experiments.evermemos_track_c.adapter` and is
not interchangeable with this implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from bench.agent_memory.table5_track_c.dataset import (
    EventItem,
    LoCoMoCorpus,
    QueryItem,
)
from bench.agent_memory.table5_track_c.tracing import stage_span

_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"
_EVENT_TAG = re.compile(r"\[event_id=([^\]]+)\]")
_SNAPSHOT_SCHEMA = "evermemos_paper_track_c_snapshot_v0.1.0"
_PREPARED_ITEM_BOUNDARY = "prepared_memory_item_insertion_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_time(item: EventItem) -> str:
    base = datetime.strptime(item.timestamp, _TIMESTAMP_FORMAT).replace(
        tzinfo=timezone.utc
    )
    turn = int(item.metadata.get("turn_number") or item.ordinal % 1_000_000)
    return (base + timedelta(microseconds=turn)).isoformat()


def message_body(item: EventItem) -> str:
    return f"[event_id={item.event_id}] {item.text}"


@dataclass(frozen=True)
class EverMemOSPaperConfig:
    dataset_path: str
    base_url: str = "http://127.0.0.1:8195"
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    source_root: str = (
        "/localhome/hza214/agent-memory-table5/src/evermemos_paper_806ad055"
    )
    retrieve_method: str = "rrf"
    build_scope_workers: int = 5
    request_timeout: float = 600.0
    require_answer_endpoint: bool = True

    def __post_init__(self) -> None:
        if self.retrieve_method != "rrf":
            raise ValueError("the frozen paper-era operating point requires rrf")


class EverMemOSPaperAdapter:
    name = "evermemos_paper_proxy"

    def __init__(self, config: EverMemOSPaperConfig) -> None:
        import httpx

        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self._client = httpx.Client(timeout=config.request_timeout)
        self._api = f"{config.base_url.rstrip('/')}/api/v1"
        self._build_stats: dict[str, Any] = {}
        self._groups: set[str] = set()
        self._stage_tracing = False

    def enable_stage_tracing(self) -> None:
        self._stage_tracing = True

    def _source_identity(self) -> dict[str, Any]:
        root = Path(self.config.source_root)
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"root": str(root), "commit": commit, "dirty": dirty}

    @staticmethod
    def _endpoint_models(base_url: str) -> list[str]:
        import urllib.request

        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as fp:
            payload = json.load(fp)
        return sorted(str(row["id"]) for row in payload.get("data", []))

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(
            method,
            f"{self._api}{path}",
            json=payload,
            params=params,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"EverMemOS {method} {path} failed {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        if response.status_code == 202:
            raise RuntimeError(
                "synchronous EverMemOS request unexpectedly entered background mode: "
                f"{data!r}"
            )
        return data

    def init_schema(self) -> dict[str, Any]:
        response = self._client.get(f"{self.config.base_url.rstrip('/')}/health")
        if response.status_code >= 400:
            raise RuntimeError(f"EverMemOS health failed: {response.text[:500]}")
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
        return {
            "ok": True,
            "backend": "evermemos-paper-http",
            "health": response.json(),
            "answer_models": answer_models,
            "answer_endpoint_required": self.config.require_answer_endpoint,
            "embedding_models": embedding_models,
            "source": self._source_identity(),
        }

    @staticmethod
    def _payload(item: EventItem) -> dict[str, Any]:
        return {
            "group_id": item.sample_id,
            "group_name": item.sample_id,
            "message_id": item.event_id,
            "create_time": reference_time(item),
            "sender": f"{item.sample_id}:{item.role}",
            "sender_name": item.role,
            "role": "user",
            "content": message_body(item),
            "refer_list": [],
        }

    @staticmethod
    def _prepared_item_payload(item: EventItem) -> dict[str, Any]:
        return {
            "prepared_item_id": item.event_id,
            "group_id": item.sample_id,
            "group_name": item.sample_id,
            "user_id": f"{item.sample_id}:{item.role}",
            "timestamp": reference_time(item),
            "role": item.role,
            "content": item.text,
            "session_id": item.session_id,
            "ordinal": item.ordinal,
        }

    @staticmethod
    def add_semantics() -> dict[str, Any]:
        return {
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "input_unit": "one frozen LoCoMo event rendered as one EpisodeMemory",
            "construction_llm_policy": "forbidden_fail_closed",
            "included_stages": [
                "item_embedding",
                "mongodb_persistence",
                "elasticsearch_index",
                "milvus_index",
            ],
            "excluded_stages": [
                "boundary_detection",
                "episode_extraction",
                "atomic_fact_extraction",
                "profile_or_foresight_extraction",
            ],
            "native_call": "save_memory_docs([EpisodeMemory])",
        }

    def _ingest_group(self, events: Sequence[EventItem]) -> dict[str, Any]:
        extracted = 0
        for item in events:
            # The upstream decorator uses sync_mode=false to disable its
            # five-second background handoff. This makes the build boundary
            # deterministic and is verified against the pinned source.
            result = self._request(
                "POST",
                "/memories",
                payload=self._payload(item),
                params={"sync_mode": "false"},
            )
            extracted += int(((result.get("result") or {}).get("count") or 0))
        return {"events": len(events), "extracted": extracted}

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        grouped: dict[str, list[EventItem]] = {}
        for item in events:
            grouped.setdefault(item.sample_id, []).append(item)
            self._groups.add(item.sample_id)
        workers = min(max(1, self.config.build_scope_workers), len(grouped))
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reports = list(executor.map(self._ingest_group, grouped.values()))
        self._build_stats = {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "groups": len(grouped),
            "scope_workers": workers,
            "extracted": sum(row["extracted"] for row in reports),
        }
        return self._build_stats

    def finalize_build(self) -> dict[str, Any]:
        # Every build request ran with sync_mode=false, which disables the
        # upstream timeout-to-background transition. No server-side queue is
        # left to drain at this boundary.
        return {
            "converged": True,
            "boundary": "all POST /memories calls completed synchronously",
        }

    def _search_response(self, item: QueryItem, top_k: int) -> dict[str, Any]:
        return self._request(
            "GET",
            "/memories/search",
            params={
                "query": item.question,
                "group_id": item.sample_id,
                "user_id": "",
                "retrieve_method": self.config.retrieve_method,
                "top_k": max(1, min(int(top_k), 100)),
                "include_metadata": "true",
            },
        )

    @staticmethod
    def _flatten_memories(result: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group in result.get("memories") or []:
            if not isinstance(group, dict):
                continue
            for memories in group.values():
                rows.extend(row for row in memories or [] if isinstance(row, dict))
        return rows

    @staticmethod
    def _strings(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from EverMemOSPaperAdapter._strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from EverMemOSPaperAdapter._strings(nested)

    @classmethod
    def _event_ids(cls, value: Any) -> list[str]:
        found: list[str] = []
        for text in cls._strings(value):
            for match in _EVENT_TAG.finditer(text):
                if match.group(1) not in found:
                    found.append(match.group(1))
        return found

    @staticmethod
    def _context(row: dict[str, Any]) -> str:
        fields = [
            row.get("timestamp"),
            row.get("summary"),
            row.get("episode"),
            row.get("content"),
        ]
        return "\n".join(str(value) for value in fields if value)

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        with stage_span(
            "fusion",
            "evermemos_paper.search.rrf",
            backend="evermemos-paper-http",
            attributes={
                "includes": ["embedding", "vector", "keyword", "rrf"],
                "observable_call_kind": "opaque_native_api",
            },
        ):
            response = self._search_response(item, top_k)
        result = response.get("result") or {}
        rows = self._flatten_memories(result)
        hit_ids = self._event_ids(result.get("original_data") or rows)
        contexts = [self._context(row) for row in rows]
        return {
            "result_count": len(rows),
            "hit_ids": hit_ids,
            "memory_ids": [
                str(row.get("event_id") or row.get("id") or index)
                for index, row in enumerate(rows)
            ],
            "contexts": contexts,
            "context_tokens": None,
            "empty": not rows,
            "reranker": "rrf",
            "total_count": int(result.get("total_count") or len(rows)),
        }

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if visibility:
            raise RuntimeError(
                "paper-era EverMemOS source-to-searchable Add is outside the selected "
                "Native Add experiment"
            )
        with stage_span(
            "framework_other",
            "evermemos_paper.add.prepared_item_http",
            backend="evermemos-paper-http",
            attributes={
                "includes": ["embedding", "mongodb", "elasticsearch", "milvus"],
                "excludes": ["llm", "memory_extraction"],
                "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
                "observable_call_kind": "opaque_native_api",
            },
        ):
            response = self._client.post(
                f"{self._api}/benchmark/prepared-items",
                json=self._prepared_item_payload(item),
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"EverMemOS Native Add failed {response.status_code}: "
                    f"{response.text[:500]}"
                )
            body = response.json()
        committed_at_ns = time.perf_counter_ns()
        accepted_background = response.status_code == 202
        if accepted_background:
            raise RuntimeError(
                "controlled synchronous Native Add unexpectedly entered background mode"
            )
        if body.get("semantic_boundary") != _PREPARED_ITEM_BOUNDARY:
            raise RuntimeError(f"EverMemOS prepared-item boundary mismatch: {body!r}")
        if body.get("llm_call_count") != 0:
            raise RuntimeError(f"EverMemOS prepared-item Add called an LLM: {body!r}")
        count = int(body.get("created_memory_count") or 0)
        receipt = {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": None,
            "visibility_probe": None,
            "visibility_error": None,
            "created_memory_count": count,
            "created_node_count": int(body.get("created_node_count") or 0),
            "created_edge_count": int(body.get("created_edge_count") or 0),
            "creation_counts_available": True,
            "creation_count_source": "prepared_item_proxy_official_save_memory_docs",
            "http_status": response.status_code,
            "accepted_background": accepted_background,
            "request_id": body.get("request_id"),
            "foreground_boundary": (
                "EverMemOS benchmark proxy -> official save_memory_docs completion"
            ),
            "semantic_boundary": _PREPARED_ITEM_BOUNDARY,
            "construction_llm_policy": "forbidden_fail_closed",
            "llm_call_count": 0,
            "prepared_item_id": item.event_id,
            "created_ids": body.get("created_ids") or [],
            "updated_stores": body.get("updated_stores") or [],
            "server_stage_ms": body.get("stage_ms") or {},
        }
        if progress is not None:
            progress(receipt)
        return receipt

    def _census(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group_id in sorted(self._groups):
            probe = QueryItem(
                sample_id=group_id,
                question_id="fingerprint",
                question="memory",
                answer=None,
                category=None,
                evidence_ids=(),
                ordinal=0,
            )
            payload = self._search_response(probe, 1).get("result") or {}
            result[group_id] = int(payload.get("total_count") or 0)
        return result

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        """Reconstruct persisted conversation scopes after process restart.

        ``_groups`` is populated as events are ingested, but a Search process
        that reuses an existing build never calls :meth:`ingest_history`.
        The group ids are a deterministic property of the frozen corpus, so
        rebinding them here lets the reuse gate census the durable backend
        rather than incorrectly treating an empty client-side set as an empty
        snapshot.
        """
        observed_sha256 = _sha256(self.dataset_path)
        if corpus.sha256 != observed_sha256:
            raise RuntimeError(
                "reused EverMemOS corpus checksum mismatch: "
                f"{corpus.sha256} != {observed_sha256}"
            )
        self._groups = set(corpus.sample_ids)

    def snapshot_fingerprint(self) -> dict[str, Any]:
        census = self._census()
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "dataset_sha256": _sha256(self.dataset_path),
            "source": self._source_identity(),
            "retrieve_method": self.config.retrieve_method,
            "group_counts": census,
            "total_count": sum(census.values()),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "display_label": "EverMemOS (paper-era official-network fork)",
            "source": self._source_identity(),
            "build": self._build_stats,
            "groups": sorted(self._groups),
            "stage_tracing": {
                "enabled": self._stage_tracing,
                "instrumented": False,
                "limitation": "server-internal stages are opaque at the HTTP boundary",
            },
            "add_semantics": self.add_semantics(),
        }

    def close(self) -> None:
        self._client.close()
