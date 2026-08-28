"""EverMemOS (EverOS OSS) adapter for the controlled-model Track C protocol.

EverOS is a *server*, not a library: the OSS edition exposes
``POST /api/v2/memory/add``, ``/flush``, ``/search`` and ``GET /health``
(see ``src/everos/entrypoints/api/routes/``). This adapter is therefore a
thin HTTP client rather than an in-process embedding of the framework, and
every stage boundary we can observe is an HTTP call.

Two consequences the receipts must carry honestly:

* EverOS extraction is **asynchronous**. ``/add`` only buffers; boundary
  detection runs on ``/flush``, and both the OME strategy engine and md ->
  LanceDB projection (cascade) continue in the background. ``finalize_build``
  first drains OME through its synchronization endpoint and then observes
  ``cascade.pending == 0`` twice in a row. This prevents construction-serving
  overlap while preserving EverOS's documented cascade convergence contract.
* EverOS partitions memory by *owner*, derived from a message's
  ``sender_id``. LoCoMo conversations have two speakers; we map the whole
  conversation onto one owner (``sender_id = sample_id``) and keep the
  speaker in ``sender_name``. This is the same scoping choice the Mem0
  adapter makes (``user_id=item.sample_id``), so the retrieval unit stays
  comparable across systems.
"""

from __future__ import annotations

import hashlib
import json
import re
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
_PATH_SAFE = re.compile(r"^[a-zA-Z0-9_.@+-]+$")

# EverOS caps /add at 500 messages; its own LoCoMo runner uses 25.
_ADD_BATCH = 25

_SNAPSHOT_SCHEMA = "table5_track_c_evermemos_snapshot_v0.1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_epoch_ms(item: EventItem) -> int:
    """Parse LoCoMo's frozen timestamp into the epoch-ms contract of /add.

    The intra-session turn offset is preserved as microseconds so that two
    turns sharing a printed minute still order deterministically, matching
    how the Graphiti adapter derives ``reference_time``.
    """
    base = datetime.strptime(item.timestamp, _TIMESTAMP_FORMAT).replace(
        tzinfo=timezone.utc
    )
    turn = int(item.metadata.get("turn_number") or item.ordinal % 1_000_000)
    return int((base + timedelta(microseconds=turn)).timestamp() * 1000)


def message_body(item: EventItem) -> str:
    """The turn text, verbatim.

    Graphiti's adapter prefixes an ``[event_id=...]`` tag so retrieved
    material can be traced back to evidence turns. That device does **not**
    work here: EverOS does not store turns, it stores LLM-rewritten episode
    summaries, and the tag is dropped during extraction (verified against a
    live server — no retrieved episode, atomic fact, or field carried it).
    Keeping the tag would therefore only inject noise into the text the
    extractor reads, so the turn is sent unmodified and provenance is
    reported at the session granularity EverOS actually preserves.
    """
    return item.text


def _path_safe(value: str, *, field: str) -> str:
    if not _PATH_SAFE.fullmatch(value) or len(value) > 128:
        raise ValueError(f"unsafe EverOS {field}: {value!r}")
    return value


@dataclass(frozen=True)
class EverMemOSTrackCConfig:
    dataset_path: str
    base_url: str = "http://127.0.0.1:8020"
    api_version: str = "v2"
    app_id: str = "tridb_trackc"
    project_id: str = "locomo"
    search_method: str = "hybrid"
    include_profile: bool = False
    enable_llm_rerank: bool = False
    answer_base_url: str = "http://127.0.0.1:8000/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    evermemos_source_root: str = "/localhome/hza214/agent-memory-table5/src/evermemos"
    build_scope_workers: int = 5
    request_timeout: float = 300.0
    drain_timeout: float = 7200.0
    drain_poll_seconds: float = 5.0

    def __post_init__(self) -> None:
        _path_safe(self.app_id, field="app_id")
        _path_safe(self.project_id, field="project_id")
        if self.search_method not in {"keyword", "vector", "hybrid", "agentic"}:
            raise ValueError(f"unknown EverOS search method: {self.search_method!r}")


class EverMemOSTrackCAdapter:
    """Track C adapter over the EverOS OSS HTTP surface."""

    name = "evermemos"

    def __init__(self, config: EverMemOSTrackCConfig) -> None:
        import httpx

        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self._api = f"{config.base_url.rstrip('/')}/api/{config.api_version}/memory"
        self._client = httpx.Client(timeout=config.request_timeout)
        self._build_stats: dict[str, Any] = {}
        self._stage_tracing = False
        self._sessions: set[str] = set()

    # ── identity ────────────────────────────────────────────────────────

    def _adapter_sha256(self) -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _source_identity(self) -> dict[str, Any]:
        import subprocess

        root = Path(self.config.evermemos_source_root)
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit, dirty = "", False
        versions: dict[str, str] = {}
        import importlib.metadata as metadata

        for dist in ("everos", "everalgo-core", "everalgo-user-memory", "everalgo-rank"):
            try:
                versions[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                versions[dist] = "absent"
        return {
            "root": str(root),
            "commit": commit,
            "tracked_dirty": dirty,
            "distributions": versions,
        }

    # ── HTTP helpers ────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(f"{self._api}{path}", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"EverOS POST {path} failed {response.status_code}: {response.text[:400]}"
            )
        return response.json()

    def _health(self) -> dict[str, Any]:
        response = self._client.get(f"{self.config.base_url.rstrip('/')}/health")
        if response.status_code >= 400:
            raise RuntimeError(
                f"EverOS /health failed {response.status_code}: {response.text[:400]}"
            )
        return response.json()

    def _ome_drain(self, timeout: float) -> dict[str, Any]:
        response = self._client.post(
            f"{self.config.base_url.rstrip('/')}/api/{self.config.api_version}/ome/drain",
            json={"timeout": timeout},
            timeout=max(self.config.request_timeout, timeout + 30.0),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                "EverOS OME drain failed "
                f"{response.status_code}: {response.text[:400]}"
            )
        return response.json()

    @staticmethod
    def _endpoint_models(base_url: str) -> list[str]:
        import urllib.request

        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as fp:
            payload = json.load(fp)
        return sorted(str(row["id"]) for row in payload.get("data", []))

    # ── protocol surface ────────────────────────────────────────────────

    def enable_stage_tracing(self) -> None:
        """EverOS is opaque behind HTTP; record that limit rather than fake it.

        Every retrieval stage (embed, vector, fusion, rerank) happens inside
        the server process. A client-side span can only bracket the whole
        request, so we mark the boundary as a single opaque native call and
        emit no per-stage attribution we cannot actually observe.
        """
        self._stage_tracing = True

    def init_schema(self) -> dict[str, Any]:
        health = self._health()
        capabilities = health.get("capabilities") or {}
        if not capabilities.get("llm") or not capabilities.get("embed"):
            raise RuntimeError(f"EverOS capabilities not ready: {capabilities!r}")
        answer_models = self._endpoint_models(self.config.answer_base_url)
        embedding_models = self._endpoint_models(self.config.embedding_base_url)
        if answer_models != [self.config.answer_model]:
            raise RuntimeError(f"answer endpoint mismatch: {answer_models!r}")
        if embedding_models != [self.config.embedding_model]:
            raise RuntimeError(f"embedding endpoint mismatch: {embedding_models!r}")
        return {
            "ok": True,
            "backend": "everos-oss-http",
            "adapter_sha256": self._adapter_sha256(),
            "server_version": health.get("version"),
            "capabilities": capabilities,
            "disabled_features": health.get("disabled_features") or [],
            "answer_models": answer_models,
            "embedding_models": embedding_models,
            "evermemos_source": self._source_identity(),
        }

    def _session_id(self, item: EventItem) -> str:
        return _path_safe(f"{item.sample_id}__{item.session_id}", field="session_id")

    def _message(self, item: EventItem) -> dict[str, Any]:
        return {
            # One owner per conversation; the speaker survives as sender_name.
            "sender_id": _path_safe(item.sample_id, field="sender_id"),
            "sender_name": item.role,
            "role": "user",
            "timestamp": reference_epoch_ms(item),
            "content": message_body(item),
        }

    def _ingest_scope(self, scope_events: Sequence[EventItem]) -> dict[str, Any]:
        by_session: dict[str, list[EventItem]] = {}
        for item in scope_events:
            by_session.setdefault(self._session_id(item), []).append(item)
        added = 0
        extracted = 0
        flushed = 0
        for session_id, items in by_session.items():
            self._sessions.add(session_id)
            for start in range(0, len(items), _ADD_BATCH):
                batch = items[start : start + _ADD_BATCH]
                payload = self._post(
                    "/add",
                    {
                        "session_id": session_id,
                        "app_id": self.config.app_id,
                        "project_id": self.config.project_id,
                        "messages": [self._message(entry) for entry in batch],
                    },
                )
                data = payload.get("data") or {}
                added += int(data.get("message_count") or 0)
                extracted += int(data.get("status") == "extracted")
            payload = self._post(
                "/flush",
                {
                    "session_id": session_id,
                    "app_id": self.config.app_id,
                    "project_id": self.config.project_id,
                },
            )
            flushed += int((payload.get("data") or {}).get("status") == "extracted")
        return {
            "sessions": len(by_session),
            "messages_accepted": added,
            "mid_stream_extractions": extracted,
            "flush_extractions": flushed,
        }

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        grouped: dict[str, list[EventItem]] = {}
        for item in events:
            grouped.setdefault(item.sample_id, []).append(item)
        started = time.perf_counter()
        workers = max(1, min(self.config.build_scope_workers, len(grouped)))
        # Conversations are independent owners, so they may ingest
        # concurrently; turns inside one conversation stay strictly ordered
        # because EverOS boundary detection is order-sensitive.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reports = list(executor.map(self._ingest_scope, grouped.values()))
        self._build_stats = {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "samples": len(grouped),
            "scope_workers": workers,
            "sessions": sum(report["sessions"] for report in reports),
            "messages_accepted": sum(report["messages_accepted"] for report in reports),
            "flush_extractions": sum(report["flush_extractions"] for report in reports),
        }
        return self._build_stats

    def finalize_build(self) -> dict[str, Any]:
        """Block until OME construction and cascade projection converge.

        The order matters: OME construction writes markdown that the cascade
        subsequently projects. EverOS documents cascade convergence as *two
        consecutive* zero samples of ``cascade.pending``: a single zero can be
        read inside the watcher's input window, before the next batch is
        enqueued.
        """
        started = time.perf_counter()
        deadline = started + self.config.drain_timeout
        ome = self._ome_drain(max(0.001, deadline - time.perf_counter()))
        if not ome.get("idle"):
            raise RuntimeError(
                f"EverOS OME did not drain within {self.config.drain_timeout}s: {ome!r}"
            )
        consecutive_zero = 0
        samples = 0
        last: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            health = self._health()
            cascade = health.get("cascade") or {}
            last = cascade
            samples += 1
            pending = int(cascade.get("pending") or 0)
            retryable = int(cascade.get("failed_retryable") or 0)
            if pending == 0 and retryable == 0:
                consecutive_zero += 1
                if consecutive_zero >= 2:
                    break
            else:
                consecutive_zero = 0
            time.sleep(self.config.drain_poll_seconds)
        else:
            raise RuntimeError(
                f"EverOS cascade did not drain within {self.config.drain_timeout}s: {last!r}"
            )
        # Re-check OME after the watcher-safe cascade samples. This is normally
        # immediate, but guards against a late strategy dispatch at the edge of
        # the first drain observation.
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RuntimeError("EverOS build drain budget expired after cascade convergence")
        final_ome = self._ome_drain(remaining)
        if not final_ome.get("idle"):
            raise RuntimeError(f"EverOS OME became active after cascade drain: {final_ome!r}")
        return {
            "wall_seconds": time.perf_counter() - started,
            "health_samples": samples,
            "ome": ome,
            "final_ome": final_ome,
            "cascade": last,
            "converged": consecutive_zero >= 2,
        }

    # ── retrieval ───────────────────────────────────────────────────────

    def _search_payload(self, item: QueryItem, top_k: int) -> dict[str, Any]:
        return {
            "user_id": item.sample_id,
            "app_id": self.config.app_id,
            "project_id": self.config.project_id,
            "query": item.question,
            "method": self.config.search_method,
            # EverOS validates top_k as -1 or 1..100.
            "top_k": max(1, min(int(top_k), 100)),
            "include_profile": self.config.include_profile,
            "enable_llm_rerank": self.config.enable_llm_rerank,
        }

    @staticmethod
    def _render_episode(row: dict[str, Any]) -> str:
        parts: list[str] = []
        subject = str(row.get("subject") or "").strip()
        summary = str(row.get("summary") or "").strip()
        episode = str(row.get("episode") or "").strip()
        if subject:
            parts.append(subject)
        if summary and summary != subject:
            parts.append(summary)
        if episode:
            parts.append(episode)
        for fact in row.get("atomic_facts") or []:
            content = str((fact or {}).get("content") or "").strip()
            if content:
                parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _hit_sessions(episodes: Iterable[dict[str, Any]]) -> list[str]:
        """Session ids behind the retrieved episodes.

        This is the finest provenance EverOS preserves. It is deliberately
        NOT returned as ``hit_ids``: an evidence-recall metric keyed on
        whole sessions would score far more generously than the turn-level
        ids every other adapter reports, so the two must not be conflated.
        """
        seen: list[str] = []
        for row in episodes:
            session = row.get("session_id")
            if session and session not in seen:
                seen.append(str(session))
        return seen

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        with stage_span(
            "framework_other",
            "evermemos.search.http",
            backend="everos-oss-http",
            attributes={
                "includes": ["embedding", "vector", "keyword", "fusion"],
                "method": self.config.search_method,
                "user_id": item.sample_id,
                # The whole pipeline runs server-side; a client span cannot
                # attribute time to individual stages.
                "observable_call_kind": "opaque_native_api",
            },
        ):
            payload = self._post("/search", self._search_payload(item, top_k))
        data = payload.get("data") or {}
        episodes = list(data.get("episodes") or [])
        contexts = [self._render_episode(row) for row in episodes]
        return {
            "result_count": len(episodes),
            # EverOS retrieval units are LLM-rewritten episode summaries, so
            # no turn-level evidence id survives to match against
            # ``QueryItem.evidence_ids``. Reporting [] keeps evidence recall
            # honestly undefined rather than silently zero-or-inflated.
            "hit_ids": [],
            "hit_sessions": self._hit_sessions(episodes),
            "provenance": "session_level_only",
            "memory_ids": [str(row.get("id")) for row in episodes],
            "contexts": contexts,
            "context_tokens": None,
            "empty": not episodes,
            "rerank": bool(self.config.enable_llm_rerank),
            "request_id": payload.get("request_id"),
            "retrieval_rounds": data.get("retrieval_rounds"),
            "round_boundaries_ns": list(data.get("round_boundaries_ns") or []),
            "agentic_decision": data.get("agentic_decision"),
        }

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        session_id = self._session_id(item)
        with stage_span(
            "framework_other",
            "evermemos.add.http",
            backend="everos-oss-http",
            attributes={
                "includes": ["llm", "embedding", "persistence"],
                "observable_call_kind": "opaque_native_api",
            },
        ):
            payload = self._post(
                "/add",
                {
                    "session_id": session_id,
                    "app_id": self.config.app_id,
                    "project_id": self.config.project_id,
                    "messages": [self._message(item)],
                },
            )
        data = payload.get("data") or {}
        result = {
            "status": data.get("status"),
            "message_count": int(data.get("message_count") or 0),
            "session_id": session_id,
            "request_id": payload.get("request_id"),
        }
        if visibility:
            # `source_to_searchable` needs the turn to be *queryable*, not just
            # buffered: force boundary detection and wait for the projection.
            with stage_span(
                "framework_other",
                "evermemos.add.flush_and_drain",
                backend="everos-oss-http",
                attributes={"observable_call_kind": "opaque_native_api"},
            ):
                flush = self._post(
                    "/flush",
                    {
                        "session_id": session_id,
                        "app_id": self.config.app_id,
                        "project_id": self.config.project_id,
                    },
                )
                drain = self.finalize_build()
            result["flush_status"] = (flush.get("data") or {}).get("status")
            result["drain_seconds"] = drain["wall_seconds"]
        if progress is not None:
            progress(dict(result))
        return result

    # ── receipts ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        health = self._health()
        return {
            "config": asdict(self.config),
            "display_label": "EverMemOS (EverOS OSS)",
            "adapter_sha256": self._adapter_sha256(),
            "evermemos_source": self._source_identity(),
            "server_health": health,
            "build": self._build_stats,
            "sessions_written": len(self._sessions),
            "stage_tracing": {
                "enabled": self._stage_tracing,
                "instrumented": False,
                "boundaries": ["HTTP /add", "HTTP /flush", "HTTP /search"],
                "limitation": (
                    "EverOS OSS runs retrieval in a separate server process; "
                    "per-stage attribution is not observable from the client."
                ),
            },
        }

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        """Rebuild the session index from the corpus for a reuse run.

        `self._sessions` is normally filled by `ingest_history`. A `search`
        run reusing a frozen build never ingests, so without this hook the
        session set stays empty and `snapshot_fingerprint` reports zero
        episodes, failing the reuse gate against a perfectly good build.
        The ids are a pure function of the corpus, so recomputing them here
        reproduces exactly what the build process recorded.
        """
        self._sessions = {
            self._session_id(item)
            for events in corpus.events_by_sample.values()
            for item in events
        }

    def snapshot_fingerprint(self) -> dict[str, Any]:
        """Identify the built corpus so a later `search` run can prove reuse.

        EverOS exposes no corpus-wide count endpoint, so the fingerprint is
        assembled from what is observable: the dataset checksum, the adapter
        and server identity, and the per-owner episode counts obtained by
        probing `/search` once per conversation. `top_k` is capped at 100 by
        the request DTO, so the count saturates there and is recorded as a
        lower bound rather than passed off as exact.
        """
        counts = self._episode_census()
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "dataset_sha256": _sha256_file(self.dataset_path),
            "adapter_sha256": self._adapter_sha256(),
            "server_version": (self._health() or {}).get("version"),
            "evermemos_distributions": self._source_identity()["distributions"],
            "search_method": self.config.search_method,
            "sessions_written": sorted(self._sessions),
            "episode_counts": counts,
            "episodes_total": sum(counts.values()),
        }

    def _episode_census(self) -> dict[str, int]:
        """Exact per-owner episode counts.

        Taken from ``/get``'s ``total_count`` rather than a ``/search``
        probe: ``/get`` is a direct paginated fetch, so the count is the
        stored total instead of whatever a ranked query happened to
        surface under its ``top_k`` cap.
        """
        census: dict[str, int] = {}
        owners = sorted({session.split("__", 1)[0] for session in self._sessions})
        for owner in owners:
            payload = self._post(
                "/get",
                {
                    "user_id": owner,
                    "app_id": self.config.app_id,
                    "project_id": self.config.project_id,
                    "memory_type": "episode",
                    "page": 1,
                    "page_size": 1,
                },
            )
            data = payload.get("data") or {}
            census[owner] = int(data.get("total_count") or 0)
        return census

    def close(self) -> None:
        self._client.close()
