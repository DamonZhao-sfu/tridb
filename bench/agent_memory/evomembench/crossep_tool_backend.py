"""GEM adapter for EvoMemBench's CrossEp-Tool Memory interface.

The official runner calls ``begin_sample → utilize → update → drain_usage``.
This adapter additionally requires ``set_sample_context`` so the experimental
``gem_task_seed`` mode can use the current first user turn and visible tool
schema. ``upstream_parity`` uses the exact string supplied by the official
runner (system prompt in prompting mode, function docs in FC mode).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.extractor import extract_observed_trajectory
from bench.agent_memory.evomembench.injection import fit_injection, pad_to_token_budget
from bench.agent_memory.evomembench.load import admit_extracted_experience
from bench.agent_memory.evomembench.modeling import experience_query
from bench.agent_memory.evomembench.task_signature import tool_task_signature
from bench.agent_memory.evomembench.system_protocol import embedding_sha256
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import RetrievalMode

QUERY_MODES = frozenset({"upstream_parity", "gem_task_seed"})
MEMORY_ARMS = frozenset({"recent_fifo", "vector_only", "graph_relational", "gem_fused"})


def _normalize_trajectory_message(message: Any, *, index: int) -> dict[str, Any]:
    """Convert an official-runner message into a stable JSON-shaped mapping.

    BFCL keeps assistant responses as OpenAI SDK/Pydantic message objects until
    after ``external_memory.update``.  User and tool messages are already plain
    mappings, so a completed trajectory can legitimately contain both shapes.
    Normalize that boundary here, before role filtering and leakage checks.
    """

    if hasattr(message, "model_dump"):
        message = message.model_dump(exclude_none=True)
    elif hasattr(message, "to_dict"):
        message = message.to_dict()
    if not isinstance(message, Mapping):
        raise TypeError(
            f"trajectory[{index}] must be a mapping or SDK message object, "
            f"got {type(message).__name__}"
        )

    def normalize(value: Any, *, location: str) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        elif hasattr(value, "to_dict"):
            value = value.to_dict()
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item, location=f"{location}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                normalize(item, location=f"{location}[{offset}]")
                for offset, item in enumerate(value)
            ]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"{location} is not JSON-compatible: {type(value).__name__}")

    return normalize(message, location=f"trajectory[{index}]")


def normalize_tool_trajectory(trajectory: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize the official handler's mixed SDK/dict online trajectory."""

    return [
        _normalize_trajectory_message(message, index=index)
        for index, message in enumerate(trajectory)
    ]


@dataclass(frozen=True)
class ToolSampleContext:
    sample_id: str
    episode_uid: str
    ordinal: int
    environment: str
    question: Sequence[Sequence[Mapping[str, object]]]
    involved_classes: Sequence[str]
    allowed_functions: Sequence[str] = ()

    @property
    def task_signature(self) -> str:
        return tool_task_signature(
            question=self.question,
            involved_classes=self.involved_classes,
            allowed_functions=self.allowed_functions,
        )

    @property
    def schema_hash(self) -> str:
        payload = json.dumps(
            {
                "classes": sorted(str(x) for x in self.involved_classes),
                "functions": sorted(str(x) for x in self.allowed_functions),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class MemoryUsageLog:
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    n_llm_calls: int = 0
    n_embed_calls: int = 0
    n_utilize: int = 0
    n_update: int = 0


@dataclass(frozen=True)
class BankSnapshot:
    scope_id: str
    source_environment: str
    unit_ids: tuple[int, ...]
    experience_count: int
    max_ordinal: int
    transition_id: int | None
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossEpToolGemMemory:
    """Duck-compatible official Memory backend backed by one TriDB database."""

    def __init__(
        self,
        *,
        memory: TriDBGovernedMemory,
        scope_id: str,
        source_environment: str,
        count_tokens: Any,
        query_mode: str = "gem_task_seed",
        retrieval_mode: RetrievalMode = RetrievalMode.FUSED,
        arm: str | None = None,
        top_k: int = 3,
        hops: int = 2,
        m_seeds: int = 4,
        term_cond: int = 32,
        graph_scoring: str = "membership",
        graph_work_budget: int = 65536,
        query_embedder: Any | None = None,
        token_budget: int = 2048,
        readonly: bool = False,
        snapshot: BankSnapshot | None = None,
        owns_memory: bool = False,
        token_matched_slot: bool = True,
    ) -> None:
        if query_mode not in QUERY_MODES:
            raise ValueError(f"unknown CrossEp-Tool query mode: {query_mode}")
        inferred_arm = {
            RetrievalMode.VECTOR: "vector_only",
            RetrievalMode.GRAPH: "graph_relational",
            RetrievalMode.FUSED: "gem_fused",
        }.get(retrieval_mode)
        self.arm = arm or inferred_arm
        if self.arm not in MEMORY_ARMS:
            raise ValueError(f"unknown CrossEp-Tool memory arm: {self.arm}")
        if snapshot is not None and snapshot.scope_id != scope_id:
            raise ValueError("snapshot scope does not match backend scope")
        if snapshot is not None and not readonly:
            raise ValueError("a frozen source-bank snapshot must be opened readonly")
        if graph_scoring != "membership":
            raise ValueError("formal multi-system parity requires membership scoring")
        if graph_work_budget < 128:
            raise ValueError("tjs.graph_work_budget must be at least 128")
        self.memory = memory
        self.scope_id = scope_id
        self.source_environment = source_environment
        self.query_mode = query_mode
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.hops = hops
        self.m_seeds = m_seeds
        self.term_cond = term_cond
        self.graph_scoring = graph_scoring
        self.graph_work_budget = graph_work_budget
        self.query_embedder = query_embedder
        self.token_budget = token_budget
        self.count_tokens = count_tokens
        self.readonly = readonly
        self.snapshot = snapshot
        self.owns_memory = owns_memory
        self.token_matched_slot = token_matched_slot
        self._tls = threading.local()
        self._db_lock = threading.RLock()
        self.receipts: list[dict[str, Any]] = []

    def close(self) -> None:
        if self.owns_memory:
            self.memory.close()

    def set_sample_context(self, context: ToolSampleContext) -> None:
        if context.ordinal < 0:
            raise ValueError("sample ordinal must be non-negative")
        self._tls.context = context

    def _context(self) -> ToolSampleContext:
        context = getattr(self._tls, "context", None)
        if context is None:
            raise RuntimeError("set_sample_context must run before utilize/update")
        return context

    def begin_sample(self) -> None:
        self._tls.usage = MemoryUsageLog()

    def _usage(self) -> MemoryUsageLog:
        usage = getattr(self._tls, "usage", None)
        if usage is None:
            usage = MemoryUsageLog()
            self._tls.usage = usage
        return usage

    def drain_usage(self) -> MemoryUsageLog:
        usage = self._usage()
        self._tls.usage = MemoryUsageLog()
        return usage

    def _load(self, unit_ids: Sequence[int]) -> list[dict[str, Any]]:
        if not unit_ids:
            return []
        rows = self.memory.store.conn.execute(
            "SELECT u.id, u.title, (u.metadata->>'experience_ordinal')::integer,"
            " fv.value FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id=u.id"
            " WHERE u.id=ANY(%s) AND u.scope_id=%s AND fv.field='memory_payload'"
            " AND fv.valid_to IS NULL ORDER BY array_position(%s::bigint[], u.id)",
            (list(unit_ids), self.scope_id, list(unit_ids)),
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

    def utilize(self, upstream_query: str) -> str:
        context = self._context()
        usage = self._usage()
        started = time.perf_counter()
        cutoff = self.snapshot.max_ordinal + 1 if self.snapshot else context.ordinal
        query_text = (
            upstream_query
            if self.query_mode == "upstream_parity"
            else context.task_signature
        )
        if cutoff <= 0:
            usage.n_utilize += 1
            if not self.token_matched_slot:
                return ""
            slot, padding_tokens = pad_to_token_budget(
                "", token_budget=self.token_budget, codec=self.count_tokens
            )
            self.receipts.append(
                {
                    "sample_id": context.sample_id,
                    "source_environment": self.source_environment,
                    "target_environment": context.environment,
                    "query_mode": self.query_mode,
                    "arm": self.arm,
                    "task_signature": query_text,
                    "task_signature_sha256": hashlib.sha256(
                        query_text.encode()
                    ).hexdigest(),
                    "cutoff_ordinal": cutoff,
                    "selected_unit_ids": [],
                    "selected_episode_uids": [],
                    "graph_scoring": self.graph_scoring,
                    "graph_work_budget": self.graph_work_budget,
                    "injection_tokens": 0,
                    "injection_text": "",
                    "injection_sha256": hashlib.sha256(b"").hexdigest(),
                    "slot_tokens": self.token_budget,
                    "padding_tokens": padding_tokens,
                    "readonly": self.readonly,
                    "probes": {"termination_reason": "empty_snapshot"},
                    "latency_ms": 0.0,
                }
            )
            return slot
        query_embedding: list[float] | None = None
        query_embedding_ms = 0.0
        query_embedding_tokens = 0
        database_retrieval_ms = 0.0
        with self._db_lock:
            if self.arm == "recent_fifo":
                rows = self.memory.store.conn.execute(
                    "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
                    " AND metadata->>'node_kind'='experience'"
                    " AND (metadata->>'experience_ordinal')::integer < %s"
                    " ORDER BY (metadata->>'experience_ordinal')::integer DESC, id DESC"
                    " LIMIT %s",
                    (self.scope_id, cutoff, self.top_k),
                ).fetchall()
                unit_ids = [int(row[0]) for row in rows]
                probes = {"mode": "relational", "termination_reason": "limit"}
                embed_calls = 0
                embed_tokens = 0
            else:
                anchor_id = None
                mode = {
                    "vector_only": RetrievalMode.VECTOR,
                    "graph_relational": RetrievalMode.GRAPH,
                    "gem_fused": RetrievalMode.FUSED,
                }[self.arm]
                if mode is RetrievalMode.GRAPH:
                    anchor = self.memory.store.conn.execute(
                        "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
                        " AND metadata->>'node_kind'='experience'"
                        " AND (metadata->>'experience_ordinal')::integer < %s"
                        " ORDER BY (metadata->>'experience_ordinal')::integer DESC, id DESC"
                        " LIMIT 1",
                        (self.scope_id, cutoff),
                    ).fetchone()
                    if anchor is None:
                        unit_ids = []
                        probes = {
                            "mode": "graph",
                            "termination_reason": "empty_snapshot",
                        }
                        embed_calls = 0
                        embed_tokens = 0
                        result = None
                    else:
                        anchor_id = int(anchor[0])
                if mode is not RetrievalMode.GRAPH or anchor_id is not None:
                    if (
                        mode in (RetrievalMode.VECTOR, RetrievalMode.FUSED)
                        and self.query_embedder is not None
                    ):
                        embedding_started = time.perf_counter()
                        query_embedding = self.query_embedder.encode([query_text])[0]
                        query_embedding_ms = (
                            time.perf_counter() - embedding_started
                        ) * 1000
                        record = self.query_embedder.client.ledger.records[-1]
                        query_embedding_tokens = int(
                            record.get("detail", {})
                            .get("usage", {})
                            .get("prompt_tokens", 0)
                        )
                    query = experience_query(
                        scope_id=self.scope_id,
                        task_signature=query_text,
                        cutoff_ordinal=cutoff,
                        mode=mode,
                        top_k=self.top_k,
                        hops=self.hops,
                        m_seeds=self.m_seeds,
                        term_cond=self.term_cond,
                        reinforce=False,
                        source_phases=("in_env",),
                    )
                    if query_embedding is not None:
                        query = replace(
                            query,
                            text=None,
                            embedding=tuple(query_embedding),
                        )
                    if anchor_id is not None:
                        query = replace(query, text=None, anchor_id=anchor_id)
                    database_started = time.perf_counter()
                    result = self.memory.retrieve(query)
                    database_retrieval_ms = (
                        time.perf_counter() - database_started
                    ) * 1000
                    if not result.committed:
                        raise RuntimeError(result.aborted_reason)
                    unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
                    probes = dict(result.probes)
                    probe_read_ms = float(
                        probes.get("instrumentation_probe_read_ms") or 0.0
                    )
                    probes["raw_database_retrieval_including_instrumentation_ms"] = (
                        database_retrieval_ms
                    )
                    database_retrieval_ms = max(
                        0.0, database_retrieval_ms - probe_read_ms
                    )
                    embed_calls = (
                        1 if query_embedding is not None else result.cost.embed_calls
                    )
                    embed_tokens = (
                        query_embedding_tokens
                        if query_embedding is not None
                        else result.cost.embed_input_tokens
                    )
            if self.snapshot is not None and not set(unit_ids).issubset(
                self.snapshot.unit_ids
            ):
                raise RuntimeError("retrieval escaped the frozen source-bank snapshot")
            items = self._load(unit_ids)
        accepted, injection, injection_tokens = fit_injection(
            items,
            max_items=self.top_k,
            token_budget=self.token_budget,
            count_tokens=self.count_tokens,
        )
        if self.token_matched_slot:
            slot, padding_tokens = pad_to_token_budget(
                injection, token_budget=self.token_budget, codec=self.count_tokens
            )
        else:
            slot, padding_tokens = injection, 0
        elapsed = time.perf_counter() - started
        probe_read_seconds = (
            float(probes.get("instrumentation_probe_read_ms") or 0.0) / 1000
        )
        usage.latency_s += max(0.0, elapsed - probe_read_seconds)
        usage.n_utilize += 1
        usage.n_embed_calls += embed_calls
        usage.embedding_tokens += embed_tokens
        self.receipts.append(
            {
                "sample_id": context.sample_id,
                "source_environment": self.source_environment,
                "target_environment": context.environment,
                "query_mode": self.query_mode,
                "arm": self.arm,
                "m_seeds": self.m_seeds,
                "term_cond": self.term_cond,
                "graph_scoring": self.graph_scoring,
                "graph_work_budget": self.graph_work_budget,
                "top_k": self.top_k,
                "hops": self.hops,
                "token_budget": self.token_budget,
                "task_signature_sha256": hashlib.sha256(
                    query_text.encode()
                ).hexdigest(),
                "task_signature": query_text,
                "task_embedding": query_embedding,
                "task_embedding_sha256": (
                    None
                    if query_embedding is None
                    else embedding_sha256(query_embedding)
                ),
                "query_embedding_ms": query_embedding_ms,
                "query_embedding_input_tokens": query_embedding_tokens,
                "database_retrieval_ms": database_retrieval_ms,
                "cutoff_ordinal": cutoff,
                "selected_unit_ids": [item["unit_id"] for item in accepted],
                "selected_episode_uids": [item["episode_uid"] for item in accepted],
                "selected_ordinals": [int(item["ordinal"]) for item in accepted],
                "injection_tokens": injection_tokens,
                "injection_text": injection,
                "injection_sha256": hashlib.sha256(injection.encode()).hexdigest(),
                "slot_tokens": self.count_tokens(slot),
                "padding_tokens": padding_tokens,
                "snapshot_digest": None
                if self.snapshot is None
                else self.snapshot.digest,
                "readonly": self.readonly,
                "probes": probes,
                "latency_ms": max(0.0, elapsed * 1000 - probe_read_seconds * 1000),
            }
        )
        return slot

    def update(self, trajectory: Sequence[Any]) -> None:
        usage = self._usage()
        usage.n_update += 1
        if self.readonly:
            return
        context = self._context()
        if context.environment != self.source_environment:
            raise RuntimeError(
                "writable in-environment bank received a cross-environment target"
            )
        started = time.perf_counter()
        concepts = [("environment", context.environment)]
        concepts.extend(("tool", item) for item in context.involved_classes)
        concepts.extend(("function", item) for item in context.allowed_functions)
        normalized_trajectory = normalize_tool_trajectory(trajectory)
        online_trajectory = [
            message
            for message in normalized_trajectory
            if not (
                self.token_matched_slot
                and str(message.get("role", "")).casefold() == "system"
            )
        ]
        extracted = extract_observed_trajectory(
            task_signature=context.task_signature,
            trajectory=online_trajectory,
            concept_hints=concepts,
            source_external_ids=[context.sample_id],
        )
        valid_from = (
            datetime(2000, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=context.ordinal)
        ).isoformat()
        with self._db_lock:
            admission = admit_extracted_experience(
                self.memory,
                uid=context.episode_uid,
                scope_id=self.scope_id,
                ordinal=context.ordinal,
                task_signature=context.task_signature,
                extracted=extracted,
                valid_from=valid_from,
                metadata={
                    "benchmark": "CrossEp-Tool",
                    "environment": context.environment,
                    "source_phase": "in_env",
                    "tool_schema_hash": context.schema_hash,
                },
            )
        if not admission.result.committed:
            raise RuntimeError(admission.result.aborted_reason)
        usage.latency_s += time.perf_counter() - started
        usage.n_embed_calls += admission.result.cost.embed_calls
        usage.embedding_tokens += admission.result.cost.embed_input_tokens

    def freeze(self) -> BankSnapshot:
        """Freeze a logical source bank without copying topology out of Postgres."""
        with self._db_lock:
            rows = self.memory.store.conn.execute(
                "SELECT u.id, (u.metadata->>'experience_ordinal')::integer,"
                " u.metadata::text, fv.value FROM gem_unit u"
                " JOIN gem_field_value fv ON fv.unit_id=u.id"
                " WHERE u.scope_id=%s AND u.state='active'"
                " AND u.metadata->>'node_kind'='experience'"
                " AND fv.field='memory_payload' AND fv.valid_to IS NULL ORDER BY u.id",
                (self.scope_id,),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            edge_rows = (
                self.memory.store.conn.execute(
                    "SELECT src, dst, edge_type, rel, weight FROM gem_edge"
                    " WHERE tombstoned_at IS NULL AND (src=ANY(%s) OR dst=ANY(%s))"
                    " ORDER BY src, dst, edge_type",
                    (ids, ids),
                ).fetchall()
                if ids
                else []
            )
            transition = self.memory.store.conn.execute(
                "SELECT max(t) FROM gem_transition WHERE scope_id=%s AND committed",
                (self.scope_id,),
            ).fetchone()[0]
        unit_ids = tuple(ids)
        max_ordinal = max((int(row[1]) for row in rows), default=-1)
        canonical = json.dumps(
            {
                "scope_id": self.scope_id,
                "experiences": [
                    {
                        "id": int(row[0]),
                        "ordinal": int(row[1]),
                        "metadata": row[2],
                        "payload_sha256": hashlib.sha256(
                            str(row[3]).encode()
                        ).hexdigest(),
                    }
                    for row in rows
                ],
                "incident_edges": [list(row) for row in edge_rows],
            },
            sort_keys=True,
            default=str,
        )
        return BankSnapshot(
            scope_id=self.scope_id,
            source_environment=self.source_environment,
            unit_ids=unit_ids,
            experience_count=len(rows),
            max_ordinal=max_ordinal,
            transition_id=None if transition is None else int(transition),
            digest=hashlib.sha256(canonical.encode()).hexdigest(),
        )

    def assert_snapshot_unchanged(self) -> None:
        if self.snapshot is None:
            raise RuntimeError("backend was not opened from a frozen snapshot")
        observed = self.freeze()
        if observed.digest != self.snapshot.digest:
            raise RuntimeError(
                "readonly target evaluation mutated its source bank: "
                f"{observed.digest} != {self.snapshot.digest}"
            )

    @staticmethod
    def write_snapshot(path: str | Path, snapshot: BankSnapshot) -> None:
        Path(path).write_text(json.dumps(snapshot.as_dict(), indent=2) + "\n")

    @staticmethod
    def read_snapshot(path: str | Path) -> BankSnapshot:
        payload = json.loads(Path(path).read_text())
        payload["unit_ids"] = tuple(payload["unit_ids"])
        return BankSnapshot(**payload)

    @classmethod
    def load_from_disk(
        cls, *, snapshot_path: str | Path, **kwargs: Any
    ) -> "CrossEpToolGemMemory":
        """Official interface spelling; loads a logical snapshot descriptor."""
        snapshot = cls.read_snapshot(snapshot_path)
        return cls(
            scope_id=snapshot.scope_id,
            source_environment=snapshot.source_environment,
            readonly=True,
            snapshot=snapshot,
            **kwargs,
        )
