"""Strict full-history baselines for EvoMemBench cross-episode tracks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.crossep_tool_backend import (
    MemoryUsageLog,
    ToolSampleContext,
)
from bench.agent_memory.evomembench.dataset import EvoEpisode
from bench.agent_memory.evomembench.system_protocol import (
    HistoryItem,
    LongContextMaterialization,
    canonical_digest,
    materialize_long_context,
)


class LongContextOverflow(RuntimeError):
    """The complete eligible history cannot fit; truncation is forbidden."""


def knowledge_history_item(episode: EvoEpisode, response: str) -> HistoryItem:
    """Serialize one complete, label-free Know episode as raw prior history."""
    if not response.strip():
        raise ValueError("knowledge long-context history requires an observed response")
    messages = [asdict(message) for message in episode.messages]
    text = "\n".join(
        [
            f"Prior episode {episode.source_task_id}:",
            "Original messages:",
            json.dumps(messages, ensure_ascii=False),
            "Observed assistant response:",
            response,
        ]
    )
    return HistoryItem(
        episode_uid=episode.episode_uid,
        ordinal=episode.ordinal,
        text=text,
    )


@dataclass(frozen=True)
class LongContextSnapshot:
    source_environment: str
    records: tuple[HistoryItem, ...]
    digest: str

    @classmethod
    def build(
        cls, *, source_environment: str, records: Sequence[HistoryItem]
    ) -> "LongContextSnapshot":
        ordered = tuple(sorted(records, key=lambda item: item.ordinal))
        unsigned = {
            "source_environment": source_environment,
            "records": [asdict(item) for item in ordered],
        }
        return cls(
            source_environment=source_environment,
            records=ordered,
            digest=canonical_digest(unsigned),
        )

    def verify(self) -> None:
        if (
            self.build(
                source_environment=self.source_environment, records=self.records
            ).digest
            != self.digest
        ):
            raise ValueError("long-context snapshot digest mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "evomembench_long_context_snapshot_v0.1.0",
            "source_environment": self.source_environment,
            "records": [asdict(item) for item in self.records],
            "digest": self.digest,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path: str | Path) -> "LongContextSnapshot":
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != "evomembench_long_context_snapshot_v0.1.0":
            raise ValueError("unsupported long-context snapshot schema")
        snapshot = cls(
            source_environment=str(payload["source_environment"]),
            records=tuple(HistoryItem(**row) for row in payload["records"]),
            digest=str(payload["digest"]),
        )
        snapshot.verify()
        return snapshot


def trajectory_history_item(
    context: ToolSampleContext, trajectory: Sequence[Mapping[str, Any]]
) -> HistoryItem:
    """Store this episode once; strip injected system history to prevent recursion."""
    online = [
        dict(message)
        for message in trajectory
        if str(message.get("role", "")).casefold() != "system"
    ]
    text = "\n".join(
        [
            f"Prior task {context.sample_id} (environment={context.environment}):",
            context.task_signature,
            "Observed trajectory:",
            json.dumps(online, ensure_ascii=False, default=str),
        ]
    )
    return HistoryItem(
        episode_uid=context.episode_uid,
        ordinal=context.ordinal,
        text=text,
    )


class CrossEpToolLongContextMemory:
    """Official Tool Memory duck type that injects every eligible source record."""

    arm = "long_context"

    def __init__(
        self,
        *,
        source_environment: str,
        count_tokens: Any,
        context_window_tokens: int,
        reserved_generation_tokens: int,
        safety_tokens: int = 256,
        readonly: bool = False,
        snapshot: LongContextSnapshot | None = None,
    ) -> None:
        if snapshot is not None:
            snapshot.verify()
            if snapshot.source_environment != source_environment:
                raise ValueError("long-context snapshot environment mismatch")
            if not readonly:
                raise ValueError("frozen long-context snapshots must be readonly")
        self.source_environment = source_environment
        self.count_tokens = count_tokens
        self.context_window_tokens = context_window_tokens
        self.reserved_generation_tokens = reserved_generation_tokens
        self.safety_tokens = safety_tokens
        self.readonly = readonly
        self.snapshot = snapshot
        self.records = list(snapshot.records if snapshot else ())
        self.receipts: list[dict[str, Any]] = []
        self._tls = threading.local()
        self._lock = threading.RLock()

    def set_sample_context(self, context: ToolSampleContext) -> None:
        if context.ordinal < 0:
            raise ValueError("sample ordinal must be non-negative")
        self._tls.context = context

    def set_base_prompt_tokens(self, value: int) -> None:
        if value < 0:
            raise ValueError("base prompt tokens must be non-negative")
        self._tls.base_prompt_tokens = int(value)

    def _context(self) -> ToolSampleContext:
        context = getattr(self._tls, "context", None)
        if context is None:
            raise RuntimeError("set_sample_context must run before utilize/update")
        return context

    def begin_sample(self) -> None:
        self._tls.usage = MemoryUsageLog()

    def set_finalized_preview(self, materialized: LongContextMaterialization) -> None:
        """Bind the runner's exact chat-template/tool-schema overflow result once."""
        self._tls.finalized_preview = materialized

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

    def preview(self) -> LongContextMaterialization:
        context = self._context()
        cutoff = (
            max((record.ordinal for record in self.records), default=-1) + 1
            if self.readonly
            else context.ordinal
        )
        with self._lock:
            visible = tuple(
                record for record in self.records if record.ordinal < cutoff
            )
        return materialize_long_context(
            visible,
            count_tokens=self.count_tokens,
            base_prompt_tokens=int(getattr(self._tls, "base_prompt_tokens", 0)),
            context_window_tokens=self.context_window_tokens,
            reserved_generation_tokens=self.reserved_generation_tokens,
            safety_tokens=self.safety_tokens,
        )

    def _record(self, materialized: LongContextMaterialization) -> None:
        context = self._context()
        self.receipts.append(
            {
                "schema_version": "evomembench_long_context_receipt_v0.1.0",
                "sample_id": context.sample_id,
                "source_environment": self.source_environment,
                "target_environment": context.environment,
                "arm": self.arm,
                "selected_episode_uids": list(materialized.episode_uids),
                "selected_ordinals": list(materialized.ordinals),
                "history_episodes": len(materialized.episode_uids),
                "history_tokens": materialized.history_tokens,
                "history_bytes": materialized.history_bytes,
                "base_prompt_tokens": materialized.base_prompt_tokens,
                "projected_input_tokens": materialized.projected_input_tokens,
                "answer_prompt_tokens": (
                    materialized.projected_input_tokens
                    - materialized.reserved_generation_tokens
                    - materialized.safety_tokens
                ),
                "answer_prompt_token_source": "live_vllm_chat_template_with_tools",
                "context_window_tokens": materialized.context_window_tokens,
                "reserved_generation_tokens": materialized.reserved_generation_tokens,
                "status": materialized.status,
                "readonly": self.readonly,
                "snapshot_digest": None
                if self.snapshot is None
                else self.snapshot.digest,
                "latency_ms": materialized.assembly_ms,
                "injection_sha256": hashlib.sha256(
                    materialized.text.encode()
                ).hexdigest(),
                "probes": {
                    "mode": "complete_eligible_raw_history",
                    "vector_search": False,
                    "graph_traversal": False,
                    "relation_scan_or_filter": False,
                    "truncation": False,
                },
            }
        )

    def record_overflow(self, materialized: LongContextMaterialization) -> None:
        if not materialized.overflow:
            raise ValueError("record_overflow requires an overflowing materialization")
        self._record(materialized)

    def utilize(self, _upstream_query: str) -> str:
        usage = self._usage()
        materialized = getattr(self._tls, "finalized_preview", None)
        if materialized is None:
            materialized = self.preview()
        else:
            del self._tls.finalized_preview
        usage.latency_s += materialized.assembly_ms / 1000
        usage.n_utilize += 1
        self._record(materialized)
        if materialized.overflow:
            raise LongContextOverflow(
                f"full history needs {materialized.projected_input_tokens} tokens; "
                f"window is {materialized.context_window_tokens}"
            )
        return materialized.text

    def update(self, trajectory: list[dict]) -> None:
        usage = self._usage()
        usage.n_update += 1
        if self.readonly:
            return
        context = self._context()
        if context.environment != self.source_environment:
            raise RuntimeError(
                "long-context source bank received cross-environment write"
            )
        began = time.perf_counter()
        item = trajectory_history_item(context, trajectory)
        with self._lock:
            expected = len(self.records)
            if item.ordinal != expected:
                raise RuntimeError(
                    f"long-context append order violation: expected {expected}, got {item.ordinal}"
                )
            self.records.append(item)
        usage.latency_s += time.perf_counter() - began

    def freeze(self) -> LongContextSnapshot:
        with self._lock:
            return LongContextSnapshot.build(
                source_environment=self.source_environment,
                records=tuple(self.records),
            )

    def assert_snapshot_unchanged(self) -> None:
        if self.snapshot is None:
            raise RuntimeError("backend was not opened from a frozen snapshot")
        if self.freeze().digest != self.snapshot.digest:
            raise RuntimeError("readonly long-context source bank was mutated")

    def close(self) -> None:
        return None
