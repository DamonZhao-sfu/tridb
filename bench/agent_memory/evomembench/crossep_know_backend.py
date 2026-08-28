"""GEM adapter for CrossEp-Know's official ``retrieve/extract`` interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import time
from typing import Any, Callable

from bench.agent_memory.evomembench.extractor import extract_observed_trajectory
from bench.agent_memory.evomembench.injection import fit_injection
from bench.agent_memory.evomembench.load import admit_extracted_experience
from bench.agent_memory.evomembench.modeling import experience_query
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import RetrievalMode


@dataclass(frozen=True)
class KnowledgeSampleContext:
    task_id: str
    episode_uid: str
    ordinal: int
    category: str
    subcategory: str
    task_signature: str


class CrossEpKnowGemMemory:
    """One context-isolated GEM bank with the upstream backend surface."""

    def __init__(
        self,
        *,
        memory: TriDBGovernedMemory,
        scope_id: str,
        count_tokens: Callable[[str], int],
        query_mode: str = "gem_task_seed",
        retrieval_mode: RetrievalMode = RetrievalMode.FUSED,
        top_k: int = 10,
        hops: int = 2,
        m_seeds: int = 4,
        term_cond: int = 32,
        token_budget: int = 4096,
    ) -> None:
        if query_mode not in {"upstream_parity", "gem_task_seed"}:
            raise ValueError(f"unknown CrossEp-Know query mode: {query_mode}")
        self.memory = memory
        self.scope_id = scope_id
        self.count_tokens = count_tokens
        self.query_mode = query_mode
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.hops = hops
        self.m_seeds = m_seeds
        self.term_cond = term_cond
        self.token_budget = token_budget
        self._context: KnowledgeSampleContext | None = None
        self._next_ordinal = 0
        self.memory_bank: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []

    def set_sample_context(self, context: KnowledgeSampleContext) -> None:
        if context.ordinal != self._next_ordinal:
            raise ValueError(
                f"context order violation: expected ordinal {self._next_ordinal}, got {context.ordinal}"
            )
        self._context = context

    def retrieve(self, query: str) -> tuple[str, dict[str, Any]]:
        context = self._context
        cutoff = self._next_ordinal if context is None else context.ordinal
        query_text = (
            query
            if self.query_mode == "upstream_parity"
            else self._require_context().task_signature
        )
        started = time.perf_counter()
        if cutoff == 0:
            return "", {
                "latency_s": 0.0,
                "embed_usage": {},
                "termination_reason": "empty_snapshot",
            }
        result = self.memory.retrieve(
            experience_query(
                scope_id=self.scope_id,
                task_signature=query_text,
                cutoff_ordinal=cutoff,
                mode=self.retrieval_mode,
                top_k=self.top_k,
                hops=self.hops,
                m_seeds=self.m_seeds,
                term_cond=self.term_cond,
                reinforce=False,
            )
        )
        if not result.committed:
            raise RuntimeError(result.aborted_reason)
        unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
        rows = self.memory.store.conn.execute(
            "SELECT u.id, u.title, (u.metadata->>'experience_ordinal')::integer,"
            " fv.value FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id=u.id"
            " WHERE u.id=ANY(%s) AND u.scope_id=%s AND fv.field='memory_payload'"
            " AND fv.valid_to IS NULL ORDER BY array_position(%s::bigint[], u.id)",
            (unit_ids, self.scope_id, unit_ids),
        ).fetchall()
        items = [
            {
                "unit_id": int(row[0]),
                "episode_uid": row[1],
                "ordinal": int(row[2]),
                "text": row[3],
            }
            for row in rows
        ]
        accepted, injection, injection_tokens = fit_injection(
            items,
            max_items=self.top_k,
            token_budget=self.token_budget,
            count_tokens=self.count_tokens,
        )
        latency = time.perf_counter() - started
        self.receipts.append(
            {
                "task_id": None if context is None else context.task_id,
                "cutoff_ordinal": cutoff,
                "query_mode": self.query_mode,
                "query_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
                "selected_unit_ids": [item["unit_id"] for item in accepted],
                "injection_tokens": injection_tokens,
                "probes": {
                    **dict(result.probes),
                    "experience_overflow_policy": "truncate_head_tail",
                    "truncated_experiences": sum(
                        bool(item.get("injection_truncated")) for item in accepted
                    ),
                },
            }
        )
        return (("\n\n" + injection) if injection else ""), {
            "latency_s": latency,
            "embed_usage": {
                "total_tokens": result.cost.embed_input_tokens,
                "calls": result.cost.embed_calls,
            },
            "probes": dict(result.probes),
        }

    def _require_context(self) -> KnowledgeSampleContext:
        if self._context is None:
            raise RuntimeError(
                "gem_task_seed requires set_sample_context before retrieve"
            )
        return self._context

    def extract(self, content: str, **kwargs: Any) -> dict[str, Any]:
        context = self._context
        if context is None:
            task_id = str(kwargs.get("task_id", f"ordinal-{self._next_ordinal}"))
            category = str(kwargs.get("context_category", ""))
            subcategory = str(kwargs.get("sub_category", ""))
            query = str(kwargs.get("query", ""))
            context = KnowledgeSampleContext(
                task_id=task_id,
                episode_uid=f"evomembench:crossep_know:episode:{task_id}",
                ordinal=self._next_ordinal,
                category=category,
                subcategory=subcategory,
                task_signature="\n".join(
                    [
                        "benchmark: CrossEp-Know",
                        f"category: {category}",
                        f"subcategory: {subcategory}",
                        f"user: {query}",
                    ]
                ),
            )
        started = time.perf_counter()
        extracted = extract_observed_trajectory(
            task_signature=context.task_signature,
            trajectory=[{"role": "observed_trajectory", "content": content}],
            concept_hints=[
                ("category", context.category),
                ("skill", context.subcategory),
            ],
            source_external_ids=[context.task_id],
        )
        valid_from = (
            datetime(2000, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=context.ordinal)
        ).isoformat()
        admission = admit_extracted_experience(
            self.memory,
            uid=context.episode_uid,
            scope_id=self.scope_id,
            ordinal=context.ordinal,
            task_signature=context.task_signature,
            extracted=extracted,
            valid_from=valid_from,
            metadata={
                "benchmark": "CrossEp-Know",
                "category": context.category,
                "subcategory": context.subcategory,
            },
        )
        if not admission.result.committed:
            raise RuntimeError(admission.result.aborted_reason)
        self.memory_bank.append(
            {"id": context.task_id, "unit_ids": list(admission.result.units)}
        )
        self._next_ordinal += 1
        self._context = None
        return {
            "latency_s": time.perf_counter() - started,
            "llm_usage": {},
            "embed_usage": {
                "total_tokens": admission.result.cost.embed_input_tokens,
                "calls": admission.result.cost.embed_calls,
            },
            "extractor_schema": extracted.extractor_schema,
            "extractor_capped": extracted.capped,
        }
