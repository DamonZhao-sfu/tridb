"""The operating-point matrix — [AM]'s four paradigms as GEM settings.

[AM] measures nine memory *systems*; this repository has one. What it can
reproduce honestly is the paper's own explanation for the spread: paradigm
membership. ``memory.py``'s docstring already states the mapping, and this
module makes it executable — every row below changes ``ingest``'s strategy slot
and the retrieval mode, and nothing else.

===========================  ========  ======  ======  ======  =========
[AM] system / paradigm       ingest    mode    revise  forget  reinforce
===========================  ========  ======  ======  ======  =========
II embedRAG                  determ.   VECTOR  off     off     off
III.a GraphRAG-like          llm/batch FUSED   off     off     off
III.b Mem0-like              llm/seq   VECTOR  on      off     off
IV agentic                   agentic   FUSED   on      on      off
**GEM-conformant**           determ.   FUSED   **on**  **on**  **on**
===========================  ========  ======  ======  ======  =========

**These are proxies, not the systems.** A row labelled ``IIIa_graphrag_like``
reproduces GraphRAG's *cost shape* — one extraction call per chunk, batched
embedding, a graph-bearing retrieval leg — not GraphRAG. Numbers from this
harness belong beside [AM]'s paradigm claims (Insight 1, 2, 8), never beside its
per-system bars. Every emitted record carries ``paradigm_proxy: true`` so the
distinction cannot be lost downstream.

The GEM-conformant row deliberately shares Paradigm II's ingest strategy. Its
delta against ``II_embedrag`` is then exactly the cost of governance —
reinforcement, revision, forgetting — with construction held identical, which is
the one comparison [AM]'s Table 1 has no row for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bench.agent_memory.gem.types import RetrievalMode, RetrievalRoute

INGEST_DETERMINISTIC = "deterministic"
INGEST_LLM_BATCH = "llm_batch"
INGEST_LLM_SEQUENTIAL = "llm_sequential"
INGEST_AGENTIC = "agentic"

#: [AM] Recommendation 10: an LLM-bounded phase without an external cap has no
#: worst case. These are the caps, stated here rather than defaulted inside the
#: strategy, and they travel into every manifest.
DEFAULT_AGENTIC_MAX_ROUNDS = 8
DEFAULT_AGENTIC_MAX_TOOL_CALLS = 24


@dataclass(frozen=True)
class OperatingPoint:
    key: str
    paradigm: str
    label: str
    ingest: str
    mode: RetrievalMode
    route: RetrievalRoute = RetrievalRoute.TOPIC
    reinforce: bool = False
    revise: bool = False
    forget: bool = False

    @property
    def uses_llm_construction(self) -> bool:
        return self.ingest != INGEST_DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "paradigm": self.paradigm,
            "label": self.label,
            "ingest_strategy": self.ingest,
            "retrieval_mode": self.mode.value,
            "retrieval_route": self.route.value,
            "reinforce": self.reinforce,
            "revise": self.revise,
            "forget": self.forget,
            "paradigm_proxy": True,
            "proxy_note": (
                "A GEM setting reproducing this paradigm's cost shape on TriDB. "
                "Not the [AM] system named in the label."
            ),
        }


POINTS: tuple[OperatingPoint, ...] = (
    OperatingPoint(
        key="II_embedrag",
        paradigm="II",
        label="embedRAG-like (deterministic ingest, vector retrieval)",
        ingest=INGEST_DETERMINISTIC,
        mode=RetrievalMode.VECTOR,
    ),
    OperatingPoint(
        key="IIIa_graphrag_like",
        paradigm="III.a",
        label="GraphRAG-like (batched LLM extraction, fused retrieval)",
        ingest=INGEST_LLM_BATCH,
        mode=RetrievalMode.FUSED,
    ),
    OperatingPoint(
        key="IIIb_mem0_like",
        paradigm="III.b",
        label="Mem0-like (sequential LLM extraction, vector retrieval)",
        ingest=INGEST_LLM_SEQUENTIAL,
        mode=RetrievalMode.VECTOR,
        revise=True,
    ),
    OperatingPoint(
        key="IV_agentic",
        paradigm="IV",
        label="Agentic (LLM-controlled writes, capped)",
        ingest=INGEST_AGENTIC,
        mode=RetrievalMode.FUSED,
        revise=True,
        forget=True,
    ),
    OperatingPoint(
        key="gem_conformant",
        paradigm="GEM",
        label="GEM-conformant (deterministic ingest + governance on)",
        ingest=INGEST_DETERMINISTIC,
        mode=RetrievalMode.FUSED,
        reinforce=True,
        revise=True,
        forget=True,
    ),
)

POINTS_BY_KEY = {point.key: point for point in POINTS}


def resolve(keys: list[str] | None) -> list[OperatingPoint]:
    if not keys:
        return list(POINTS)
    resolved = []
    for key in keys:
        if key not in POINTS_BY_KEY:
            raise ValueError(
                f"unknown operating point {key!r}; choose from {sorted(POINTS_BY_KEY)}"
            )
        resolved.append(POINTS_BY_KEY[key])
    return resolved


def build_strategy(
    point: OperatingPoint,
    *,
    extractor: Any,
    embedder: Any,
    construction_model: str,
    chunk_tokens: int,
    embedding_batch_size: int,
    max_rounds: int = DEFAULT_AGENTIC_MAX_ROUNDS,
    max_tool_calls: int = DEFAULT_AGENTIC_MAX_TOOL_CALLS,
) -> Any:
    """Instantiate the ingest strategy this operating point names.

    A fresh strategy per history: ``cost``, ``rejections``, ``capped`` and the
    agentic ``transcript`` are per-invocation accumulators, and reusing one
    instance across histories would silently pool five histories' construction
    cost into the last one's transition.
    """
    from bench.agent_memory.gem.strategies import (
        AgenticIngestStrategy,
        DeterministicIngestStrategy,
        LLMMediatedIngestStrategy,
    )
    from bench.agent_memory.gem.strategies.llm_mediated import (
        MODE_BATCH,
        MODE_SEQUENTIAL,
    )

    if point.ingest == INGEST_DETERMINISTIC:
        return DeterministicIngestStrategy(
            chunk_tokens=chunk_tokens,
            batch_size=embedding_batch_size,
        )
    if point.ingest == INGEST_LLM_BATCH:
        return LLMMediatedIngestStrategy(
            client=extractor,
            model=construction_model,
            mode=MODE_BATCH,
            chunk_tokens=chunk_tokens,
        )
    if point.ingest == INGEST_LLM_SEQUENTIAL:
        return LLMMediatedIngestStrategy(
            client=extractor,
            model=construction_model,
            mode=MODE_SEQUENTIAL,
            # III.b embeds each extracted fact BEFORE its similarity search
            # resolves ADD/UPDATE — the strategy refuses to run without an
            # embedder rather than degrading into III.a under a III.b label.
            embedder=embedder,
            chunk_tokens=chunk_tokens,
        )
    if point.ingest == INGEST_AGENTIC:
        return AgenticIngestStrategy(
            client=extractor,
            model=construction_model,
            max_rounds=max_rounds,
            max_tool_calls=max_tool_calls,
            chunk_tokens=chunk_tokens,
        )
    raise ValueError(f"unknown ingest strategy {point.ingest!r}")
