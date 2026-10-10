"""Figure 1 semantic-operator layer: LLM ops behind a PG-resident cache (ADR-0026).

- :mod:`llm` — OpenAI-compatible client and a deterministic mock.
- :mod:`cache` — ``gem_sem_cache`` + the cost meter.
- :mod:`ops` — sem_extract / filter / match / conflict / rank / topk / summarize.
- :mod:`pending` — the pending set Q (``gem_pending``).
- :mod:`planner` — predicate placement via ``tridb_place_sem`` (or its Python spec).
- :mod:`stream` — streaming semantic filter over a server-side portal.
"""

from bench.agent_memory.semantic.cache import SemCache, SemMeter, ensure_schema
from bench.agent_memory.semantic.llm import LLMReply, MockLLM, OpenAIChatClient
from bench.agent_memory.semantic.ops import CacheMiss, SemOps
from bench.agent_memory.semantic.pending import PendingItem, PendingQueue
from bench.agent_memory.semantic.planner import Placement, PredicateLeg, SemPlanner
from bench.agent_memory.semantic.stream import StreamReceipt, sem_filter_stream

__all__ = [
    "CacheMiss",
    "LLMReply",
    "MockLLM",
    "OpenAIChatClient",
    "PendingItem",
    "PendingQueue",
    "Placement",
    "PredicateLeg",
    "SemCache",
    "SemMeter",
    "SemOps",
    "SemPlanner",
    "StreamReceipt",
    "ensure_schema",
    "sem_filter_stream",
]
