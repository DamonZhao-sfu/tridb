"""Ingest strategies — Omri et al.'s four construction forms.

The taxonomy is a SETTING here, not four systems: each module below fills
``ingest``'s strategy slot and nothing else changes. That is the design thesis
of docs/agent_memory_gem_interface_v0.1.0.md §1.

    absent          Paradigm I    — no strategy; the long-context arm (M3)
    deterministic   Paradigm II   — chunk/index only, no LLM  (embedRAG)
    llm_mediated    Paradigm III  — LLM as a fixed extractor, batch|sequential
    agentic         Paradigm IV   — LLM-controlled writes, hard-capped
"""

from bench.agent_memory.gem.strategies.agentic import AgenticIngestStrategy
from bench.agent_memory.gem.strategies.deterministic import DeterministicIngestStrategy
from bench.agent_memory.gem.strategies.llm_mediated import (
    LLMMediatedIngestStrategy,
    OpenAIChatExtractor,
)

__all__ = [
    "AgenticIngestStrategy",
    "DeterministicIngestStrategy",
    "LLMMediatedIngestStrategy",
    "OpenAIChatExtractor",
]
