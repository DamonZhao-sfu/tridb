"""Paradigm II construction: chunk, embed, index. No LLM.

This is TriDB's CURRENT behaviour moved behind the GEM interface, and that is
the whole point of it. Milestone G2's gate is:

    with reinforce=False, revise off and forget off, a GEM run must reproduce
    the existing embedRAG retrieval numbers.

If those numbers move when nothing semantic has changed, every measurement
taken afterwards is uninterpretable — so this strategy is deliberately the
dullest code in the package, and it shares
:class:`~bench.agent_memory.tridbBackend.chunking.TiktokenSentenceChunker` with the old
pipeline rather than reimplementing chunking.

**Embedding source.** Here ``gem_unit.embedding`` is the CHUNK TEXT vector, not
a title+summary vector as in the LLM-mediated strategies. Two runs with
different ``embedding_source`` are different operating points, so the strategy
records the value in ``gem_unit.metadata`` and nothing may compare across it
silently.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.protocols import MemoryView
from bench.agent_memory.gem.types import InteractionEvent

#: The MemoryAgentBench streaming protocol's construction unit.
DEFAULT_CHUNK_TOKENS = 4096
DEFAULT_BATCH_SIZE = 64

EMBEDDING_SOURCE = "chunk_text"


class DeterministicIngestStrategy:
    """One unit per chunk, a single ``content`` field, no edges, no LLM.

    Implements the ``DeterministicIngest`` Protocol. ``plan`` is pure: it reads
    nothing from the view and calls no model, so the operator's transaction
    stays short and the construction cost is exactly chunking plus one batched
    embed issued by the operator.
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        chunker: Any | None = None,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.batch_size = batch_size
        # Lazily constructed: TiktokenSentenceChunker imports tiktoken + nltk,
        # which are optional extras (requirements-agent-memory.txt). Tests inject
        # a fake, and the unit suite must not need the extras installed.
        self._chunker = chunker
        self.cost: dict[str, int] = {}

    @property
    def chunker(self) -> Any:
        if self._chunker is None:
            from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker

            self._chunker = TiktokenSentenceChunker(chunk_size=self.chunk_tokens)
        return self._chunker

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> list[Mapping[str, Any]]:
        """Chunk each event; emit one unit + one ``content`` field per chunk.

        ``view`` is accepted and ignored — deterministic construction is by
        definition blind to existing state, which is exactly [GEM]'s Failure ②
        (an updated fact is appended beside the outdated one) and why this
        configuration is NOT GEM-conformant. Keeping it faithful is the point.
        """
        ops: list[Mapping[str, Any]] = []
        for event in events:
            chunks = self.chunker.chunk(event.content)
            for position, chunk in enumerate(chunks):
                ref = f"{event.external_id}#{position}"
                ops.append(
                    planmod.upsert_unit(
                        scope_id=event.scope_id,
                        # The chunk IS the topic here; there is no extraction to
                        # name one, so the external id keeps units addressable
                        # and the (scope_id, title) unique constraint honest.
                        title=ref,
                        summary="",
                        ref=ref,
                        embed_text=chunk,
                        metadata={
                            "embedding_source": EMBEDDING_SOURCE,
                            "strategy": self.name,
                            "external_id": event.external_id,
                            "session_id": event.session_id,
                            "role": event.role,
                            "kind": event.kind,
                            "event_time": event.event_time,
                            "event_order": event.event_order,
                            "chunk_index": position,
                            **dict(event.metadata),
                        },
                    )
                )
                ops.append(
                    planmod.append_field_value(
                        ref=ref,
                        field="content",
                        value=chunk,
                        valid_from=event.event_time or "-infinity",
                        supersede_current=False,
                        provenance={
                            "source_external_ids": [event.external_id],
                            "operator": "ingest",
                        },
                    )
                )
        return ops
