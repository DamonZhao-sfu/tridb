"""Cross-session memory for OpenEvolve, injected through the inspiration channel.

WHY THIS EXISTS
---------------
Stock OpenEvolve has no cross-session memory at all. `_create_database_snapshot()`
serialises `self.database.programs`, and the worker resolves inspirations with
`[programs[pid] for pid in inspiration_ids if pid in programs]` -- so every reference
the model ever sees comes from the run that is currently executing. Each run starts
from nothing. That gap is the experiment: arm A keeps it, arms B and C fill it from a
memory system, and the only thing that differs between arms is where the references
came from.

REPLACEMENT, NOT ADDITION
-------------------------
`PromptSampler.build_prompt` renders `num_top_programs` (3) + `num_diverse_programs`
(2) programs from the current run before any memory is involved, so arm A is "context
from this run only", never "no context". If memory were APPENDED the prompt would grow,
and any outcome difference could be explained by context length instead of memory
quality. So `sample_from_island` returns the same COUNT it was asked for, with the
first `k` entries swapped for retrieved ones.

WHAT AN INJECTED PROGRAM MUST NEVER BECOME
------------------------------------------
A parent. If evolution could continue directly from another session's code, that is
seeding, not memory, and the arms stop being comparable. Injected programs are
therefore added to `self.programs` (the snapshot needs them, or the worker's
`if pid in programs` silently drops every one) and to nothing else -- not an island,
not the archive, not the MAP-Elites feature map, not `best_program_id`.

`ProgramDatabase._sample_random_parent()` picks uniformly from `self.programs`, which
is reachable from `sample()` when an island is empty, so that one path is overridden
rather than trusted. `sample_from_island` also asserts on the parent it returns: a
silent breach here would corrupt every downstream number.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from openevolve.database import Program, ProgramDatabase

logger = logging.getLogger(__name__)

from bench.agent_memory.gem_oe.constants import (  # noqa: E402
    EXTERNAL_FLAG,
    EXTERNAL_PREFIX,
    is_external,
)

__all__ = [
    "EXTERNAL_FLAG",
    "EXTERNAL_PREFIX",
    "GemMemoryDatabase",
    "NullRetriever",
    "RetrievedProgram",
    "Retriever",
    "is_external",
]


@dataclass(frozen=True)
class RetrievedProgram:
    """One historical attempt, as the memory system returned it."""

    uid: str
    code: str
    language: str = "python"
    metrics: dict[str, float] = field(default_factory=dict)
    changes_description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """What an arm must implement. One call per evolution iteration."""

    #: Short arm label recorded in every trace row (e.g. "gem", "polyglot", "none").
    name: str

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]: ...


class NullRetriever:
    """Arm A. Returns nothing, so `sample_from_island` degrades to stock behaviour."""

    name = "none"

    def retrieve(
        self, *, task_uid: str, parent: Program, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        return []


class GemMemoryDatabase(ProgramDatabase):
    """`ProgramDatabase` whose inspirations may come from another session.

    Everything else -- islands, archive, MAP-Elites, migration, best-program tracking,
    checkpointing -- is inherited untouched, so arm A and arms B/C differ in exactly
    one place.
    """

    def __init__(
        self,
        config: Any,
        *,
        retriever: Retriever,
        task_uid: str,
        max_injected: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._retriever = retriever
        self._task_uid = task_uid
        #: How many of the requested inspirations to replace. None = all of them.
        self._max_injected = max_injected
        self._external_ids: set[str] = set()
        #: One row per iteration; the Phase 0 injection assertion reads this and checks
        #: each id actually reached the rendered prompt.
        self.injection_trace: list[dict[str, Any]] = []
        self._iteration = 0

    # -- the one overridden decision -------------------------------------

    def sample_from_island(
        self, island_id: int, num_inspirations: int | None = None
    ) -> tuple[Program, list[Program]]:
        parent, own = super().sample_from_island(island_id, num_inspirations)
        if is_external(parent.id):
            # Unreachable by construction; if it ever fires, the run is void rather
            # than quietly measuring a seeded agent.
            raise RuntimeError(
                f"injected program {parent.id} was selected as a parent; "
                "memory has become seeding and this run cannot be compared"
            )

        # `wanted` bounds the RETRIEVAL, not the returned count: asking the memory
        # system for more than the prompt can hold is wasted work.
        wanted = len(own) if num_inspirations is None else num_inspirations
        budget = wanted if self._max_injected is None else min(wanted, self._max_injected)
        self._iteration += 1

        retrieved: list[RetrievedProgram] = []
        error: str | None = None
        if budget > 0:
            try:
                retrieved = self._retriever.retrieve(
                    task_uid=self._task_uid,
                    parent=parent,
                    k=budget,
                    iteration=self._iteration,
                )[:budget]
            except Exception as exc:  # noqa: BLE001
                # Recorded, never swallowed into "memory had nothing useful": a
                # retrieval outage that silently degrades arm B into arm A is the
                # failure mode that makes a whole comparison meaningless.
                error = f"{type(exc).__name__}: {exc}"
                logger.error("retrieval failed at iteration %d: %s", self._iteration, error)
                raise

        injected = [self._register(item) for item in retrieved]
        # Replace, never append. The invariant is "the same NUMBER of inspirations arm
        # A would have rendered", not "the number the caller asked for" -- the base
        # class routinely returns fewer than `num_inspirations` (the island may hold
        # too few programs, and the parent is excluded), so slicing to `wanted` would
        # make arm B's prompt a different length from arm A's in exactly the cases
        # where the island is small. Swap the first n entries and keep the rest.
        n_replace = min(len(injected), len(own))
        merged = injected[:n_replace] + own[n_replace:]

        self.injection_trace.append(
            {
                "iteration": self._iteration,
                "arm": self._retriever.name,
                "parent_id": parent.id,
                "requested": wanted,
                "rendered": len(merged),
                "budget": budget,
                "injected_ids": [p.id for p in injected],
                "own_ids": [p.id for p in merged if not is_external(p.id)],
                "provenance": [item.provenance for item in retrieved],
                "error": error,
            }
        )
        return parent, merged

    # -- registration ----------------------------------------------------

    def _register(self, item: RetrievedProgram) -> Program:
        """Put a retrieved program where the worker can resolve it, and nowhere else."""
        pid = f"{EXTERNAL_PREFIX}{item.uid}"
        existing = self.programs.get(pid)
        if existing is not None:
            return existing
        program = Program(
            id=pid,
            code=item.code,
            language=item.language,
            changes_description=item.changes_description,
            metrics=dict(item.metrics),
            # generation/iteration_found stay at 0 and parent_id at None: these belong
            # to another run's history and must not read as this run's lineage.
            metadata={EXTERNAL_FLAG: True, "provenance": item.provenance},
        )
        # `self.programs` only. Deliberately NOT: self.islands, self.archive,
        # self.feature_map, self.best_program_id, self.island_best_programs.
        self.programs[pid] = program
        self._external_ids.add(pid)
        return program

    # -- guards ----------------------------------------------------------

    def _sample_random_parent(self) -> Program:
        """Base class picks uniformly from `self.programs`, which now holds externals.

        Reachable from `sample()` when an island is empty, so it is narrowed rather
        than trusted to be unreachable.
        """
        own = [pid for pid in self.programs if not is_external(pid)]
        if not own:
            raise ValueError("No programs available for sampling")
        return self.programs[random.choice(own)]

    # -- housekeeping ----------------------------------------------------

    @property
    def external_ids(self) -> set[str]:
        return set(self._external_ids)

    def own_programs(self) -> dict[str, Program]:
        """Everything this run actually produced -- what any fitness stat must use."""
        return {pid: p for pid, p in self.programs.items() if not is_external(pid)}
