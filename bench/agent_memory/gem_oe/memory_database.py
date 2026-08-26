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

import hashlib
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


#: Lines that carry no identity: nearly every program in this corpus opens with the
#: same imports and a banner comment, so a window anchored on them would match a
#: prompt that never contained the injected program.
SKIP_PREFIXES = ("#", "import ", "from ") + tuple(q * 3 for q in ('"', "'"))


def code_fingerprint(code: str, *, lines: int = 3, chars: int = 120) -> str:
    """A CONTIGUOUS slice of real code, for locating it inside a rendered prompt.

    Contiguous is the whole point, and the first version got it wrong: it FILTERED
    uninteresting lines and joined what was left, producing a string that never
    appears anywhere in the original -- `def f():` welded onto a docstring body with
    the intervening quote line removed. The gate then reported 100% of injections
    absent from a prompt that visibly contained every one of them, which is exactly
    the false alarm a hard gate must never raise.

    So: skip forward to the first line that carries identity, then take the window
    unmodified. The result is a substring of `code`, which `assert_is_substring`
    below is the standing proof of.
    """
    rows = code.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(rows)
            if line.strip() and not line.lstrip().startswith(SKIP_PREFIXES)
        ),
        0,
    )
    return "\n".join(rows[start : start + lines])[:chars]


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
        policy: str = "fixed",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if policy not in {"fixed", "match_baseline"}:
            raise ValueError(f"unknown injection policy: {policy}")
        self._retriever = retriever
        self._task_uid = task_uid
        self._policy = policy
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

        # Two defensible policies, and the choice changes what is being measured.
        #
        # `match_baseline` -- render exactly as many inspirations as arm A would, and
        #   swap the first n for retrieved ones. Prompt length is identical across
        #   arms, so no outcome difference can be blamed on context length. The cost
        #   is real and was measured: with 5 islands and a nearly empty database the
        #   base class returns ZERO own inspirations, so arm B injects nothing and is
        #   byte-identical to arm A for exactly the early iterations where memory
        #   should matter most.
        #
        # `fixed` -- render up to `budget` retrieved programs regardless of how many
        #   the run itself can offer. Arm B's prompt is longer while the islands are
        #   sparse. That asymmetry IS the treatment (memory supplies context the run
        #   does not have), which is also how AlphaEvolve's own "No context in the
        #   prompt" ablation is framed; it is reported per-iteration rather than
        #   assumed away.
        if self._policy == "match_baseline":
            n_replace = min(len(injected), len(own))
            merged = injected[:n_replace] + own[n_replace:]
        elif self._policy == "fixed":
            keep = max(0, len(own) - len(injected))
            merged = injected + own[len(own) - keep:] if keep else injected + []
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"unknown injection policy: {self._policy}")
        rendered_external = [p.id for p in merged if is_external(p.id)]

        self.injection_trace.append(
            {
                "iteration": self._iteration,
                "arm": self._retriever.name,
                "parent_id": parent.id,
                "requested": wanted,
                "rendered": len(merged),
                "budget": budget,
                "policy": self._policy,
                "retrieved": len(retrieved),
                # What actually reached the prompt, NOT what was retrieved. Recording
                # the retrieved count here reported "injected=5, rendered=0" for three
                # straight iterations while arm B was byte-identical to arm A.
                "injected_ids": rendered_external,
                # The corpus uids behind those ids. The prompt renders the CODE, not
                # the id, so the assertion in gate_injection needs both.
                "injected_uids": [
                    p.metadata.get("uid", "") for p in merged if is_external(p.id)
                ],
                # What the PROMPT can actually be searched for. OpenEvolve's
                # INSPIRATION_PROGRAM_TEMPLATE renders {program_snippet} and a score
                # -- never the program id -- so an assertion that greps for the id
                # fails on a run where injection worked perfectly.
                "injected_fingerprints": [
                    code_fingerprint(p.code) for p in merged if is_external(p.id)
                ],
                # Exact identity of what was injected. The agent-run analogue of
                # `groundtruth.is_trivial_hit()`: that helper asks whether a retrieved
                # candidate already exists in the TARGET SESSION, which cannot be
                # asked here because the target session is this run and is not in the
                # corpus. What can be asked is whether the program the agent produced
                # is byte-identical to one it was handed, and that needs the hash.
                "injected_sha256": [
                    hashlib.sha256(p.code.encode("utf-8")).hexdigest()
                    for p in merged
                    if is_external(p.id)
                ],
                "own_ids": [p.id for p in merged if not is_external(p.id)],
                "provenance": [item.provenance for item in retrieved],
                "error": error,
            }
        )
        return parent, merged

    # -- registration ----------------------------------------------------

    @staticmethod
    def external_id(uid: str) -> str:
        """A filesystem-safe id for a corpus node uid.

        `ProgramDatabase.save()` writes one file per program id, so an id is a
        FILENAME. Corpus uids look like
        `349117b0:openevolve_native/circle_packing_..._056bfb#945f52e4-...`, whose `/`
        is read as a directory separator -- checkpointing died with FileNotFoundError
        on a path that had never been created. Hashing keeps the id short, safe and
        stable; the real uid travels in `metadata["uid"]` and in the injection trace,
        so nothing is lost.
        """
        digest = hashlib.sha1(uid.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return f"{EXTERNAL_PREFIX}{digest}"

    def _register(self, item: RetrievedProgram) -> Program:
        """Put a retrieved program where the worker can resolve it, and nowhere else."""
        pid = self.external_id(item.uid)
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
            metadata={
                EXTERNAL_FLAG: True,
                "uid": item.uid,
                "provenance": item.provenance,
            },
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
