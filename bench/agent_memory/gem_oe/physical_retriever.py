"""GEM retrieval through the Python physical executor (`--physical-plan`).

Kept out of `retrievers.py` on purpose (ADR-0023 gate 6): the formal native-fused
path must never import the Python plan executor, so the only module that does is
this one, and `run_arm` imports it only when a cell asks for `--physical-plan`.
"""

from __future__ import annotations

from dataclasses import dataclass

from bench.agent_memory.gem_eg.oracle import QuerySpec
from bench.agent_memory.gem_eg.physical import PhysicalExecutor, PhysicalPlan
from bench.agent_memory.gem_eg.query import EngineResult
from bench.agent_memory.gem_oe.retrievers import GemRetriever


@dataclass
class PhysicalGemRetriever(GemRetriever):
    """Node-seeded GEM retrieval executed by one named Python physical plan."""

    #: `vfwd` / `rrev` / `aivg`; None keeps the base retriever's behaviour.
    physical_plan: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.physical_plan is None:
            return
        if self.native_fused_plan is not None:
            raise ValueError("physical_plan and native_fused_plan are exclusive")
        self.physical_plan = PhysicalPlan.parse(self.physical_plan).value
        assert self._engine is not None
        PhysicalExecutor(self._engine).ensure_ready(self.physical_plan)

    def _execute(self, spec: QuerySpec) -> EngineResult:
        if self.physical_plan is None:
            return super()._execute(spec)
        assert self._engine is not None
        return self._engine.run(spec, self.knobs, self.physical_plan)
