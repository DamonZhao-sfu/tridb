"""LongMemEval_S* analogues of [AM] §4.1, §4.2, §4.7 and §4.8.

[AM] is arXiv:2606.06448, "Agent Memory: Characterization and System
Implications of Stateful Long-Horizon Workloads". It characterizes nine memory
systems; this package reproduces the three sections that a single-system
repository can honestly reproduce, using GEM's ingest-strategy slot to stand in
for the paper's four construction paradigms:

    §4.1  per-query serving latency (retrieval + generation) vs accuracy,
          construction excluded. The long-context arm is NOT run.
    §4.2  construction / retrieval / generation phase split, lifecycle calls,
          tokens, GPU energy, and joules per correct answer.
    §4.8  effective TTFT structure and QA tail width (p95/p50).

    §4.7  complete-session 64K..1M construction, footprint and retrieval
          scaling, run separately by :mod:`.scaling`.

Everything that is not a memory system — workload loading, streaming
generation, TTFT timing, judging — comes from :mod:`bench.agent_memory.serving`,
which the embedRAG pipeline also imports, so the two arms cannot drift apart.
"""

from bench.agent_memory.gem_bench.points import POINTS, POINTS_BY_KEY, OperatingPoint

__all__ = ["POINTS", "POINTS_BY_KEY", "OperatingPoint"]
