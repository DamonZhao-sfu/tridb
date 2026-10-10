"""The Experience Graph sub-model of GEM, per arXiv:2606.29823.

A sibling of :mod:`bench.agent_memory.gem`, not a replacement. GEM's ``SemanticUnit``
carries *knowledge*; this package carries *search experience* — attempts, rewards,
failures and causal lineage — which the paper treats as a distinct, reward-bearing
layer. Distilled knowledge derived from an experience path still lands in ``gem_unit``,
linked back by provenance.
"""

from bench.agent_memory.gem_eg.store import (
    EG_EDGE_TYPES,
    EgStore,
    LoadReceipt,
)

__all__ = ["EG_EDGE_TYPES", "EgStore", "LoadReceipt"]
