"""Joint trajectory plan/configuration optimization (quality and LLM tokens).

Imports are deliberately lightweight: importing this package starts no model,
DB connection, background thread, or experiment.
"""

from trimem.optimizer.schema import CandidateSpec

__all__ = ["CandidateSpec"]
