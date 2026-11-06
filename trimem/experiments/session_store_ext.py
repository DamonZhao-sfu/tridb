"""Scalar-quality evaluation for ``SessionProfileStore`` (environment datasets).

SWE-QA keeps the three-judge ``evaluate``; ALFWorld / ScienceWorld have a deterministic simulator
reward, so the row is evaluated from one native metric. Same table, same training view.
"""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from trimem.experiments.session_profile_store import SessionProfileStore


class ScalarSessionStore(SessionProfileStore):
    def evaluate_scalar(
        self,
        rid: str,
        quality: float,
        native: dict,
        baseline_quality: float | None = None,
    ) -> None:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be normalized to [0, 1]")
        gain = None if baseline_quality is None else quality - baseline_quality
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "UPDATE opt.session_retrieval SET status='evaluated', judge=%s, quality=%s, paired_gain=%s, "
                "updated_at=now() WHERE retrieval_id=%s AND generation IS NOT NULL RETURNING retrieval_id",
                (Jsonb({"kind": "simulator", **native}), quality, gain, rid),
            ).fetchone()
            if not row:
                raise ValueError("cannot score a missing or unfinished generation")
