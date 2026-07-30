"""The ``forget`` operator — graded attenuation by relevance.

    M_{t+1} = U(M_t, {}, F)

By RELEVANCE, never by age or capacity, at per-field granularity. The ladder,
applied to both ``gem_unit`` and ``gem_field_value``::

    s < theta_summary  -> compress history: keep first + current + N most
                          salient, archive the middle   (state='compressed')
    s < theta_remove   -> state='hidden'    (excluded by the retrieval predicate)
    s < theta_archive  -> state='archived'; tombstone the unit's edges; the row
                          REMAINS                        (C5 recoverable)

**Never DELETE.** C4 requires the provenance chain of anything still reachable
to survive, and C5 requires archived content to remain recoverable. There is no
``DELETE`` statement in this module and there must never be one.

**Decay is lazy** — computed here from ``last_access`` at tick time, never on
the read path. The consequence belongs in every run manifest: salience is only
current as of the last forget tick.

**Benchmark isolation.** C5 changes the corpus between queries while
LongMemEval/LoCoMo assume a fixed corpus per question, so in paradigm-faithful
[AM] reproductions ``forget`` is OFF and its effect is measured separately.
Never silently on.
"""

from __future__ import annotations

import time
from typing import Any

from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import GemStore, Tx
from bench.agent_memory.gem.types import ForgetResult, PhaseCost, StateDelta

#: How many of the most salient middle entries a compressed history keeps,
#: on top of the first and the current one.
DEFAULT_KEEP_SALIENT = 2


class ForgetOperator:
    """Walks the graded ladder for one scope."""

    def __init__(
        self,
        store: GemStore,
        *,
        salience: ExponentialSalience | None = None,
        keep_salient: int = DEFAULT_KEEP_SALIENT,
    ) -> None:
        self.store = store
        self.salience = salience or ExponentialSalience()
        self.keep_salient = keep_salient

    def forget(self, scope_id: str, *, now: str | None = None) -> ForgetResult:
        started = time.perf_counter()
        demoted: list[dict[str, Any]] = []
        try:
            with self.store.transition("forget", scope_id, "construction") as tx:
                tx.lock_writer()
                self._apply_decay(tx, scope_id, now)
                demoted += self._compress_histories(tx, scope_id)
                demoted += self._demote_units(tx, scope_id)
                demoted += self._demote_fields(tx, scope_id)

            # Captured AFTER the envelope closes: transition() samples the C5
            # counts, evaluates P_t, and writes the log during __exit__, so
            # anything read inside the body is stale.
            delta = tx.delta.freeze()
            cost = tx.meter.freeze(time.perf_counter() - started)
            transition_id = tx.transition_id
            policies = tuple(tx.policies_evaluated)
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            return ForgetResult(
                operator="forget",
                committed=False,
                delta=StateDelta(),
                cost=PhaseCost(
                    phase="construction", seconds=time.perf_counter() - started
                ),
                aborted_reason=f"{type(exc).__name__}: {exc}",
            )

        return ForgetResult(
            operator="forget",
            committed=True,
            delta=delta,
            cost=cost,
            transition_id=transition_id,
            policies_evaluated=policies,
            demoted=tuple(demoted),
        )

    # -- decay -------------------------------------------------------------

    def _apply_decay(self, tx: Tx, scope_id: str, now: str | None) -> None:
        """Fold elapsed idle time into salience, in SQL, for the whole scope.

        Done as one statement rather than row-by-row in Python: the decay is
        ``s * exp(-lam * seconds_idle)`` and PostgreSQL can evaluate that over
        the relation without shipping every salience value to the client.
        Units never accessed decay from ``created_at``.
        """
        reference = "%s::timestamptz" if now else "now()"
        params: list[Any] = [self.salience.lam]
        if now:
            params.append(now)
        tx.execute(
            "UPDATE gem_unit SET salience = salience * exp(-%s * GREATEST("
            f" EXTRACT(EPOCH FROM ({reference} - COALESCE(last_access, created_at)))"
            ", 0)) WHERE scope_id = %s AND state <> 'archived'",
            (*params, scope_id),
        )
        # Field salience decays on the parent's access clock: a field has no
        # last_access of its own, and inventing one would be a second, silently
        # different signal.
        tx.execute(
            "UPDATE gem_field_value fv SET salience = fv.salience * exp(-%s *"
            " GREATEST(EXTRACT(EPOCH FROM ("
            f" {reference} - COALESCE(u.last_access, u.created_at))), 0))"
            " FROM gem_unit u WHERE u.id = fv.unit_id AND u.scope_id = %s"
            "   AND fv.state <> 'archived'",
            (*params, scope_id),
        )

    # -- the ladder ---------------------------------------------------------

    def _compress_histories(self, tx: Tx, scope_id: str) -> list[dict[str, Any]]:
        """Below theta_summary: keep first + current + N most salient.

        The middle entries are ARCHIVED, not removed, so C4's provenance chain
        and C5's recoverability both survive compression.
        """
        rows = tx.execute(
            "WITH ranked AS ("
            "  SELECT fv.id, fv.unit_id, fv.field, fv.valid_to, fv.salience,"
            "    row_number() OVER (PARTITION BY fv.unit_id, fv.field"
            "      ORDER BY fv.valid_from, fv.id) AS oldest,"
            "    row_number() OVER (PARTITION BY fv.unit_id, fv.field"
            "      ORDER BY fv.salience DESC, fv.id) AS by_salience"
            "  FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
            "  WHERE u.scope_id = %s AND fv.state = 'active')"
            " UPDATE gem_field_value SET state = 'compressed' FROM ranked"
            " WHERE gem_field_value.id = ranked.id"
            "   AND ranked.salience < %s"
            "   AND ranked.oldest > 1"  # keep the first
            "   AND ranked.valid_to IS NOT NULL"  # keep the current
            "   AND ranked.by_salience > %s"  # keep the N most salient
            " RETURNING gem_field_value.id, gem_field_value.unit_id,"
            "   gem_field_value.field",
            (scope_id, self.salience.theta_summary, self.keep_salient),
        ).fetchall()
        return [
            {
                "kind": "compress_history",
                "value_id": int(row[0]),
                "unit_id": int(row[1]),
                "field": row[2],
                "to_state": "compressed",
            }
            for row in rows
        ]

    def _demote_units(self, tx: Tx, scope_id: str) -> list[dict[str, Any]]:
        """Hide, then archive. Archiving tombstones the unit's edges.

        The row REMAINS in both cases — hidden and archived units are excluded
        by the retrieval predicate, not deleted, so an explicit lookup still
        recovers them (C5).
        """
        demoted: list[dict[str, Any]] = []

        hidden = tx.execute(
            "UPDATE gem_unit SET state = 'hidden' WHERE scope_id = %s"
            "  AND state = 'active' AND salience < %s AND salience >= %s"
            " RETURNING id, salience",
            (scope_id, self.salience.theta_remove, self.salience.theta_archive),
        ).fetchall()
        for row in hidden:
            demoted.append(
                {
                    "kind": "demote_unit",
                    "unit_id": int(row[0]),
                    "salience": float(row[1]),
                    "to_state": "hidden",
                }
            )
        tx.delta.units_updated += len(hidden)

        archived = tx.execute(
            "UPDATE gem_unit SET state = 'archived' WHERE scope_id = %s"
            "  AND state <> 'archived' AND salience < %s"
            " RETURNING id, salience",
            (scope_id, self.salience.theta_archive),
        ).fetchall()
        for row in archived:
            unit_id = int(row[0])
            self._tombstone_edges(tx, unit_id)
            demoted.append(
                {
                    "kind": "demote_unit",
                    "unit_id": unit_id,
                    "salience": float(row[1]),
                    "to_state": "archived",
                }
            )
        tx.delta.units_archived += len(archived)
        return demoted

    def _demote_fields(self, tx: Tx, scope_id: str) -> list[dict[str, Any]]:
        """The same ladder at sub-unit granularity.

        [GEM] §3.2 requires it: "part of a unit may be attenuated while the rest
        stays current". Only non-current entries are hidden or archived — the
        current value of a live unit is what C1 answers with, so attenuating it
        would make the default query wrong rather than smaller.
        """
        demoted: list[dict[str, Any]] = []
        for state, floor, ceiling in (
            ("hidden", self.salience.theta_archive, self.salience.theta_remove),
            ("archived", None, self.salience.theta_archive),
        ):
            params: list[Any] = [state, scope_id, ceiling]
            clause = ""
            if floor is not None:
                clause = " AND fv.salience >= %s"
                params.append(floor)
            rows = tx.execute(
                "UPDATE gem_field_value fv SET state = %s FROM gem_unit u"
                " WHERE u.id = fv.unit_id AND u.scope_id = %s"
                "   AND fv.state NOT IN ('archived')"
                "   AND fv.valid_to IS NOT NULL"
                "   AND fv.salience < %s"
                + clause
                + " RETURNING fv.id, fv.unit_id, fv.field, fv.salience",
                tuple(params),
            ).fetchall()
            demoted.extend(
                {
                    "kind": "demote_field",
                    "value_id": int(row[0]),
                    "unit_id": int(row[1]),
                    "field": row[2],
                    "salience": float(row[3]),
                    "to_state": state,
                }
                for row in rows
            )
        return demoted

    def _tombstone_edges(self, tx: Tx, unit_id: int) -> None:
        """Tombstone the archived unit's native edges; keep the metadata row.

        The AM has no edge UPDATE, so tombstone-and-reinsert is the documented
        two-step. ``gem_edge`` keeps its row with ``tombstoned_at`` set, which
        is what makes an archived unit's structure recoverable rather than
        merely gone.
        """
        rows = tx.execute(
            "SELECT src, dst, edge_type FROM gem_edge"
            " WHERE (src = %s OR dst = %s) AND tombstoned_at IS NULL",
            (unit_id, unit_id),
        ).fetchall()
        for src, dst, edge_type in rows:
            tx.execute(
                "SELECT graph_store.gph_tombstone_edge(%s, %s, %s)",
                (int(src), int(dst), int(edge_type)),
            )
        if rows:
            tx.execute(
                "UPDATE gem_edge SET tombstoned_at = now()"
                " WHERE (src = %s OR dst = %s) AND tombstoned_at IS NULL",
                (unit_id, unit_id),
            )
            tx.delta.edges_tombstoned += len(rows)
