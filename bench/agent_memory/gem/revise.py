"""The ``revise`` operator — reconcile and propagate.

    M_{t+1} = U(M_t, {}, rev(delta))

Four repair kinds, run in this order, each idempotent:

==========  ==================================================  ================
Repair      Detection                                           Action
==========  ==================================================  ================
conflict    two candidate current values for one (unit, field)  supersede the
                                                                older, chain
                                                                superseded_by (C4)
duplicate   two units, same scope, cosine >= theta and a title  merge: union
            match                                               histories,
                                                                re-point edges,
                                                                archive the loser
propagate   a field changed on u_i                              walk EXTENSION
                                                                out-edges, halt
                                                                where the policy
                                                                condition does
                                                                not fire (C3)
split       a field subset accumulates >= N references          promote to a new
                                                                unit, re-point
                                                                edges
==========  ==================================================  ================

Two engine realities shape the implementation:

**The propagation walk materialises once.** The graph leg is commit-visible,
not snapshot-isolated (proven, interface §6.5), so a walk that re-traverses can
see edges appear mid-walk. One ``gph_traverse_bfs``, then process the frozen
set.

**Re-embedding is batched.** It is 0% HOT and grew the relation ~369 B per
update (measured, §6.3), so every unit whose fields changed goes into a set and
the operator issues ONE embedding call plus one bulk UPDATE at the end. Never
re-embed per field change.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem.store import GemStore, Tx, vec_literal
from bench.agent_memory.gem.types import (
    PhaseCost,
    RevisionResult,
    StateDelta,
)

DEFAULT_DUP_THRESHOLD = 0.05  # cosine DISTANCE; <= this is a duplicate candidate
DEFAULT_SPLIT_REFERENCES = 5


class ReviseOperator:
    """Detects evidence in ``M_t`` and applies the matching repair."""

    def __init__(
        self,
        store: GemStore,
        *,
        embedder: Any | None = None,
        dup_threshold: float = DEFAULT_DUP_THRESHOLD,
        split_references: int = DEFAULT_SPLIT_REFERENCES,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.dup_threshold = dup_threshold
        self.split_references = split_references

    def revise(
        self,
        scope_id: str,
        *,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        max_hops: int = 3,
        kinds: Sequence[str] = ("conflict", "duplicate", "propagate", "split"),
    ) -> RevisionResult:
        started = time.perf_counter()
        repairs: list[Mapping[str, Any]] = []
        try:
            with self.store.transition("revise", scope_id, "construction") as tx:
                tx.lock_writer()
                # Units whose vector is now stale. Collected across every
                # repair and re-embedded ONCE at the end.
                dirty: set[int] = set()

                if "conflict" in kinds:
                    repairs += self._repair_conflicts(tx, scope_id, dirty)
                if "duplicate" in kinds:
                    repairs += self._repair_duplicates(tx, scope_id, dirty)
                if "propagate" in kinds:
                    repairs += self._propagate(tx, scope_id, evidence, max_hops)
                if "split" in kinds:
                    repairs += self._split_topics(tx, scope_id, dirty)

                self._reembed(tx, dirty)

                delta = tx.delta.freeze()
                cost = tx.meter.freeze(time.perf_counter() - started)
                transition_id = tx.transition_id
                policies = tuple(tx.policies_evaluated)
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            return RevisionResult(
                operator="revise",
                committed=False,
                delta=StateDelta(),
                cost=PhaseCost(
                    phase="construction", seconds=time.perf_counter() - started
                ),
                aborted_reason=f"{type(exc).__name__}: {exc}",
            )

        return RevisionResult(
            operator="revise",
            committed=True,
            delta=delta,
            cost=cost,
            transition_id=transition_id,
            policies_evaluated=policies,
            repairs=tuple(repairs),
        )

    # -- conflict ---------------------------------------------------------

    def _repair_conflicts(
        self, tx: Tx, scope_id: str, dirty: set[int]
    ) -> list[Mapping[str, Any]]:
        """Two current values for one ``(unit, field)`` — supersede the older.

        The partial unique index normally prevents this from ever existing, so
        this repair matters for state written before the index, or restored
        from a dump. If a repair produced a wrong result the index makes the
        transaction abort, which is the intended safety net.
        """
        rows = tx.execute(
            "SELECT fv.unit_id, fv.field, array_agg(fv.id ORDER BY fv.valid_from,"
            " fv.id) FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
            " WHERE u.scope_id = %s AND fv.valid_to IS NULL"
            "   AND fv.state <> 'archived'"
            " GROUP BY fv.unit_id, fv.field HAVING count(*) > 1",
            (scope_id,),
        ).fetchall()

        repairs: list[Mapping[str, Any]] = []
        for unit_id, field_name, value_ids in rows:
            ordered = [int(v) for v in value_ids]
            winner = ordered[-1]  # newest valid_from wins
            losers = ordered[:-1]
            tx.execute(
                "UPDATE gem_field_value SET valid_to = ("
                "  SELECT valid_from FROM gem_field_value WHERE id = %s),"
                " superseded_by = %s WHERE id = ANY(%s)",
                (winner, winner, losers),
            )
            tx.delta.values_superseded += len(losers)
            dirty.add(int(unit_id))
            repairs.append(
                {
                    "kind": "conflict",
                    "unit_id": int(unit_id),
                    "field": field_name,
                    "kept": winner,
                    "superseded": losers,
                }
            )
        return repairs

    # -- duplicate --------------------------------------------------------

    def _repair_duplicates(
        self, tx: Tx, scope_id: str, dirty: set[int]
    ) -> list[Mapping[str, Any]]:
        """Merge near-identical units: union histories, re-point edges, archive.

        The loser is ARCHIVED, never deleted — C4 requires the provenance chain
        of anything still reachable to survive, and the merged unit references
        the loser's values.
        """
        rows = tx.execute(
            "SELECT a.id, b.id, (a.embedding <=> b.embedding) AS dist"
            " FROM gem_unit a JOIN gem_unit b"
            "   ON a.scope_id = b.scope_id AND a.id < b.id"
            " WHERE a.scope_id = %s AND a.state = 'active' AND b.state = 'active'"
            "   AND (a.embedding <=> b.embedding) <= %s"
            "   AND lower(a.title) = lower(b.title)"
            " ORDER BY dist, a.id, b.id",
            (scope_id, self.dup_threshold),
        ).fetchall()

        repairs: list[Mapping[str, Any]] = []
        merged: set[int] = set()
        for keep_raw, drop_raw, distance in rows:
            keep, drop = int(keep_raw), int(drop_raw)
            if keep in merged or drop in merged:
                continue  # one merge per unit per pass; idempotent across passes
            merged.add(drop)

            # Move only the loser's fields that the winner does not already
            # hold currently — otherwise the partial unique index aborts, which
            # would be correct but unhelpful.
            tx.execute(
                "UPDATE gem_field_value SET unit_id = %s WHERE unit_id = %s"
                " AND field NOT IN (SELECT field FROM gem_field_value"
                "   WHERE unit_id = %s AND valid_to IS NULL)",
                (keep, drop, keep),
            )
            # The loser's remaining values become history under the winner.
            tx.execute(
                "UPDATE gem_field_value SET unit_id = %s, valid_to = COALESCE("
                " valid_to, now()) WHERE unit_id = %s",
                (keep, drop),
            )
            tx.execute(
                "UPDATE gem_edge SET src = %s WHERE src = %s AND dst <> %s",
                (keep, drop, keep),
            )
            tx.execute(
                "UPDATE gem_edge SET dst = %s WHERE dst = %s AND src <> %s",
                (keep, drop, keep),
            )
            tx.execute("UPDATE gem_unit SET state = 'archived' WHERE id = %s", (drop,))
            tx.delta.units_archived += 1
            tx.delta.units_updated += 1
            dirty.add(keep)
            repairs.append(
                {
                    "kind": "duplicate",
                    "kept": keep,
                    "archived": drop,
                    "distance": float(distance),
                }
            )
        return repairs

    # -- propagate (C3) ---------------------------------------------------

    def _propagate(
        self,
        tx: Tx,
        scope_id: str,
        evidence: Sequence[Mapping[str, Any]] | None,
        max_hops: int,
    ) -> list[Mapping[str, Any]]:
        """Walk EXTENSION out-edges from each changed unit, once.

        Association edges are never traversed: [GEM] §4.1 requires propagation
        to follow entailment, not relatedness. The single-type filter makes
        that exact — ``gph_traverse_bfs`` takes the extension type id, so an
        association edge is invisible to this walk rather than filtered after.
        """
        seeds = self._propagation_seeds(tx, scope_id, evidence)
        if not seeds:
            return []
        extension_type = self.store.edge_type_id("extension")

        repairs: list[Mapping[str, Any]] = []
        for seed in seeds:
            # ONE traversal, then process the frozen set — never re-traverse.
            # The graph leg is commit-visible, so a re-traversal could observe
            # a topology newer than this transaction's snapshot.
            rows = tx.execute(
                "SELECT graph_store.gph_traverse_bfs(%s, %s, %s)",
                (int(seed), int(max_hops), int(extension_type)),
            ).fetchall()
            reach = [
                int(row[0])
                for row in rows
                if row[0] is not None and int(row[0]) != seed
            ]
            if reach:
                tx.execute(
                    "UPDATE gem_unit SET metadata = jsonb_set(metadata,"
                    " '{needs_revision}', 'true'::jsonb) WHERE id = ANY(%s)",
                    (reach,),
                )
                tx.delta.propagated_units.extend(reach)
            # Frontier size per revision is recorded so a dense extension graph
            # shows up as a number rather than as a slow run.
            repairs.append(
                {
                    "kind": "propagate",
                    "seed": int(seed),
                    "frontier": len(reach),
                    "reached": reach,
                    "max_hops": max_hops,
                }
            )
            tx.execute(
                "UPDATE gem_unit SET metadata = metadata - 'needs_revision'"
                " WHERE id = %s",
                (int(seed),),
            )
        return repairs

    def _propagation_seeds(
        self,
        tx: Tx,
        scope_id: str,
        evidence: Sequence[Mapping[str, Any]] | None,
    ) -> list[int]:
        if evidence:
            return sorted(
                {
                    int(item["unit_id"])
                    for item in evidence
                    if item.get("unit_id") is not None
                }
            )
        rows = tx.execute(
            "SELECT id FROM gem_unit WHERE scope_id = %s"
            "  AND metadata->>'needs_revision' = 'true' ORDER BY id",
            (scope_id,),
        ).fetchall()
        return [int(row[0]) for row in rows]

    # -- split ------------------------------------------------------------

    def _split_topics(
        self, tx: Tx, scope_id: str, dirty: set[int]
    ) -> list[Mapping[str, Any]]:
        """Promote a heavily-referenced field subset to its own unit.

        [GEM] Figure 3: "Alice splits out of Website Redesign once enough
        interactions reference her directly." Detection here is deliberately
        crude — a field whose history has accumulated at least
        ``split_references`` entries — because the paper specifies the
        mechanism, not the trigger heuristic, and an invented heuristic
        presented as theirs would be a misattribution.
        """
        rows = tx.execute(
            "SELECT fv.unit_id, fv.field, count(*) FROM gem_field_value fv"
            " JOIN gem_unit u ON u.id = fv.unit_id"
            " WHERE u.scope_id = %s AND u.state = 'active'"
            " GROUP BY fv.unit_id, fv.field HAVING count(*) >= %s",
            (scope_id, self.split_references),
        ).fetchall()

        repairs: list[Mapping[str, Any]] = []
        for unit_id, field_name, references in rows:
            parent = int(unit_id)
            title = f"{field_name}@{parent}"
            existing = tx.execute(
                "SELECT id FROM gem_unit WHERE scope_id = %s AND title = %s",
                (scope_id, title),
            ).fetchone()
            if existing is not None:
                continue  # idempotent: already split

            new_id = tx.allocate_vertex()
            tx.execute(
                "INSERT INTO gem_unit (id, scope_id, title, summary, embedding,"
                " metadata) SELECT %s, %s, %s, %s, embedding,"
                " jsonb_build_object('split_from', %s::bigint, 'field', %s::text)"
                " FROM gem_unit WHERE id = %s",
                (
                    new_id,
                    scope_id,
                    title,
                    f"split of {field_name}",
                    parent,
                    field_name,
                    parent,
                ),
            )
            tx.delta.units_created += 1
            tx.execute(
                "UPDATE gem_field_value SET unit_id = %s WHERE unit_id = %s"
                "  AND field = %s",
                (new_id, parent, field_name),
            )
            # The parent entails the split-out unit: a change in the parent
            # topic requires re-evaluating the promoted one. Orientation is
            # parent -> child so revision walks out-edges (interface §6.4).
            self.store.link(tx, parent, new_id, kind="extension", rel="split_of")
            dirty.add(new_id)
            dirty.add(parent)
            repairs.append(
                {
                    "kind": "split",
                    "parent": parent,
                    "new_unit": new_id,
                    "field": field_name,
                    "references": int(references),
                }
            )
        return repairs

    # -- batched re-embedding ---------------------------------------------

    def _reembed(self, tx: Tx, dirty: set[int]) -> None:
        """ONE embedding call and ONE bulk UPDATE for every touched unit.

        Re-embedding is the expensive write: 0% HOT, ~369 B of relation growth
        per update, and an HNSW insert plus a dead tuple each time. Batching is
        not an optimisation here, it is the difference between revision being
        affordable and revision degrading recall over a trajectory
        (``bench/recall_decay.py`` measures the drift).
        """
        if not dirty or self.embedder is None:
            return
        ids = sorted(dirty)
        rows = tx.execute(
            "SELECT u.id, u.title, u.summary, COALESCE(string_agg(fv.value, ' '"
            "  ORDER BY fv.field), '') FROM gem_unit u"
            " LEFT JOIN gem_field_value fv ON fv.unit_id = u.id"
            "   AND fv.valid_to IS NULL AND fv.state = 'active'"
            " WHERE u.id = ANY(%s) GROUP BY u.id, u.title, u.summary",
            (ids,),
        ).fetchall()
        if not rows:
            return
        texts = [f"{row[1]}\n{row[2]}\n{row[3]}".strip() for row in rows]
        vectors = self.embedder.encode(texts)
        tx.meter.embed_calls += 1
        tx.meter.embed_sequences += len(texts)

        tx.execute(
            "UPDATE gem_unit SET embedding = v.embedding::vector FROM ("
            " SELECT * FROM unnest(%s::bigint[], %s::text[]) AS s(id, embedding)"
            ") v WHERE gem_unit.id = v.id",
            (
                [int(row[0]) for row in rows],
                [vec_literal(vector) for vector in vectors],
            ),
        )
        tx.delta.units_updated += len(rows)
