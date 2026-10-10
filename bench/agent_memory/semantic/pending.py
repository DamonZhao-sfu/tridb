"""The pending set Q (paper: deferred derivations), as the heap table ``gem_pending``.

Ingest under a deferred plan enqueues a derivation instead of running it; a background or
batch pass, or a strong-mode read, resolves it later. Every call uses the caller's
connection and transaction, so enqueue commits atomically with the item it belongs to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

TRIGGERS = ("background", "batch", "on_read")


@dataclass(frozen=True)
class PendingItem:
    id: int
    scope_id: str
    derivation: str
    target_ref: str
    trigger: str
    snapshot_seq: int
    payload: dict[str, Any]


class PendingQueue:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def enqueue(
        self,
        scope_id: str,
        derivation: str,
        target_ref: str,
        *,
        trigger: str,
        snapshot_seq: int,
        payload: dict[str, Any] | None = None,
    ) -> int | None:
        """Returns the new id, or None if that derivation is already queued/resolved."""
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
        from psycopg.types.json import Jsonb

        row = self.conn.execute(
            "INSERT INTO gem_pending"
            " (scope_id, derivation, target_ref, trigger, snapshot_seq, payload)"
            " VALUES (%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (scope_id, derivation, target_ref) DO NOTHING RETURNING id",
            (
                scope_id,
                derivation,
                target_ref,
                trigger,
                snapshot_seq,
                Jsonb(payload or {}),
            ),
        ).fetchone()
        return None if row is None else int(row[0])

    def open_items(
        self,
        scope_id: str,
        *,
        derivation: str | None = None,
        targets: Sequence[str] | None = None,
        limit: int | None = None,
        lock: bool = True,
    ) -> list[PendingItem]:
        """Pending items in snapshot (FIFO) order. ``lock`` takes row locks (SKIP LOCKED)."""
        sql = [
            "SELECT id, scope_id, derivation, target_ref, trigger, snapshot_seq, payload"
            " FROM gem_pending WHERE scope_id = %s AND status = 'pending'"
        ]
        args: list[Any] = [scope_id]
        if derivation is not None:
            sql.append(" AND derivation = %s")
            args.append(derivation)
        if targets is not None:
            if not targets:
                return []
            sql.append(" AND target_ref = ANY(%s)")
            args.append(list(targets))
        sql.append(" ORDER BY snapshot_seq, id")
        if limit is not None:
            sql.append(" LIMIT %s")
            args.append(int(limit))
        if lock:
            sql.append(" FOR UPDATE SKIP LOCKED")
        rows = self.conn.execute("".join(sql), args).fetchall()
        return [
            PendingItem(int(r[0]), r[1], r[2], r[3], r[4], int(r[5]), dict(r[6] or {}))
            for r in rows
        ]

    def resolve(self, ids: Sequence[int], *, by: str) -> int:
        if not ids:
            return 0
        cur = self.conn.execute(
            "UPDATE gem_pending SET status = 'resolved', resolved_tx = txid_current(),"
            " resolved_by = %s WHERE id = ANY(%s) AND status = 'pending'",
            (by, list(ids)),
        )
        return int(cur.rowcount or 0)

    def fail(self, item_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE gem_pending SET status = 'failed', error = %s WHERE id = %s",
            (error[:2000], item_id),
        )

    def count_open(self, scope_id: str, *, targets: Sequence[str] | None = None) -> int:
        if targets is not None:
            if not targets:
                return 0
            row = self.conn.execute(
                "SELECT count(*) FROM gem_pending WHERE scope_id = %s"
                " AND status = 'pending' AND target_ref = ANY(%s)",
                (scope_id, list(targets)),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT count(*) FROM gem_pending WHERE scope_id = %s AND status = 'pending'",
                (scope_id,),
            ).fetchone()
        return int(row[0])
