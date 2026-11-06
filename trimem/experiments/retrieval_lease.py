"""retrieval-slot-v2 (user 2026-10-21): hold a host-wide retrieval slot -- and the PG 55450 connection it
bounds (max_connections 32) -- only while a retrieval does DATABASE work.

Before v2 a retrieval held its slot from adapter creation to the last store write, including 1-7 min of
SemFilter judge calls, so all slots were held by SemFilter retrievals and fast (VGR) episodes queued.
``RetrievalLease`` is entered around the retrieval (acquires a slot); ``session_retrieval.retrieve`` enters
``lease.pause()`` around every SemFilter / SemRank call, after hydrating the cards those calls see. ``pause``
closes the adapter's read-only connection (no transaction stays open across the LLM wait) and frees the slot;
the adapter's next query (``_LazyConnection``) or the next store write (``GatedStore``) re-acquires a slot and,
for the adapter, opens a fresh connection with the adapter's session settings (``readonly_connect``). The
adapter calls, their order and their SQL are unchanged, so receipts and selected cards are identical.
"""

from __future__ import annotations

import fcntl
import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

VERSION = "retrieval-slot-v2"

# semfilter-cap-v1 (user 2026-10-21): with the PG slot released, every SemFilter retrieval judged at once and
# saturated vLLM (agent calls starved). A SEPARATE host-wide flock pool bounds concurrent SemFilter calls (all
# processes: ALF / Sci session eval and the τ² runner_pool); ``$TRIMEM_SEMFILTER_SLOTS`` (default 24, the
# effective SemFilter concurrency under retrieval-slot-v1; <= 0 = no cap). ColBERT (own servers) is not capped.
SEMFILTER_SLOT_DIR = Path("/tmp/trimem-semfilter-slots")
SEMFILTER_SLOTS_ENV = "TRIMEM_SEMFILTER_SLOTS"
SEMFILTER_CAP_VERSION = "semfilter-cap-v1"


class SemFilterSlot:
    """One of ``n`` host-wide SemFilter slots (flock files; released on exit or when the holder dies)."""

    def __init__(self, n: int | None = None, folder: Path | None = None):
        if n is None:
            n = int(os.environ.get(SEMFILTER_SLOTS_ENV, "24"))
        self.n, self.folder, self.f = n, Path(folder or SEMFILTER_SLOT_DIR), None
        self.waited_s = 0.0

    def __enter__(self) -> SemFilterSlot:
        if self.n <= 0:
            return self
        self.folder.mkdir(parents=True, exist_ok=True)
        order = list(range(self.n))
        random.shuffle(order)
        t0 = time.monotonic()
        while True:
            for i in order:
                f = open(self.folder / f"{i}.lock", "w")  # noqa: SIM115 - held until __exit__
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    self.waited_s = time.monotonic() - t0
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(0.2)

    def __exit__(self, *exc) -> bool:
        if self.f is not None:
            self.f.close()
            self.f = None
        return False


# Session settings of the read-only adapter connection (as ``AgentEnvTaskAdapter`` / ``Tau2TaskAdapter``).
READONLY_SETUP = (
    "SET TRANSACTION READ ONLY",
    "SET LOCAL statement_timeout='60s'",
    "SET LOCAL hnsw.ef_search=1000",
)


def readonly_connect(dsn: str):
    """A connection configured exactly like the retrieval adapters' own (dict rows, read-only transaction)."""
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(dsn, row_factory=dict_row)
    for sql in READONLY_SETUP:
        conn.execute(sql)
    return conn


class RetrievalLease:
    """A retrieval's claim on one host-wide slot, held only during DB phases.

    ``slot_factory()`` returns a fresh slot context manager (``agentgym_session_eval.retrieval_slot`` or
    ``runner_pool.HostSlots.slot``). ``reconnect()`` opens a replacement adapter connection; without it
    ``pause`` frees only the slot and leaves the adapter connection alone (fake adapters in tests)."""

    def __init__(
        self,
        slot_factory: Callable[[], AbstractContextManager],
        reconnect: Callable[[], object] | None = None,
        semfilter_slot: Callable[[], AbstractContextManager] | None = SemFilterSlot,
    ):
        self._factory, self._reconnect = slot_factory, reconnect
        self._semfilter_slot = semfilter_slot
        self._slot: AbstractContextManager | None = None
        self.adapter = None
        self.acquisitions = (
            0  # slot acquisitions of this retrieval (1 + re-acquisitions)
        )
        self.pauses = 0

    @property
    def held(self) -> bool:
        return self._slot is not None

    def acquire(self) -> None:
        if self._slot is None:
            slot = self._factory()
            slot.__enter__()
            self._slot = slot
            self.acquisitions += 1

    def release(self) -> None:
        if self._slot is not None:
            slot, self._slot = self._slot, None
            slot.__exit__(None, None, None)

    def __enter__(self) -> RetrievalLease:
        self.acquire()
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False

    def bind(self, adapter):
        """The adapter whose connection ``pause`` closes; returns it."""
        self.adapter = adapter
        return adapter

    def connect(self):
        """A fresh adapter connection under a (re-)acquired slot."""
        self.acquire()
        return self._reconnect()

    @contextmanager
    def pause(self, kind: str | None = None) -> Iterator[None]:
        """No DB work inside: close the adapter connection (rolls back its read-only transaction) and free
        the slot. Re-acquisition is lazy: the next adapter query or store write takes a slot again.
        ``kind == "semfilter"``: the call also runs under a host-wide SemFilter slot (semfilter-cap-v1)."""
        a = self.adapter
        if a is not None and self._reconnect is not None:
            a.conn.close()
            a.conn = _LazyConnection(self)
        self.release()
        self.pauses += 1
        cap = (
            self._semfilter_slot()
            if kind == "semfilter" and self._semfilter_slot is not None
            else nullcontext()
        )
        with cap:
            yield

    def gated(self, store) -> GatedStore:
        return GatedStore(store, self)


class _LazyConnection:
    """Stand-in for a paused adapter's connection: the first ``execute`` re-acquires a slot and connects."""

    def __init__(self, lease: RetrievalLease):
        self._lease, self._conn = lease, None

    def execute(self, *args, **kwargs):
        if self._conn is None:
            self._conn = self._lease.connect()
        return self._conn.execute(*args, **kwargs)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class GatedStore:
    """Store proxy: every method call (each opens its own short PG connection) runs under a held slot."""

    def __init__(self, store, lease: RetrievalLease):
        self._store, self._lease = store, lease

    def __getattr__(self, name: str):
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self._lease.acquire()
            return attr(*args, **kwargs)

        return call
