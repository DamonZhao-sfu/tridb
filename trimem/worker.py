"""The SemFilter worker: answers the trimem_pg shared-memory mailbox.

    python -m trimem.worker --dsn postgresql://.../db [--model ...] [--base-url ...]

Loop: ``trimem.take()`` (blocks on the mailbox CV) -> re-check ``trimem.sem_cache`` -> judge the
misses with the LLM -> commit cache rows (own transaction; the committed row is authoritative)
-> ``trimem.complete()`` wakes the waiting query. Invalid LLM replies answer ``false`` for this
query but are not cached. A request for another model fails closed (NULL verdict -> ERROR).

``sem_extract`` requests carry no value back through the mailbox: the label is committed into
``trimem.sem_cache.extracted`` first and ``trimem.complete_extracts()`` only signals completion;
the SemExtract node then reads its keys back. An unparsable reply is not cached, so the node sees
no label and returns NULL for that input.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field

import psycopg

from trimem import DEFAULT_MODEL, OP_EXTRACT, OP_FILTER, OP_SCORE
from trimem.keys import sem_key
from trimem.llm import Judge, OpenAIJudge

log = logging.getLogger("trimem.worker")

_COLUMN = {OP_FILTER: "verdict", OP_SCORE: "score", OP_EXTRACT: "extracted"}
# answer for this query when the LLM reply is invalid (never cached)
_FALLBACK: dict[str, object] = {OP_FILTER: False, OP_SCORE: 0.5, OP_EXTRACT: None}


@dataclass
class WorkerStats:
    requests: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    invalid: int = 0
    rejected: int = 0
    llm_seconds: float = 0.0
    batches: list[int] = field(default_factory=list)


class Worker:
    def __init__(
        self, dsn: str, judge: Judge, *, max_take: int = 64, take_ms: int = 500
    ) -> None:
        self.dsn = dsn
        self.judge = judge
        self.max_take = max_take
        self.take_ms = take_ms
        self.stats = WorkerStats()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, *, flush_every: float = 5.0) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            last_flush = time.monotonic()
            while not self._stop.is_set():
                try:
                    self.step(conn)
                except psycopg.Error:
                    # Never die silently: waiting queries would only see a timeout.
                    log.exception("worker step failed; continuing")
                    time.sleep(0.5)
                if time.monotonic() - last_flush >= flush_every:
                    conn.execute("SELECT trimem.flush_stats()")
                    last_flush = time.monotonic()
            conn.execute("SELECT trimem.flush_stats()")

    def step(self, conn: psycopg.Connection) -> int:
        rows = conn.execute(
            "SELECT slot, seq, op, model, prompt_version, predicate, input "
            "FROM trimem.take(%s, %s)",
            (self.max_take, self.take_ms),
        ).fetchall()
        if not rows:
            return 0
        self.stats.requests += len(rows)
        self.stats.batches.append(len(rows))
        for op, complete, cast in (
            (OP_FILTER, "trimem.complete", "boolean[]"),
            (OP_SCORE, "trimem.complete_scores", "float8[]"),
        ):
            part = [r for r in rows if r[2] == op]
            if part:
                answers = self._answer(conn, op, part)
                conn.execute(
                    f"SELECT {complete}(%s::int[], %s::bigint[], %s::{cast})",
                    (
                        [r[0] for r in part],
                        [r[1] for r in part],
                        [answers.get(r[0]) for r in part],
                    ),
                )
        part = [r for r in rows if r[2] == OP_EXTRACT]
        if part:
            self._extract(conn, part)
        unknown = [r for r in rows if r[2] not in (OP_FILTER, OP_SCORE, OP_EXTRACT)]
        if unknown:  # fail closed
            self.stats.rejected += len(unknown)
            self._fail(conn, unknown)
        return len(rows)

    @staticmethod
    def _fail(conn: psycopg.Connection, rows: list) -> None:
        """NULL verdict: the waiting query raises instead of guessing."""
        conn.execute(
            "SELECT trimem.complete(%s::int[], %s::bigint[], %s::boolean[])",
            ([r[0] for r in rows], [r[1] for r in rows], [None] * len(rows)),
        )

    def _extract(self, conn: psycopg.Connection, rows: list) -> None:
        """Commit labels into sem_cache, then signal completion (the label is read from the cache)."""
        if not hasattr(self.judge, "extract_many"):
            log.error(
                "judge %r cannot extract labels; failing %d request(s)",
                self.judge,
                len(rows),
            )
            self.stats.rejected += len(rows)
            self._fail(conn, rows)
            return
        ok = [r for r in rows if r[3] == self.judge.model]
        rejected = [r for r in rows if r[3] != self.judge.model]
        if ok:
            self._answer(conn, OP_EXTRACT, ok)  # commits before completing
            conn.execute(
                "SELECT trimem.complete_extracts(%s::int[], %s::bigint[])",
                ([r[0] for r in ok], [r[1] for r in ok]),
            )
        if rejected:
            log.error(
                "rejecting %d %s request(s) for another model (worker serves %s)",
                len(rejected),
                OP_EXTRACT,
                self.judge.model,
            )
            self.stats.rejected += len(rejected)
            self._fail(conn, rejected)

    def _answer(
        self, conn: psycopg.Connection, op: str, rows: list
    ) -> dict[int, object]:
        """slot -> verdict (filter) / score (score) / label (extract); None makes the waiting
        filter/score query fail closed (extract completion goes through ``_extract``)."""
        col = _COLUMN[op]
        out: dict[int, object] = {}
        keyed = []
        for slot, _seq, _op, model, pv, pred, text in rows:
            if model != self.judge.model:
                log.error(
                    "rejecting %s request for model %s (worker serves %s)",
                    op,
                    model,
                    self.judge.model,
                )
                self.stats.rejected += 1
                out[slot] = None
                continue
            keyed.append(
                (slot, sem_key(op, model, pv, pred, text), model, pv, pred, text)
            )
        cached = self._cached(conn, [k for _, k, *_ in keyed], col)
        misses = [r for r in keyed if r[1] not in cached]
        self.stats.cache_hits += len(keyed) - len(misses)
        unique = {}
        for r in misses:
            unique.setdefault(r[1], r)
        fresh: dict[str, object] = {}
        if unique:
            pairs = [(r[4], r[5]) for r in unique.values()]
            if op == OP_FILTER:
                answers = self.judge.judge_many(pairs)
            elif op == OP_SCORE:
                answers = self.judge.score_many(pairs)
            else:
                answers = self.judge.extract_many(pairs)
            self.stats.llm_calls += len(answers)
            with conn.transaction():
                for (key, model, pv, pred, _t), v in zip(
                    [r[1:] for r in unique.values()], answers, strict=True
                ):
                    self.stats.llm_seconds += v.seconds
                    if v.value is None:
                        # answer this query (reject / neutral), never cache an invalid reply
                        self.stats.invalid += 1
                        fresh[key] = _FALLBACK[op]
                        continue
                    conn.execute(
                        f"INSERT INTO trimem.sem_cache (key, op, model, prompt_version, predicate, "
                        f"{col}, prompt_tokens, completion_tokens, seconds) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (key) DO NOTHING",
                        (
                            key,
                            op,
                            model,
                            pv,
                            pred,
                            v.value,
                            v.prompt_tokens,
                            v.completion_tokens,
                            v.seconds,
                        ),
                    )
            # The committed row is authoritative (a concurrent writer may have won the race).
            fresh.update(self._cached(conn, [k for k in unique if k not in fresh], col))
        for slot, key, *_ in keyed:
            out[slot] = cached[key] if key in cached else fresh.get(key, _FALLBACK[op])
        return out

    @staticmethod
    def _cached(
        conn: psycopg.Connection, keys: list[str], col: str = "verdict"
    ) -> dict[str, object]:
        if not keys:
            return {}
        cur = conn.execute(
            f"SELECT key, {col} FROM trimem.sem_cache WHERE key = ANY(%s) AND {col} IS NOT NULL",
            (keys,),
        )
        return {k: v for k, v in cur.fetchall()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-take", type=int, default=64)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    judge = OpenAIJudge(args.base_url, args.model, concurrency=args.concurrency)
    worker = Worker(args.dsn, judge, max_take=args.max_take)
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    log.info("stats: %s", worker.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
