"""Thin native VSGT client: one retrieval request, no Python query-plan stages."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Sequence
from uuid import uuid4

from psycopg.types.json import Jsonb

if TYPE_CHECKING:
    from psycopg import Connection

STRATEGIES = frozenset(
    {"seed_cursor_f", "seed_cursor_b", "graph_only_f", "graph_only_b", "vector_only"}
)
SEARCH_SEED = 42


class VsgtClient:
    """Use inside a transaction; native cursor owns all retrieval and hydration."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.retrieval_requests = 0

    @contextmanager
    def stream(
        self, request: dict[str, Any], query_vector: Sequence[float] | None = None
    ) -> Iterator[Iterator[dict[str, Any]]]:
        strategy = request.get("strategy")
        if strategy == "own_context":
            yield iter(())
            return
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported native strategy: {strategy}")
        if strategy.startswith("graph_only") and query_vector is not None:
            raise ValueError("GraphOnly must not prepare a query vector")
        vector = None
        if query_vector is not None:
            values = [float(x) for x in query_vector]
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError("Query vector must be nonempty and finite")
            vector = "[" + ",".join(map(str, values)) + "]"
        if not strategy.startswith("graph_only") and vector is None:
            raise ValueError("Vector strategy requires an embedding")
        with self.connection.cursor(name="vsgt_" + uuid4().hex) as cursor:
            cursor.itersize = 1
            self.retrieval_requests += 1
            cursor.execute(
                "SELECT result FROM public.tjs_vsgt(%s::jsonb, %s::vector)",
                (Jsonb(request), vector),
            )
            yield (row[0] for row in cursor)

    def retrieve(
        self, request: dict[str, Any], query_vector: Sequence[float] | None = None
    ) -> list[dict[str, Any]]:
        with self.stream(request, query_vector) as rows:
            return list(rows)

    def stats(self, request_id: str) -> dict[str, Any] | None:
        return self.connection.execute(
            "SELECT public.tjs_vsgt_stats(%s)", (request_id,)
        ).fetchone()[0]
