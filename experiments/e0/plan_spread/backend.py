"""Backend protocol and factory for E0 plan execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .model import PlanSpec, QuerySpec


class DatasetBackend(Protocol):
    backend_name: str
    valid_for_system_latency_claims: bool

    def load_queries(self, path: Path) -> list[QuerySpec]: ...

    def prepare_query(self, query: QuerySpec, hops: set[int]) -> None: ...

    def execute(
        self, query: QuerySpec, plan: PlanSpec, *, top_n: int
    ) -> dict[str, Any]: ...


def build_backend(name: str, dataset: str, spec: dict[str, Any]) -> DatasetBackend:
    if name == "parquet_reference":
        from .reference_backend import ReferenceDataset

        return ReferenceDataset(dataset, spec)
    if name == "polyglot_live":
        from .live_backend import PolyglotLiveDataset

        return PolyglotLiveDataset(dataset, spec)
    if name == "tridb_live":
        from .tridb_backend import TriDBLiveDataset

        return TriDBLiveDataset(dataset, spec)
    raise ValueError(f"unknown backend: {name}")
