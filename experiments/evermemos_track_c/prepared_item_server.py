"""Benchmark-only HTTP boundary for official EverMemOS prepared-item persistence.

Run this file with the pinned paper-era repository's ``src`` directory on
``PYTHONPATH``.  Startup follows the repository's official ``src/run.py``
sequence; this module only adds the narrow composition that the upstream
repository already uses to persist an ``EpisodeMemory`` to MongoDB,
Elasticsearch, and Milvus.  It deliberately imports no memory extractor or LLM
service.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field


SEMANTIC_BOUNDARY = "prepared_memory_item_insertion_v1"


class PreparedItemRequest(BaseModel):
    prepared_item_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    group_name: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    timestamp: datetime
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    ordinal: int


async def persist_prepared_item(request: PreparedItemRequest) -> dict[str, Any]:
    """Generate only the embedding, then call official multi-store persistence."""
    from api_specs.memory_models import MemoryType
    from api_specs.memory_types import EpisodeMemory, RawDataType
    from biz_layer.mem_db_operations import _convert_episode_memory_to_doc
    from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs
    from core.di.utils import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )

    rendered = f"[event_id={request.prepared_item_id}] {request.content}"
    repository = get_bean_by_type(EpisodicMemoryRawRepository)

    embedding_started = time.perf_counter_ns()
    embedding = await repository.vectorize_service.get_embedding(rendered)
    embedding_completed = time.perf_counter_ns()
    vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    document = EpisodeMemory(
        memory_type=MemoryType.EPISODIC_MEMORY,
        user_id=request.user_id,
        timestamp=request.timestamp,
        ori_event_id_list=[request.prepared_item_id],
        group_id=request.group_id,
        group_name=request.group_name,
        participants=[request.role],
        type=RawDataType.CONVERSATION,
        keywords=[],
        linked_entities=[],
        memcell_event_id_list=[request.prepared_item_id],
        user_name=request.role,
        extend={
            "source": "locomo",
            "session_id": request.session_id,
            "event_order": request.ordinal,
            "semantic_boundary": SEMANTIC_BOUNDARY,
        },
        vector_model=repository.vectorize_service.get_model_name(),
        vector=vector,
        subject=request.role,
        summary=rendered,
        episode=rendered,
    )

    persistence_document = _convert_episode_memory_to_doc(
        document, current_time=request.timestamp
    )
    persistence_started = time.perf_counter_ns()
    saved = await save_memory_docs(
        [
            MemoryDocPayload(
                memory_type=MemoryType.EPISODIC_MEMORY, doc=persistence_document
            )
        ]
    )
    persistence_completed = time.perf_counter_ns()
    rows = saved.get(MemoryType.EPISODIC_MEMORY) or []
    if len(rows) != 1 or rows[0] is None:
        raise HTTPException(status_code=500, detail="prepared item was not persisted")
    saved_id = str(rows[0].id)
    return {
        "status": "complete",
        "semantic_boundary": SEMANTIC_BOUNDARY,
        "construction_llm_policy": "forbidden_by_code_path_and_unserved_endpoint",
        "llm_call_count": 0,
        "prepared_item_id": request.prepared_item_id,
        "created_memory_count": 1,
        "created_node_count": 0,
        "created_edge_count": 0,
        "created_ids": [saved_id],
        "updated_stores": [
            "mongodb_episodic_memory",
            "elasticsearch_episodic_memory",
            "milvus_episodic_memory",
        ],
        "stage_ms": {
            "embedding": (embedding_completed - embedding_started) / 1_000_000,
            "persistence_and_indexes": (persistence_completed - persistence_started)
            / 1_000_000,
            "server_total": (persistence_completed - embedding_started) / 1_000_000,
        },
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--host", default="127.0.0.1")
    root.add_argument("--port", type=int, default=8196)
    root.add_argument("--env-file", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    from common_utils.load_env import setup_environment

    setup_environment(
        load_env_file_name=args.env_file,
        check_env_var="MONGODB_HOST",
        service_name="web",
    )
    from application_startup import setup_all

    setup_all()
    from core.oxm.mongo.migration.manager import MigrationManager

    MigrationManager.run_migrations_on_startup(enabled=True)
    from app import app

    app.add_api_route(
        "/api/v1/benchmark/prepared-items",
        persist_prepared_item,
        methods=["POST"],
        response_model=dict[str, Any],
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
