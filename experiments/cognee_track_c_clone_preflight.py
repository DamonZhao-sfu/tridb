"""Verify that a Cognee PostgreSQL snapshot clone is independently searchable."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bench.agent_memory.table5_track_c.adapters.cognee import (
    CogneeAdapter,
    CogneeConfig,
)
from bench.agent_memory.table5_track_c.dataset import load_locomo
from bench.agent_memory.table5_track_c.protocol import _retrieval_query

DEFAULT_DATASET = (
    "/localhome/hza214/Mandol/experimental/self_host_benchmarks/locomo/"
    "data/locomo10.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=55432)
    parser.add_argument("--db-user", default="table5c_cognee_user")
    parser.add_argument("--answer-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    args = parser.parse_args()

    build = json.loads(Path(args.build_receipt).read_text(encoding="utf-8"))
    corpus = load_locomo(args.dataset)
    adapter = CogneeAdapter(
        CogneeConfig(
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=os.environ["COGNEE_DB_PASSWORD"],
            db_name=args.database_name,
            dataset_prefix=args.dataset_prefix,
            answer_base_url=args.answer_base_url,
            embedding_base_url=args.embedding_base_url,
        )
    )
    try:
        schema = adapter.init_schema()
        adapter.prepare_reused_build(corpus)
        observed = adapter.snapshot_fingerprint()
        expected = build["snapshot_fingerprint"]
        if observed != expected:
            raise RuntimeError(
                f"Cognee clone fingerprint mismatch: {observed!r} != {expected!r}"
            )
        result = adapter.search(_retrieval_query(corpus.formal_queries[0]), top_k=35)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "database_name": args.database_name,
                    "schema": schema,
                    "clone_rebind": adapter.stats()["clone_rebind"],
                    "result_count": result["result_count"],
                    "empty": result["empty"],
                },
                sort_keys=True,
            )
        )
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
