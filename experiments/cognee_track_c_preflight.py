"""Create Cognee's default user once before concurrent Track C ingestion."""

from __future__ import annotations

import argparse
import os

from bench.agent_memory.table5_track_c.adapters.cognee import (
    CogneeAdapter,
    CogneeConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=55432)
    parser.add_argument("--db-user", default="table5c_cognee_user")
    parser.add_argument("--answer-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    args = parser.parse_args()

    adapter = CogneeAdapter(
        CogneeConfig(
            db_host=args.db_host,
            db_port=args.db_port,
            db_user=args.db_user,
            db_password=os.environ["COGNEE_DB_PASSWORD"],
            db_name=args.database_name,
            dataset_prefix="trackc_preflight",
            answer_base_url=args.answer_base_url,
            embedding_base_url=args.embedding_base_url,
        )
    )
    adapter.init_schema()
    from cognee.modules.users.methods import get_default_user

    user = adapter.bridge.run(get_default_user())
    print(f"Cognee default user ready: {user.id}")
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
