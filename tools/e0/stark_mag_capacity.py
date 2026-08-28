"""Estimate whether the host can hold the full STARK-MAG E1 live deployment.

The estimate scales measured STARK-PRIME physical footprints by the actual MAG
candidate-vector and edge ratios.  It is deliberately conservative: E1 needs
TriDB, Milvus, Neo4j, and baseline pgvector resident at the same time, plus room
for WAL, transaction logs, and index-build scratch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tools.e0.common import environment_record, write_json

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[2]


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def estimate() -> dict[str, Any]:
    import psycopg

    prime = ROOT / "data/e0/stark_prime/normalized"
    mag = ROOT / "data/e0/stark_mag/normalized"
    prime_vectors = _parquet_rows(prime / "embeddings.parquet")
    mag_vectors = _parquet_rows(mag / "embeddings.parquet")
    prime_edges = _parquet_rows(prime / "edges.parquet")
    mag_edges = _parquet_rows(mag / "edges.parquet")
    vector_ratio = (mag_vectors * 1536) / (prime_vectors * 1024)
    edge_ratio = mag_edges / prime_edges

    with psycopg.connect(
        host=str(ROOT / ".tridb-pgdata"), port=55432, dbname="tridb_e0_stark"
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_total_relation_size('public.e0_node')")
            tridb_relational_prime = int(cursor.fetchone()[0])
            cursor.execute("SELECT pg_total_relation_size('graph_store.gstore')")
            tridb_graph_prime = int(cursor.fetchone()[0])

    with psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="tridb_wiki",
        user="postgres",
        password="postgres",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_total_relation_size('public.e0_stark_prime_node')"
            )
            polyglot_pg_prime = int(cursor.fetchone()[0])

    estimates = {
        "tridb_relational_and_hnsw": round(tridb_relational_prime * vector_ratio),
        "tridb_native_graph": round(tridb_graph_prime * edge_ratio),
        "polyglot_pgvector": round(polyglot_pg_prime * vector_ratio),
        "polyglot_milvus": round(
            _tree_bytes(ROOT / "baseline/volumes/milvus") * vector_ratio
        ),
        "polyglot_neo4j": round(
            _tree_bytes(ROOT / "baseline/volumes/neo4j/data") * edge_ratio
        ),
    }
    steady = sum(estimates.values())
    headroom = max(10 * GIB, round(steady * 0.20))
    free = shutil.disk_usage(ROOT).free
    required = steady + headroom
    tridb_steady = (
        estimates["tridb_relational_and_hnsw"] + estimates["tridb_native_graph"]
    )
    polyglot_steady = (
        estimates["polyglot_pgvector"]
        + estimates["polyglot_milvus"]
        + estimates["polyglot_neo4j"]
    )

    def isolated_mode(steady_bytes: int) -> dict[str, Any]:
        isolated_headroom = max(5 * GIB, round(steady_bytes * 0.15))
        isolated_required = steady_bytes + isolated_headroom
        return {
            "estimated_steady_state_bytes": steady_bytes,
            "required_build_headroom_bytes": isolated_headroom,
            "required_free_bytes": isolated_required,
            "shortfall_bytes": max(0, isolated_required - free),
            "ready": free >= isolated_required,
        }

    modes = {
        "simultaneous": {
            "estimated_steady_state_bytes": steady,
            "required_build_headroom_bytes": headroom,
            "required_free_bytes": required,
            "shortfall_bytes": max(0, required - free),
            "ready": free >= required,
        },
        "tridb_isolated": isolated_mode(tridb_steady),
        "polyglot_isolated": isolated_mode(polyglot_steady),
    }
    return {
        "schema_version": "e0-stark-mag-capacity-v0.1.0",
        "environment": environment_record(),
        "method": "scale measured PRIME physical footprints by MAG vector-bytes and edge ratios",
        "ratios": {
            "candidate_vector_bytes_mag_over_prime": vector_ratio,
            "directed_edges_mag_over_prime": edge_ratio,
        },
        "estimated_additional_bytes": estimates,
        "estimated_steady_state_bytes": steady,
        "required_build_headroom_bytes": headroom,
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "shortfall_bytes": max(0, required - free),
        "ready": free >= required,
        "modes": modes,
        "scope": "simultaneous TriDB + Polyglot-Tuned MAG deployment",
        "warning": (
            None
            if free >= required
            else (
                "simultaneous capacity gate failed; use audited system-isolated "
                "execution only when both isolated mode gates pass"
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/e0/stark_mag/capacity_preflight.json"),
    )
    args = parser.parse_args(argv)
    report = estimate()
    write_json(args.out, report)
    print(
        f"ready={report['ready']} free={report['observed_free_bytes'] / GIB:.1f} GiB "
        f"required={report['required_free_bytes'] / GIB:.1f} GiB "
        f"shortfall={report['shortfall_bytes'] / GIB:.1f} GiB"
    )
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
