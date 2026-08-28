"""Capture machine, database, client, and serving evidence for a formal run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence
import urllib.request


def _command(argv: Sequence[str], *, timeout: float = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "argv": list(argv),
            "exit_status": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"argv": list(argv), "error": f"{type(exc).__name__}: {exc}"}


def _endpoint(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            return {"url": url, "payload": json.load(response)}
    except Exception as exc:
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def _model_processes() -> dict[str, Any]:
    observed = _command(("ps", "-eo", "pid=,lstart=,args="))
    if "stdout" in observed:
        observed["stdout"] = "\n".join(
            line
            for line in str(observed["stdout"]).splitlines()
            if any(token in line.casefold() for token in ("vllm", "api_server"))
        )
    return observed


def _postgres(dsn: str) -> dict[str, Any]:
    from psycopg import connect

    try:
        with connect(dsn) as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT version(),current_setting('server_version'),"
                    " current_setting('block_size'),current_setting('wal_level')"
                )
                version, server_version, block_size, wal_level = cursor.fetchone()
                cursor.execute(
                    "SELECT extname,extversion FROM pg_extension ORDER BY extname"
                )
                extensions = [
                    {"name": str(name), "version": str(value)}
                    for name, value in cursor.fetchall()
                ]
        return {
            "version": str(version),
            "server_version": str(server_version),
            "block_size": int(block_size),
            "wal_level": str(wal_level),
            "extensions": extensions,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _baseline_versions() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    try:
        from pymilvus import connections, utility

        alias = "evomembench_hardware_capture"
        connections.connect(
            alias=alias,
            host=os.environ.get("MILVUS_HOST", "127.0.0.1"),
            port=os.environ.get("MILVUS_PORT", "19530"),
        )
        evidence["milvus"] = {"server_version": utility.get_server_version(using=alias)}
        connections.disconnect(alias)
    except Exception as exc:
        evidence["milvus"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        driver = GraphDatabase.driver(
            uri,
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "testpassword"),
            ),
        )
        with driver.session() as session:
            components = [dict(row) for row in session.run("CALL dbms.components()")]
        driver.close()
        evidence["neo4j"] = {"uri": uri, "components": components}
    except Exception as exc:
        evidence["neo4j"] = {"error": f"{type(exc).__name__}: {exc}"}
    return evidence


def capture(args: argparse.Namespace) -> dict[str, Any]:
    packages = {}
    for name in ("neo4j", "pymilvus", "psycopg", "tokenizers", "openai", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": "evomembench_system_hardware_v0.1.0",
        "claim_mode": args.hardware_mode,
        "platform": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "uname": list(platform.uname()),
        },
        "cpu": _command(("lscpu", "-J")),
        "memory": {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:"))
        },
        "gpu": {
            "inventory": _command(
                (
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
                    "--format=csv,noheader",
                )
            ),
            "cuda": _command(("nvidia-smi",)),
        },
        "tridb": {
            "git_commit": _command(("git", "rev-parse", "HEAD")),
            "git_status": _command(("git", "status", "--porcelain")),
            "build_flags": args.tridb_build_flags,
            "native_database": _postgres(args.native_dsn),
            "scale_database": _postgres(args.scale_dsn),
            "pg_config": _command(("pg_config", "--configure")),
        },
        "multi_system": {
            **_baseline_versions(),
            "postgresql": _postgres(args.baseline_dsn),
            "tuning": {
                "milvus_index": "HNSW",
                "M": 16,
                "efConstruction": 200,
                "efSearch": 128,
                "ann_overfetch": 8,
                "neo4j_graph_work_budget": 65536,
            },
        },
        "serving": {
            "answer": _endpoint(args.answer_base_url),
            "embedding": _endpoint(args.embedding_base_url),
            "processes": _model_processes(),
            "answer_context_limit": 262144,
            "answer_max_tokens": 4096,
            "judge_max_tokens": 4096,
            "judge_max_tokens_source": "vLLM generation_config.max_new_tokens",
            "temperature": 0.0,
            "prefix_cache_policy": "enabled_equally_on shared answer endpoint",
        },
        "python_packages": packages,
        "protocol": json.loads(Path(args.protocol).read_text()),
        "timeouts": {
            "client_seconds": 300,
            "formal_hard_timeout_seconds": 604800,
            "tool_timeout_source": "pinned runner configuration",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hardware-mode", choices=("gx10", "off_target"), required=True
    )
    parser.add_argument("--native-dsn", required=True)
    parser.add_argument("--scale-dsn", required=True)
    parser.add_argument("--baseline-dsn", required=True)
    parser.add_argument("--answer-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tridb-build-flags", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing hardware evidence: {args.output}")
    args.output.write_text(
        json.dumps(capture(args), ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
