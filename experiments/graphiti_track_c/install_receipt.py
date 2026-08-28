"""Write an immutable-style receipt for the isolated Graphiti installation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_NEO4J_VERSION = "5.26.6"
EXPECTED_CYPHER_SHELL_VERSION = "Cypher-Shell 5.26.6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*command: str) -> str:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _backend_identity(neo4j_home: Path) -> dict[str, str]:
    neo4j = neo4j_home / "bin/neo4j"
    cypher_shell = neo4j_home / "bin/cypher-shell"
    missing = [str(path) for path in (neo4j, cypher_shell) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Neo4j runtime is incomplete: {missing}")
    neo4j_version = _run(str(neo4j), "--version")
    cypher_shell_version = _run(str(cypher_shell), "--version")
    if neo4j_version != EXPECTED_NEO4J_VERSION:
        raise RuntimeError(
            f"Neo4j version mismatch: {neo4j_version} != {EXPECTED_NEO4J_VERSION}"
        )
    if cypher_shell_version != EXPECTED_CYPHER_SHELL_VERSION:
        raise RuntimeError(
            "Cypher Shell version mismatch: "
            f"{cypher_shell_version} != {EXPECTED_CYPHER_SHELL_VERSION}"
        )
    return {
        "home": str(neo4j_home),
        "neo4j_version": neo4j_version,
        "cypher_shell_version": cypher_shell_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--neo4j-home", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    venv = Path(args.venv).resolve()
    neo4j_home = Path(args.neo4j_home).resolve()
    output = Path(args.output).resolve()
    commit = _run("git", "-C", str(source), "rev-parse", "HEAD")
    tracked_status = _run(
        "git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"
    )
    distributions = sorted(
        {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }.items()
    )
    import graphiti_core

    payload = {
        "schema_version": "table5_track_c_install_manifest_v0.2.0",
        "system": "graphiti_zep_oss_proxy",
        "display_label": "Graphiti (Zep OSS proxy)",
        "interpretation_boundary": "not production Zep",
        "status": "installed",
        "installed": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/getzep/graphiti.git",
            "root": str(source),
            "commit": commit,
            "tracked_dirty": bool(tracked_status),
            "pyproject_sha256": _sha256(source / "pyproject.toml"),
            "uv_lock_sha256": _sha256(source / "uv.lock"),
        },
        "environment": {
            "venv": str(venv),
            "python": platform.python_version(),
            "python_executable": str(Path(os.sys.executable).absolute()),
            "python_executable_resolved": str(Path(os.sys.executable).resolve()),
            "platform": platform.platform(),
            "graphiti_core_version": importlib.metadata.version("graphiti-core"),
            "graphiti_core_module": str(Path(graphiti_core.__file__).resolve()),
            "distributions": [
                {"name": name, "version": version} for name, version in distributions
            ],
        },
        "models": {
            "llm": "Qwen/Qwen3-32B (FP8 verified by endpoint receipt)",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
        },
        "backend": "Neo4j Community 5.26.6",
        "backend_identity": _backend_identity(neo4j_home),
    }
    _write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
