"""Content-addressed experiment contract for the EvoMemBench GEM track."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "evomembench_gem_manifest_v0.1.0"


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported EvoMemBench GEM manifest")
    return value


def manifest_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_assets(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    base = Path(root).resolve()
    observed: dict[str, str] = {}
    for asset in manifest.get("assets", []):
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = (base / relative).resolve()
        if base not in path.parents:
            raise ValueError(f"asset escapes source root: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"asset checksum mismatch: {relative}")
        observed[str(relative)] = digest
    return observed
