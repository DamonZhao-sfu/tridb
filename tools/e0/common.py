"""Shared, deterministic artifact helpers for E0 data preparation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import urllib.request
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    label = path.relative_to(relative_to) if relative_to else path
    return {
        "path": str(label),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "openevolve", "pyarrow", "stark-qa")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def endpoint_models(base_url: str, timeout: int = 10) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return list(json.loads(response.read().decode("utf-8")).get("data", []))


def hf_snapshot_revisions(model_artifact: str) -> list[str]:
    """Return locally visible HF snapshot revisions for an org/model artifact."""
    cache_name = f"models--{model_artifact.replace('/', '--')}"
    user = Path.home().name
    candidates = [
        Path(os.environ.get("HF_HOME", "")) / "hub"
        if os.environ.get("HF_HOME")
        else None,
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "hf-cache" / "hub",
        Path("/localhome") / user / "hf-cache" / "hub",
    ]
    revisions: set[str] = set()
    for hub in candidates:
        if hub is None:
            continue
        model_dir = hub / cache_name
        ref = model_dir / "refs" / "main"
        if ref.exists():
            revisions.add(ref.read_text(encoding="utf-8").strip())
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            revisions.update(path.name for path in snapshots.iterdir() if path.is_dir())
    return sorted(value for value in revisions if value)
