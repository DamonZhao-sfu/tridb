"""Hash every Graphiti adapter file that can affect a formal measurement."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PACKAGES = (
    "experiments/graphiti_track_c",
    "experiments/graphiti_track_c_formal",
)


def graphiti_adapter_sha256(repository: str | Path) -> str:
    """Return one path-sensitive digest for the base and formal guard packages."""
    root = Path(repository).resolve()
    digest = hashlib.sha256()
    for relative_root in _PACKAGES:
        package = root / relative_root
        for path in sorted(package.glob("*.py")):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()
