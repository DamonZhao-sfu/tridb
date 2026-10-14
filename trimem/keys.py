"""Cache key shared with the C extension (``trimem_key_hex`` in src/trimem_pg/trimem_pg.c)."""

from __future__ import annotations

import hashlib

_SEP = b"\x1f"


def sem_key(op: str, model: str, prompt_version: str, predicate: str, text: str) -> str:
    """sha256 hex of ``op ␟ model ␟ prompt_version ␟ predicate ␟ input`` (UTF-8)."""
    parts = (op, model, prompt_version, predicate, text)
    return hashlib.sha256(_SEP.join(p.encode("utf-8") for p in parts)).hexdigest()
