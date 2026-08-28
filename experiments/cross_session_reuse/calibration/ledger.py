"""Append-only evaluator ledger for OpenEvolve calibration sessions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_attempt(
    program_path: str | Path,
    *,
    metrics: dict[str, Any],
    status: str,
    error_message: str = "",
) -> None:
    """Persist one completed evaluator call when a ledger path is configured."""
    ledger_value = os.environ.get("CSR_ATTEMPT_LEDGER")
    if not ledger_value:
        return

    program_sha256 = sha256_file(program_path)
    ledger = Path(ledger_value)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        program_role = "initial" if handle.tell() == 0 else "child"
        row = {
            "schema_version": "csr-calibration-attempt-v0.1.0",
            "recorded_at_unix_ns": time.time_ns(),
            "task_id": os.environ.get("CSR_TASK_ID", "unknown"),
            "session_id": os.environ.get("CSR_SESSION_ID", "unknown"),
            "program_sha256": program_sha256,
            "program_bytes": Path(program_path).stat().st_size,
            "program_role": program_role,
            "initial_sha256_matches": (
                None
                if program_role != "initial"
                else program_sha256 == os.environ.get("CSR_INITIAL_SHA256")
            ),
            "status": status,
            "error_message": error_message[:4000],
            "metrics": metrics,
        }
        payload = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
