"""Preparation completion marker of a TRACE v0.7.0 run (fix round 1, finding 4).

``prepared.json`` is written LAST by ``tools.prepare_trace_v070_campaign`` (after the manifest, the main ledger,
the imported configurations, the NoMemory ledger, ``samples.json`` and every post-check). The queue and the
launcher test the marker, never the manifest. A run directory without the marker is an interrupted preparation:
``set_aside_incomplete`` renames it to ``<dir>.incomplete-<epoch>`` (only if no evaluation ever started there) so
the preparation can be replayed from scratch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

MARKER = "prepared.json"


def _manifest_sha(run_dir: Path) -> str:
    return hashlib.sha256((Path(run_dir) / "manifest.json").read_bytes()).hexdigest()


def mark_prepared(run_dir, manifest, *, kind: str) -> dict:
    from .trace_v070_baseline import baseline_entry

    run_dir = Path(run_dir)
    nm = baseline_entry(run_dir, "TRAIN")
    if nm is None:
        raise ValueError("cannot mark a run prepared without its NoMemory TRAIN ledger")
    body = {"kind": kind, "prepared_at": time.time(), "manifest_sha256": _manifest_sha(run_dir),
            "protocol": manifest["protocol"], "nomemory_key": nm["key"],
            "nomemory_train_complete": nm["complete"], "search_backend": manifest.get("search_backend")}
    tmp = run_dir / (MARKER + ".tmp")
    tmp.write_text(json.dumps(body, indent=1) + "\n")
    os.replace(tmp, run_dir / MARKER)
    return body


def check_prepared(run_dir) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / MARKER
    if not path.exists():
        raise SystemExit(f"{run_dir}: no {MARKER} (preparation incomplete); re-run the prepare command")
    body = json.loads(path.read_text())
    if body["manifest_sha256"] != _manifest_sha(run_dir):
        raise SystemExit(f"{run_dir}: manifest.json changed after preparation ({MARKER} sha mismatch)")
    return body


def evaluation_started(run_dir) -> bool:
    run_dir = Path(run_dir)
    return any(p.is_dir() for p in (run_dir / "episodes", run_dir / "baselines/nomemory/episodes")) or (
        run_dir / "status.json").exists()


def set_aside_incomplete(run_dir) -> Path | None:
    """Rename an interrupted preparation (no marker, no evaluation) aside; refuse when evaluation started."""
    run_dir = Path(run_dir)
    if not run_dir.exists() or (run_dir / MARKER).exists():
        return None
    if evaluation_started(run_dir):
        raise SystemExit(f"{run_dir}: no {MARKER} but evaluation evidence exists; inspect before re-preparing")
    dst = run_dir.with_name(f"{run_dir.name}.incomplete-{int(time.time())}")
    run_dir.rename(dst)
    return dst
