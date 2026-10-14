"""Atomic, resumable research trial records (not a second database graph store)."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from trimem.optimizer.schema import CandidateSpec, stable_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".trial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrialArchive:
    """A run is one fixed protocol/cohort/objective/search manifest.

    Advisory locking coordinates research-controller writers. Evaluation never
    runs while the lock is held. Existing observations cannot be overwritten.
    """

    def __init__(self, path: str | Path, manifest: dict):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.path / "trials"
        self.trials_path.mkdir(exist_ok=True)
        if not manifest.get("cohort_id") or not manifest.get("protocol_id"):
            raise ValueError("manifest requires cohort_id and protocol_id")
        # Copy through JSON so caller mutation cannot change the archive identity.
        self.manifest = json.loads(json.dumps(manifest, allow_nan=False))
        self.manifest_hash = stable_hash(self.manifest)
        with self._lock():
            manifest_path = self.path / "manifest.json"
            if manifest_path.exists():
                stored = json.loads(manifest_path.read_text())
                digest_path = self.path / "manifest.sha256"
                if (
                    not digest_path.exists()
                    or digest_path.read_text().strip() != stable_hash(stored)
                ):
                    raise ValueError("archive manifest integrity check failed")
                if stored != self.manifest:
                    raise ValueError(
                        "archive manifest differs; start a new run directory"
                    )
            else:
                _write_atomic(manifest_path, self.manifest)
                (self.path / "manifest.sha256").write_text(self.manifest_hash + "\n")

    @contextmanager
    def _lock(self):
        with (self.path / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _trial_path(self, trial_id: str) -> Path:
        try:
            normalized = uuid.UUID(trial_id).hex
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("invalid trial_id") from exc
        if trial_id != normalized:
            raise ValueError("trial_id must be canonical UUID hex")
        return self.trials_path / f"{normalized}.json"

    def records(self) -> list[dict]:
        # Atomic rename ensures each reader sees either the old or new whole row.
        return sorted(
            (
                self._validate_record(json.loads(p.read_text()))
                for p in self.trials_path.glob("*.json")
            ),
            key=lambda row: (row["created_at"], row["trial_id"]),
        )

    def _validate_record(self, row: dict) -> dict:
        if row.get("manifest_hash") != self.manifest_hash:
            raise ValueError("trial manifest identity differs from archive")
        if CandidateSpec.from_dict(row["action"]).action_hash != row["action_hash"]:
            raise ValueError("trial action identity mismatch")
        for key in ("protocol_id", "cohort_id"):
            if row[key] != self.manifest[key]:
                raise ValueError("trial protocol/cohort identity mismatch")
        if (
            row.get("observation") is not None
            and stable_hash(row["observation"]) != row["observation_hash"]
        ):
            raise ValueError("stored observation integrity check failed")
        return row

    def get(self, trial_id: str) -> dict:
        return self._validate_record(json.loads(self._trial_path(trial_id).read_text()))

    def reserve(self, action: CandidateSpec, *, proposal: dict | None = None) -> dict:
        with self._lock():
            if any(
                r["action_hash"] == action.action_hash
                and (
                    r["status"] == "pending"
                    or not self.manifest.get("search_settings", {}).get(
                        "allow_repeats", False
                    )
                )
                for r in self.records()
            ):
                raise ValueError(
                    "this action is pending or was already reserved without repeats"
                )
            row = {
                "schema_version": 1,
                "manifest_hash": self.manifest_hash,
                "trial_id": uuid.uuid4().hex,
                "action": action.to_dict(),
                "action_hash": action.action_hash,
                "cohort_id": self.manifest["cohort_id"],
                "protocol_id": self.manifest["protocol_id"],
                "status": "pending",
                "proposal": proposal or {},
                "created_at": _now(),
            }
            _write_atomic(self._trial_path(row["trial_id"]), row)
            return row

    def tell(self, trial_id: str, observation: dict) -> dict:
        observation = json.loads(json.dumps(observation, allow_nan=False))
        with self._lock():
            row = self.get(trial_id)
            for key in ("action_hash", "protocol_id", "cohort_id"):
                if observation.get(key) != row[key]:
                    raise ValueError(f"observation {key} differs from reserved trial")
            if row["status"] != "pending":
                if row.get("observation") == observation:
                    return row  # idempotent delivery after reconnect/restart
                raise ValueError("completed trial cannot be overwritten")
            row.update(
                status="observed",
                observation=observation,
                observed_at=_now(),
                observation_hash=stable_hash(observation),
            )
            _write_atomic(self._trial_path(trial_id), row)
            return row

    def fail(self, trial_id: str, reason: str, *, evidence: dict | None = None) -> dict:
        """Explicitly close a crashed trial, retaining any available cost evidence."""
        if not reason:
            raise ValueError("failure requires a reason")
        with self._lock():
            row = self.get(trial_id)
            if row["status"] != "pending":
                raise ValueError("only a pending trial can fail")
            row.update(
                status="failed",
                error=reason,
                evidence=evidence or {},
                observed_at=_now(),
            )
            _write_atomic(self._trial_path(trial_id), row)
            return row
