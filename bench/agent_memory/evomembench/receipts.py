"""Content-addressed receipts for GEM Experience Graph retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "evomembench_gem_retrieval_receipt_v0.1.0"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExperienceRetrievalReceipt:
    run_id: str
    manifest_sha256: str
    track: str
    arm: str
    target_episode_uid: str
    cutoff_ordinal: int
    task_signature_sha256: str
    selected_unit_ids: tuple[int, ...]
    selected_episode_uids: tuple[str, ...]
    selected_ordinals: tuple[int, ...]
    injection_sha256: str
    injection_tokens: int
    latency_ms: float
    probes: Mapping[str, Any]

    def unsigned_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def as_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        return {**payload, "receipt_sha256": _digest(payload)}


def build_experience_receipt(
    *,
    run_id: str,
    manifest_sha256: str,
    track: str,
    arm: str,
    target_episode_uid: str,
    cutoff_ordinal: int,
    task_signature: str,
    selected_unit_ids: Sequence[int],
    selected_episode_uids: Sequence[str],
    selected_ordinals: Sequence[int],
    injection_text: str,
    injection_tokens: int,
    latency_ms: float,
    probes: Mapping[str, Any],
) -> ExperienceRetrievalReceipt:
    if not (
        len(selected_unit_ids) == len(selected_episode_uids) == len(selected_ordinals)
    ):
        raise ValueError("selected receipt columns have different lengths")
    if any(value >= cutoff_ordinal for value in selected_ordinals):
        raise ValueError("retrieval receipt contains current/future experience")
    receipt = ExperienceRetrievalReceipt(
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        track=track,
        arm=arm,
        target_episode_uid=target_episode_uid,
        cutoff_ordinal=cutoff_ordinal,
        task_signature_sha256=_hash(task_signature),
        selected_unit_ids=tuple(int(value) for value in selected_unit_ids),
        selected_episode_uids=tuple(selected_episode_uids),
        selected_ordinals=tuple(int(value) for value in selected_ordinals),
        injection_sha256=_hash(injection_text),
        injection_tokens=int(injection_tokens),
        latency_ms=float(latency_ms),
        probes=dict(probes),
    )
    return receipt


def verify_receipt(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("receipt_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return isinstance(observed, str) and observed == _digest(unsigned)
