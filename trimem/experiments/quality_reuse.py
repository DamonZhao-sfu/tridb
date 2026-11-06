"""Quality reuse ``quality-reuse-v1`` (RAG-Stack ``gen_fingerprint_dedup``; user decision 2026-10-20).

When a configuration's retrieval for a (task, trial) injects EXACTLY the same memory as an earlier evaluated
episode of another configuration, the agent prompt is identical, so the episode is not run: the earlier episode's
outcome (reward, agent tokens, episode latency) is reused and the new result file is a reuse record.

Key (``episode_key``): experiment, split, task key, trial, trial seed, run memory policy, model and the MEMORY
KEY = sha256 over (memory_form, render version, exact injected text). The concurrency regime is NOT part of the
key (coordinator decision 2026-10-20, "b"): episodes of the c64 rounds 0-5 are reusable from round 6 (c100) on;
every reuse record carries ``source_regime`` so latency can be load-adjusted later. Train and test never share an
index (one file per split, and the split is in the key). NoMemory episodes are never indexed.

The index is a host-wide JSONL file (``ReuseIndex``): one line per CLAIM (key -> the episode that runs it). A
claim is held with ``flock`` on ``pending/<key>.lock`` while its episode runs; a second episode with the same key
waits for it (``ReuseIndex.acquire``), then reuses its result -- or, when the claimer died or failed (no valid
result file), claims the key itself. Lines are appended under an exclusive ``flock`` (``.lock``).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = "quality-reuse-v1"
EMPTY_MEMORY_RULE = (
    "an empty-memory episode is reused only from another MEMORY configuration's empty-memory episode (same key), "
    "never from NoMemory: the prompt is byte-identical, but NoMemory is the paired baseline (its reuse would force "
    "a paired gain of exactly 0 and correlate the baseline with the configuration), and its episodes ran under "
    "another agent class / no-retrieval latency"
)


def memory_key(text: str, memory_form: str, render_version: str) -> str:
    """sha256 over the exact injected string plus the memory_form / render version."""
    h = hashlib.sha256()
    h.update(json.dumps([PROTOCOL, memory_form, render_version]).encode())
    h.update(b"\0")
    h.update(text.encode())
    return h.hexdigest()


def episode_key(
    *,
    experiment: str,
    split: str,
    task: str,
    trial: int,
    trial_seed: int,
    run_policy: str,
    model: str,
    mem_key: str,
) -> str:
    fields = {
        "protocol": PROTOCOL,
        "experiment": experiment,
        "split": split,
        "task": task,
        "trial": int(trial),
        "trial_seed": int(trial_seed),
        "run_policy": run_policy,
        "model": model,
        "memory_key": mem_key,
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def index_path(folder: Path, experiment: str, split: str) -> Path:
    """One index per (experiment, split): train and test never share a file."""
    if split not in ("train", "test"):
        raise ValueError(f"bad split {split!r}")
    eid = hashlib.sha256(experiment.encode()).hexdigest()[:12]
    return Path(folder) / f"index_{split}_{eid}.jsonl"


@dataclass
class Claim:
    key: str
    lock: object  # the open, flock-ed pending file

    def release(self) -> None:
        if self.lock is not None:
            try:
                self.lock.close()
            finally:
                self.lock = None


class ReuseIndex:
    """Host-wide (experiment, split) reuse index: key -> claim records (latest last)."""

    def __init__(
        self, path: Path, wait_poll_s: float = 5.0, max_wait_s: float = 3 * 3600
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending = self.path.parent / "pending"
        self.pending.mkdir(exist_ok=True)
        self.wait_poll_s, self.max_wait_s = wait_poll_s, max_wait_s
        self._mu = threading.Lock()
        self._offset = 0
        self._entries: dict[str, list[dict]] = {}

    # ------------------------------------------------------------ file
    def _refresh(self) -> None:
        with self._mu:
            try:
                with open(self.path, "rb") as f:
                    f.seek(self._offset)
                    data = f.read()
            except FileNotFoundError:
                return
            end = data.rfind(b"\n")
            if end < 0:
                return
            for line in data[: end + 1].splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._entries.setdefault(e["key"], []).append(e)
            self._offset += end + 1

    def append(self, entry: dict) -> None:
        line = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode()
        with open(self.path.with_suffix(".lock"), "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            with open(self.path, "ab") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def entries(self, key: str) -> list[dict]:
        self._refresh()
        with self._mu:
            return list(self._entries.get(key, ()))

    def __len__(self) -> int:
        self._refresh()
        with self._mu:
            return len(self._entries)

    # ------------------------------------------------------------ claims
    def _try_lock(self, key: str):
        f = open(self.pending / f"{key}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            f.close()
            return None

    def find(
        self, key: str, valid: Callable[[dict], dict | None]
    ) -> tuple[dict, dict] | None:
        """(entry, source result) of the newest claim of ``key`` whose source result is valid, else None."""
        for e in reversed(self.entries(key)):
            src = valid(e)
            if src is not None:
                return e, src
        return None

    def acquire(
        self,
        key: str,
        valid: Callable[[dict], dict | None],
        entry: dict,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[str, object]:
        """('hit', (entry, source)) when an evaluated episode has this key; else ('claim', Claim) after recording
        ``entry`` as this key's claim (the caller runs the episode and releases the claim after its result file is
        written). Waits while another live episode holds the key (at most ``max_wait_s``, then runs anyway)."""
        waited = 0.0
        while True:
            hit = self.find(key, valid)
            if hit is not None:
                return "hit", hit
            lock = self._try_lock(key)
            if lock is not None:
                hit = self.find(key, valid)  # finished between the lookup and the lock
                if hit is not None:
                    lock.close()
                    return "hit", hit
                self.append({**entry, "key": key, "kind": "claim", "at": now()})
                return "claim", Claim(key, lock)
            if waited >= self.max_wait_s:
                self.append(
                    {**entry, "key": key, "kind": "claim-unlocked", "at": now()}
                )
                return "claim", Claim(key, None)
            sleep(self.wait_poll_s)
            waited += self.wait_poll_s


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
