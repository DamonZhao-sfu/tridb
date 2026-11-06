"""Host-wide episode -> endpoint affinity for the τ² pool round (protocol ``ROUTING``).

Every LLM call of one episode (agent, user simulator, official NL judge, SemFilter) goes to ONE endpoint;
that already held per process with ``runner.EndpointPool``, which picks the endpoint with the fewest
in-flight episodes OF ITS OWN PROCESS. The pool round runs one process per arm (11 in round 0, 5 episodes
each), so every process broke the tie the same way and the host ran 33 episodes on 8030 vs 22 on 8031
(results/tau2_pool_20261019, first 20 minutes). ``HostEndpointPool`` counts in-flight episodes HOST-wide:

* one lease file per in-flight episode under ``<folder>/<endpoint>/``, held with ``flock`` for the episode's
  lifetime (released by the kernel when the holder dies; such stale files are removed on the next count);
* the choice (count live leases, pick the least-loaded endpoint, create + lock the lease) runs under one
  host-wide ``flock`` on ``<folder>/.lock``, so concurrent processes never pick on a stale count;
* ties go to the endpoint this process has assigned fewer episodes to, then to the configured order.

Transport only: no prompt, sampling parameter, seed or evaluator changes (unit test
``test_tau2_endpoint_affinity``: the per-episode official payloads equal the single-endpoint ones except
``api_base``). Same interface as ``runner.EndpointPool`` (``bases``, ``assigned``, ``load``, ``lease()``).
"""

from __future__ import annotations

import fcntl
import os
import re
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROUTING = "episode-sticky-hostwide-least-inflight-v1"
LEASE_DIR = Path("/tmp/trimem-tau2-endpoint-leases")


def _slug(base: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


class HostEndpointPool:
    """Pins each episode to the endpoint with the fewest in-flight episodes on this HOST (all processes)."""

    def __init__(self, bases: list[str] | tuple[str, ...], folder: Path = LEASE_DIR):
        bases = tuple(b.rstrip("/") for b in bases)
        if not bases or len(set(bases)) != len(bases):
            raise ValueError("endpoint pool needs distinct endpoints")
        self.bases, self.folder = bases, Path(folder)
        self.dirs = {b: self.folder / _slug(b) for b in bases}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.load = dict.fromkeys(bases, 0)  # this process's in-flight episodes
        self.assigned = dict.fromkeys(bases, 0)  # this process's episodes so far
        self._lock = (
            threading.Lock()
        )  # flock is per open file: threads need their own guard

    @contextmanager
    def _host_lock(self) -> Iterator[None]:
        with self._lock, open(self.folder / ".lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def live(self) -> dict[str, int]:
        """In-flight episodes per endpoint, host-wide (removes leases whose holder is gone)."""
        out = dict.fromkeys(self.bases, 0)
        for b, d in self.dirs.items():
            for p in d.iterdir():
                try:
                    f = open(p)
                except FileNotFoundError:
                    continue
                with f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        out[b] += 1  # held by a live episode
                        continue
                    p.unlink(
                        missing_ok=True
                    )  # stale: its holder exited without releasing
        return out

    @contextmanager
    def lease(self) -> Iterator[str]:
        with self._host_lock():
            live = self.live()
            base = min(
                self.bases,
                key=lambda b: (live[b], self.assigned[b], self.bases.index(b)),
            )
            path = self.dirs[base] / f"{os.getpid()}-{uuid.uuid4().hex}"
            f = open(path, "w")
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.load[base] += 1
            self.assigned[base] += 1
        try:
            yield base
        finally:
            with self._host_lock():
                path.unlink(missing_ok=True)
                f.close()
                self.load[base] -= 1
