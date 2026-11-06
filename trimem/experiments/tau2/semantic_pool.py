"""Request-level semantic routing, preserving episode-local dialogue endpoints."""

from contextlib import contextmanager
import threading
import time

import httpx


class SemanticRequestPool:
    """Combine server running+waiting counts with local reservations between polls.

    Only SemFilter requests lease this pool. External load includes the pinned
    agent/user/judge requests, measured once per second. No retry or output cache.
    """

    def __init__(self, bases, *, poll_seconds=1.0, client=None):
        self.bases = tuple(b.rstrip("/").removesuffix("/v1") for b in bases)
        if len(self.bases) < 2 or len(set(self.bases)) != len(self.bases):
            raise ValueError("semantic routing needs at least two distinct endpoints")
        self.local = dict.fromkeys(self.bases, 0)
        self.external = dict.fromkeys(self.bases, 0)
        self.observed = dict.fromkeys(self.bases, 0)
        self.updated = dict.fromkeys(self.bases, 0.0)
        self.failures = dict.fromkeys(self.bases, 0)
        self.assigned = dict.fromkeys(self.bases, 0)
        self.generation_failed = set()
        self.poll_seconds = poll_seconds
        self.client = client or httpx.Client(timeout=1.0, trust_env=False)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._turn = 0

    def refresh(self):
        for base in self.bases:
            try:
                response = self.client.get(base + "/metrics")
                response.raise_for_status()
                counts = {}
                for line in response.text.splitlines():
                    name = line.split("{", 1)[0]
                    if name in (
                        "vllm:num_requests_running",
                        "vllm:num_requests_waiting",
                    ):
                        counts[name] = counts.get(name, 0) + float(
                            line.rsplit(" ", 1)[1]
                        )
                if len(counts) != 2 or any(v < 0 for v in counts.values()):
                    raise ValueError("missing or invalid vLLM load metrics")
                total = sum(counts.values())
                with self._lock:
                    self.observed[base] = total
                    # Do not count our own active requests twice. Local leases
                    # correct this estimate as requests start/end before the next poll.
                    self.external[base] = max(0, total - self.local[base])
                    self.updated[base] = time.monotonic()
                    self.failures[base] = 0
            except Exception:
                with self._lock:
                    self.failures[base] += 1

    def _healthy(self, base):
        return (
            base not in self.generation_failed
            and self.failures[base] < 3
            and time.monotonic() - self.updated[base] < 5
        )

    def unhealthy_reason(self):
        with self._lock:
            bad = [base for base in self.bases if not self._healthy(base)]
        return "unavailable model endpoints: " + ", ".join(bad) if bad else None

    def start(self):
        self.refresh()
        if reason := self.unhealthy_reason():
            self.client.close()
            raise RuntimeError(reason)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(self.poll_seconds):
            self.refresh()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.client.close()

    @contextmanager
    def lease(self):
        with self._lock:
            rotated = self.bases[self._turn :] + self.bases[: self._turn]
            available = [base for base in rotated if self._healthy(base)]
            if not available:
                raise RuntimeError("no healthy semantic endpoint")
            base = min(available, key=lambda b: self.external[b] + self.local[b])
            self._turn = (self.bases.index(base) + 1) % len(self.bases)
            self.local[base] += 1
            self.assigned[base] += 1
        try:
            yield base
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500:
                with self._lock:
                    self.generation_failed.add(base)
            raise
        finally:
            with self._lock:
                self.local[base] -= 1
