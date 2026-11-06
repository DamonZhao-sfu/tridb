"""Process memory accounting + a self-check for long-running experiment workers (stdlib only).

Used by the AppWorld and τ² runners (each venv imports it from the repo):

- ``usage()``: RSS of this process and of its live descendants, peak RSS, open fds and threads, read
  from ``/proc`` (Linux). Written into every per-task result so a slow leak is visible in the data.
- ``RssGuard``: a ``--max_rss_gb`` budget. ``soft_exceeded()`` is checked between tasks (stop taking
  work at ``SOFT_FRACTION`` of the budget, so the worker exits cleanly before reaching it; the per-task
  result file is the resume unit, so a rerun picks up). ``start_watchdog()`` is the backstop for a
  single task that blows through the hard budget: it prints a clear error and ends the process with
  ``EXIT_RSS`` (the in-flight task writes no result file, so resume re-runs it).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
SOFT_FRACTION = 0.9
EXIT_RSS = 3  # stopped on the RSS budget without finishing any task in this life (do not respawn)
EXIT_RECYCLE = (
    75  # EX_TEMPFAIL: stopped between tasks after progress (respawn to continue)
)


def _rss_mb(pid: int | str) -> float:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * PAGE / 2**20
    except (OSError, ValueError, IndexError):
        return 0.0


def _status(pid: int | str) -> dict[str, str]:
    out = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            k, _, v = line.partition(":")
            out[k] = v.strip()
    except OSError:
        pass
    return out


def children_of(pid: int) -> list[int]:
    """Live descendants of ``pid`` (via ``/proc/<pid>/task/*/children``)."""
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        kids = []
        for t in Path(f"/proc/{p}/task").glob("*/children"):
            try:
                kids += [int(x) for x in t.read_text().split()]
            except (OSError, ValueError):
                continue
        for k in kids:
            if k not in seen:
                seen.append(k)
                stack.append(k)
    return seen


def fd_count(pid: int | str = "self") -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return -1


def usage(pid: int | None = None) -> dict:
    """``{rss_mb, children_rss_mb, total_rss_mb, peak_rss_mb, fds, threads, children}`` of ``pid``
    (default: this process)."""
    pid = os.getpid() if pid is None else pid
    st = _status(pid)
    kids = children_of(pid)
    own = _rss_mb(pid)
    kid_rss = sum(_rss_mb(k) for k in kids)
    hwm = st.get("VmHWM", "0 kB").split()[0]
    return {
        "rss_mb": round(own, 1),
        "children_rss_mb": round(kid_rss, 1),
        "total_rss_mb": round(own + kid_rss, 1),
        "peak_rss_mb": round(int(hwm) / 1024, 1) if hwm.isdigit() else None,
        "fds": fd_count(pid),
        "threads": int(st.get("Threads", "0") or 0),
        "children": len(kids),
    }


class RssGuard:
    """``--max_rss_gb`` budget on this process + its children. ``max_rss_gb`` None/0 = disabled."""

    def __init__(
        self, max_rss_gb: float | None, *, soft_fraction: float = SOFT_FRACTION
    ):
        if max_rss_gb is not None and max_rss_gb < 0:
            raise ValueError("--max_rss_gb must be >= 0")
        if not 0 < soft_fraction <= 1:
            raise ValueError("soft_fraction must be in (0, 1]")
        self.limit_mb = (max_rss_gb or 0) * 1024
        self.soft_mb = self.limit_mb * soft_fraction
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.limit_mb > 0

    def soft_exceeded(self, total_rss_mb: float | None = None) -> bool:
        if not self.enabled:
            return False
        rss = usage()["total_rss_mb"] if total_rss_mb is None else total_rss_mb
        return rss >= self.soft_mb

    def message(self, total_rss_mb: float, what: str) -> str:
        return (
            f"RSS {total_rss_mb:.0f} MB (process + children) {what} --max_rss_gb "
            f"{self.limit_mb / 1024:g} GB; stopping this worker. Finished tasks are saved; "
            "re-run the same command to resume."
        )

    def start_watchdog(
        self, interval_s: float = 2.0, exit_fn=os._exit
    ) -> threading.Thread | None:
        """Hard backstop: a daemon thread that ends the process once the hard budget is reached."""
        if not self.enabled:
            return None

        def watch():
            while not self._stop.wait(interval_s):
                rss = usage()["total_rss_mb"]
                if rss >= self.limit_mb:
                    print(
                        "FATAL: " + self.message(rss, "reached"),
                        file=sys.stderr,
                        flush=True,
                    )
                    for k in children_of(os.getpid()):
                        try:
                            os.kill(k, 9)
                        except OSError:
                            pass
                    exit_fn(EXIT_RSS)
                    return

        t = threading.Thread(target=watch, name="rss-watchdog", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
