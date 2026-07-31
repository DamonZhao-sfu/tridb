"""GPU energy attribution for [AM] §4.2 — the paper's "hardware telemetry".

[AM] §3.3 samples "device power, GPU utilization, VRAM footprint, SM activity,
tensor-core activity, and HBM bandwidth", aligns samples with phase markers, and
integrates power over each interval. Table 3 and Fig. 4 are the result: total kJ
per arm, and joules per correct answer. Without this module those two columns
are simply absent — which is the state the GEM implementation status doc records
as open gate **M2**.

Two acquisition methods, and which one ran is reported rather than assumed:

``energy_counter``
    ``nvmlDeviceGetTotalEnergyConsumption`` — a monotonic millijoule counter
    maintained by the driver since boot (Volta and later). Windowing is a
    subtraction, so a phase shorter than the poll interval is still exact up to
    interpolation of the counter, and no power spike is ever missed between
    polls. Preferred whenever the device supports it.

``power_integration``
    Trapezoidal integration of ``nvmlDeviceGetPowerUsage``. The fallback. A
    100 ms poll cannot see sub-100 ms transients, so a short retrieval phase is
    an estimate. ``coverage`` says how much of the requested window actually had
    samples; a caller reporting joules from a window with coverage < 1 is
    reporting an extrapolation.

**Absent NVML is not an error.** ``pynvml`` is an optional extra and the
hardware-independent test suite must run without a GPU. The sampler then reports
``available=False`` and every window returns ``joules=None``, which propagates to
``PhaseCost.gpu_joules`` — the same NULL the schema has always allowed. A run
without energy is a run with three of Table 3's five columns, not a failed run.

**Multi-GPU honesty.** This box has two devices; [AM] ran one GPU per SLURM job.
Summing every visible device charges an idle neighbour's ~30 W to the workload.
The default therefore follows ``CUDA_VISIBLE_DEVICES`` when set, and whichever
devices were sampled are named in :meth:`describe`.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

#: Poll period. [AM] does not publish theirs; 100 ms keeps NVML overhead under a
#: percent of a core while still resolving a ~1 s retrieval phase.
DEFAULT_INTERVAL_SECONDS = 0.1

METHOD_COUNTER = "energy_counter"
METHOD_POWER = "power_integration"


@dataclass(frozen=True)
class EnergyWindow:
    """Energy attributed to one phase interval on the caller's timeline."""

    joules: float | None
    mean_watts: float | None
    samples: int
    coverage: float
    method: str | None

    @classmethod
    def unavailable(cls) -> EnergyWindow:
        return cls(joules=None, mean_watts=None, samples=0, coverage=0.0, method=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joules": self.joules,
            "mean_watts": self.mean_watts,
            "samples": self.samples,
            "coverage": self.coverage,
            "method": self.method,
        }


def _visible_device_indices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    indices: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            indices.append(int(piece))
        except ValueError:
            # A UUID form (GPU-xxxx) cannot be mapped to an NVML index here.
            # Fall back to "all devices" and let describe() show the truth.
            return None
    return indices or None


class GpuEnergySampler:
    """Background NVML poller with phase-window attribution.

    Start it once for the whole run and ask for windows afterwards; the caller
    stamps phase boundaries with ``time.perf_counter()`` and this class keeps
    its samples on that same monotonic clock, which is what [AM] §3.3 means by
    "recorded on the same monotonic timeline".
    """

    def __init__(
        self,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        device_indices: Sequence[int] | None = None,
        enabled: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self._requested_indices = (
            list(device_indices) if device_indices is not None else None
        )
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._watts: list[float] = []
        self._millijoules: list[float] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._nvml: Any = None
        self._handles: list[Any] = []
        self._names: list[str] = []
        self._method: str | None = None
        self._init_error: str | None = None
        self._setup()

    # -- lifecycle -------------------------------------------------------

    def _setup(self) -> None:
        if not self.enabled:
            self._init_error = "disabled by caller"
            return
        try:
            import pynvml
        except ImportError as exc:
            self._init_error = f"pynvml not installed ({exc})"
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            wanted = self._requested_indices
            if wanted is None:
                wanted = _visible_device_indices()
            if wanted is None:
                wanted = list(range(count))
            handles = []
            names = []
            for index in wanted:
                if not 0 <= index < count:
                    raise RuntimeError(
                        f"device index {index} out of range 0..{count - 1}"
                    )
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                handles.append(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                names.append(name.decode() if isinstance(name, bytes) else str(name))
            if not handles:
                raise RuntimeError("no NVML devices selected")
            # Probe the counter once; a device that raises here falls back for
            # the whole run rather than switching method mid-flight.
            method = METHOD_COUNTER
            try:
                for handle in handles:
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:  # noqa: BLE001 — capability probe, not a failure
                method = METHOD_POWER
            self._nvml = pynvml
            self._handles = handles
            self._names = names
            self._method = method
        except Exception as exc:  # noqa: BLE001 — reported via available/describe
            self._init_error = f"{type(exc).__name__}: {exc}"
            self._nvml = None
            self._handles = []

    @property
    def available(self) -> bool:
        return bool(self._handles)

    @property
    def method(self) -> str | None:
        return self._method

    def start(self) -> GpuEnergySampler:
        if not self.available or self._thread is not None:
            return self
        self._stop.clear()
        # Sample synchronously before the thread exists. Otherwise the series
        # starts one poll period late and every phase in that opening window —
        # the first history's construction, typically — falls outside the
        # samples and reports joules=None. Measured: a 17 ms construction at a
        # 50 ms interval lost its whole reading.
        try:
            self._sample_once()
        except Exception:  # noqa: BLE001 — a lost first sample only costs
            pass  # coverage, which window() reports honestly.
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-energy-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        # Symmetric to start(): close the series at the end of the run so the
        # final phase is not truncated by up to one poll period.
        try:
            self._sample_once()
        except Exception:  # noqa: BLE001 — same trade as the opening sample
            pass
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 — teardown must not mask a result
                pass

    def mark(self) -> float:
        """Force a sample now, and return the instant it was taken.

        Polling alone brackets a phase only to within one interval, so a window
        closed the moment a phase ends can miss up to ``interval_seconds`` of it
        — a systematic UNDER-count of every short phase, and at a 100 ms poll
        that is 10% of a one-second query. Marking both boundaries costs two
        NVML reads and makes the window exact.

        Safe when the sampler is unavailable: it records nothing and just
        returns the clock.
        """
        if self.available:
            try:
                self._sample_once()
            except Exception:  # noqa: BLE001 — a lost mark costs coverage only
                pass
        return time.perf_counter()

    def __enter__(self) -> GpuEnergySampler:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- sampling --------------------------------------------------------

    def _sample_once(self) -> None:
        watts = 0.0
        millijoules = 0.0
        for handle in self._handles:
            watts += self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            if self._method == METHOD_COUNTER:
                millijoules += float(
                    self._nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                )
        stamp = time.perf_counter()
        with self._lock:
            self._stamps.append(stamp)
            self._watts.append(watts)
            self._millijoules.append(millijoules)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:  # noqa: BLE001 — a dropped sample degrades
                # coverage, which the window reports; it must not kill the run.
                pass
            self._stop.wait(self.interval_seconds)

    # -- attribution -----------------------------------------------------

    def window(self, start: float, end: float) -> EnergyWindow:
        """Energy between two ``perf_counter`` stamps."""
        if not self.available or end <= start:
            return EnergyWindow.unavailable()
        with self._lock:
            stamps = list(self._stamps)
            watts = list(self._watts)
            millijoules = list(self._millijoules)
        if len(stamps) < 2:
            return EnergyWindow.unavailable()

        low = max(start, stamps[0])
        high = min(end, stamps[-1])
        if high <= low:
            return EnergyWindow(
                joules=None,
                mean_watts=None,
                samples=0,
                coverage=0.0,
                method=self._method,
            )
        coverage = (high - low) / (end - start)
        inside = sum(1 for stamp in stamps if low <= stamp <= high)

        if self._method == METHOD_COUNTER:
            joules = (
                _interpolate(stamps, millijoules, high)
                - _interpolate(stamps, millijoules, low)
            ) / 1000.0
        else:
            joules = _trapezoid(stamps, watts, low, high)
        span = high - low
        return EnergyWindow(
            joules=joules,
            mean_watts=None if span <= 0 else joules / span,
            samples=inside,
            coverage=coverage,
            method=self._method,
        )

    def describe(self) -> dict[str, Any]:
        with self._lock:
            samples = len(self._stamps)
        return {
            "available": self.available,
            "method": self._method,
            "interval_seconds": self.interval_seconds,
            "devices": list(self._names),
            "device_count": len(self._handles),
            "samples": samples,
            "init_error": self._init_error,
            "note": (
                "Energy is attributed by integrating over phase markers on the "
                "run's monotonic clock. Absolute joules are hardware-specific "
                "and must not be compared against [AM]'s H100 numbers; only the "
                "spread across arms measured on this box is comparable."
            ),
        }


def _interpolate(stamps: Sequence[float], values: Sequence[float], at: float) -> float:
    """Piecewise-linear value of a monotone series at ``at`` (clamped)."""
    if at <= stamps[0]:
        return float(values[0])
    if at >= stamps[-1]:
        return float(values[-1])
    lower = 0
    upper = len(stamps) - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if stamps[middle] <= at:
            lower = middle
        else:
            upper = middle
    span = stamps[upper] - stamps[lower]
    if span <= 0:
        return float(values[upper])
    fraction = (at - stamps[lower]) / span
    return float(values[lower]) + fraction * (
        float(values[upper]) - float(values[lower])
    )


def _trapezoid(
    stamps: Sequence[float],
    watts: Sequence[float],
    low: float,
    high: float,
) -> float:
    """Integrate a piecewise-linear power series over ``[low, high]``."""
    total = 0.0
    for index in range(len(stamps) - 1):
        left, right = stamps[index], stamps[index + 1]
        if right <= low or left >= high:
            continue
        segment_low = max(left, low)
        segment_high = min(right, high)
        if segment_high <= segment_low:
            continue
        total += (
            (
                _interpolate(stamps, watts, segment_low)
                + _interpolate(stamps, watts, segment_high)
            )
            / 2.0
            * (segment_high - segment_low)
        )
    return total
