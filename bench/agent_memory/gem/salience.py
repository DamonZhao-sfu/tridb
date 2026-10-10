"""The salience signal — the C5/C6 coupling.

C6 requires that repeated retrieval **strictly** reduces a unit's eligibility
for attenuation. That single word is what makes retention relevance-driven
rather than age-driven, and it is why :meth:`ExponentialSalience.reinforce`
carries a positive ``floor`` term: without it the last-ranked hit of a result
set (``rank == k - 1``) would gain exactly zero and C6 would fail for precisely
the hits that are hardest to notice. The property is asserted in the unit suite,
not merely documented here.

**Decay is lazy.** Applying decay on read would touch every row on every query,
which would turn retrieval into a full-relation write. Instead ``forget``
computes decay from ``last_access`` at tick time. The consequence must be stated
in every run manifest: **salience is only current as of the last forget tick.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Defaults are deliberately conservative: a gain that cannot invert the
#: ranking within one retrieval, and a half-life on the order of a day.
DEFAULT_GAIN = 1.0
DEFAULT_FLOOR = 0.05
DEFAULT_LAMBDA = math.log(2) / 86_400.0  # 24h half-life


@dataclass(frozen=True)
class ExponentialSalience:
    """Rises on access, decays exponentially on disuse.

    Implements the ``SaliencePolicy`` Protocol. Applied at BOTH granularities:
    per unit and per field, because [GEM] §3.2 requires sub-unit granularity —
    "part of a unit may be attenuated while the rest stays current".

    The three thresholds are the graded ladder ``forget`` walks. They must stay
    ordered ``theta_archive <= theta_remove <= theta_summary``; the constructor
    checks it, because an out-of-order ladder silently skips rungs instead of
    failing.
    """

    gain: float = DEFAULT_GAIN
    floor: float = DEFAULT_FLOOR
    lam: float = DEFAULT_LAMBDA
    theta_summary: float = 0.50  # below -> compress the history
    theta_remove: float = 0.20  # below -> hide from active retrieval
    theta_archive: float = 0.05  # below -> archive, still recoverable (C5)

    def __post_init__(self) -> None:
        if self.floor <= 0.0:
            # C6 is "strictly reduces eligibility for attenuation". A zero floor
            # makes the last-ranked hit gain nothing, which is not strict.
            raise ValueError("floor must be > 0 so reinforce is STRICTLY increasing")
        if self.gain < 0.0:
            raise ValueError("gain must be >= 0")
        if self.lam < 0.0:
            raise ValueError("lam must be >= 0")
        if not (self.theta_archive <= self.theta_remove <= self.theta_summary):
            raise ValueError(
                "ladder must be ordered theta_archive <= theta_remove <= "
                f"theta_summary; got {self.theta_archive}, {self.theta_remove}, "
                f"{self.theta_summary}"
            )

    def reinforce(self, current: float, *, rank: int, k: int) -> float:
        """Salience after one retrieval hit at ``rank`` out of ``k``.

        Rank-weighted so a top hit gains more than a marginal one, but the
        ``floor`` guarantees the strict increase C6 demands for every hit.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not (0 <= rank < k):
            raise ValueError(f"rank {rank} out of range for k={k}")
        return current + self.gain * (1.0 - rank / k) + self.floor

    def decay(self, current: float, *, seconds_idle: float) -> float:
        """Exponential decay from time since ``last_access``.

        Computed at ``forget`` tick time, never on the read path.
        """
        if seconds_idle < 0.0:
            raise ValueError("seconds_idle must be >= 0")
        return current * math.exp(-self.lam * seconds_idle)

    def rung(self, salience: float) -> str | None:
        """Which ladder rung this salience falls to, or None to stay active.

        Returns the target ``UnitState`` value. Checked most-severe first so a
        salience below every threshold archives rather than merely compressing.
        """
        if salience < self.theta_archive:
            return "archived"
        if salience < self.theta_remove:
            return "hidden"
        if salience < self.theta_summary:
            return "compressed"
        return None
