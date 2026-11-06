"""Trust region, proposal compute budget and anchor order (spec §8).

All state is a pure function of the round history (``replay``), so resume
needs no state file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch

from trimem.optimizer.latent_plan_surrogate import measured_hv
from trimem.optimizer.plan_schema import digest

H0, H_MIN, H_MAX, GAIN_EPS, STREAK = 0.4, 0.05, 1.0, 1e-4, 3
ANCHOR_CALL_CAP = 100


@dataclass(frozen=True)
class TrustState:
    h: float = H0
    up: int = 0
    down: int = 0
    resets: int = 0


def step(s: TrustState, gain: float) -> TrustState:
    if gain > GAIN_EPS:
        s = replace(s, up=s.up + 1, down=0)
    else:
        s = replace(s, down=s.down + 1, up=0)
    if s.up >= STREAK:
        s = replace(s, h=min(2 * s.h, H_MAX), up=0)
    if s.down >= STREAK:
        if s.h <= H_MIN * (1 + 1e-9):
            s = replace(s, h=H0, resets=s.resets + 1, down=0)
        else:
            s = replace(s, h=max(s.h / 2, H_MIN), down=0)
    return s


def replay(gains: Sequence[float]) -> TrustState:
    s = TrustState()
    for g in gains:
        s = step(s, g)
    return s


def box(anchor: torch.Tensor, h: float) -> torch.Tensor:
    """R(a) = [a - h/2, a + h/2] clipped to [0,1]^d, as a (2, d) tensor."""
    lo = (anchor - h / 2).clamp(0.0, 1.0)
    hi = (anchor + h / 2).clamp(0.0, 1.0)
    return torch.stack([lo, hi])


def batch_gains(rounds, observations, objective, *, key: Callable | None = None) -> list[float]:
    """HV gain of each complete round: HV(archive ∪ selected) - HV(archive).

    ``rounds``: (archive_keys, selected_keys) pairs. ``key`` maps an
    observation to its config key (default ``obs.key``; pass
    ``lambda o: space.key(o.config)`` when a JointSpace is at hand).
    Rounds with any selected key lacking an observation are skipped.
    """
    keyf = key or (lambda o: o.key)
    by_key = {keyf(o): o for o in observations}
    gains = []
    for archive, selected in rounds:
        if not all(k in by_key for k in selected):
            continue
        base = [by_key[k] for k in archive if k in by_key]
        after = base + [by_key[k] for k in selected if k not in archive]
        gains.append(measured_hv(after, objective) - measured_hv(base, objective))
    return gains


class BatchBudget:
    """Per-batch wall-clock and objective-call budget with carry-forward."""

    def __init__(self, seconds: float | None = 300.0, calls: int = 10_000):
        self.seconds = seconds
        self.calls = calls

    def slot_share(self, remaining_slots: int) -> tuple[float | None, int]:
        n = max(1, remaining_slots)
        sec = None if self.seconds is None else max(self.seconds, 0.0) / n
        return sec, max(self.calls, 0) // n

    def anchor_share(self, slot_share, remaining_anchors: int) -> tuple[float | None, int]:
        """Split what is left of a slot share evenly over the remaining anchors."""
        sec, calls = slot_share
        n = max(1, remaining_anchors)
        return (None if sec is None else sec / n), min(ANCHOR_CALL_CAP, max(calls, 0) // n)

    def charge(self, seconds: float, calls: int) -> None:
        if self.seconds is not None:
            self.seconds -= seconds
        self.calls -= calls

    def exhausted(self) -> bool:
        return (self.seconds is not None and self.seconds <= 0) or self.calls <= 0


def anchor_order(n: int, *, seed: int, round_index: int, slot: int) -> list[int]:
    order = list(range(n))
    random.Random(digest([seed, round_index, slot])).shuffle(order)
    return order
