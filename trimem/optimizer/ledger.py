"""Actual request-level token accounting for trajectory optimizer evaluations.

The ledger never estimates missing provider usage and never drops failed requests.
Callers must register requests before dispatch and finish each physical retry with
its own request ID. Embeddings, evaluation judges, and optimizer exploration are
reported separately from the agent's construction/retrieval/generation objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any

OBJECTIVE_CATEGORIES = frozenset(
    {
        "construction",
        "retrieval",
        "semantic_filter",
        "semantic_rank",
        "generation",
        "retry",
        "reflection",
    }
)
OVERHEAD_CATEGORIES = frozenset({"embedding", "evaluation", "optimizer"})
REQUEST_STATUSES = frozenset({"pending", "success", "failure", "cancelled"})


class MissingUsageError(ValueError):
    """An actual objective token total cannot be established."""


class TokenBudgetExhausted(RuntimeError):
    """There is no room for another admitted request."""


def _tokens(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class UsageEvent:
    request_id: str
    category: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.category not in OBJECTIVE_CATEGORIES | OVERHEAD_CATEGORIES:
            raise ValueError(f"unknown token category: {self.category}")
        if self.status not in REQUEST_STATUSES:
            raise ValueError(f"unknown request status: {self.status}")
        _tokens(self.input_tokens, "input_tokens")
        _tokens(self.output_tokens, "output_tokens")

    @property
    def usage_complete(self) -> bool:
        return (
            self.status != "pending"
            and self.input_tokens is not None
            and self.output_tokens is not None
        )

    @property
    def total_tokens(self) -> int | None:
        if not self.usage_complete:
            return None
        return self.input_tokens + self.output_tokens  # type: ignore[operator]


class TokenLedger:
    """Thread-safe bookkeeping; transport instrumentation remains caller-owned.

    ``request_allowance`` supports serial request admission. It is not an atomic
    reservation API for concurrent requests; a caller using concurrency must
    reserve its own shared budget before dispatch. The continuation adapter is
    sequential. No token estimator is silently substituted for actual usage.
    """

    def __init__(self, token_budget: int | None = None) -> None:
        _tokens(token_budget, "token_budget")
        self.token_budget = token_budget
        self._events: dict[str, UsageEvent] = {}
        self._lock = RLock()

    @property
    def events(self) -> tuple[UsageEvent, ...]:
        with self._lock:
            return tuple(self._events.values())

    def start(self, request_id: str, category: str) -> None:
        self._insert(UsageEvent(request_id, category))

    def _insert(self, event: UsageEvent) -> None:
        with self._lock:
            if event.request_id in self._events:
                raise ValueError(f"duplicate request_id: {event.request_id}")
            self._events[event.request_id] = event

    def finish(
        self,
        request_id: str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        status: str = "success",
    ) -> None:
        if status == "pending":
            raise ValueError("finish requires a terminal status")
        with self._lock:
            old = self._events[request_id]
            if old.status != "pending":
                raise ValueError(f"request already finished: {request_id}")
            self._events[request_id] = replace(
                old,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status=status,
            )

    def record(
        self,
        request_id: str,
        category: str,
        input_tokens: int | None,
        output_tokens: int | None,
        status: str = "success",
    ) -> None:
        """Record one completed physical request from a trusted usage receipt."""
        self._insert(
            UsageEvent(
                request_id,
                category,
                input_tokens,
                output_tokens,
                status,
            )
        )

    def record_event(self, event: Mapping[str, Any]) -> None:
        self.record(
            request_id=event["request_id"],
            category=event["category"],
            input_tokens=event.get("input_tokens"),
            output_tokens=event.get("output_tokens"),
            status=event.get("status", "success"),
        )

    @property
    def objective_usage_complete(self) -> bool:
        return all(
            e.usage_complete for e in self.events if e.category in OBJECTIVE_CATEGORIES
        )

    @property
    def total_tokens(self) -> int:
        relevant = [e for e in self.events if e.category in OBJECTIVE_CATEGORIES]
        missing = [e.request_id for e in relevant if not e.usage_complete]
        if missing:
            raise MissingUsageError(f"missing actual usage: {', '.join(missing)}")
        return sum(e.total_tokens for e in relevant)  # type: ignore[misc]

    @property
    def remaining_tokens(self) -> int | None:
        total = self.total_tokens
        return None if self.token_budget is None else max(0, self.token_budget - total)

    def request_allowance(self, input_tokens: int, requested_output: int) -> int:
        """Return an output cap fitting the remaining serial-request budget.

        ``input_tokens`` must include the actual model's complete prompt framing.
        Post-request provider usage remains the authoritative measured cost.
        """
        _tokens(input_tokens, "input_tokens")
        _tokens(requested_output, "requested_output")
        remaining = self.remaining_tokens
        allowed = (
            requested_output
            if remaining is None
            else min(
                requested_output,
                remaining - input_tokens,
            )
        )
        if allowed <= 0:
            raise TokenBudgetExhausted("no room for prompt plus generated output")
        return allowed

    def to_dict(self) -> dict[str, Any]:
        totals: dict[str, int | None] = {}
        for category in sorted(OBJECTIVE_CATEGORIES | OVERHEAD_CATEGORIES):
            events = [e for e in self.events if e.category == category]
            totals[category] = (
                sum(e.total_tokens for e in events)  # type: ignore[misc]
                if all(e.usage_complete for e in events)
                else None
            )
        return {
            "token_budget": self.token_budget,
            "events": [asdict(e) for e in self.events],
            "objective_usage_complete": self.objective_usage_complete,
            "total_tokens": self.total_tokens
            if self.objective_usage_complete
            else None,
            "category_tokens": totals,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TokenLedger:
        ledger = cls(value.get("token_budget"))
        for event in value.get("events", []):
            ledger._insert(UsageEvent(**event))
        return ledger
