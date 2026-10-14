"""Paired, isolated continuation evaluation for quality / total-LLM-token search.

This module deliberately imports no replay task runner, opens no connection, and
starts no worker. Real transports, evaluators, and isolation are explicit injected
callbacks. A branch lease describes a caller-established boundary; it cannot prove
database snapshot isolation by metadata alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol

from .ledger import MissingUsageError, TokenBudgetExhausted, TokenLedger


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


class CandidateLike(Protocol):
    action_hash: str


@dataclass(frozen=True)
class EvalUnit:
    task_id: str
    checkpoint_id: str
    graph_snapshot: str
    seed: int
    initial_fitness: float | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.checkpoint_id or not self.graph_snapshot:
            raise ValueError(
                "task, checkpoint, and graph snapshot identities are required"
            )
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.initial_fitness is not None:
            _finite(self.initial_fitness, "initial_fitness")

    @property
    def unit_id(self) -> str:
        return _identity(asdict(self))


@dataclass(frozen=True)
class EvaluationProtocol:
    """Frozen protocol identity; normalization constants must be fixed in advance.

    Per-task quality is ``direction * (raw - offset) / scale``. Do not derive
    offsets/scales from the current candidate archive: doing so changes past
    objectives. Equal task weights are the default regardless of replicate count.
    """

    name: str
    model_version: str
    prompt_version: str
    tokenizer_version: str
    token_budget: int
    baseline_policy_version: str = "frozen-v1"
    quality_kind: str = "best_valid_fitness"
    quality_direction: str = "maximize"
    horizon_kind: str = "token_budget"
    max_steps: int = 100
    task_weights: tuple[tuple[str, float], ...] = ()
    quality_scales: tuple[tuple[str, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not all(
            (
                self.name,
                self.model_version,
                self.prompt_version,
                self.tokenizer_version,
                self.baseline_policy_version,
            )
        ):
            raise ValueError("all protocol versions must be explicit")
        if type(self.token_budget) is not int or self.token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if self.quality_kind != "best_valid_fitness":
            raise ValueError("first edition supports best_valid_fitness")
        if self.quality_direction not in {"maximize", "minimize"}:
            raise ValueError("quality_direction must be maximize or minimize")
        if self.horizon_kind not in {"token_budget", "fixed_steps"}:
            raise ValueError("unsupported horizon_kind")
        weights = tuple((task, float(weight)) for task, weight in self.task_weights)
        scales = tuple(
            (task, float(offset), float(scale))
            for task, offset, scale in self.quality_scales
        )
        object.__setattr__(self, "task_weights", tuple(sorted(weights)))
        object.__setattr__(self, "quality_scales", tuple(sorted(scales)))
        if len(dict(weights)) != len(weights):
            raise ValueError("duplicate task weights")
        if len({task for task, _, _ in scales}) != len(scales):
            raise ValueError("duplicate quality scales")
        for _, weight in weights:
            _finite(weight, "task weight")
            if weight <= 0:
                raise ValueError("task weights must be positive")
        for _, offset, scale in scales:
            _finite(offset, "quality offset")
            _finite(scale, "quality scale")
            if scale <= 0:
                raise ValueError("quality scale must be positive")

    @property
    def protocol_id(self) -> str:
        return _identity(asdict(self))

    def normalize_quality(self, task: str, raw: float) -> float:
        offset, scale = next(
            (
                (offset, scale)
                for name, offset, scale in self.quality_scales
                if name == task
            ),
            (0.0, 1.0),
        )
        direction = 1 if self.quality_direction == "maximize" else -1
        return direction * (raw - offset) / scale

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationProtocol:
        return cls(**value)


@dataclass(frozen=True)
class IsolatedBranch:
    """Caller-owned isolation lease, validated for identity reuse and snapshot.

    ``resource`` can be a cloned in-memory state, a private DB schema, or another
    real isolation handle. The factory must establish/clean it; this type does not
    create a snapshot or make shared mutable state safe.
    """

    branch_id: str
    frozen_snapshot: str
    resource: Any = None
    isolation_evidence: str = ""


@dataclass(frozen=True)
class EvalContext:
    protocol: EvaluationProtocol
    unit: EvalUnit
    ledger: TokenLedger
    isolated_branch: IsolatedBranch

    @property
    def frozen_snapshot(self) -> str:
        return self.unit.graph_snapshot


@dataclass(frozen=True)
class RolloutResult:
    quality_raw: float | None
    valid: bool
    status: str = "completed"
    stop_reason: str = "done"
    steps: int = 0
    valid_programs: int = 0
    attempted_programs: int = 0
    error: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quality_raw is not None:
            _finite(self.quality_raw, "quality_raw")
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        if self.valid and self.quality_raw is None:
            raise ValueError("valid quality requires a raw fitness")
        for name in ("steps", "valid_programs", "attempted_programs"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.valid_programs > self.attempted_programs:
            raise ValueError("valid_programs cannot exceed attempted_programs")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("rollout status must be a nonempty string")


@dataclass(frozen=True)
class UnitObservation:
    unit: EvalUnit
    branch_id: str
    rollout: RolloutResult
    ledger: dict[str, Any]
    quality: float | None
    total_tokens: int | None

    @property
    def complete_for_model(self) -> bool:
        return (
            self.rollout.status == "completed"
            and self.rollout.valid
            and self.quality is not None
            and self.total_tokens is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UnitObservation:
        return cls(
            unit=EvalUnit(**value["unit"]),
            branch_id=value["branch_id"],
            rollout=RolloutResult(**value["rollout"]),
            ledger=dict(value["ledger"]),
            quality=value["quality"],
            total_tokens=value["total_tokens"],
        )


@dataclass(frozen=True)
class CohortObservation:
    action_hash: str
    protocol_id: str
    cohort_id: str
    units: tuple[UnitObservation, ...]
    quality_mean: float | None
    total_tokens_mean: float | None
    quality_variance_of_mean: float | None
    tokens_variance_of_mean: float | None
    task_weights: tuple[tuple[str, float], ...]

    @property
    def n(self) -> int:
        return len(self.units)

    @property
    def complete_for_model(self) -> bool:
        return bool(self.units) and all(unit.complete_for_model for unit in self.units)

    @property
    def quality_sem(self) -> float | None:
        return (
            None
            if self.quality_variance_of_mean is None
            else math.sqrt(self.quality_variance_of_mean)
        )

    @property
    def tokens_sem(self) -> float | None:
        return (
            None
            if self.tokens_variance_of_mean is None
            else math.sqrt(self.tokens_variance_of_mean)
        )

    @property
    def objective_vector(self) -> tuple[float, float]:
        if not self.complete_for_model:
            raise ValueError(
                "incomplete cohort cannot enter the quality/token surrogate"
            )
        return (self.quality_mean, -self.total_tokens_mean)  # type: ignore[return-value,operator]

    @property
    def validity_rate(self) -> float | None:
        attempts = sum(unit.rollout.attempted_programs for unit in self.units)
        return (
            None
            if attempts == 0
            else sum(unit.rollout.valid_programs for unit in self.units) / attempts
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CohortObservation:
        return cls(
            action_hash=value["action_hash"],
            protocol_id=value["protocol_id"],
            cohort_id=value["cohort_id"],
            units=tuple(UnitObservation.from_dict(unit) for unit in value["units"]),
            quality_mean=value["quality_mean"],
            total_tokens_mean=value["total_tokens_mean"],
            quality_variance_of_mean=value["quality_variance_of_mean"],
            tokens_variance_of_mean=value["tokens_variance_of_mean"],
            task_weights=tuple(tuple(pair) for pair in value["task_weights"]),
        )


def _aggregate(
    observations: Sequence[UnitObservation],
    weights: Mapping[str, float],
    name: str,
) -> tuple[float, float | None]:
    """Fixed-cohort macro mean and sampling variance across per-task replicates.

    A task with one replicate has unknown variance, not a fictitious zero SEM.
    Between-task differences are not replicate noise for this fixed task cohort.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        values[observation.unit.task_id].append(getattr(observation, name))
    total_weight = sum(weights[task] for task in values)
    mean = sum(
        weights[task] / total_weight * statistics.mean(rows)
        for task, rows in values.items()
    )
    variance = (
        None
        if any(len(rows) < 2 for rows in values.values())
        else sum(
            (weights[task] / total_weight) ** 2 * statistics.variance(rows) / len(rows)
            for task, rows in values.items()
        )
    )
    return mean, variance


def _require_consistent(actual: Any, expected: Any, path: str) -> None:
    """Compare derived JSON fields without treating booleans as token counts."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"inconsistent {path}: mapping keys do not match")
        for key, value in expected.items():
            _require_consistent(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"inconsistent {path}: sequence length does not match")
        for index, (item, value) in enumerate(zip(actual, expected, strict=True)):
            _require_consistent(item, value, f"{path}[{index}]")
        return
    if isinstance(expected, bool):
        same = type(actual) is bool and actual == expected
    elif isinstance(expected, (int, float)):
        _finite(actual, path)
        same = actual == expected
    else:
        same = type(actual) is type(expected) and actual == expected
    if not same:
        raise ValueError(f"inconsistent {path}: supplied and recomputed values differ")


def validate_observation(
    observation: CohortObservation,
    protocol: EvaluationProtocol,
    units: Sequence[EvalUnit],
) -> CohortObservation:
    """Recompute imported observation objectives from raw rollouts and events.

    This checks internal consistency, not the authenticity of original experiment
    receipts or a caller's claimed isolation. Every cached ledger field, per-unit
    objective, aggregate, variance, cohort ID, and protocol ID must agree with the
    raw events and the caller-supplied frozen protocol/cohort. Return a freshly
    reconstructed observation only after those checks pass.
    """
    frozen_units = tuple(sorted(units, key=lambda unit: unit.unit_id))
    unit_ids = [unit.unit_id for unit in frozen_units]
    if not frozen_units or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("frozen cohort must be nonempty with unique units")
    observed_ids = [item.unit.unit_id for item in observation.units]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("observation contains duplicate units")
    if set(observed_ids) != set(unit_ids):
        raise ValueError("observation units differ from the frozen cohort")
    if not isinstance(observation.action_hash, str) or not observation.action_hash:
        raise ValueError("action_hash must be a nonempty string")
    tasks = {unit.task_id for unit in frozen_units}
    weights = dict(protocol.task_weights) or dict.fromkeys(tasks, 1.0)
    if set(weights) != tasks:
        raise ValueError("protocol task weights do not cover the frozen cohort")
    if protocol.quality_scales and {row[0] for row in protocol.quality_scales} != tasks:
        raise ValueError("protocol quality scales do not cover the frozen cohort")
    cohort_id = _identity(
        {
            "protocol_id": protocol.protocol_id,
            "units": [asdict(unit) for unit in frozen_units],
        }
    )
    _require_consistent(observation.protocol_id, protocol.protocol_id, "protocol_id")
    _require_consistent(observation.cohort_id, cohort_id, "cohort_id")
    _require_consistent(
        sorted(observation.task_weights), sorted(weights.items()), "task_weights"
    )
    observed = {item.unit.unit_id: item for item in observation.units}
    rebuilt_units = []
    for unit in frozen_units:
        item = observed[unit.unit_id]
        if not isinstance(item.branch_id, str):
            raise ValueError("branch_id must be a string")  # noqa: TRY004 -- imported data validation
        # Reconstructing the rollout re-runs finite/value/count validation even
        # if an importer bypassed normal frozen dataclass construction.
        result = RolloutResult(**asdict(item.rollout))
        ledger = TokenLedger.from_dict(item.ledger)
        if ledger.token_budget != protocol.token_budget:
            raise ValueError("ledger token budget differs from the frozen protocol")
        canonical_ledger = ledger.to_dict()
        _require_consistent(item.ledger, canonical_ledger, "ledger")
        try:
            token_total: int | None = ledger.total_tokens
        except MissingUsageError as exc:
            token_total = None
            result = replace(
                result,
                status="missing_usage",
                stop_reason="missing_usage",
                error=str(exc),
            )
        if token_total is not None and token_total > protocol.token_budget:
            result = replace(
                result,
                status="budget_exceeded",
                stop_reason="budget_exceeded",
                error="actual usage exceeded the frozen token budget",
            )
        if result.status == "missing_usage" and token_total is not None:
            raise ValueError("missing_usage status contradicts complete actual usage")
        if result.status == "budget_exceeded" and (
            token_total is None or token_total <= protocol.token_budget
        ):
            raise ValueError("budget_exceeded status contradicts actual usage")
        if (
            result.status == "completed"
            and result.stop_reason == "step_limit"
            and protocol.horizon_kind == "token_budget"
            and token_total is not None
            and token_total < protocol.token_budget
        ):
            result = replace(result, status="horizon_incomplete")
        if (
            result.status == "completed"
            and result.stop_reason == "token_budget"
            and protocol.horizon_kind == "fixed_steps"
            and result.steps < protocol.max_steps
        ):
            result = replace(result, status="horizon_incomplete")
        _require_consistent(asdict(item.rollout), asdict(result), "rollout")
        if result.status == "completed":
            if not item.branch_id:
                raise ValueError(
                    "completed rollout requires an isolated branch identity"
                )
            if result.steps > protocol.max_steps:
                raise ValueError(
                    "completed rollout exceeded the frozen step safety cap"
                )
            if unit.initial_fitness is not None:
                if not result.valid or result.quality_raw is None:
                    raise ValueError(
                        "best valid fitness cannot discard a valid initial baseline"
                    )
                direction = 1 if protocol.quality_direction == "maximize" else -1
                if direction * result.quality_raw < direction * unit.initial_fitness:
                    raise ValueError(
                        "best valid fitness is worse than the initial baseline"
                    )
        quality = (
            protocol.normalize_quality(unit.task_id, result.quality_raw)
            if (
                result.status == "completed"
                and result.valid
                and result.quality_raw is not None
            )
            else None
        )
        if quality is not None:
            _finite(quality, "normalized quality")
        if item.total_tokens is not None and type(item.total_tokens) is not int:
            raise ValueError("unit total_tokens must be an integer or None")
        _require_consistent(item.quality, quality, "unit.quality")
        _require_consistent(item.total_tokens, token_total, "unit.total_tokens")
        rebuilt_units.append(
            UnitObservation(
                unit,
                item.branch_id,
                result,
                canonical_ledger,
                quality,
                token_total,
            )
        )
    complete = all(item.complete_for_model for item in rebuilt_units)
    quality_mean, quality_var = (
        _aggregate(rebuilt_units, weights, "quality") if complete else (None, None)
    )
    tokens_mean, tokens_var = (
        _aggregate(rebuilt_units, weights, "total_tokens") if complete else (None, None)
    )
    rebuilt = CohortObservation(
        observation.action_hash,
        protocol.protocol_id,
        cohort_id,
        tuple(rebuilt_units),
        quality_mean,
        tokens_mean,
        quality_var,
        tokens_var,
        tuple(sorted(weights.items())),
    )
    for name in (
        "quality_mean",
        "total_tokens_mean",
        "quality_variance_of_mean",
        "tokens_variance_of_mean",
    ):
        _require_consistent(getattr(observation, name), getattr(rebuilt, name), name)
    return rebuilt


class CohortEvaluator:
    """Evaluate all candidates on exactly the same versioned task/seed cohort."""

    def __init__(
        self,
        protocol: EvaluationProtocol,
        units: Sequence[EvalUnit],
        runner: Callable[[CandidateLike, EvalContext], RolloutResult],
        branch_factory: Callable[
            [EvalUnit, str], AbstractContextManager[IsolatedBranch]
        ],
    ) -> None:
        self.protocol = protocol
        self.units = tuple(sorted(units, key=lambda unit: unit.unit_id))
        if not self.units or len({unit.unit_id for unit in self.units}) != len(
            self.units
        ):
            raise ValueError(
                "cohort must be nonempty and contain unique evaluation units"
            )
        tasks = {unit.task_id for unit in self.units}
        if protocol.task_weights and set(dict(protocol.task_weights)) != tasks:
            raise ValueError(
                "explicit task weights must cover exactly the evaluation cohort"
            )
        if (
            protocol.quality_scales
            and {row[0] for row in protocol.quality_scales} != tasks
        ):
            raise ValueError(
                "explicit quality scales must cover exactly the evaluation cohort"
            )
        self.task_weights = MappingProxyType(
            dict(protocol.task_weights) or dict.fromkeys(tasks, 1.0)
        )
        self.protocol_id = protocol.protocol_id
        self.cohort_id = _identity(
            {
                "protocol_id": self.protocol_id,
                "units": [asdict(unit) for unit in self.units],
            }
        )
        self.runner = runner
        self.branch_factory = branch_factory
        self._used_branches: set[str] = set()

    def evaluate(self, candidate: CandidateLike) -> CohortObservation:
        if (
            self.protocol.protocol_id != self.protocol_id
            or _identity(
                {
                    "protocol_id": self.protocol_id,
                    "units": [asdict(unit) for unit in self.units],
                }
            )
            != self.cohort_id
        ):
            raise ValueError("evaluation protocol/cohort changed after initialization")
        observations = tuple(self._run(candidate, unit) for unit in self.units)
        complete = all(unit.complete_for_model for unit in observations)
        quality, quality_var = (
            _aggregate(observations, self.task_weights, "quality")
            if complete
            else (None, None)
        )
        tokens, tokens_var = (
            _aggregate(observations, self.task_weights, "total_tokens")
            if complete
            else (None, None)
        )
        return CohortObservation(
            candidate.action_hash,
            self.protocol_id,
            self.cohort_id,
            observations,
            quality,
            tokens,
            quality_var,
            tokens_var,
            tuple(sorted(self.task_weights.items())),
        )

    def _run(self, candidate: CandidateLike, unit: EvalUnit) -> UnitObservation:
        ledger = TokenLedger(self.protocol.token_budget)
        branch_id = ""
        result = RolloutResult(
            None,
            False,
            status="infrastructure_error",
            stop_reason="missing_result",
            error="branch context ended without a runner result",
        )
        try:
            with self.branch_factory(unit, candidate.action_hash) as branch:
                branch_id = branch.branch_id
                if not branch_id or branch_id in self._used_branches:
                    raise ValueError(
                        "isolated branch identity must be unique for every evaluation"
                    )
                if (
                    branch.frozen_snapshot != unit.graph_snapshot
                    or not branch.isolation_evidence
                ):
                    raise ValueError(
                        "branch snapshot/evidence does not match the frozen unit"
                    )
                self._used_branches.add(branch_id)
                context = EvalContext(self.protocol, unit, ledger, branch)
                result = self.runner(candidate, context)
                if not isinstance(result, RolloutResult):
                    raise TypeError("runner must return RolloutResult")
        except Exception as exc:  # noqa: BLE001 -- retain callback failures as failed observations
            result = RolloutResult(
                None,
                False,
                status="infrastructure_error",
                stop_reason="exception",
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            tokens: int | None = ledger.total_tokens
        except MissingUsageError as exc:
            tokens = None
            result = replace(
                result,
                status="missing_usage",
                stop_reason="missing_usage",
                error=str(exc),
            )
        if tokens is not None and tokens > self.protocol.token_budget:
            result = replace(
                result,
                status="budget_exceeded",
                stop_reason="budget_exceeded",
                error="actual usage exceeded the frozen token budget",
            )
        quality = (
            self.protocol.normalize_quality(unit.task_id, result.quality_raw)
            if (
                result.status == "completed"
                and result.valid
                and result.quality_raw is not None
            )
            else None
        )
        return UnitObservation(
            unit, branch_id, result, ledger.to_dict(), quality, tokens
        )


@dataclass(frozen=True)
class ContinuationStep:
    state: Any
    fitness: float | None = None
    valid: bool = False
    done: bool = False


class ContinuationAdapter:
    """Execute one chosen memory action, then a frozen continuation callback.

    ``initialize(context)`` loads the checkpoint into the isolated branch.
    ``retrieve(candidate, context, state)`` performs only the chosen memory action.
    ``step(state, memory, context, remaining_tokens)`` follows the versioned fixed
    baseline policy (which may itself retrieve using a fixed policy). All physical
    LLM calls, including failed attempts, must use ``context.ledger``. Callbacks
    should cap each generation via ``request_allowance`` and may raise
    ``TokenBudgetExhausted`` before dispatch. That is an ordinary completed horizon.

    The adapter does not introspect a callback or prove that its transport has
    been instrumented. Production callback implementations need integration tests.

    ``retrieve_each_step=True`` is the per-round routed mode: ``retrieve`` runs
    before every step (the candidate is then a policy that chooses an action
    from the current state). The default retrieves exactly once, before round 1.
    """

    def __init__(
        self,
        initialize: Callable[[EvalContext], Any],
        retrieve: Callable[[CandidateLike, EvalContext, Any], Any],
        step: Callable[[Any, Any, EvalContext, int], ContinuationStep],
        *,
        retrieve_each_step: bool = False,
    ) -> None:
        self.initialize = initialize
        self.retrieve = retrieve
        self.step = step
        self.retrieve_each_step = retrieve_each_step

    def __call__(self, candidate: CandidateLike, context: EvalContext) -> RolloutResult:
        best = context.unit.initial_fitness
        steps = valid_programs = 0
        reason = "step_limit"
        try:
            state = self.initialize(context)
            memory = (
                None
                if self.retrieve_each_step
                else self.retrieve(candidate, context, state)
            )
            for _ in range(context.protocol.max_steps):
                remaining = context.ledger.remaining_tokens
                if remaining == 0:
                    reason = "token_budget"
                    break
                if self.retrieve_each_step:
                    memory = self.retrieve(candidate, context, state)
                    remaining = context.ledger.remaining_tokens
                    if remaining == 0:
                        reason = "token_budget"
                        break
                outcome = self.step(state, memory, context, remaining)
                if not isinstance(outcome, ContinuationStep):
                    raise TypeError("step callback must return ContinuationStep")
                state = outcome.state
                steps += 1
                if outcome.valid:
                    if outcome.fitness is None:
                        raise ValueError("valid program requires a finite fitness")
                    _finite(outcome.fitness, "program fitness")
                    valid_programs += 1
                    if best is None:
                        best = outcome.fitness
                    elif context.protocol.quality_direction == "maximize":
                        best = max(best, outcome.fitness)
                    else:
                        best = min(best, outcome.fitness)
                if outcome.done:
                    reason = "done"
                    break
            if context.ledger.remaining_tokens == 0:
                reason = "token_budget"
        except TokenBudgetExhausted:
            reason = "token_budget"
        return RolloutResult(
            best,
            best is not None,
            status="horizon_incomplete"
            if (
                reason == "step_limit"
                and context.protocol.horizon_kind == "token_budget"
                or reason == "token_budget"
                and context.protocol.horizon_kind == "fixed_steps"
                and steps < context.protocol.max_steps
            )
            else "completed",
            stop_reason=reason,
            steps=steps,
            valid_programs=valid_programs,
            attempted_programs=steps,
        )
