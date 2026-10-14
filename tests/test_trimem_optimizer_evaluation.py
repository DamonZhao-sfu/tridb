"""Quality/token objectives use paired continuations and actual request costs."""

import json
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import pytest

from trimem.optimizer.evaluator import (
    CohortEvaluator,
    CohortObservation,
    ContinuationAdapter,
    ContinuationStep,
    EvaluationProtocol,
    EvalUnit,
    IsolatedBranch,
    RolloutResult,
    validate_observation,
)
from trimem.optimizer.ledger import (
    MissingUsageError,
    TokenBudgetExhausted,
    TokenLedger,
)


def protocol(**kwargs):
    return EvaluationProtocol(
        "math-ale-v1", "model-v1", "prompt-v1", "tokenizer-v1", **kwargs
    )


def branches():
    sequence = count()

    @contextmanager
    def factory(unit, action_hash):
        resource = {"state": unit.initial_fitness}
        yield IsolatedBranch(
            f"{action_hash}:{next(sequence)}",
            unit.graph_snapshot,
            resource,
            "fresh independent in-memory dictionary; test fixture only",
        )
        resource.clear()

    return factory


def action(name="action-a"):
    return SimpleNamespace(action_hash=name)


def test_actual_tokens_include_failed_requests_and_retries_but_separate_overhead():
    ledger = TokenLedger(1000)
    ledger.start("first", "generation")
    ledger.finish("first", input_tokens=100, output_tokens=10, status="failure")
    ledger.record("retry", "retry", 100, 20)
    ledger.record("filter", "semantic_filter", 8, 2)
    ledger.record("embed", "embedding", 200, 0)
    ledger.record("judge", "evaluation", 300, 10)
    ledger.record("search", "optimizer", 70, 30)
    assert ledger.total_tokens == 240
    assert ledger.to_dict()["category_tokens"]["evaluation"] == 310
    assert TokenLedger.from_dict(ledger.to_dict()).to_dict() == ledger.to_dict()


def test_pending_or_missing_usage_is_never_imputed_as_zero():
    ledger = TokenLedger()
    ledger.start("request", "generation")
    with pytest.raises(MissingUsageError):
        _ = ledger.total_tokens
    ledger.finish("request", input_tokens=None, output_tokens=5, status="failure")
    assert ledger.to_dict()["total_tokens"] is None
    assert ledger.to_dict()["events"][0]["output_tokens"] == 5
    with pytest.raises(MissingUsageError):
        ledger.request_allowance(2, 10)


def test_actual_known_zero_failure_is_allowed_and_duplicate_retry_ids_rejected():
    ledger = TokenLedger()
    ledger.record("preflight", "generation", 0, 0, "failure")
    assert ledger.total_tokens == 0
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record("preflight", "generation", 2, 1)
    with pytest.raises(ValueError):
        ledger.record("bad", "generation", True, 1)


def test_serial_request_allowance_caps_output_before_dispatch():
    ledger = TokenLedger(100)
    ledger.record_event(
        {
            "request_id": "first",
            "category": "retrieval",
            "input_tokens": 20,
            "output_tokens": 10,
            "status": "success",
        }
    )
    assert ledger.request_allowance(25, 100) == 45
    with pytest.raises(TokenBudgetExhausted):
        ledger.request_allowance(70, 10)


def test_protocol_and_cohort_ids_include_versions_budget_and_checkpoint():
    p = protocol(token_budget=100)
    unit = EvalUnit("math:a", "checkpoint-1", "snapshot-1", 42, -3)

    def runner(candidate, context):
        return RolloutResult(-3, True)

    original = CohortEvaluator(p, [unit], runner, branches())
    different_prompt = CohortEvaluator(
        replace(p, prompt_version="v2"), [unit], runner, branches()
    )
    different_checkpoint = CohortEvaluator(
        p, [replace(unit, checkpoint_id="cp2")], runner, branches()
    )
    assert original.protocol_id != different_prompt.protocol_id
    assert original.cohort_id != different_prompt.cohort_id
    assert original.cohort_id != different_checkpoint.cohort_id
    assert p.protocol_id != replace(p, baseline_policy_version="different").protocol_id
    assert p.protocol_id != replace(p, token_budget=200).protocol_id
    assert EvaluationProtocol.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_invalid_program_does_not_improve_valid_negative_baseline():
    calls = []

    def retrieve(candidate, context, state):
        calls.append(candidate.action_hash)
        context.ledger.record("semfilter", "retrieval", 5, 5)
        return "chosen-memory"

    def step(state, memory, context, remaining):
        assert memory == "chosen-memory"
        assert remaining == 90
        context.ledger.record("generation", "generation", 30, 60)
        return ContinuationStep(state, fitness=0, valid=False)

    adapter = ContinuationAdapter(
        lambda context: context.isolated_branch.resource, retrieve, step
    )
    observation = CohortEvaluator(
        protocol(token_budget=100),
        [EvalUnit("math:a", "cp", "graph", 1, -7)],
        adapter,
        branches(),
    ).evaluate(action())
    assert observation.complete_for_model
    assert observation.quality_mean == -7
    assert observation.total_tokens_mean == 100
    assert observation.objective_vector == (-7, -100)
    assert observation.validity_rate == 0
    assert observation.units[0].rollout.stop_reason == "token_budget"
    assert calls == ["action-a"]


def test_minimizing_quality_is_explicit_and_negative_fitness_remains_valid():
    def step(state, memory, context, remaining):
        context.ledger.record("g", "generation", 1, 2)
        return ContinuationStep(state, fitness=-6, valid=True, done=True)

    observation = CohortEvaluator(
        protocol(token_budget=100, quality_direction="minimize"),
        [EvalUnit("ale:a", "cp", "graph", 1, -3)],
        ContinuationAdapter(lambda context: {}, lambda *args: [], step),
        branches(),
    ).evaluate(action())
    assert observation.units[0].rollout.quality_raw == -6
    assert observation.quality_mean == 6
    assert observation.validity_rate == 1


def test_budget_end_before_new_request_is_a_completed_observation():
    def step(state, memory, context, remaining):
        context.ledger.request_allowance(100, 5)
        raise AssertionError("request should never dispatch")

    observation = CohortEvaluator(
        protocol(token_budget=100),
        [EvalUnit("math:a", "cp", "graph", 1, 2)],
        ContinuationAdapter(lambda context: {}, lambda *args: [], step),
        branches(),
    ).evaluate(action())
    assert observation.complete_for_model
    assert observation.units[0].rollout.stop_reason == "token_budget"
    assert observation.quality_mean == 2
    assert observation.total_tokens_mean == 0


def test_missing_usage_and_infrastructure_failures_do_not_become_bad_quality_labels():
    def missing(candidate, context):
        context.ledger.record("g", "generation", None, 4, "failure")
        return RolloutResult(-3, True)

    evaluator = CohortEvaluator(
        protocol(token_budget=100),
        [EvalUnit("math:a", "cp", "graph", 1, -4)],
        missing,
        branches(),
    )
    observation = evaluator.evaluate(action())
    assert not observation.complete_for_model
    assert observation.quality_mean is None
    assert observation.units[0].rollout.status == "missing_usage"
    assert observation.units[0].rollout.quality_raw == -3
    with pytest.raises(ValueError, match="incomplete"):
        _ = observation.objective_vector

    def broken(candidate, context):
        raise ConnectionError("offline test service")

    evaluator.runner = broken
    observation = evaluator.evaluate(action("action-b"))
    assert not observation.complete_for_model
    assert observation.units[0].rollout.status == "infrastructure_error"
    assert observation.quality_mean is None


def test_budget_overshoot_and_safety_step_limit_are_not_valid_horizons():
    def too_much(candidate, context):
        context.ledger.record("g", "generation", 60, 60)
        return RolloutResult(10, True)

    unit = EvalUnit("math:a", "cp", "graph", 1, 1)
    observation = CohortEvaluator(
        protocol(token_budget=100),
        [unit],
        too_much,
        branches(),
    ).evaluate(action())
    assert observation.units[0].rollout.status == "budget_exceeded"
    assert observation.units[0].total_tokens == 120
    assert not observation.complete_for_model

    adapter = ContinuationAdapter(
        lambda context: {},
        lambda *args: [],
        lambda state, *_: ContinuationStep(state, 2, True),
    )
    observation = CohortEvaluator(
        protocol(token_budget=100, max_steps=2),
        [unit],
        adapter,
        branches(),
    ).evaluate(action())
    assert observation.units[0].rollout.status == "horizon_incomplete"
    assert not observation.complete_for_model


def test_task_macro_weights_normalization_and_replicate_sem():
    p = protocol(
        token_budget=1000,
        task_weights=(("a", 1), ("b", 3)),
        quality_scales=(("a", 10, 10), ("b", 0, 1)),
    )
    units = [
        EvalUnit(task, "cp", "graph", seed) for task in ("a", "b") for seed in (1, 2)
    ]

    def runner(candidate, context):
        task, seed = context.unit.task_id, context.unit.seed
        raw = {("a", 1): 10, ("a", 2): 30, ("b", 1): 2, ("b", 2): 4}[(task, seed)]
        context.ledger.record("g", "generation", seed * 100, 0)
        return RolloutResult(raw, True)

    observation = CohortEvaluator(p, units, runner, branches()).evaluate(action())
    assert observation.quality_mean == 2.5  # 1/4 * 1 + 3/4 * 3
    assert observation.quality_variance_of_mean == pytest.approx(0.625)
    assert observation.total_tokens_mean == 150
    assert observation.tokens_variance_of_mean == pytest.approx(1562.5)
    assert observation.quality_sem == pytest.approx(0.625**0.5)
    restored = CohortObservation.from_dict(
        json.loads(json.dumps(observation.to_dict()))
    )
    assert restored == observation
    assert restored.objective_vector == observation.objective_vector


def test_single_replicate_variance_is_unknown_and_partial_cohort_not_averaged():
    units = [EvalUnit("a", "cp", "graph", 1), EvalUnit("b", "cp", "graph", 1)]

    def good(candidate, context):
        return RolloutResult(1, True)

    evaluator = CohortEvaluator(protocol(token_budget=100), units, good, branches())
    observation = evaluator.evaluate(action())
    assert observation.complete_for_model
    assert observation.quality_variance_of_mean is None
    assert observation.quality_sem is None

    evaluator.runner = lambda candidate, context: RolloutResult(
        1 if context.unit.task_id == "a" else None, context.unit.task_id == "a"
    )
    observation = evaluator.evaluate(action("b"))
    assert not observation.complete_for_model
    assert observation.quality_mean is None
    assert observation.total_tokens_mean is None


def test_isolation_lease_requires_correct_snapshot_evidence_and_unique_identity():
    unit = EvalUnit("a", "cp", "graph", 1, 0)

    def runner(candidate, context):
        return RolloutResult(1, True)

    @contextmanager
    def reused(unit, action_hash):
        yield IsolatedBranch("reused", unit.graph_snapshot, {}, "test clone")

    evaluator = CohortEvaluator(protocol(token_budget=100), [unit], runner, reused)
    assert evaluator.evaluate(action()).complete_for_model
    assert not evaluator.evaluate(action("b")).complete_for_model

    @contextmanager
    def wrong(unit, action_hash):
        yield IsolatedBranch("wrong", "different-snapshot", {}, "test clone")

    observation = CohortEvaluator(
        protocol(token_budget=100),
        [unit],
        runner,
        wrong,
    ).evaluate(action())
    assert not observation.complete_for_model
    assert "snapshot/evidence" in observation.units[0].rollout.error


def test_each_candidate_gets_fresh_state_and_factory_cleanup():
    unit = EvalUnit("a", "cp", "graph", 1, 0)
    seen = []

    def runner(candidate, context):
        resource = context.isolated_branch.resource
        assert resource == {"state": 0}
        resource["modified"] = True
        seen.append(resource)
        return RolloutResult(1, True)

    evaluator = CohortEvaluator(protocol(token_budget=100), [unit], runner, branches())
    evaluator.evaluate(action("a"))
    evaluator.evaluate(action("b"))
    assert seen == [{}, {}]  # both context managers cleaned their owned resource
    assert seen[0] is not seen[1]


def test_evaluator_rejects_changed_protocol_after_identity_is_frozen():
    evaluator = CohortEvaluator(
        protocol(token_budget=100),
        [EvalUnit("a", "cp", "graph", 1, 0)],
        lambda candidate, context: RolloutResult(1, True),
        branches(),
    )
    evaluator.protocol = replace(evaluator.protocol, token_budget=200)
    with pytest.raises(ValueError, match="changed after initialization"):
        evaluator.evaluate(action())


def validation_fixture():
    p = protocol(token_budget=100)
    units = [EvalUnit("math:a", "cp", "graph", seed, -3) for seed in (1, 2)]

    def runner(candidate, context):
        context.ledger.record("generation", "generation", context.unit.seed * 10, 5)
        return RolloutResult(
            -2 + context.unit.seed,
            True,
            steps=1,
            valid_programs=1,
            attempted_programs=1,
        )

    observation = CohortEvaluator(p, units, runner, branches()).evaluate(action())
    return p, units, observation


def test_import_validation_accepts_full_json_roundtrip_and_recomputes_identity():
    p, units, observation = validation_fixture()
    imported = CohortObservation.from_dict(
        json.loads(json.dumps(observation.to_dict()))
    )
    assert validate_observation(imported, p, units) == observation
    assert (
        validate_observation(
            replace(imported, units=tuple(reversed(imported.units))), p, units
        )
        == observation
    )
    with pytest.raises(ValueError, match="protocol_id"):
        validate_observation(imported, replace(p, prompt_version="different"), units)
    with pytest.raises(ValueError, match="frozen cohort"):
        validate_observation(
            imported, p, [replace(unit, checkpoint_id="other") for unit in units]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("quality_mean", 100),
        ("total_tokens_mean", 0),
        ("quality_variance_of_mean", 0),
        ("tokens_variance_of_mean", -1),
        ("quality_mean", float("nan")),
        ("total_tokens_mean", False),
        ("quality_variance_of_mean", float("inf")),
    ],
)
def test_import_validation_rejects_tampered_aggregates(field, value):
    p, units, observation = validation_fixture()
    with pytest.raises(ValueError):
        validate_observation(replace(observation, **{field: value}), p, units)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_tokens", 0),
        ("total_tokens", -1),
        ("total_tokens", False),
        ("quality", 100),
        ("quality", float("nan")),
    ],
)
def test_import_validation_rejects_tampered_unit_objectives(field, value):
    p, units, observation = validation_fixture()
    rows = list(observation.units)
    rows[0] = replace(rows[0], **{field: value})
    with pytest.raises(ValueError):
        validate_observation(replace(observation, units=tuple(rows)), p, units)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_tokens", 0),
        ("objective_usage_complete", False),
        ("token_budget", 200),
        ("category_tokens", {}),
    ],
)
def test_import_validation_rejects_tampered_ledger_caches(field, value):
    p, units, observation = validation_fixture()
    rows = list(observation.units)
    rows[0] = replace(rows[0], ledger=rows[0].ledger | {field: value})
    with pytest.raises(ValueError):
        validate_observation(replace(observation, units=tuple(rows)), p, units)


def test_import_validation_rejects_missing_usage_disguised_as_complete_zero_cost():
    p, units, observation = validation_fixture()
    payload = json.loads(json.dumps(observation.to_dict()))
    for row in payload["units"]:
        row["ledger"]["events"][0]["input_tokens"] = None
        row["ledger"]["total_tokens"] = 0
        row["ledger"]["category_tokens"]["generation"] = 0
        row["total_tokens"] = 0
    payload["total_tokens_mean"] = 0
    payload["tokens_variance_of_mean"] = 0
    with pytest.raises(ValueError, match="ledger"):
        validate_observation(CohortObservation.from_dict(payload), p, units)
    # Even correctly recomputed ledger caches cannot make an incomplete rollout
    # become a completed observation merely by leaving its status unchanged.
    for row in payload["units"]:
        row["ledger"] = TokenLedger.from_dict(row["ledger"]).to_dict()
        row["total_tokens"] = None
    with pytest.raises(ValueError, match="rollout"):
        validate_observation(CohortObservation.from_dict(payload), p, units)


def test_import_validation_preserves_genuine_missing_usage_failure():
    p = protocol(token_budget=100)
    units = [EvalUnit("math:a", "cp", "graph", 1, -3)]

    def runner(candidate, context):
        context.ledger.start("failed", "generation")
        raise ConnectionError("lost provider usage")

    observation = CohortEvaluator(p, units, runner, branches()).evaluate(action())
    assert observation.units[0].rollout.status == "missing_usage"
    assert validate_observation(observation, p, units) == observation


def test_import_validation_rejects_duplicate_units_and_changed_weights():
    p, units, observation = validation_fixture()
    with pytest.raises(ValueError, match="duplicate units"):
        validate_observation(
            replace(observation, units=(observation.units[0],) * 2), p, units
        )
    with pytest.raises(ValueError, match="task_weights"):
        validate_observation(
            replace(observation, task_weights=(("math:a", 2),)), p, units
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", -1),
        ("steps", True),
        ("valid_programs", 2),
        ("attempted_programs", -1),
        ("quality_raw", float("nan")),
        ("valid", 1),
    ],
)
def test_import_rejects_invalid_raw_rollout_counts_and_values(field, value):
    _, _, observation = validation_fixture()
    payload = observation.to_dict()
    payload["units"][0]["rollout"][field] = value
    with pytest.raises(ValueError):
        CohortObservation.from_dict(payload)


def test_import_validation_rejects_negative_usage_and_false_completed_budget_status():
    p, units, observation = validation_fixture()
    payload = json.loads(json.dumps(observation.to_dict()))
    payload["units"][0]["ledger"]["events"][0]["input_tokens"] = -1
    with pytest.raises(ValueError):
        validate_observation(CohortObservation.from_dict(payload), p, units)
    payload = json.loads(json.dumps(observation.to_dict()))
    for row in payload["units"]:
        row["ledger"]["events"][0]["input_tokens"] = 200
        row["ledger"] = TokenLedger.from_dict(row["ledger"]).to_dict()
        row["total_tokens"] = row["ledger"]["total_tokens"]
    with pytest.raises(ValueError, match="rollout"):
        validate_observation(CohortObservation.from_dict(payload), p, units)
