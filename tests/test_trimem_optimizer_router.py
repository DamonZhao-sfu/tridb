"""Per-round routed policy (Branch B): router training and routed runtime mode.

No live provider is contacted. The private-PG test needs
TRIMEM_OPTIMIZER_TEST_PG_PREFIX and never touches a shared cluster.
"""

import copy
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_trimem_optimizer_runtime import config as runtime_config  # noqa: F401
from test_trimem_optimizer_runtime import fake_models

from trimem.optimizer import cli
from trimem.optimizer.evaluator import (
    ContinuationAdapter,
    ContinuationStep,
    EvalContext,
    EvaluationProtocol,
    EvalUnit,
    IsolatedBranch,
)
from trimem.optimizer.ledger import TokenLedger
from trimem.optimizer.router import (
    STATE_FEATURES,
    CatalogEntry,
    RoundSample,
    RoutedPolicy,
    _choose,
    artifact_digest,
    build_artifact,
    deployment_invariants,
    simulate_unit,
    state_features,
    unit_samples,
)
from trimem.optimizer.runtime import math_ale_factory
from trimem.optimizer.schema import CandidateSpec

GOOD = CandidateSpec("vgr_v1", seed_count=8, fanout=4, top_k=3)
BAD = CandidateSpec("vgr_v1", seed_count=24, fanout=8, top_k=8)
SEM = CandidateSpec("seed_sem_expand_v1", seed_count=8, fanout=4, top_k=3)
NONE = CandidateSpec("no_memory_v1")


@pytest.fixture
def config(runtime_config):  # noqa: F811 -- pytest fixture injection
    """The runtime test's frozen-input fixture config (fresh per test)."""
    return runtime_config


def features(round_index):
    # Only the round index varies, so trees can only split on it and the action.
    return state_features(
        round_index=round_index,
        best_before=0.0,
        corpus_best=1.0,
        tokens_so_far=0,
        last_valid=False,
        last_improved=False,
        direction=1,
    )


def synthetic(gain, tokens, retrieval, trials=6, rounds=3):
    """gain/tokens/retrieval: action -> callable(round_index) / constant."""
    samples = []
    for action in gain:
        for t in range(trials):
            for r in range(rounds):
                samples.append(
                    RoundSample(
                        f"{action.action_hash[:8]}-{t}",
                        f"unit-{t}",
                        action.action_hash,
                        features(r),
                        float(gain[action](r)),
                        tokens[action],
                        retrieval[action] if r == 0 else 0,
                    )
                )
    actions = {action.action_hash: action for action in gain}
    return samples, actions, {"source": "synthetic test data, not evidence"}


def test_router_training_is_deterministic_and_picks_obviously_better_action():
    samples, actions, provenance = synthetic(
        {GOOD: lambda r: 0.5, BAD: lambda r: 0.0, NONE: lambda r: 0.0},
        {GOOD: 1000, BAD: 3000, NONE: 900},
        {GOOD: 100, BAD: 5000, NONE: 0},
    )
    first = build_artifact(samples, actions, provenance, seed=7)
    second = build_artifact(samples, actions, provenance, seed=7)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["lambda"]["is_offline_proxy"] is True
    assert {row["action_hash"] for row in first["catalog"]} == set(actions)
    no_memory = next(
        r for r in first["catalog"] if r["action"]["plan_id"] == "no_memory_v1"
    )
    assert no_memory["retrieval_tokens"] == 0
    for multiplier in (0.0, 1.0):
        policy = RoutedPolicy.from_artifact(
            first, lambda_=multiplier * first["lambda"]["scale"]
        )
        for r in range(3):
            assert policy.choose(features(r)).action == GOOD
    chosen = RoutedPolicy.from_artifact(first)
    assert chosen.lambda_ == first["lambda"]["chosen"]
    assert (
        RoutedPolicy.from_artifact(first, lambda_=1.0).action_hash
        != RoutedPolicy.from_artifact(first, lambda_=0.0).action_hash
    )
    tampered = json.loads(json.dumps(first))
    tampered["catalog"][0]["retrieval_tokens"] += 1
    with pytest.raises(ValueError, match="sha256"):
        RoutedPolicy.from_artifact(tampered)


def test_lambda_trades_gain_against_measured_retrieval_cost():
    # SEM gains slightly more but its retrieval is far more expensive.
    samples, actions, provenance = synthetic(
        {GOOD: lambda r: 0.30, SEM: lambda r: 0.32},
        {GOOD: 1000, SEM: 1000},
        {GOOD: 0, SEM: 50000},
    )
    artifact = build_artifact(samples, actions, provenance, seed=0)
    zero = RoutedPolicy.from_artifact(artifact, lambda_=0.0)
    costly = RoutedPolicy.from_artifact(artifact, lambda_=1e-4)
    assert artifact["no_memory_in_catalog"] is False
    assert zero.choose(features(0)).action == SEM
    assert costly.choose(features(0)).action == GOOD
    # Once SEM's memory is cached its retrieval cost is no longer charged.
    assert costly.choose(features(0), cached={SEM.action_hash}).action == SEM


def test_unit_samples_targets_and_features_follow_round_history():
    events = [
        {
            "request_id": "s1",
            "category": "semantic_filter",
            "input_tokens": 50,
            "output_tokens": 5,
            "status": "success",
        },
        {
            "request_id": "g1",
            "category": "generation",
            "input_tokens": 100,
            "output_tokens": 10,
            "status": "success",
        },
        {
            "request_id": "g2",
            "category": "generation",
            "input_tokens": 200,
            "output_tokens": 20,
            "status": "success",
        },
        {
            "request_id": "g3",
            "category": "generation",
            "input_tokens": 300,
            "output_tokens": 30,
            "status": "success",
        },
    ]
    history = [
        {"request_ids": ["g1"], "evaluation": {"valid": True}, "best_fitness": 0.4},
        {"request_ids": ["g2"], "evaluation": {"valid": False}, "best_fitness": 0.4},
        {"request_ids": ["g3"], "evaluation": {"valid": True}, "best_fitness": 0.7},
    ]
    unit = SimpleNamespace(
        unit=EvalUnit("math:x", "c", "s", 42, 0.0),
        rollout=SimpleNamespace(status="completed", telemetry={"history": history}),
        ledger={"events": events},
    )
    rows = unit_samples("trial", SEM, unit, corpus_best=0.9, direction=1)
    assert [r.gain for r in rows] == pytest.approx([0.4, 0.0, 0.3])
    assert [r.generation_tokens for r in rows] == [110, 220, 330]
    assert [r.retrieval_tokens for r in rows] == [55, 0, 0]
    assert [r.features["tokens_so_far"] for r in rows] == [0, 165, 385]
    assert [r.features["last_valid"] for r in rows] == [0, 1, 0]
    assert [r.features["last_improved"] for r in rows] == [0, 1, 0]
    assert rows[2].features["gap_to_corpus_best"] == pytest.approx(0.5)
    assert tuple(rows[0].features) == STATE_FEATURES


def _adapter_run(retrieve_each_step):
    calls = []
    protocol = EvaluationProtocol(
        "p", "m", "v", "t", 10_000, horizon_kind="fixed_steps", max_steps=3
    )
    unit = EvalUnit("math:x", "c", "s", 1, 0.0)
    context = EvalContext(
        protocol, unit, TokenLedger(10_000), IsolatedBranch("b", "s", None, "e")
    )

    def retrieve(candidate, ctx, state):
        calls.append(state["round"])
        return "memory"

    def step(state, memory, ctx, remaining):
        state = {"round": state["round"] + 1}
        return ContinuationStep(state, 1.0, True, done=state["round"] >= 3)

    adapter = ContinuationAdapter(
        lambda ctx: {"round": 0}, retrieve, step, retrieve_each_step=retrieve_each_step
    )
    result = adapter(SimpleNamespace(action_hash="x"), context)
    return calls, result


def test_frozen_adapter_retrieves_once_and_routed_adapter_every_round():
    calls, result = _adapter_run(False)
    assert calls == [0] and result.steps == 3
    calls, result = _adapter_run(True)
    assert calls == [0, 1, 2] and result.steps == 3


def _bindir():
    return str(
        Path(os.environ["TRIMEM_OPTIMIZER_TEST_PG_PREFIX"])
        / "usr/lib/postgresql/16/bin"
    )


def _count_clusters(monkeypatch):
    from trimem.optimizer import private_pg

    started = []
    original = private_pg.PrivatePostgres

    def counting(**kwargs):
        started.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(private_pg, "PrivatePostgres", counting)
    return started


def _count_retrievals(monkeypatch):
    from trimem.optimizer import adapters

    executed = []
    original = adapters.NativeRetrievalAdapter.__call__

    def counting(self, action, context, state):
        executed.append(action.action_hash)
        return original(self, action, context, state)

    monkeypatch.setattr(adapters.NativeRetrievalAdapter, "__call__", counting)
    return executed


pg_gate = pytest.mark.skipif(
    not os.environ.get("TRIMEM_OPTIMIZER_TEST_PG_PREFIX"),
    reason="private PG gate opt-in",
)


@pg_gate
def test_frozen_mode_retrieves_once_per_branch_on_private_pg(config, monkeypatch):
    fake_models(monkeypatch)
    config["postgres"]["bindir"] = _bindir()
    config["protocol"]["max_steps"] = 3
    clusters = _count_clusters(monkeypatch)
    executed = _count_retrievals(monkeypatch)
    with math_ale_factory(config) as evaluator:
        result = evaluator.evaluate(GOOD)
    assert result.complete_for_model, result.to_dict()
    assert len(clusters) == 1 and executed == [GOOD.action_hash]
    telemetry = result.units[0].rollout.telemetry
    assert set(telemetry) == {
        "checkpoint_digests",
        "best_code",
        "best_fitness",
        "rounds",
        "history",
        "retrieval",
        "application_mode",
        "graph_policy",
        "runtime_config_sha256",
    }
    assert all("routing" not in row for row in telemetry["history"])


@pg_gate
@pytest.mark.parametrize("ordering", ["stream", "gain"])
def test_routed_mode_retrieves_per_round_caches_and_attributes_tokens(
    config,
    monkeypatch,
    tmp_path,
    ordering,
):
    good = replace(GOOD, ordering=ordering)
    sem = replace(SEM, ordering=ordering)
    fake_models(monkeypatch)
    config["postgres"]["bindir"] = _bindir()
    config["protocol"]["max_steps"] = 3
    config["application_mode"] = "per_round_routed_policy"
    # Router: good in rounds 1 and 3, sem in round 2; no-memory is never best.
    samples, actions, provenance = synthetic(
        {
            good: lambda r: 1.0 if r != 1 else 0.0,
            sem: lambda r: 1.0 if r == 1 else 0.0,
            NONE: lambda r: -1.0,
        },
        {good: 100, sem: 100, NONE: 100},
        {good: 0, sem: 0, NONE: 0},
    )
    # Invariants come from the FROZEN-mode config: only application_mode differs.
    provenance = {**provenance, "deployment_invariants": invariants_for(config)}
    artifact = build_artifact(samples, actions, provenance, seed=0)
    router = tmp_path / "router.json"
    router.write_text(json.dumps(artifact))
    run_config = tmp_path / "run.json"
    run_config.write_text(json.dumps({"evaluation": config}))
    archive = tmp_path / "policy_archive"
    clusters = _count_clusters(monkeypatch)
    executed = _count_retrievals(monkeypatch)
    code = cli.main(
        [
            "run-policy",
            "--factory",
            "trimem.optimizer.runtime:math_ale_factory",
            "--config",
            str(run_config),
            "--router",
            str(router),
            "--router-sha256",
            artifact["artifact_sha256"],
            "--lambda",
            "0",
            "--archive",
            str(archive),
        ]
    )
    assert code == 0
    assert len(clusters) == 1  # one private cluster reused across rounds
    assert executed == [good.action_hash, sem.action_hash]  # round 3 is cached
    (trial_path,) = (archive / "trials").glob("*.json")
    record = json.loads(trial_path.read_text())
    assert record["status"] == "observed"
    assert record["action"]["kind"] == "per_round_routed_policy"
    assert record["proposal"] == {"strategy": "fixed_routed_policy_evaluation"}
    unit = record["observation"]["units"][0]
    telemetry = unit["rollout"]["telemetry"]
    assert telemetry["application_mode"] == "per_round_routed_policy"
    assert telemetry["policy"] == record["action"]
    routing = [row["routing"] for row in telemetry["history"]]
    assert [r["action_hash"] for r in routing] == [
        good.action_hash,
        sem.action_hash,
        good.action_hash,
    ]
    assert [r["cache_hit"] for r in routing] == [False, False, True]
    assert routing[2]["receipt"] is None and routing[2]["retrieval_tokens"] == 0
    assert routing[2]["packed_ids"] == routing[0]["packed_ids"]
    assert routing[2]["execution_id"] == routing[0]["execution_id"]
    assert routing[1]["receipt"]["actual_executions"] == 1
    for row in (routing[0], routing[1]):
        assert row["receipt"]["ranked_by"] == ordering
        if ordering == "gain":
            assert 0 < row["receipt"]["pool_distinct"] <= row["receipt"]["pool_rows"]
            assert row["receipt"]["pool_rows"] <= 64  # candidate_visit_cap
    # Each actual execution's receipt is kept once in unit telemetry.
    assert [r["receipt"]["execution_id"] for r in telemetry["retrievals"]] == [
        routing[0]["execution_id"],
        routing[1]["execution_id"],
    ]
    assert routing[1]["retrieval_tokens"] > 0  # SemFilter judge calls
    assert set(routing[0]["scores"]) == {
        good.action_hash,
        sem.action_hash,
        NONE.action_hash,
    }
    assert [r["features"]["round_index"] for r in routing] == [0, 1, 2]
    assert len(telemetry["retrievals"]) == 2 and telemetry["retrieval"] is None
    # Ledger consistency: every objective request belongs to exactly one round.
    events = {e["request_id"]: e for e in unit["ledger"]["events"]}
    per_round = [
        set(row["request_ids"]) | set(row["routing"]["retrieval_request_ids"])
        for row in telemetry["history"]
    ]
    assert sum(len(ids) for ids in per_round) == len(events)
    assert set().union(*per_round) == set(events)
    for row in telemetry["history"]:
        ids = row["routing"]["retrieval_request_ids"]
        assert row["routing"]["retrieval_tokens"] == sum(
            events[i]["input_tokens"] + events[i]["output_tokens"] for i in ids
        )
        assert all(events[i]["category"] != "generation" for i in ids)
    assert (
        unit["total_tokens"]
        == unit["ledger"]["total_tokens"]
        == sum(e["input_tokens"] + e["output_tokens"] for e in events.values())
    )
    status = cli.main(["status", "--archive", str(archive)])
    assert status == 0


def test_routed_mode_rejects_a_plain_candidate(config, monkeypatch):
    fake_models(monkeypatch)
    config["application_mode"] = "per_round_routed_policy"
    with math_ale_factory(config) as evaluator:
        result = evaluator.evaluate(NONE)
    assert result.units[0].rollout.status == "infrastructure_error"
    assert "RoutedPolicy" in result.units[0].rollout.error


def invariants_for(config):
    frozen = copy.deepcopy(config)
    frozen["application_mode"] = "first_decision_then_frozen_policy"
    with math_ale_factory(frozen) as evaluator:
        return deployment_invariants(
            evaluator.protocol.to_dict(), [asdict(u) for u in evaluator.units], frozen
        )


def test_offline_replay_cost_equals_online_routed_cost_with_perfect_predictions():
    logged, other = GOOD, SEM
    entries = (CatalogEntry(logged, 1000.0, 1), CatalogEntry(other, 500.0, 1))
    # Truth per round: `other` is better in rounds 0-2, the logged action in 3.
    gain = {logged: [0.0, 0.0, 0.0, 1.0], other: [1.0, 1.0, 1.0, 0.0]}
    generation = {logged: [10, 20, 30, 40], other: [11, 21, 31, 41]}

    def oracle(feature_row):
        r = int(feature_row["round_index"])
        return [(e, gain[e.action][r], float(generation[e.action][r])) for e in entries]

    rounds = [
        RoundSample(
            "t",
            "u",
            logged.action_hash,
            features(r),
            gain[logged][r],
            generation[logged][r],
            1000 if r == 0 else 0,
        )
        for r in range(4)
    ]
    for lam in (0.0, 1e-5):
        offline_gain, offline_cost, choices = simulate_unit(rounds, oracle, lam)
        # Online routed runtime: cache from the router's own choices.
        cached, online_gain, online_cost, online = set(), 0.0, 0.0, []
        for r in range(4):
            entry, _ = _choose(oracle(features(r)), lam, frozenset(cached))
            online_gain += gain[entry.action][r]
            online_cost += generation[entry.action][r]
            online_cost += (
                0 if entry.action.action_hash in cached else entry.retrieval_tokens
            )
            cached.add(entry.action.action_hash)
            online.append(entry.action.action_hash)
        assert choices == online == [other.action_hash] * 3 + [logged.action_hash]
        assert offline_gain == online_gain == 4.0
        # 500 once for `other`, 1000 once for the logged action on its first pick.
        assert offline_cost == online_cost == 11 + 21 + 31 + 40 + 500 + 1000


def test_router_deployment_invariants_and_run_policy_refusal(
    config, monkeypatch, tmp_path
):
    fake_models(monkeypatch)
    config["generation"]["temperature"] = 0.7
    config["semantic"]["temperature"] = 0.0
    samples, actions, provenance = synthetic(
        {GOOD: lambda r: 1.0, NONE: lambda r: 0.0},
        {GOOD: 1, NONE: 1},
        {GOOD: 0, NONE: 0},
    )
    invariants = invariants_for(config)
    artifact = build_artifact(
        samples, actions, {**provenance, "deployment_invariants": invariants}, seed=0
    )
    policy = RoutedPolicy.from_artifact(artifact)
    routed = copy.deepcopy(config)
    routed["application_mode"] = "per_round_routed_policy"
    routed["seeds"] = [7, 8]  # noise-control repeats may use other seeds
    routed["trace_directory"] = str(tmp_path / "other_traces")
    routed["protocol"]["name"] = "branch-b"
    with math_ale_factory(routed) as evaluator:
        protocol = evaluator.protocol.to_dict()
        units = [asdict(u) for u in evaluator.units]
    frozen_units = invariants_for(config)["units"]
    assert units != frozen_units  # other seeds, same task/checkpoint/snapshot
    # Allowed: application_mode, seeds, trace directory, protocol name.
    policy.check_deployment(protocol, units, routed)
    with pytest.raises(ValueError, match="protocol.model_version"):
        policy.check_deployment({**protocol, "model_version": "other"}, units, routed)
    with pytest.raises(ValueError, match="protocol.quality_direction"):
        policy.check_deployment(
            {**protocol, "quality_direction": "minimize"}, units, routed
        )
    for key, value in (("checkpoint_id", "x"), ("graph_snapshot", "sha256:x")):
        with pytest.raises(ValueError, match="units"):
            policy.check_deployment(protocol, [{**units[0], key: value}], routed)
    # The reproduced review cases: same ids, different execution settings.
    for path, value, field in (
        (("generation", "temperature"), 1.8, "generation.temperature"),
        (("generation", "max_output_tokens"), 1, "generation.max_output_tokens"),
        (("semantic", "temperature"), 0.9, "semantic.temperature"),
        (("semantic", "max_output_tokens"), 1, "semantic.max_output_tokens"),
        (("generation", "base_url"), "http://other/v1", "generation.base_url"),
        (("checkpoints", 0, "code_sha256"), "0" * 64, "checkpoints"),
        (("checkpoints", 0, "spec_sha256"), "0" * 64, "checkpoints"),
        (("checkpoints", 0, "evaluator_sha256"), "0" * 64, "checkpoints"),
        (("checkpoints", 0, "initial_fitness"), -3, "checkpoints"),
        (("snapshots", "math:fixture", "sha256"), "0" * 64, "snapshots"),
        (("scorer", "timeout"), 1, "scorer.timeout"),
    ):
        changed = copy.deepcopy(routed)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=field):
            policy.check_deployment(protocol, units, changed)
    changed = copy.deepcopy(routed)
    changed["dependency_files"] = [{"path": "/x", "sha256": "0" * 64}]
    with pytest.raises(ValueError, match="dependency_files"):
        policy.check_deployment(protocol, units, changed)
    stale_code = json.loads(json.dumps(artifact))
    stale_code["training_data"]["deployment_invariants"]["execution"]["code"][
        "packing.py"
    ] = "0" * 64
    stale_code["artifact_sha256"] = artifact_digest(stale_code)
    with pytest.raises(ValueError, match=r"code\.packing\.py"):
        RoutedPolicy.from_artifact(stale_code).check_deployment(protocol, units, routed)
    no_provenance = build_artifact(samples, actions, provenance, seed=0)
    with pytest.raises(ValueError, match="invariants"):
        RoutedPolicy.from_artifact(no_provenance).check_deployment(
            protocol, units, routed
        )

    # run-policy refuses a B config whose model differs from the router's data.
    for section in ("generation", "semantic"):
        routed[section]["model"] = "mock2"
    routed["protocol"]["model_version"] = "mock2"
    router = tmp_path / "router.json"
    router.write_text(json.dumps(artifact))
    run_config = tmp_path / "run.json"
    run_config.write_text(json.dumps({"evaluation": routed}))
    archive = tmp_path / "policy_archive"
    with pytest.raises(ValueError, match="model_version"):
        cli.main(
            [
                "run-policy",
                "--factory",
                "trimem.optimizer.runtime:math_ale_factory",
                "--config",
                str(run_config),
                "--router",
                str(router),
                "--archive",
                str(archive),
            ]
        )
    assert not archive.exists()


def test_sklearn_mismatch_message_names_exact_version():
    samples, actions, provenance = synthetic(
        {GOOD: lambda r: 1.0, NONE: lambda r: 0.0},
        {GOOD: 1, NONE: 1},
        {GOOD: 0, NONE: 0},
    )
    artifact = build_artifact(samples, actions, provenance, seed=0)
    artifact["sklearn_version"] = "0.0.1"
    artifact["artifact_sha256"] = artifact_digest(artifact)
    with pytest.raises(ValueError, match=r"scikit-learn==0\.0\.1"):
        RoutedPolicy.from_artifact(artifact)


def test_train_router_publish_is_exclusive_and_leaves_no_partial_file(tmp_path):
    target = tmp_path / "router.json"
    cli._publish_exclusive(target, "first\n")
    with pytest.raises(ValueError, match="exists"):
        cli._publish_exclusive(target, "second\n")
    assert target.read_text() == "first\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["router.json"]
