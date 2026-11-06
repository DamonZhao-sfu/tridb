import json
from types import SimpleNamespace

import pytest
import torch

from trimem.experiments.trace_staged_plan_search import (
    get_space,
    prepare_run,
    run_round,
    source_identity,
)
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.joint_staged_analyzer import advise
from trimem.optimizer.plan_schema import JointSpace


def manifest():
    return {
        "space": JointSpace.live().manifest(),
        "tasks": ["0", "1"],
        "protocol": "staged-test",
        "smoke": True,
        "budget": 14,
        "q": 4,
        "task_concurrency": 2,
        "sources": source_identity(),
    }


def evaluate(config, task, settings):
    return {
        "task_id": task,
        "split": "TRAIN",
        "protocol": settings["protocol"],
        "synthetic": True,
        "raw_score": -10 + int(task),
        "total_tokens": 100,
        "latency_seconds": 2,
    }


def test_two_initialization_batches_have_a_real_feedback_barrier(tmp_path):
    frozen = prepare_run(tmp_path, manifest())
    assert frozen["search_schedule"]["sobol_count"] == 5
    assert frozen["search_schedule"]["agent_count"] == 5
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert len(ledger.pending()) == 10  # five configurations, two fixture tasks
    first = run_round(tmp_path, evaluator=evaluate)
    assert first["complete_configs"] == 5
    assert not (tmp_path / "objective.json").exists()
    waiting = run_round(tmp_path)
    assert waiting["status"] == "awaiting_initialization_advice"
    space = get_space(frozen)
    second = space.sample(5, seed=751, avoid=frozen["initial_keys"])
    result = run_round(tmp_path, evaluator=evaluate, agent=second)
    assert result["complete_configs"] == 10
    assert (tmp_path / "objective.json").exists()
    receipt = json.loads((tmp_path / "initialization_agent.json").read_text())
    assert set(receipt["archive_keys"]) == set(frozen["initial_keys"])
    assert len(receipt["configs"]) == 5
    assert len(set(receipt["keys"]) | set(frozen["initial_keys"])) == 10


def test_bad_second_batch_does_not_reserve_any_config(tmp_path):
    frozen = prepare_run(tmp_path, manifest())
    run_round(tmp_path, evaluator=evaluate)
    space = get_space(frozen)
    with pytest.raises(ValueError, match="five|5|distinct"):
        run_round(tmp_path, agent=[space.default()] * 5)
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert len(ledger.complete_configs()) == 5 and not ledger.pending()


def test_init_analyzer_can_return_five_after_reading_first_batch():
    space = JointSpace.live()
    configs = space.sample(5, seed=42)
    obs = [SimpleNamespace(config=c, mean=(-10, 100, 2)) for c in configs]

    class Client:
        def chat(self, system, user, temperature):
            body = json.loads(user)
            assert len(body["complete_train_archive"]) == 5
            assert body["requested_proposals"] == 5
            assert body["purpose"] == "initialization"
            return SimpleNamespace(
                text=json.dumps({"proposals": space.sample(5, seed=11)}),
                error=None,
                input_tokens=1,
                output_tokens=2,
            )

    candidates, _ = advise(space, obs, Client(), n_propose=5, purpose="initialization")
    assert len(candidates) == 5


def test_stagnation_counts_completed_ordinary_configurations_and_resets():
    from trimem.optimizer.joint_search_protocol import stagnation_status

    events = [{"kind": "ordinary", "nhv": 1.0} for _ in range(3)]
    assert stagnation_status(1.0, events)["force_next"]
    assert not stagnation_status(1.0, events + [{"kind": "forced", "nhv": 1.0}])[
        "force_next"
    ]
    assert not stagnation_status(1.0, events + [{"kind": "ordinary", "nhv": 1.1}])[
        "force_next"
    ]


def test_forced_choice_uses_uncertainty_and_leaves_other_slots_pending_aware():
    from trimem.optimizer.joint_search_protocol import select_with_escape

    space = JointSpace.live()
    configs = [space.default(seed_count=n) for n in (4, 8, 16, 32, 64)]

    class Snapshot:
        identity = "fixture"
        pending = ()

        def score(self, cs):
            return [c["seed_count"] for c in cs]

        def posterior_stats(self, cs):
            return torch.zeros(len(cs), 3), torch.tensor(
                [[100.0 if c["seed_count"] == 4 else 1.0] * 3 for c in cs]
            )

        def with_pending(self, pending):
            child = Snapshot()
            child.pending = pending
            child.identity = str(len(pending))
            return child

    selected = select_with_escape(space, Snapshot(), configs, configs, [], q=4)
    assert selected[0]["config"]["seed_count"] == 4
    assert selected[0]["selection_rule"] == "forced_uncertainty_novelty"
    assert selected[1]["config"]["seed_count"] == 64
    assert len({r["key"] for r in selected}) == 4


def test_staged_resume_batch_size_and_forced_exploration(tmp_path, monkeypatch):
    import trimem.experiments.trace_staged_plan_search as driver

    frozen = prepare_run(tmp_path, {**manifest(), "budget": 20})
    space = get_space(frozen)
    calls = []

    def evaluator(c, t, m):
        calls.append((space.key(c), t))
        return evaluate(c, t, m)

    run_round(tmp_path, evaluator=evaluator)
    second = space.sample(5, seed=751, avoid=frozen["initial_keys"])
    run_round(tmp_path, agent=second, evaluator=evaluator)
    generated = []

    def search(directory, m, entries, *, seed, round_file, agent):
        count = m["q"]
        forced = m["_search_progress"]["force_next"]
        generated.append((count, forced))
        configs = space.sample(
            count, seed=100 + len(entries), avoid=[e["key"] for e in entries]
        )
        selected = [
            {
                "config": c,
                "key": space.key(c),
                "selection_rule": "forced_uncertainty_novelty"
                if forced and i == 0
                else "qLogNEHVI",
            }
            for i, c in enumerate(configs)
        ]
        result = {"archive_keys": [e["key"] for e in entries], "selected": selected}
        driver.atomic_json(directory / (round_file + ".json"), result)
        return result, {}

    monkeypatch.setattr(driver, "guarded_search", search)
    # Crash/restart after generating/reserving a batch must not select it twice.
    run_round(tmp_path)
    assert generated == [(4, False)]
    assert run_round(tmp_path, evaluator=evaluator)["complete_configs"] == 14
    assert generated == [(4, False)]
    assert run_round(tmp_path, evaluator=evaluator)["complete_configs"] == 18
    assert run_round(tmp_path, evaluator=evaluator)["complete_configs"] == 20
    assert generated == [(4, False), (4, True), (2, True)]
    assert run_round(tmp_path, evaluator=evaluator)["status"] == "budget_complete"
    assert len(calls) == len(set(calls)) == 40


def test_second_advice_survives_crash_before_reservation(tmp_path, monkeypatch):
    frozen = prepare_run(tmp_path, manifest())
    run_round(tmp_path, evaluator=evaluate)
    space = get_space(frozen)
    second = space.sample(5, seed=751, avoid=frozen["initial_keys"])
    original = SearchLedger.reserve

    def fail(*args, **kwargs):
        raise RuntimeError("fixture crash before reserve")

    monkeypatch.setattr(SearchLedger, "reserve", fail)
    with pytest.raises(RuntimeError, match="fixture crash"):
        run_round(tmp_path, agent=second)
    monkeypatch.setattr(SearchLedger, "reserve", original)
    assert run_round(tmp_path, evaluator=evaluate)["complete_configs"] == 10


def test_uncertain_first_batch_blocks_second_batch(tmp_path):
    frozen = prepare_run(tmp_path, manifest())
    space = get_space(frozen)
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        item = ledger.pending()[0]
        ledger.claim(item["key"], item["task_id"])
    second = space.sample(5, seed=751, avoid=frozen["initial_keys"])
    result = run_round(tmp_path, evaluator=evaluate, agent=second)
    assert result["status"] == "pending"
    assert result["complete_configs"] == 4
    assert not (tmp_path / "initialization_agent.json").exists()


def test_tampered_initialization_advice_is_rejected(tmp_path):
    frozen = prepare_run(tmp_path, manifest())
    space = get_space(frozen)
    run_round(tmp_path, evaluator=evaluate)
    run_round(tmp_path, agent=space.sample(5, seed=751, avoid=frozen["initial_keys"]))
    path = tmp_path / "initialization_agent.json"
    record = json.loads(path.read_text())
    record["archive_digest"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        run_round(tmp_path, evaluator=evaluate)


def test_prepare_cannot_overwrite_existing_run(tmp_path):
    prepare_run(tmp_path, manifest())
    with pytest.raises(ValueError, match="new empty"):
        prepare_run(tmp_path, manifest())


@pytest.mark.parametrize(("forced", "q"), [(True, 4), (False, 3)])
def test_real_gp_pr_and_acquisition_selection(forced, q):
    from trimem.optimizer.joint_search_protocol import (
        EscapeSnapshot,
        propose_with_escape,
    )
    from trimem.optimizer.plan_codec import JointCodec
    from trimem.optimizer.plan_surrogate import (
        ObjectiveProtocol,
        aggregate_panel,
        fit_model,
    )

    torch.set_num_threads(4)
    space = JointSpace.live()
    configs = space.sample(5, seed=44)
    obs = []
    for i, config in enumerate(configs):
        rows = [
            {
                **evaluate(config, str(t), {"protocol": "staged-test"}),
                "raw_score": -20 + 10 * i + t,
                "total_tokens": 100 + 20 * i + t,
                "latency_seconds": 2 + 0.2 * i + 0.01 * t,
            }
            for t in (0, 1)
        ]
        obs.append(
            aggregate_panel(
                space,
                config,
                rows,
                task_ids=["0", "1"],
                protocol="staged-test",
                allow_synthetic=True,
            )
        )
    codec = JointCodec(space)
    objective = ObjectiveProtocol.from_initial(obs)
    model = fit_model(codec, obs, objective, maxiter=2)
    snapshot = EscapeSnapshot(codec, model, obs, objective, mc_samples=4)
    result = propose_with_escape(
        space,
        snapshot,
        configs,
        force_exploration=forced,
        q=q,
        steps=1,
        samples=4,
        pr_seconds=2,
        seed=56,
    )
    selected = result["selected"]
    assert len({r["key"] for r in selected}) == q
    assert not {r["key"] for r in selected} & {space.key(c) for c in configs}
    assert all(space.canonicalize(r["config"]) == r["config"] for r in selected)
    assert all(torch.isfinite(torch.tensor(r["ell"])) for r in selected)
    assert len({r["snapshot"] for r in selected}) == q
    assert sum(
        r["selection_rule"] == "forced_uncertainty_novelty" for r in selected
    ) == int(forced)


def test_analyzer_stage_is_wired_to_measured_feedback(tmp_path, monkeypatch):
    import trimem.experiments.trace_staged_plan_search as driver
    from trimem.optimizer import analyzer_pool

    frozen = prepare_run(
        tmp_path,
        {
            **manifest(),
            "analyzer": {
                "backend": "qwen",
                "model": "fixture",
                "api_base": "http://fixture.invalid/v1",
            },
        },
    )
    space = get_space(frozen)
    second = space.sample(5, seed=751, avoid=frozen["initial_keys"])
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def chat(self, system, user, temperature):
            body = json.loads(user)
            calls.append(body)
            return SimpleNamespace(
                text=json.dumps({"proposals": second}),
                error=None,
                input_tokens=10,
                output_tokens=20,
            )

    monkeypatch.setattr(analyzer_pool, "QwenChat", Client)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)

    def guarded(directory, m, entries, *, round_file):
        driver._search_worker(directory, m, entries, [], 0, round_file)
        return json.loads((directory / (round_file + ".json")).read_text()), {}

    monkeypatch.setattr(driver, "guarded_search", guarded)
    run_round(tmp_path, evaluator=evaluate)
    assert not calls
    result = run_round(tmp_path, evaluator=evaluate)
    assert result["complete_configs"] == 10
    assert len(calls) == 1 and calls[0]["requested_proposals"] == 5
    assert len(calls[0]["complete_train_archive"]) == 5
    assert all(o["quality"] < 0 for o in calls[0]["complete_train_archive"])
