import json
from types import SimpleNamespace

import pytest

from trimem.experiments import trace_joint_plan_search as driver
from trimem.optimizer.joint_plan_analyzer import advise
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import JointSpace


def manifest():
    return {
        "space": JointSpace.live().manifest(),
        "tasks": ["a", "b"],
        "protocol": "two-stage-fixture",
        "smoke": True,
        "initial_configs": 10,
        "new_configs": 0,
        "budget": 10,
        "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5},
        "q": 4,
        "seed": 19,
        "task_concurrency": 2,
        "sources": driver.source_identity(),
    }


def fixture(c, t, m):
    return {
        "task_id": t,
        "split": m.get("evaluation_split", "TRAIN"),
        "protocol": m["protocol"],
        "synthetic": True,
        "raw_score": -10 if t == "a" else 80,
        "total_tokens": c["seed_count"],
        "latency_seconds": c["depth"],
    }


def test_two_stages_and_replay_without_extra_llm_or_episodes(tmp_path, monkeypatch):
    m = driver.prepare_run(tmp_path, manifest())
    space = driver.get_space(m)
    assert len(m["initial_keys"]) == 5
    assert set(m["initial_keys"]) == {space.key(c) for c in space.sample(5, seed=19)}
    with SearchLedger(tmp_path / "ledger.sqlite", m) as ledger:
        assert len(ledger.pending()) == 10
    assert driver.run_round(tmp_path, evaluator=fixture)["complete_configs"] == 5
    assert not (tmp_path / "objective.json").exists()
    calls = []

    def initializer(space, obs, settings):
        calls.append(len(obs))
        assert len(obs) == 5 and all(o.mean[0] == 35 for o in obs)
        return space.sample(5, seed=22, avoid={space.key(o.config) for o in obs}), {
            "status": "ok"
        }

    monkeypatch.setattr(driver, "initial_advice", initializer)
    driver.run_round(tmp_path)  # Reserve only; emulate interruption before dispatch.
    assert calls == [5]
    with SearchLedger(tmp_path / "ledger.sqlite", m) as ledger:
        assert len(ledger.pending()) == 10
    result = driver.run_round(tmp_path, evaluator=fixture)
    assert result["complete_configs"] == 10 and calls == [5]
    assert (tmp_path / "objective.json").exists()
    assert driver.run_round(tmp_path, evaluator=fixture)["status"] == "budget_complete"


def test_llm_initial_prompt_requests_five_and_sees_only_train():
    class Client:
        def chat(self, system, user, temperature):
            assert "5" in system and "initial" in system.lower()
            assert len(json.loads(user)["complete_train_archive"]) == 1
            return SimpleNamespace(
                text=json.dumps(
                    {"proposals": [{"seed_count": k} for k in [4, 8, 16, 32, 64]]}
                ),
                error=None,
                input_tokens=10,
                output_tokens=20,
            )

    space = JointSpace.live()
    obs = [SimpleNamespace(config=space.default(), mean=(-5, 20, 2))]
    proposals, _ = advise(space, obs, Client(), max_proposals=5, initialization=True)
    assert len(proposals) == 5


def test_under_delivery_fallback_is_novel_and_explicit(tmp_path, monkeypatch):
    m = driver.prepare_run(tmp_path, manifest())
    driver.run_round(tmp_path, evaluator=fixture)
    with SearchLedger(tmp_path / "ledger.sqlite", m) as ledger:
        old = ledger.complete_configs()[0]["config"]
    monkeypatch.setattr(
        driver, "initial_advice", lambda *a: ([old, old], {"status": "ok"})
    )
    driver.run_round(tmp_path, evaluator=fixture)
    proposal = json.loads((tmp_path / "initialization_llm.json").read_text())
    assert len(proposal["selected"]) == 5
    assert {r["source"] for r in proposal["selected"]} == {"sobol_fallback"}


def test_random_train_panel_and_fixed_test_are_disjoint():
    from tools.trace_joint_protocol import make_panels

    pool = {
        "items": [
            {"item_id": str(i), "split": "train", "task_key": str(i)}
            for i in range(400)
        ],
        "validation_ids": ["1"],
    }
    test_ids = [str(i) for i in range(200, 400)]
    a, b = make_panels(pool, test_ids, seed=19)
    assert len(a["items"]) == 100 and b["tasks"] == test_ids
    assert not ({r["item_id"] for r in a["items"]} & set(test_ids + ["1"]))
    assert make_panels(pool, test_ids, seed=19) == (a, b)
    assert make_panels(pool, test_ids, seed=20)[0] != a
    with pytest.raises(ValueError):
        make_panels(pool, test_ids + [test_ids[0]], seed=19)


def test_test200_is_separate_and_selected_before_any_test_result(tmp_path):
    from tools.trace_joint_protocol import run_holdout

    m = manifest()
    m.pop("initialization")
    m["holdout"] = {"tasks": [str(i) for i in range(200)], "selection": "train_pareto"}
    m = driver.prepare_run(tmp_path, m)
    with pytest.raises(ValueError, match="complete TRAIN"):
        run_holdout(tmp_path, evaluator=fixture)
    driver.run_round(tmp_path, evaluator=fixture)
    before = (tmp_path / "ledger.sqlite").read_bytes()
    calls = []

    def evaluate_test(c, t, test_manifest):
        assert test_manifest["evaluation_split"] == "TEST"
        assert test_manifest["tasks"] == m["holdout"]["tasks"]
        calls.append(t)
        return fixture(c, t, test_manifest)

    result = run_holdout(tmp_path, evaluator=evaluate_test)
    selected = json.loads((tmp_path / "holdout/selection.json").read_text())
    assert result["status"] == "test_complete"
    assert len(calls) == 200 * len(selected["configs"])
    assert (tmp_path / "ledger.sqlite").read_bytes() == before
    count = len(calls)
    assert run_holdout(tmp_path, evaluator=evaluate_test)["status"] == "test_complete"
    assert len(calls) == count


def test_train_manifest_rejects_test_leakage():
    m = manifest()
    m["holdout"] = {"tasks": ["a"], "selection": "train_pareto"}
    with pytest.raises(ValueError, match="TEST"):
        driver.validate_manifest(m)
