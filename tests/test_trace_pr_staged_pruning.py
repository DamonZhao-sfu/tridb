"""TRACE-PR staged pruning (staged-pareto-prune-v2 reused, user 2026-11-04)."""

import json

from trimem.experiments import trace_joint_plan_search as driver
from trimem.optimizer import staged_pruning as sp
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import JointSpace

TASKS = [f"t{i}" for i in range(10)]


def manifest():
    return {
        "space": JointSpace.live().manifest(),
        "tasks": TASKS,
        "protocol": "staged-fixture",
        "smoke": True,
        "initial_configs": 10,
        "new_configs": 6,
        "budget": 16,
        "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5},
        "q": 4,
        "seed": 19,
        "task_concurrency": 2,
        "sources": driver.source_identity(),
        "pruning": {
            "protocol": sp.PROTOCOL,
            "z": 1.96,
            "sizes": [4, 8, 10],
            "order": list(TASKS),
            "budget_episodes": 12 * len(TASKS),
        },
    }


def fixture(c, t, m):
    # seed_count 128 is great, anything else is poor -> poor BO configurations get pruned.
    good = c["seed_count"] == 128
    return {
        "task_id": t,
        "split": m.get("evaluation_split", "TRAIN"),
        "protocol": m["protocol"],
        "synthetic": True,
        "raw_score": (100 if good else 0) + int(t[1:]) % 2,
        "total_tokens": 1000 if good else 5000 + 10 * int(t[1:]),
        "latency_seconds": 10 if good else 50 + int(t[1:]),
    }


def test_ledger_pruned_state_semantics(tmp_path):
    m = driver.prepare_run(tmp_path, manifest())
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        key = led.pending()[0]["key"]
        assert led.prune(key, "x") == len(TASKS)
        assert all(p["key"] != key for p in led.pending())
        done = [e for e in led.complete_configs() if e["key"] == key]
        assert done and done[0]["pruned"] and done[0]["receipts"] == []


def test_partial_panel_observation_has_larger_noise():
    from trimem.optimizer.plan_surrogate import aggregate_panel

    space = JointSpace.live()
    c = space.sample(1, seed=3)[0]
    rows = [fixture(c, t, {"protocol": "p"}) for t in TASKS]
    for r in rows:  # same spread in every prefix: alternating 0 / 100
        r["raw_score"] = 100 * (int(r["task_id"][1:]) % 2)
    full = aggregate_panel(space, c, rows, task_ids=TASKS, protocol="p", allow_synthetic=True)
    part = aggregate_panel(space, c, rows[:4], task_ids=TASKS, protocol="p", allow_synthetic=True,
                           allow_partial=True)
    assert part.variance[0] > full.variance[0] and part.panel_identity == full.panel_identity


def test_staged_round_prunes_dominated_and_keeps_budget(tmp_path, monkeypatch):
    m = driver.prepare_run(tmp_path, manifest())
    space = driver.get_space(m)
    monkeypatch.setattr(
        driver, "initial_advice",
        lambda space, obs, settings: (
            [{**space.sample(1, seed=7)[0], "seed_count": 128}] + space.sample(4, seed=22, avoid={space.key(o.config) for o in obs}),
            {"status": "ok"},
        ),
    )
    # stub the search: propose 4 poor configurations (seed_count 4) -> dominated by the good init config
    def fake_search(directory, man, complete, *, agent, seed, round_file):
        avoid = {e["key"] for e in complete}
        cands = [dict(c, seed_count=4) for c in space.sample(40, seed=seed + 100, avoid=avoid)]
        uniq, seen = [], set(avoid)
        for c in cands:
            c = space.canonicalize(c)
            if space.key(c) not in seen:
                seen.add(space.key(c))
                uniq.append(c)
        sel = [{"config": c, "key": space.key(c), "sources": ["pr"]} for c in uniq[: man["q"]]]
        res = {"selected": sel, "archive_keys": sorted(avoid)}
        (directory / f"{round_file}.json").write_text(json.dumps(res))
        return res, None

    monkeypatch.setattr(driver, "guarded_search", fake_search)
    for _ in range(3):
        driver.run_round(tmp_path, evaluator=fixture)
    log = [json.loads(x) for x in (tmp_path / "pruning_log.jsonl").read_text().splitlines()]
    assert any(d["prune"] for d in log)
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        used = led.db.execute("SELECT count(*) FROM episodes WHERE state='done'").fetchone()[0]
        n_pruned = led.db.execute("SELECT count(DISTINCT config) FROM episodes WHERE state='pruned'").fetchone()[0]
        init = set(driver.initialization_keys(tmp_path, m, space))
        pruned_keys = {k for (k,) in led.db.execute("SELECT DISTINCT config FROM episodes WHERE state='pruned'")}
    assert n_pruned > 0 and not (pruned_keys & init)  # initialization is protected
    assert used <= m["pruning"]["budget_episodes"]
    # observations include pruned configurations as partial panels
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        obs = driver.observations(led, space, m)
        complete = led.complete_configs()
    assert len(obs) == len(complete) and any(e["pruned"] for e in complete)


def _run_staged(tmp_path, monkeypatch, rounds=3):
    m = driver.prepare_run(tmp_path, manifest())
    space = driver.get_space(m)
    monkeypatch.setattr(
        driver, "initial_advice",
        lambda space, obs, settings: (
            [{**space.sample(1, seed=7)[0], "seed_count": 128}]
            + space.sample(4, seed=22, avoid={space.key(o.config) for o in obs}),
            {"status": "ok"},
        ),
    )

    def fake_search(directory, man, complete, *, agent, seed, round_file):
        avoid = {e["key"] for e in complete}
        uniq, seen = [], set(avoid)
        for c in space.sample(40, seed=seed + 100, avoid=avoid):
            c = space.canonicalize(dict(c, seed_count=4))
            if space.key(c) not in seen:
                seen.add(space.key(c))
                uniq.append(c)
        sel = [{"config": c, "key": space.key(c), "sources": ["pr"]} for c in uniq[: man["q"]]]
        res = {"selected": sel, "archive_keys": sorted(avoid)}
        (directory / f"{round_file}.json").write_text(json.dumps(res))
        return res, None

    monkeypatch.setattr(driver, "guarded_search", fake_search)
    results = [driver.run_round(tmp_path, evaluator=fixture) for _ in range(rounds)]
    return m, space, results


def test_resume_does_not_redecide_passed_checkpoints(tmp_path, monkeypatch):
    m, space, _ = _run_staged(tmp_path, monkeypatch)
    log_before = (tmp_path / "pruning_log.jsonl").read_text()
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        full = [e for e in led.complete_configs() if not e["pruned"]][-1]["key"]
        # emulate a lost receipt of the last task: the batch resumes at its last stage
        led.db.execute("UPDATE episodes SET state='ready', receipt=NULL WHERE config=? AND task='t9'", (full,))
        pending = led.pending()
    driver.staged_evaluate(tmp_path, m, space, pending, fixture)
    assert (tmp_path / "pruning_log.jsonl").read_text() == log_before  # no new decision


def test_stage_barrier_stops_on_unresolved_claim(tmp_path, monkeypatch):
    import pytest

    m, space, _ = _run_staged(tmp_path, monkeypatch, rounds=2)
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        key = [e for e in led.complete_configs() if not e["pruned"]][-1]["key"]
        led.db.execute("UPDATE episodes SET state='ready', receipt=NULL WHERE config=? AND task IN ('t5','t6')", (key,))
        led.db.execute("UPDATE episodes SET state='claimed' WHERE config=? AND task='t5'", (key,))
        pending = led.pending()
    with pytest.raises(RuntimeError, match="incomplete"):
        driver.staged_evaluate(tmp_path, m, space, pending, fixture)


def test_prune_covers_failed_and_is_atomic(tmp_path):
    m = driver.prepare_run(tmp_path, manifest())
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        key = led.pending()[0]["key"]
        led.db.execute("UPDATE episodes SET state='failed', retryable=1 WHERE config=? AND task='t0'", (key,))
        out = led.prune_many({key: "r"})
        assert out[key] == len(TASKS)
        led.retry(max_attempts=5)
        assert all(p["key"] != key for p in led.pending())


def test_final_selection_accepts_episode_budget_termination(tmp_path, monkeypatch):
    from tools.trace_pr_final_select import fit_posterior

    m, space, results = _run_staged(tmp_path, monkeypatch, rounds=10)
    assert results[-1]["status"] == "budget_complete"
    with SearchLedger(tmp_path / "ledger.sqlite", m) as led:
        n = len(led.complete_configs())
    assert n < m["budget"]  # stopped by the episode budget, not the config cap
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    rows, _ = fit_posterior(tmp_path)
    assert any(r["pruned"] for r in rows)
