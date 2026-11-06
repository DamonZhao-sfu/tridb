"""Per-configuration stratified TRAIN samples (Amendment A.4): draw, prefixes, ledger, samples.json, aggregation."""

import json
from collections import Counter

import pytest

import tools.prepare_trace_v070_campaign as prep
from tests import v070_fixtures as fx
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_v070_samples as ts
from trimem.experiments.trace_v070_baseline import baseline_key, train_sample
from trimem.experiments.trace_v070_ledger import (
    SampledLedger,
    read_samples,
    write_samples_json,
)
from trimem.experiments.trace_v070_runtime import v070_patches
from trimem.optimizer.plan_surrogate import aggregate_panel


@pytest.fixture(scope="module")
def block100():
    return fx.tiny_sampling(size=100)[0]


def test_quota_follows_test_mix_not_pool(block100):
    # TEST mix A 20 / B 20 / C 40 / D 20 (pool is A 40 / B 30 / C 20 / D 10)
    assert block100["quota"] == {"A": 20, "B": 20, "C": 40, "D": 20}
    assert block100["eligible_count"] == 299  # tiny_7 (validation) and the TEST id are excluded
    assert "tiny_7" not in block100["strata"] and fx.HOLDOUT[0] not in block100["strata"]


def test_two_configs_get_different_samples_with_the_test_mix(block100):
    a, b = ts.config_sample(block100, "key-a"), ts.config_sample(block100, "key-b")
    assert a != b and set(a) != set(b)
    for s in (a, b):
        assert len(s) == len(set(s)) == 100
        assert dict(Counter(block100["strata"][t] for t in s)) == block100["quota"]
    assert ts.config_sample(block100, "key-a") == a  # deterministic in (seed, key, pool)
    other = dict(block100, seed=block100["seed"] + 1)
    assert ts.config_sample(other, "key-a") != a


def test_every_20_task_prefix_follows_the_mix(block100):
    for i in range(50):
        s = ts.config_sample(block100, f"k{i}")
        assert ts.max_prefix_deviation(block100, s, [20, 40, 60, 80, 100]) <= 1.0
        assert ts.prefix_mix(s, block100["strata"], 20) == {"A": 4, "B": 4, "C": 8, "D": 4}


def test_validate_sampling(block100):
    ts.validate_sampling(block100, list(block100["strata"]))
    with pytest.raises(ValueError):
        ts.validate_sampling(block100, list(block100["strata"])[:-1])
    with pytest.raises(ValueError):
        ts.validate_sampling({**block100, "quota": {**block100["quota"], "A": 999}}, list(block100["strata"]))


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    run = tmp_path_factory.mktemp("samples") / "shared"
    return run, prep.shared_prepare_run_v070(run, fx.shared_manifest(size=40))


def test_ledger_reserves_own_samples_and_records_them(shared):
    run, m = shared
    block = m["v070"]["sampling"]
    with SampledLedger(run / "ledger.sqlite", m) as led:
        keys = [k for (k,) in led.db.execute("SELECT key FROM configs")]
        assert len(keys) == 5
        for k in keys:
            tasks = sorted(t for (t,) in led.db.execute("SELECT task FROM episodes WHERE config=?", (k,)))
            assert tasks == sorted(ts.config_sample(block, k)) == sorted(led.sample_of(k))
        sets = [frozenset(led.sample_of(k)) for k in keys]
    assert len(set(sets)) == 5
    body = json.loads((run / "samples.json").read_text())
    assert set(body["configs"]) == set(keys)
    nm = body["baselines"]["NoMemory"]
    assert nm["key"] == baseline_key(m) and nm["tasks"] == train_sample(m)
    assert set(nm["tasks"]) not in [set(s) for s in sets]
    assert read_samples(run / "ledger.sqlite") == {k: v["tasks"] for k, v in body["configs"].items()}
    write_samples_json(run, m)  # idempotent and consistent


def test_aggregation_uses_own_tasks_and_one_panel_identity(shared):
    run, m = shared
    space = fx.tiny_space_v070()
    evaluate = fx.evaluator_for(space)
    with v070_patches(m):
        with SampledLedger(run / "ledger.sqlite", m) as led:
            configs = [e["config"] for e in led.pending()]
        rows = {}
        for c in {json.dumps(c, sort_keys=True): c for c in configs}.values():
            own = ts.config_sample(m["v070"]["sampling"], space.key(c))
            rows[space.key(c)] = (c, [evaluate(c, t, m) for t in own], own)
        obs = [driver.aggregate_panel(space, c, r, task_ids=m["tasks"], protocol=m["protocol"],
                                      allow_synthetic=True) for c, r, _ in rows.values()]
    assert len({o.panel_identity for o in obs}) == 1
    for o, (c, r, own) in zip(obs, rows.values()):
        stock = aggregate_panel(space, c, r, task_ids=own, protocol=m["protocol"], allow_synthetic=True)
        assert o.mean == stock.mean and o.variance == stock.variance
    c, r, own = next(iter(rows.values()))
    with v070_patches(m), pytest.raises(ValueError):  # another configuration's tasks are not its panel
        other = next(v for k, v in rows.items() if k != space.key(c))
        driver.aggregate_panel(space, c, other[1], task_ids=m["tasks"], protocol=m["protocol"],
                               allow_synthetic=True)
