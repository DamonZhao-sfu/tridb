"""T3: runtime patches are scoped, restored, refused for non-v0.7.0 manifests and re-entered in a spawned child."""

import json
import multiprocessing

import pytest

import tools.prepare_trace_v070_campaign as prep
from tests import v070_fixtures as fx
from tools import trace_joint_live as live
from tools import trace_joint_protocol as proto
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.experiments import trace_v070_runtime as rt
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

NAMES = ("JointSpace", "validate_manifest", "aggregate_panel", "SearchLedger", "pruning_budget",
         "staged_evaluate", "_search_worker")


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    run = tmp_path_factory.mktemp("rt") / "shared"
    return run, prep.shared_prepare_run_v070(run, fx.shared_manifest())


def latent_manifest(m):
    return {**m, "search_backend": rt.LATENT_BACKEND}


def test_patches_restored(shared):
    _, m = shared
    before = {n: getattr(driver, n) for n in NAMES}
    extra = (live.JointSpace, live.runner_config, proto.run_holdout)
    with rt.v070_patches(latent_manifest(m)):
        assert driver.JointSpace is JointSpaceV070 and live.JointSpace is JointSpaceV070
        assert driver._search_worker is rt.latent_search_worker_v070
        assert driver.validate_manifest is rt.validate_manifest_v070
        assert driver.SearchLedger.__name__ == "SampledLedger"
        assert proto.run_holdout.__name__ == "run_pruned_holdout_v070"
        assert all(rt.patched_names().values())
    assert {n: getattr(driver, n) for n in NAMES} == before
    assert (live.JointSpace, live.runner_config, proto.run_holdout) == extra


def test_shared_backend_refuses_search(shared):
    _, m = shared
    with rt.v070_patches(m), pytest.raises(RuntimeError, match="latent backend"):
        driver._search_worker()


def test_refuses_non_v070_and_second_run(shared):
    _, m = shared
    with pytest.raises(ValueError), rt.v070_patches({k: v for k, v in m.items() if k != "v070"}):
        pass
    with rt.v070_patches(m):
        with rt.v070_patches(m):  # same run nests
            pass
        with pytest.raises(RuntimeError, match="another v0.7.0 run"), rt.v070_patches({**m, "protocol": "other"}):
            pass
    assert rt._ACTIVE == {"protocol": None, "depth": 0}


def test_validate_manifest_v070(shared):
    _, m = shared
    rt.validate_manifest_v070(m)
    with pytest.raises(ValueError, match="100 TRAIN tasks"):
        driver.validate_manifest({**m, "smoke": False})  # stock live rule: the pool is not a 100-task panel
    with pytest.raises(ValueError, match="rank_impl"):
        driver.get_space(m)  # stock space refuses the v0.7.0 domains
    bad = json.loads(json.dumps(m))
    bad["v070"]["sampling"]["quota"] = {"A": 1}
    with pytest.raises(ValueError):
        rt.validate_manifest_v070(bad)
    bad = json.loads(json.dumps(m))
    bad["baselines"]["NoMemory"]["test"] = "pareto"
    with pytest.raises(ValueError):
        rt.validate_manifest_v070(bad)


def test_worker_reenters_patches(shared, monkeypatch, tmp_path):
    _, m = shared
    seen = {}

    def probe(directory, manifest, entries, agent, seed, round_file):
        seen.update(rt.patched_names())

    monkeypatch.setattr(tls, "latent_search_worker", probe)
    rt.latent_search_worker_v070(str(tmp_path), latent_manifest(m), [], [], 0, "round_010")
    assert seen and all(seen.values())
    marker = json.loads((tmp_path / "round_010.v070.json").read_text())
    assert marker["worker"] == "latent_search_worker_v070"
    assert driver.JointSpace is not JointSpaceV070  # restored after the worker


def _spawn(target, manifest, out):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=target, args=(manifest, str(out)))
    p.start()
    p.join(120)
    return out.read_text()


def test_spawned_child_needs_the_patches(shared, tmp_path):
    _, m = shared
    with rt.v070_patches(m):  # the parent is patched; the spawned child is not
        plain = _spawn(fx.child_get_space, m, tmp_path / "plain.txt")
        patched = _spawn(fx.child_patched_get_space, m, tmp_path / "patched.txt")
    assert plain.startswith("error") and "rank_impl" in plain
    assert patched == "ok JointSpaceV070"
