"""T9: prepare_trace_v070_campaign through the real (hashed) base.prepare, offline (memory identity stubbed).

shared-init (ALFWorld, live space) -> complete it with identity-checked non-synthetic fake receipts ->
arm (latent-mobo-v1) imports the 10 init configurations and NoMemory; explicit-pr is refused."""

import json
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_v070_campaign as prep
from tests import v070_fixtures as fx
from tools.trace_joint_live import ROOT
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_latent_search import FROZEN_AFFINE_KIND
from trimem.experiments.trace_v070_baseline import (
    baseline_entry,
    baseline_key,
    run_baseline,
)
from trimem.experiments.trace_v070_ledger import SampledLedger
from trimem.experiments.trace_v070_runtime import REQUIRED_ARTIFACTS, v070_patches
from trimem.optimizer.latent_plan_codec import Affine, file_sha256, save_affine
from trimem.optimizer.plan_schema import digest
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

TEMPLATE = ROOT / "results/trace_pr_three_dataset_prepare/template_v060.json"
pytestmark = pytest.mark.skipif(not TEMPLATE.exists(), reason="needs the v0.6.0 template")


def memory_stub(database, env):
    return {"memory_database": database, "database_oid": 1, "sessions": 10, "read_only": True,
            "memory_snapshot": "stub", "memory_writes": 0}


def fake_live(m, space):
    def evaluate(config, task, manifest):
        nm = config.get("baseline") == "NoMemory"
        r = fx.synthetic_receipt(config, task, manifest)
        return {**r, "synthetic": False, "quality_source": "environment_raw_score",
                "config_key": manifest["key"] if nm else space.key(config),
                "execution_fingerprint": manifest["space"]["execution_fingerprint"],
                "runtime": manifest["fixed"]["runtime"]}

    return evaluate


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    run = tmp_path_factory.mktemp("prep") / "trace_v070_alfworld_init_test"
    with mock.patch.object(base, "memory_identity", memory_stub):
        prep.main(["shared-init", "--dataset", "alfworld", "--template", str(TEMPLATE), "--run-dir", str(run),
                   "--memory-db", "trace_ablation_sci_48d26f8d18eba20f"])
    return run


def test_shared_init_manifest(shared):
    m = json.loads((shared / "manifest.json").read_text())
    assert m["space"]["schema"] == "trace-pr-v070-rank-optional-v1"
    assert digest(m["space"]["domains"]) == digest(JointSpaceV070.live().domains)
    assert m["fixed"]["protocol_version"] == prep.PROTOCOL_VERSION
    assert m["search_backend"] == "shared-init-v1" and m["holdout"]["selection"] == "deferred"
    assert m["budget"] == 10 and m["initialization"] == {"method": "sobol_then_llm", "sobol": 5, "llm": 5}
    assert "pruning" not in m and m["train_panel_policy"] == "per-config-stratified-test-mix-v1"
    arts = m["live"]["artifacts"]
    for p in REQUIRED_ARTIFACTS:
        assert str((ROOT / p).resolve()) in arts
    s = m["v070"]["sampling"]
    strat = json.loads((ROOT / "results/trace_latent_alfworld_init_20261104/train_panel.json").read_text())
    assert s["quota"] == strat["quota"] and len(m["tasks"]) == s["eligible_count"] == 3792
    assert set(m["tasks"]).isdisjoint(m["holdout"]["tasks"])
    panel = json.loads((shared / "train_panel.json").read_text())
    assert [r["item_id"] for r in panel["items"]] == m["tasks"]
    nm = baseline_entry(shared, "TRAIN")
    assert nm["key"] == baseline_key(m) == m["baselines"]["NoMemory"]["key"] and len(nm["tasks"]) == 100
    assert dict(Counter(s["strata"][t] for t in nm["tasks"])) == s["quota"]
    samples = json.loads((shared / "samples.json").read_text())
    assert len(samples["configs"]) == 5 and samples["baselines"]["NoMemory"]["tasks"] == nm["tasks"]


@pytest.fixture(scope="module")
def completed(shared):
    m = json.loads((shared / "manifest.json").read_text())
    with v070_patches(m):
        space = driver.get_space(m)
        ev = fake_live(m, space)
        run_baseline(shared, m, "TRAIN", ev)
        with mock.patch.object(driver, "initial_advice", fx.advice_stub()):
            for _ in range(4):
                if driver.run_round(shared, evaluator=ev)["status"] == "budget_complete":
                    break
    ck = shared.parent / "seqvae_v2_d16_fake.pt"
    ck.write_bytes(b"not a model; prepare only hashes it")
    aff = shared.parent / "affine.json"
    save_affine(aff, Affine((0.0,) * 16, (1.0,) * 16),
                {"kind": FROZEN_AFFINE_KIND, "shared_init": str(shared), "checkpoint_sha256": file_sha256(ck)})
    return shared, ck, aff


def arm_args(shared, run, ck, aff, backend="latent-mobo-v1"):
    return ["arm", "--backend", backend, "--shared-init", str(shared), "--dataset", "alfworld", "--template",
            str(TEMPLATE), "--run-dir", str(run), "--memory-db", "trace_ablation_sci_48d26f8d18eba20f",
            "--checkpoint", str(ck), "--affine", str(aff), "--panel", "stratified"]


def test_arm_imports_init_and_nomemory(completed):
    shared, ck, aff = completed
    run = shared.parent / "trace_v070_latent_alfworld_test"
    with mock.patch.object(base, "memory_identity", memory_stub):
        prep.main(arm_args(shared, run, ck, aff))
    m = json.loads((run / "manifest.json").read_text())
    src = json.loads((shared / "manifest.json").read_text())
    assert m["search_backend"] == "latent-mobo-v1" and m["holdout"]["selection"] == "train_pareto"
    assert (m["budget"], m["new_configs"], m["pruning"]["sizes"]) == (80, 70, [20, 40, 60, 80, 100])
    assert m["pruning"]["budget_episodes"] == 5000 and m["baselines"]["NoMemory"]["source"] == "imported"
    assert digest(m["v070"]["sampling"]) == digest(src["v070"]["sampling"]) and m["space"] == src["space"]
    init = set(driver.initialization_keys(shared, src, JointSpaceV070(src["space"]["domains"],
               execution_fingerprint=src["space"]["execution_fingerprint"])))
    assert set(m["initial_keys"]) == init and len(init) == 10
    with SampledLedger(run / "ledger.sqlite", m) as led:
        done = led.complete_configs()
        assert len(done) == 10 and not led.pending()
        for e in done:
            assert sorted(str(r["task_id"]) for r in e["receipts"]) == sorted(led.sample_of(e["key"]))
    nm = baseline_entry(run, "TRAIN")
    assert nm["complete"] and nm["key"] == baseline_key(m)
    assert str(Path(m["latent"]["checkpoint"]).resolve()) in m["live"]["artifacts"]
    assert m["pruning"]["latency_cap"]["factor"] == 2.0 and m["pruning"]["latency_cap"]["protocol"] == (
        "latency-cap-v070-stage-v1")
    assert json.loads((run / "prepared.json").read_text())["kind"] == "arm"


def test_explicit_pr_refused(completed):
    shared, ck, aff = completed
    with pytest.raises(SystemExit, match="refused"):
        prep.main(arm_args(shared, shared.parent / "explicit", ck, aff, backend="explicit-pr"))


def test_prepared_marker_and_replay(shared, tmp_path):
    from trimem.experiments import trace_v070_marker as mk

    body = mk.check_prepared(shared)
    assert body["kind"] == "shared-init" and body["nomemory_key"] == baseline_key(
        json.loads((shared / "manifest.json").read_text()))
    half = tmp_path / "interrupted"
    half.mkdir()
    (half / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="preparation incomplete"):
        mk.check_prepared(half)
    aside = mk.set_aside_incomplete(half)
    assert aside is not None and aside.exists() and not half.exists()
    started = tmp_path / "started"
    (started / "episodes").mkdir(parents=True)
    with pytest.raises(SystemExit, match="evaluation evidence"):
        mk.set_aside_incomplete(started)
    assert mk.set_aside_incomplete(shared) is None  # a prepared run is never moved


def test_arm_refuses_unprepared_shared_init(completed, tmp_path):
    shared, ck, aff = completed
    fake = tmp_path / "unprepared_init"
    fake.mkdir()
    (fake / "manifest.json").write_text((shared / "manifest.json").read_text())
    with pytest.raises(SystemExit, match="preparation incomplete"):
        prep.main(arm_args(fake, tmp_path / "arm_x", ck, aff))


def test_runtime_parity_enforced(shared):
    from trimem.experiments.trace_v070_baseline import load_baseline, parity_violations
    from trimem.experiments.trace_v070_runtime import validate_v070_manifest

    m = json.loads((shared / "manifest.json").read_text())
    validate_v070_manifest(m)
    for path, value in ((("task_concurrency",), 8), (("live", "agent_max_tokens"), 8192),
                        (("fixed", "agent_max_tokens"), 8192), (("baselines", "NoMemory", "latency_front"), True)):
        bad = json.loads(json.dumps(m))
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        with pytest.raises(ValueError):
            validate_v070_manifest(bad)
    bm = load_baseline(shared, "TRAIN")
    assert parity_violations(m, bm) == []
    for key, value in (("model", "other"), ("llm_bases", ["http://x/v1"]), ("environment_bases", ["http://y"]),
                       ("agent_max_tokens", 8192)):
        assert parity_violations(m, {**bm, "live": {**bm["live"], key: value}}) == [f"live.{key}"]
    assert "task_concurrency" in parity_violations(m, {**bm, "task_concurrency": 8})
