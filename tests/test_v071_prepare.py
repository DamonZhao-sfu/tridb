"""prepare_trace_v071_campaign through the real hashed base.prepare, offline: v2c init + arm; the mid-run ALFWorld arm."""

import json
from unittest import mock

import pytest

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_v071_campaign as p71
from tests import v070_fixtures as fx
from tests.test_v070_prepare import TEMPLATE, arm_args, fake_live, memory_stub
from tools.trace_joint_live import ROOT
from trimem.experiments import semfilter_v071 as v2c
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.experiments import trace_v071_runtime as r71
from trimem.experiments.trace_latent_search import FROZEN_AFFINE_KIND
from trimem.experiments.trace_v070_baseline import (
    baseline_entry,
    baseline_key,
    run_baseline,
)
from trimem.experiments.trace_v070_ledger import SampledLedger
from trimem.experiments.trace_v070_runtime import v070_patches, validate_manifest_v070
from trimem.experiments.trace_v071_search import RULE
from trimem.optimizer.latent_plan_codec import Affine, file_sha256, save_affine

pytestmark = pytest.mark.skipif(not TEMPLATE.exists(), reason="needs the v0.6.0 template")
DB = "trace_ablation_sci_48d26f8d18eba20f"


def init_args(run, *extra):
    return ["shared-init", "--dataset", "alfworld", "--template", str(TEMPLATE), "--run-dir", str(run), "--memory-db", DB,
            *extra]


def prepare_init(run, *extra):
    with mock.patch.object(base, "memory_identity", memory_stub):
        p71.main(init_args(run, *extra))
    return json.loads((run / "manifest.json").read_text())


def complete_init(run, m):
    with v070_patches(m):
        space = driver.get_space(m)
        ev = fake_live(m, space)
        run_baseline(run, m, "TRAIN", ev)
        with mock.patch.object(driver, "initial_advice", fx.advice_stub()):
            for _ in range(4):
                if driver.run_round(run, evaluator=ev)["status"] == "budget_complete":
                    break
    ck = run.parent / f"{run.name}_ck.pt"
    ck.write_bytes(b"prepare only hashes it")
    aff = run.parent / f"{run.name}_affine.json"
    save_affine(aff, Affine((0.0,) * 16, (1.0,) * 16),
                {"kind": FROZEN_AFFINE_KIND, "shared_init": str(run), "checkpoint_sha256": file_sha256(ck)})
    return ck, aff


def prepare_arm(shared, run, ck, aff, *extra):
    with mock.patch.object(base, "memory_identity", memory_stub):
        p71.main([*arm_args(shared, run, ck, aff), *extra])
    return json.loads((run / "manifest.json").read_text())


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("v071prep")


@pytest.fixture(scope="module")
def old_init(tmp):
    run = tmp / "old_init"
    m = prepare_init(run, "--semfilter", "v2")  # = the v0.7.0 command: stock SemFilter
    return run, m, *complete_init(run, m)


@pytest.fixture(scope="module")
def new_init(tmp):
    run = tmp / "new_init"
    m = prepare_init(run)  # default v2c
    return run, m, *complete_init(run, m)


def test_shared_init_v2c_manifest(new_init, old_init):
    run, m, *_ = new_init
    old = old_init[1]
    r71.validate_v071_manifest(m)
    validate_manifest_v070(m)
    assert m["v071"]["semfilter"] == v2c.BLOCK == m["fixed"]["semfilter"] and m["v071"]["stall_rule"] is None
    env = m["live"]["environment"]
    assert env["TRIMEM_SEMFILTER_TOPN"] == "16" and env["TRIMEM_SEMFILTER_JUDGE_INPUT"] == "compact-v2c"
    assert env[v2c.ENV_GATE] == "1" and env[v2c.ENV_REUSE] == "episode"
    for p in r71.REQUIRED_ARTIFACTS:
        assert str((ROOT / p).resolve()) in m["live"]["artifacts"]
    assert (run / "prepared.json").exists() and m["q"] == 4
    # the protocol changes the runtime identity: other fingerprint, keys and NoMemory sample
    assert m["space"]["execution_fingerprint"] != old["space"]["execution_fingerprint"]
    assert baseline_key(m) != baseline_key(old)


def test_semfilter_v2_is_the_v070_command(old_init):
    _, m, *_ = old_init
    assert "v071" not in m and "semfilter" not in m["fixed"]
    env = m["live"]["environment"]
    assert env["TRIMEM_SEMFILTER_TOPN"] == "32" and env["TRIMEM_SEMFILTER_JUDGE_INPUT"] == "raw"
    assert v2c.ENV_GATE not in env and r71.semfilter_mode(m) == "semfilter-v2"
    with r71.v071_patches(m):  # no-op on a plain v0.7.0 manifest
        assert driver.run_round.__name__ == "run_round" and driver.run_round.__module__.endswith("plan_search")


def test_v2c_arm_from_v2c_init(new_init, tmp):
    shared, sm, ck, aff = new_init
    run = tmp / "new_arm"
    m = prepare_arm(shared, run, ck, aff)
    tls.validate_latent_manifest(m)  # stock: q == 4
    assert m["q"] == 4 and m["v071"]["stall_rule"] == RULE and m["v071"]["init_protocol"] is None
    assert m["space"] == sm["space"] and m["fixed"]["semfilter"] == v2c.BLOCK
    assert baseline_entry(run, "TRAIN")["complete"]


def test_midrun_arm_requires_the_flag(old_init, tmp):
    shared, _, ck, aff = old_init
    with pytest.raises(ValueError, match="--old-protocol-init"):
        prepare_arm(shared, tmp / "refused", ck, aff)
    assert not (tmp / "refused" / "prepared.json").exists()


def test_midrun_arm_records_the_exemption(old_init, tmp):
    shared, sm, ck, aff = old_init
    run = tmp / "midrun_arm"
    m = prepare_arm(shared, run, ck, aff, "--old-protocol-init")
    tls.validate_latent_manifest(m)
    r71.validate_v071_manifest(m)
    validate_manifest_v070(m)
    ex = m["v071"]["init_protocol"]
    assert ex["protocol"] == "semfilter-v2" and ex["top_n"] == 32 and ex["judge_input"] == "raw" and ex["reuse"] == "off"
    assert ex["source_run"] == str(shared) and ex["exploration_protocol"] == "semfilter-v2c"
    assert m["space"] == sm["space"]  # same keys and samples as the init
    assert m["live"]["environment"]["TRIMEM_SEMFILTER_TOPN"] == "16" and m["q"] == 4
    assert m["fixed"]["semfilter"] == v2c.BLOCK and "semfilter" not in sm["fixed"]
    with SampledLedger(run / "ledger.sqlite", m) as led:
        done = led.complete_configs()
    assert len(done) == 10
    tags = {r["semfilter_protocol"] for e in done for r in e["receipts"]}
    assert tags == {"semfilter-v2"}
    assert all(r["execution_fingerprint"] == m["space"]["execution_fingerprint"] for e in done for r in e["receipts"])
    # the unchanged validators accept them against the arm manifest
    with v070_patches(m):
        space = driver.get_space(m)
        for e in done:
            for r in e["receipts"]:
                driver.check_receipt(r, {"key": e["key"], "task_id": r["task_id"]}, m, space)
    nm = baseline_entry(run, "TRAIN")
    assert nm["complete"] and nm["key"] == baseline_key(sm)
    table = r71.protocol_table(run, m)
    assert all(v["source"] == "init" and v["protocols"] == {"semfilter-v2": 100} for v in table.values())


def test_midrun_exemption_refuses_other_differences(old_init):
    _, sm, *_ = old_init
    m = json.loads(json.dumps(sm))
    m["fixed"]["semfilter"] = dict(v2c.BLOCK)
    assert p71.init_exemption(sm, m, "src")["protocol"] == "semfilter-v2"
    m["fixed"]["model"] = "another-model"
    with pytest.raises(ValueError, match="beyond semfilter"):
        p71.init_exemption(sm, m, "src")


def test_midrun_import_checks_receipts_with_the_stock_validator(old_init, tmp):
    shared, sm, *_ = old_init
    with v070_patches(sm):
        space = driver.get_space(sm)
    with p71._midrun_import_patches(), SampledLedger(shared / "ledger.sqlite", sm) as led:
        e = led.complete_configs()[0]
        bad = dict(e["receipts"][0], split="TEST")
        with pytest.raises(ValueError, match="split"):
            SampledLedger.import_reserved(led, space, e["config"], [bad])
    assert SampledLedger.import_reserved.__qualname__ == "SampledLedger.import_reserved"  # restored


def test_v2c_init_refuses_the_midrun_flag(new_init, tmp):
    shared, _, ck, aff = new_init
    with pytest.raises(ValueError, match="already runs semfilter-v2c"):
        prepare_arm(shared, tmp / "refused2", ck, aff, "--old-protocol-init")


def test_options():
    assert p71.split_options(["arm", "--semfilter", "v2", "--stall-rule", "off", "--x"]) == (
        ["arm", "--x"], {"semfilter": "v2", "stall": False, "old_protocol_init": False})
    with pytest.raises(SystemExit):
        p71.split_options(["--semfilter", "v9"])


def test_hash_guard_ignores_v071_modules_hashed_by_v071_runs(monkeypatch):
    """Review 2026-11-05: a started v0.7.1 run hashes v0.7.1 modules; that must not read as a leak into older runs."""
    from tools import trace_v071_hash_guard as g

    monkeypatch.setattr(g, "started_v070_runs", lambda: ["old", "new"])
    monkeypatch.setattr(g.hg, "report", lambda: {"ok": True, "violations": [], "protected_runs": []})
    monkeypatch.setattr(g.hg, "protected", lambda: [])
    monkeypatch.setattr(g.hg, "present", lambda runs: list(runs))
    monkeypatch.setattr(g.hg, "mismatches", lambda run: [])
    monkeypatch.setattr(g.hg, "_manifest", lambda run: {"v071": {}} if run == "new" else {})
    scanned = []
    monkeypatch.setattr(g, "v071_importers", lambda runs: scanned.extend(runs) or [])
    r = g.report()
    assert scanned == ["old"] and r["ok"]
