"""Prepare wrapper for the shared 10-config init and the latent-mobo-v1 / explicit-pr arms (Task 9)."""

import json
from pathlib import Path

import pytest

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_latent_campaign as pl
from tests.latent_fixtures import (
    enumerate_space,
    make_shared_init,
    tiny_model,
    tiny_shared_manifest,
    tiny_space,
)
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.latent_plan_codec import file_sha256, fit_affine, save_affine
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint
from trimem.optimizer.plan_schema import digest


def arm_manifest(space=None, *, protocol="arm-fixture"):
    m = tiny_shared_manifest(space, protocol=protocol)
    m.update(new_configs=40, budget=50, initialization=None, selection="channel-quota-v1")
    m["holdout"] = {**m["holdout"], "selection": "train_pareto"}
    return m


def latent_files(tmp_path, space):
    model = tiny_model(4)
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, model, space, {})
    aff = fit_affine(model, SeqGrammar(space), enumerate_space(space)[:60])
    ap = tmp_path / "affine.json"
    save_affine(ap, aff, {"checkpoint_sha256": file_sha256(ck)})
    return ck, ap


@pytest.fixture
def arm_env(monkeypatch):
    # these tests cover the arm without staged pruning (pruning: tests/test_trace_latent_pruning.py)
    for k, v in (("backend", None), ("latent", None), ("pruning", False), ("import_policy", "init")):
        monkeypatch.setitem(pl.ARM, k, v)


def run_arm(tmp_path, src, backend, manifest=None, latent=None):
    space = tiny_space()
    pl.ARM["backend"] = backend
    pl.ARM["latent"] = latent
    out = tmp_path / f"arm_{backend}"
    out.mkdir(parents=True)
    with pl.arm_patches():
        return pl.arm_import_archive(out, manifest or arm_manifest(space), src), out


@pytest.fixture(scope="module")
def src_fwd(tmp_path_factory):
    return make_shared_init(tmp_path_factory.mktemp("fwd"), forward=True)


def test_shared_init_manifest(tmp_path):
    out = tmp_path / "s"
    m = pl.shared_prepare_run(out, tiny_shared_manifest())
    driver.validate_manifest(m)
    assert (m["budget"], m["initial_configs"], m["new_configs"]) == (10, 10, 0)
    assert m["initialization"] == {"method": "sobol_then_llm", "sobol": 5, "llm": 5}
    assert m["holdout"]["selection"] == "deferred" and "pruning" not in m
    assert m["live"]["whole_card_cap"] == "none" and m["selection"] == "greedy"
    assert m["search_backend"] == "shared-init-v1"
    assert m["train_panel_policy"] == "stratified-test-type-mix-v1"
    arts = m["live"]["artifacts"]
    assert str(pl.TYPES.resolve()) in arts
    assert str(Path(pl.STRATIFIED_MODULE).resolve()) in arts


def test_shared_init_rejects_pruning(tmp_path):
    with pytest.raises(ValueError, match="pruning"):
        pl.shared_prepare_run(tmp_path / "s", {**tiny_shared_manifest(), "pruning": {}})


def test_arm_import_keeps_forward(tmp_path, src_fwd, arm_env):
    src_m = json.loads((src_fwd / "manifest.json").read_text())
    space = tiny_space()
    with SearchLedger(src_fwd / "ledger.sqlite", src_m) as led:
        fwd_keys = {e["key"] for e in led.complete_configs() if e["config"]["graph_direction"] == "Forward"}
    assert fwd_keys
    assert base.legacy_to_v060({"graph_direction": "Forward"})["graph_direction"] == "Both"  # stock behaviour
    frozen, out = run_arm(tmp_path, src_fwd, "explicit-pr")
    with SearchLedger(out / "ledger.sqlite", frozen) as led:
        got = led.complete_configs()
    assert {e["key"] for e in got} >= fwd_keys
    assert sum(e["config"]["graph_direction"] == "Forward" for e in got) >= len(fwd_keys)
    assert all(space.key(e["config"]) == e["key"] for e in got)


def test_arm_has_10_imported_and_budget_50(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    frozen, out = run_arm(tmp_path, src, "explicit-pr")
    with SearchLedger(out / "ledger.sqlite", frozen) as led:
        assert len(led.complete_configs()) == 10
    assert (frozen["budget"], frozen["new_configs"], frozen["initial_configs"], frozen["q"]) == (50, 40, 10, 4)
    assert frozen["holdout"]["selection"] == "train_pareto"
    assert frozen["live"]["whole_card_cap"] == "none" and frozen["selection"] == "greedy"
    assert "pruning" not in frozen and len(frozen["initial_keys"]) == 10
    driver.validate_manifest(json.loads((out / "manifest.json").read_text()))


def test_latent_arm_passes_validators(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    space = tiny_space()
    ck, ap = latent_files(tmp_path, space)
    block = pl.latent_block(ck, ap, 4, kernel="matern", structure="coupled-cost")
    _, out = run_arm(tmp_path, src, "latent-mobo-v1", latent=block)
    m = json.loads((out / "manifest.json").read_text())
    driver.validate_manifest(m)
    tls.validate_latent_manifest(m)
    lat = m["latent"]
    assert lat["checkpoint_sha256"] == file_sha256(ck) and lat["affine_sha256"] == file_sha256(ap)
    assert lat["surrogate"] == {"kernel": "matern", "structure": "coupled-cost",
                                "nuts": [512, 256, 16], "nuts_cost": None}
    assert lat["proposal_budget"] == tls.DEFAULT_BUDGET and (lat["anchors"], lat["h0"]) == (16, 0.4)
    assert lat["decode"] == tls.DECODE and lat["d"] == 4
    arts = m["live"]["artifacts"]
    for p in (ck, ap, pl.ROOT / "trimem/experiments/trace_latent_search.py",
              pl.ROOT / "tools/prepare_trace_latent_campaign.py",
              pl.ROOT / "trimem/optimizer/latent_plan_codec.py"):
        assert str(Path(p).resolve()) in arts
    assert m["search_backend"] == "latent-mobo-v1"


def test_arm_rejects_space_or_panel_mismatch(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    space = tiny_space()
    bad_space = arm_manifest(space)
    bad_space["space"] = {**bad_space["space"], "execution_fingerprint": "other"}
    assert digest(bad_space["space"]) != digest(space.manifest())
    with pytest.raises(ValueError, match="space"):
        run_arm(tmp_path / "a", src, "explicit-pr", manifest=bad_space)
    bad_tasks = arm_manifest(space)
    bad_tasks["tasks"] = bad_tasks["tasks"][:-1] + ["tX"]
    with pytest.raises(ValueError, match="(panel|tasks)"):
        run_arm(tmp_path / "b", src, "explicit-pr", manifest=bad_tasks)


def test_explicit_pr_arm_has_no_latent_block(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    frozen, out = run_arm(tmp_path, src, "explicit-pr", latent={"stale": 1})
    assert "latent" not in frozen and frozen["search_backend"] == "explicit-pr"
    assert "latent" not in json.loads((out / "manifest.json").read_text())


def test_arm_without_pruning_drops_stale_block_and_rejects_unknown_backend(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    frozen, _ = run_arm(tmp_path / "a", src, "explicit-pr", manifest={**arm_manifest(), "pruning": {}})
    assert "pruning" not in frozen  # --pruning none: the arm builds no block and keeps none
    with pytest.raises(ValueError, match="backend"):
        run_arm(tmp_path / "b", src, "nope")


def test_arm_patches_restore_base():
    before = (base.INITIAL, base.NEW, base.make_panels, base.legacy_to_v060, base.import_archive)
    prep_before = base.prepare_run
    with pl.arm_patches():
        assert (base.INITIAL, base.NEW) == (10, 40) and base.legacy_to_v060({"a": 1}) == {"a": 1}
    assert (base.INITIAL, base.NEW, base.make_panels, base.legacy_to_v060, base.import_archive) == before
    with pl.shared_patches():
        assert (base.INITIAL, base.NEW) == (10, 0) and base.prepare_run is pl.shared_prepare_run
    assert base.prepare_run is prep_before


# ---- live wiring (goal Step 3): saas-numpyro is the live default; cpus; artifacts
def test_latent_block_saas_numpyro_writes_cpus(tmp_path):
    ck, ap = latent_files(tmp_path, tiny_space())
    block = pl.latent_block(ck, ap, 4, kernel="saas-numpyro", cpus=8)
    assert block["surrogate"] == {"kernel": "saas-numpyro", "structure": "independent",
                                  "nuts": [512, 256, 16], "nuts_cost": None, "cpus": 8,
                                  "cpu_offset": 0, "warm_warmup": 128}
    assert pl.latent_block(ck, ap, 4, kernel="saas-numpyro")["surrogate"]["cpus"] == 12
    with pytest.raises(ValueError, match="independent"):
        pl.latent_block(ck, ap, 4, kernel="saas-numpyro", structure="coupled-cost")


def test_latent_arm_saas_numpyro_artifacts(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    ck, ap = latent_files(tmp_path, tiny_space())
    _, out = run_arm(tmp_path, src, "latent-mobo-v1", latent=pl.latent_block(ck, ap, 4, kernel="saas-numpyro", cpus=6))
    m = json.loads((out / "manifest.json").read_text())
    tls.validate_latent_manifest(m)
    assert m["latent"]["surrogate"]["kernel"] == "saas-numpyro" and m["latent"]["surrogate"]["cpus"] == 6
    arts = m["live"]["artifacts"]
    for name in ("latent_numpyro_saas.py", "latent_numpyro_warm.py", "latent_seq_vae_v2.py",
                 "latent_plan_surrogate.py"):
        assert str((pl.ROOT / "trimem/optimizer" / name).resolve()) in arts
    assert str(ck.resolve()) in arts and str(ap.resolve()) in arts


def test_cli_kernel_and_cpus_options():
    p = pl.build_parser()
    common = ["arm", "--dataset", sorted(base.PROFILES)[0], "--template", "t", "--run-dir", "r",
              "--backend", "latent-mobo-v1", "--shared-init", "s"]
    a = p.parse_args(common)
    assert (a.kernel, a.structure, a.cpus, a.cpu_offset) == ("matern", "independent", 12, 0)
    a = p.parse_args([*common, "--kernel", "saas-numpyro", "--cpus", "4", "--cpu-offset", "8"])
    assert (a.kernel, a.cpus, a.cpu_offset) == ("saas-numpyro", 4, 8)
    with pytest.raises(SystemExit):
        p.parse_args([*common, "--kernel", "map-saas"])


def test_latent_block_defaults_matern_16_anchors(tmp_path):
    ck, ap = latent_files(tmp_path, tiny_space())
    block = pl.latent_block(ck, ap, 4)  # live default (user 2026-11-04): Matern x 3, 16 anchors
    assert block["anchors"] == 16
    assert block["surrogate"] == {"kernel": "matern", "structure": "independent",
                                  "nuts": [512, 256, 16], "nuts_cost": None}


# ---- pre-launch fix round: frozen R6 affine outside smoke; trace_latent_live.py in artifacts
def test_check_frozen_affine(tmp_path):
    from trimem.optimizer.latent_plan_codec import load_affine

    ck, ap = latent_files(tmp_path, tiny_space())
    aff, meta = load_affine(ap)
    shared = tmp_path / "shared"
    shared.mkdir()
    good = tmp_path / "frozen.json"
    save_affine(good, aff, {**meta, "kind": "frozen-train-init-v1", "shared_init": str(shared)})
    pl.check_frozen_affine(good, shared)
    with pytest.raises(ValueError, match="frozen-train-init-v1"):
        pl.check_frozen_affine(ap, shared)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="shared_init"):
        pl.check_frozen_affine(good, other)


def test_arm_outside_smoke_refuses_provisional_affine(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    ck, ap = latent_files(tmp_path, tiny_space())
    man = {**arm_manifest(tiny_space()), "smoke": False}
    with pytest.raises(ValueError, match="frozen-train-init-v1"):
        run_arm(tmp_path, src, "latent-mobo-v1", manifest=man, latent=pl.latent_block(ck, ap, 4))


def test_live_driver_in_artifacts(tmp_path, arm_env):
    src = make_shared_init(tmp_path)
    ck, ap = latent_files(tmp_path, tiny_space())
    _, out = run_arm(tmp_path, src, "latent-mobo-v1", latent=pl.latent_block(ck, ap, 4))
    arts = json.loads((out / "manifest.json").read_text())["live"]["artifacts"]
    assert str((pl.ROOT / "tools/trace_latent_live.py").resolve()) in arts
