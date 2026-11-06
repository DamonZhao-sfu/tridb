"""TRACE-latent staged pruning (user 2026-11-04): pruning block, validator, partial observations in
the latent GP, staged batch sizes, TEST selection without pruned configs, resume import policy."""

import json
from pathlib import Path

import pytest

import tools.prepare_trace_latent_campaign as pl
from tests.latent_fixtures import (
    SMOKE_TASKS,
    enumerate_space,
    make_shared_init,
    synthetic_receipt,
    tiny_model,
    tiny_shared_manifest,
    tiny_space,
)
from tools import trace_joint_protocol as proto
from tools import trace_latent_live as tll
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.optimizer import staged_pruning as sp
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.latent_plan_codec import file_sha256, fit_affine, save_affine
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint


# ------------------------------------------------------------------ helpers
def write_panel(tmp_path, tasks=SMOKE_TASKS, *, stratum=True):
    items = [{"item_id": t, **({"stratum": "even" if i % 2 == 0 else "odd"} if stratum else {})}
             for i, t in enumerate(tasks)]
    path = Path(tmp_path) / "train_panel.json"
    path.write_text(json.dumps({"items": items}))
    return path


def arm_manifest(tmp_path, *, budget=50, protocol="arm-fixture", stratum=True):
    m = tiny_shared_manifest(tiny_space(), protocol=protocol)
    m.update(new_configs=budget - 10, budget=budget, initialization=None, selection="channel-quota-v1")
    m["holdout"] = {**m["holdout"], "selection": "train_pareto"}
    m["live"] = {"artifacts": {}, "panel_path": str(write_panel(tmp_path, stratum=stratum))}
    return m


def latent_files(tmp_path, *, shared=None):
    space = tiny_space()
    model = tiny_model(4)
    ck = Path(tmp_path) / "ck.pt"
    save_checkpoint(ck, model, space, {})
    ap = Path(tmp_path) / "affine.json"
    meta = {"checkpoint_sha256": file_sha256(ck)}
    if shared is not None:
        meta.update(kind="frozen-train-init-v1", shared_init=str(shared))
    save_affine(ap, fit_affine(model, SeqGrammar(space), enumerate_space(space)[:60]), meta)
    return ck, ap


@pytest.fixture
def arm_env(monkeypatch):
    for k, v in (("backend", None), ("latent", None), ("panel", "stratified"), ("pruning", True),
                 ("stage_step", 5), ("import_policy", "init")):
        monkeypatch.setitem(pl.ARM, k, v)


def run_arm(tmp_path, src, backend, *, latent=None, manifest=None, name=None, **arm):
    pl.ARM.update(backend=backend, latent=latent, **{"pruning": True, "import_policy": "init", **arm})
    out = Path(tmp_path) / (name or f"arm_{backend}")
    out.mkdir(parents=True)
    man = manifest or arm_manifest(tmp_path)
    with pl.arm_patches():
        frozen = pl.arm_import_archive(out, man, src)
    return frozen, out


def bad_evaluator(bad_key):
    space = tiny_space()

    def evaluate(config, task, m):
        r = synthetic_receipt(config, task, m)
        if space.key(config) == bad_key:
            r = {**r, "raw_score": 0, "total_tokens": 10**6, "latency_seconds": 1000}
        return r

    return evaluate


def keyed(config, task, m):
    """synthetic receipt with the config key (real receipts carry it; the stock importer restates it)."""
    return {**synthetic_receipt(config, task, m), "config_key": tiny_space().key(config)}


def freeze(run):
    m = json.loads((run / "manifest.json").read_text())
    space = driver.get_space(m)
    with SearchLedger(run / "ledger.sqlite", m) as led:
        driver.freeze_objective(run, m, driver.observations(led, space, m), space)
    return m, space


def reserve(run, m, space, configs):
    with SearchLedger(run / "ledger.sqlite", m) as led:
        led.reserve(space, configs)


def full_plan_config(run, m, space):
    """A new (unmeasured) config whose plan (filter_order) is covered by a measured config."""
    with SearchLedger(run / "ledger.sqlite", m) as led:
        seen = {e["key"] for e in led.complete_configs()}
        orders = {e["config"]["filter_order"] for e in led.complete_configs()}
    return next(c for c in enumerate_space(space)
                if space.key(c) not in seen and c["filter_order"] in orders and c["rank_impl"] == "ColBERT")


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return make_shared_init(tmp_path_factory.mktemp("shared"))


# ------------------------------------------------------------------ 1. pruning block at prepare
def test_stage_sizes_and_quality_scale():
    assert pl.stage_sizes(100, 20) == [20, 40, 60, 80, 100]
    assert pl.stage_sizes(10, 5) == [5, 10]
    assert pl.stage_sizes(10, 20) == [10]
    assert pl.quality_scale("sciworld") == 100
    assert pl.quality_scale("alfworld") == 1 and pl.quality_scale("webshop") == 1


def test_arm_default_has_stratified_pruning_block(tmp_path, shared, arm_env):
    frozen, out = run_arm(tmp_path, shared, "explicit-pr")
    p = frozen["pruning"]
    tasks = [str(t) for t in frozen["tasks"]]
    stratum = {t: ("even" if i % 2 == 0 else "odd") for i, t in enumerate(SMOKE_TASKS)}
    assert p["protocol"] == sp.PROTOCOL and p["z"] == sp.Z
    assert p["sizes"] == [5, 10]
    assert p["order"] == sp.stage_order(tasks, stratum, sp.seed_of(frozen["protocol"], "stages"))
    assert p["budget_episodes"] == 50 * len(tasks)  # the unpruned plan's episodes are kept
    assert p["quality_scale"] == 1 and p["quality_se_floor"] == "agresti-coull"
    assert p["protect"] == "initialization configurations (5 Sobol + 5 LLM)"
    assert (frozen["budget"], frozen["new_configs"], frozen["initial_configs"]) == (80, 70, 10)
    assert str((pl.ROOT / "trimem/optimizer/staged_pruning.py").resolve()) in frozen["live"]["artifacts"]
    driver.validate_manifest(json.loads((out / "manifest.json").read_text()))


def test_arm_pruning_none_keeps_old_shape(tmp_path, shared, arm_env):
    frozen, _ = run_arm(tmp_path, shared, "explicit-pr", pruning=False)
    assert "pruning" not in frozen and frozen["budget"] == 50


def test_arm_pruning_requires_panel_strata(tmp_path, shared, arm_env):
    with pytest.raises(ValueError, match="stratum"):
        run_arm(tmp_path, shared, "explicit-pr", manifest=arm_manifest(tmp_path, stratum=False))


def test_shared_init_still_refuses_pruning(tmp_path):
    with pytest.raises(ValueError, match="pruning"):
        pl.shared_prepare_run(tmp_path / "s", {**tiny_shared_manifest(), "pruning": {}})


def test_cli_pruning_and_import_policy_options():
    p = pl.build_parser()
    common = ["arm", "--dataset", "alfworld", "--template", "t", "--run-dir", "r",
              "--backend", "latent-mobo-v1", "--shared-init", "s"]
    a = p.parse_args(common)
    assert (a.pruning, a.import_policy) == ("staged", "init")  # pruning is the arm default
    a = p.parse_args([*common, "--pruning", "none", "--import-policy", "complete"])
    assert (a.pruning, a.import_policy) == ("none", "complete")


# ------------------------------------------------------------------ 2. latent validator
def live_pruned(m):
    """A non-smoke latent manifest in the live staged-pruning shape."""
    tasks = [f"x{i}" for i in range(100)]
    return {**m, "smoke": False, "initial_configs": 10, "budget": 80, "new_configs": 70,
            "fixed": {"dataset": "alfworld"},
            "latent": {**m["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET)},
            "pruning": {"protocol": sp.PROTOCOL, "z": sp.Z, "sizes": [20, 40, 60, 80, 100],
                        "order": tasks, "budget_episodes": 5000, "quality_scale": 1,
                        "quality_se_floor": "agresti-coull"}}


def test_validator_requires_live_pruning(tmp_path):
    from tests.test_trace_latent_worker import make_manifest

    ok = make_manifest(tmp_path, tiny_space())
    tls.validate_latent_manifest(ok)  # smoke without pruning: allowed
    live = live_pruned(ok)
    tls.validate_latent_manifest(live)
    with pytest.raises(ValueError, match="pruning"):
        tls.validate_latent_manifest({k: v for k, v in live.items() if k != "pruning"})
    for bad in ({"budget_episodes": 4000}, {"sizes": [50, 100]}, {"quality_scale": 100},
                {"quality_se_floor": None}, {"protocol": "staged-pareto-prune-v1"}, {"z": 1.0}):
        with pytest.raises(ValueError, match="pruning"):
            tls.validate_latent_manifest({**live, "pruning": {**live["pruning"], **bad}})
    for bad in ({"budget": 50, "new_configs": 40}, {"new_configs": 60}):
        with pytest.raises(ValueError, match="budget"):
            tls.validate_latent_manifest({**live, **bad})
    sci = {**live, "fixed": {"dataset": "sciworld"}, "pruning": {**live["pruning"], "quality_scale": 100}}
    tls.validate_latent_manifest(sci)
    smoke_pruned = {**ok, "pruning": {**live["pruning"], "sizes": [5, 10], "budget_episodes": 120}}
    tls.validate_latent_manifest(smoke_pruned)  # smoke: any stage sizes / episode budget


# ------------------------------------------------------------------ 3. worker: partial obs + batch sizes
@pytest.fixture
def latent_arm(tmp_path, shared, arm_env):
    ck, ap = latent_files(tmp_path)
    block = pl.latent_block(ck, ap, 4)
    block["proposal_budget"] = {"seconds": 20, "calls": 2400}

    def make(budget=50):
        frozen, run = run_arm(tmp_path, shared, tls.BACKEND, latent=block,
                              manifest=arm_manifest(tmp_path, budget=budget), name=f"arm{budget}")
        return run

    return make


def test_pruned_config_enters_latent_gp(latent_arm):
    run = latent_arm()
    m, space = freeze(run)
    bad = full_plan_config(run, m, space)
    reserve(run, m, space, [bad])
    status = driver.run_round(run, evaluator=bad_evaluator(space.key(bad)))  # staged evaluation
    assert status["status"] == "batch_complete"
    log = [json.loads(x) for x in (run / "pruning_log.jsonl").read_text().splitlines()]
    assert any(r["key"] == space.key(bad) and r["prune"] and r["stage"] == 5 for r in log)
    with SearchLedger(run / "ledger.sqlite", m) as led:
        entry = next(e for e in led.complete_configs() if e["key"] == space.key(bad))
    assert entry["pruned"] and len(entry["receipts"]) == 5
    with tls.install():
        status = driver.run_round(run, evaluator=synthetic_receipt)
    body = json.loads((run / "round_011.json").read_text())
    assert space.key(bad) in body["archive_keys"] and len(body["archive_keys"]) == 11
    assert len(body["selected"]) == 4 and status["complete_configs"] == 15


def test_staged_batch_size_below_q_and_budget_complete(latent_arm):
    run = latent_arm(budget=12)  # 120 episodes: init 100 -> room for 2 full configurations
    m = json.loads((run / "manifest.json").read_text())
    with tls.install():
        status = driver.run_round(run, evaluator=synthetic_receipt)
        assert len(json.loads((run / "round_010.json").read_text())["selected"]) == 2
        assert status["complete_configs"] == 12
        with SearchLedger(run / "ledger.sqlite", m) as led:
            used = led.db.execute("SELECT count(*) FROM episodes WHERE state='done'").fetchone()[0]
        if used > 110:  # both picks ran in full: no configuration fits any more
            assert driver.run_round(run, evaluator=synthetic_receipt)["status"] == "budget_complete"


def test_partially_evaluated_config_is_pending_not_measured(latent_arm, monkeypatch):
    run = latent_arm()
    m, space = freeze(run)
    with tls.install():
        reserved = driver.run_round(run)  # reserve only
    key = reserved["proposal"]["selected"][0]["key"]
    cfg = reserved["proposal"]["selected"][0]["config"]
    with SearchLedger(run / "ledger.sqlite", m) as led:
        for t in m["pruning"]["order"][:5]:
            led.claim(key, t)
            led.finish(key, t, synthetic_receipt(cfg, t, m))
    spawned = []
    monkeypatch.setattr(driver, "guarded_search", lambda *a, **k: spawned.append(1))
    with tls.install():
        again = driver.run_round(run)
    assert spawned == [] and again["status"] == "reserved"  # no proposal while a stage is open
    with SearchLedger(run / "ledger.sqlite", m) as led:
        assert key not in {e["key"] for e in led.complete_configs()}
        assert key in {p["key"] for p in led.pending()}
        assert len(driver.observations(led, space, m)) == 10


# ------------------------------------------------------------------ 4. TEST selection excludes pruned
def test_holdout_excludes_pruned_configs(tmp_path, shared, arm_env):
    frozen, run = run_arm(tmp_path, shared, "explicit-pr", manifest=arm_manifest(tmp_path, budget=11))
    m, space = freeze(run)
    great = full_plan_config(run, m, space)
    gkey = space.key(great)
    reserve(run, m, space, [great])
    with SearchLedger(run / "ledger.sqlite", m) as led:
        for t in m["pruning"]["order"][:5]:
            led.claim(gkey, t)
            led.finish(gkey, t, {**synthetic_receipt(great, t, m), "raw_score": 1000,
                                 "total_tokens": 1, "latency_seconds": 1})
        led.prune_many({gkey: "manual prune (test)"})
        assert driver.staged_batch_size(led, m) == 0 and not led.pending()
    with pytest.raises(ValueError):  # the stock selection needs budget (80) complete configurations
        proto.run_holdout(run, evaluator=synthetic_receipt)
    summary = tll.run_pruned_holdout(run, evaluator=synthetic_receipt)
    sel = json.loads((run / "holdout" / "selection.json").read_text())
    assert gkey not in {space.key(c) for c in sel["configs"]}
    assert sel["excluded_pruned"] == [gkey] and gkey in sel["train_keys"]
    assert len(sel["candidate_keys"]) == 10 and summary["status"] == "test_complete"
    assert summary["configurations"] == len(sel["configs"]) >= 1


def test_pruned_holdout_requires_budget_complete(tmp_path, shared, arm_env):
    _, run = run_arm(tmp_path, shared, "explicit-pr")
    with pytest.raises(ValueError, match="TRAIN"):
        tll.run_pruned_holdout(run, evaluator=synthetic_receipt)


def test_backend_patches_swap_holdout_for_pruned_train_pareto():
    original = proto.run_holdout
    m = {"search_backend": tls.BACKEND, "pruning": {"protocol": sp.PROTOCOL},
         "holdout": {"selection": "train_pareto"}}
    with tll.backend_patches(m):
        assert proto.run_holdout is tll.run_pruned_holdout
    assert proto.run_holdout is original
    for other in ({**m, "pruning": None}, {k: v for k, v in m.items() if k != "pruning"},
                  {**m, "holdout": {"selection": "deferred"}}):
        with tll.backend_patches(other):
            assert proto.run_holdout is original
    with tll.backend_patches({**m, "search_backend": "explicit-pr"}):
        assert proto.run_holdout is tll.run_pruned_holdout


# ------------------------------------------------------------------ 5. resume import from a latent arm
@pytest.fixture
def stopped_latent_arm(tmp_path, shared, arm_env):
    """A latent arm without pruning (like the stopped ALF arm): 10 init + 4 complete round-1
    configurations + a partially evaluated round-2 batch."""
    ck, ap = latent_files(tmp_path)
    block = pl.latent_block(ck, ap, 4)
    block["proposal_budget"] = {"seconds": 20, "calls": 2400}
    _, src = run_arm(tmp_path, shared, tls.BACKEND, latent=block, pruning=False, name="stopped")
    with tls.install():
        driver.run_round(src, evaluator=keyed)  # round_010: 4 complete
        r2 = driver.run_round(src)  # round_014 reserved
    m = json.loads((src / "manifest.json").read_text())
    key, cfg = r2["proposal"]["selected"][0]["key"], r2["proposal"]["selected"][0]["config"]
    with SearchLedger(src / "ledger.sqlite", m) as led:
        for t in m["tasks"][:3]:
            led.claim(key, t)
            led.finish(key, t, keyed(cfg, t, m))
    return src, block, key


def test_import_policy_complete_resumes_latent_arm(tmp_path, stopped_latent_arm):
    src, block, partial_key = stopped_latent_arm
    src_m = json.loads((src / "manifest.json").read_text())
    with SearchLedger(src / "ledger.sqlite", src_m) as led:
        src_complete = [e["key"] for e in led.complete_configs()]
    assert len(src_complete) == 14
    frozen, run = run_arm(tmp_path, src, tls.BACKEND, latent=block, import_policy="complete",
                          name="resumed")
    with SearchLedger(run / "ledger.sqlite", frozen) as led:
        got = [e["key"] for e in led.complete_configs()]
        assert not led.pending()
    assert sorted(got) == sorted(src_complete) and partial_key not in got
    assert sorted(frozen["initial_keys"]) == sorted(src_m["initial_keys"])  # only init is protected
    assert frozen["import_policy"] == "complete"
    assert frozen["imported_rounds"]["rounds"] == ["round_010"]
    assert (run / "round_010.json").read_bytes() == (src / "round_010.json").read_bytes()
    assert (run / "round_010.pt").exists() and not (run / "round_014.json").exists()
    assert "pruning" in frozen and frozen["budget"] == 80
    m, space = freeze(run)
    assert json.loads((run / "objective.json").read_text()) == json.loads((src / "objective.json").read_text())
    with tls.install():
        status = driver.run_round(run, evaluator=synthetic_receipt)
    body = json.loads((run / "round_014.json").read_text())
    assert len(body["trust_gains"]) == 1  # the imported round_010 batch is replayed by the trust state
    assert status["status"] == "batch_complete" and status["complete_configs"] == 18


def test_import_policy_init_is_default(tmp_path, stopped_latent_arm):
    src, block, _ = stopped_latent_arm
    frozen, run = run_arm(tmp_path, src, tls.BACKEND, latent=block, name="init_only")
    with SearchLedger(run / "ledger.sqlite", frozen) as led:
        assert len(led.complete_configs()) == 10
    assert frozen["import_policy"] == "init" and not (run / "round_010.json").exists()


def test_check_frozen_affine_accepts_latent_source_arm(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    ck, ap = latent_files(tmp_path, shared=shared)
    pl.check_frozen_affine(ap, shared)
    src = tmp_path / "stopped"
    src.mkdir()

    def write(**over):
        lat = {"affine": str(ap), "affine_sha256": file_sha256(ap), **over.pop("latent", {})}
        body = {"search_backend": tls.BACKEND, "latent": lat,
                "imported_observations": {"source_run": str(shared)}, **over}
        (src / "manifest.json").write_text(json.dumps(body))

    write()
    pl.check_frozen_affine(ap, src)  # source arm froze this affine on the same shared init
    write(latent={"affine_sha256": "0" * 64})
    with pytest.raises(ValueError, match="shared_init"):
        pl.check_frozen_affine(ap, src)
    write(imported_observations={"source_run": str(tmp_path / "elsewhere")})
    with pytest.raises(ValueError, match="shared_init"):
        pl.check_frozen_affine(ap, src)
    write(search_backend="explicit-pr")
    with pytest.raises(ValueError, match="shared_init"):
        pl.check_frozen_affine(ap, src)
