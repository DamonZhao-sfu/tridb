"""Latent-MOBO search worker at the _search_worker seam (Task 7)."""

import json

import pytest

from tests.latent_fixtures import enumerate_space, synthetic_receipt, tiny_model, tiny_space
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.latent_plan_codec import fit_affine, file_sha256, save_affine
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint

TASKS = [f"t{i}" for i in range(10)]


def make_manifest(tmp_path, space, *, surrogate=None, proposal_budget=None):
    model = tiny_model(4)
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, model, space, {})
    sha = file_sha256(ck)
    cfgs = enumerate_space(space)
    aff = fit_affine(model, SeqGrammar(space), cfgs[:60])
    ap = tmp_path / "affine.json"
    save_affine(ap, aff, {"checkpoint_sha256": sha, "kind": "frozen-train-init-v1",
                          "shared_init": str(tmp_path / "shared")})
    return {
        "space": space.manifest(),
        "tasks": TASKS,
        "protocol": "latent-fixture",
        "smoke": True,
        "initial_configs": 10,
        "new_configs": 12,
        "budget": 22,
        "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5},
        "q": 4,
        "seed": 3,
        "task_concurrency": 2,
        "sources": driver.source_identity(),
        "search_backend": tls.BACKEND,
        "latent": {
            "surrogate": surrogate or {"kernel": "matern", "structure": "independent",
                                       "nuts": [512, 256, 16], "nuts_cost": None},
            "checkpoint": str(ck), "checkpoint_sha256": sha,
            "affine": str(ap), "affine_sha256": file_sha256(ap),
            "d": 4, "decode": tls.DECODE,
            # calls cover the shared scoring reserve (~1.6k on the tiny space) + refinement
            "proposal_budget": proposal_budget or {"seconds": 20, "calls": 2400},
            "anchors": 16, "h0": 0.4,
        },
    }


def live_of(ok):
    """Non-smoke copy in the live shape (user 2026-11-04 ~23:25: staged pruning, budget = 80 configs)."""
    from trimem.optimizer import staged_pruning as sp

    return {**ok, "smoke": False, "initial_configs": 10, "budget": 80, "new_configs": 70,
            "pruning": {"protocol": sp.PROTOCOL, "z": sp.Z, "sizes": [20, 40, 60, 80, 100],
                        "order": list(ok["tasks"]), "budget_episodes": 5000, "quality_scale": 1,
                        "quality_se_floor": "agresti-coull"}}


def init_run(tmp_path, monkeypatch, **manifest_kw):
    space = tiny_space()
    run = tmp_path / "run"
    m = driver.prepare_run(run, make_manifest(tmp_path, space, **manifest_kw))
    monkeypatch.setattr(
        driver, "initial_advice",
        lambda sp, obs, settings: (
            sp.sample(5, seed=22, avoid={sp.key(o.config) for o in obs}), {"status": "ok"}),
    )
    driver.run_round(run, evaluator=synthetic_receipt)
    r = driver.run_round(run, evaluator=synthetic_receipt)
    assert r["complete_configs"] == 10
    return run, m, space


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("latent")
    mp = pytest.MonkeyPatch()
    try:
        run, m, space = init_run(tmp, mp)
        with tls.install():
            statuses = [driver.run_round(run, evaluator=synthetic_receipt) for _ in range(4)]
    finally:
        mp.undo()
    return run, m, space, statuses


def test_rounds_reserve_and_finish(finished):
    run, m, space, statuses = finished
    assert [s.get("complete_configs") for s in statuses[:3]] == [14, 18, 22]
    assert [len(s["proposal"]["selected"]) for s in statuses[:3]] == [4, 4, 4]
    assert statuses[3]["status"] == "budget_complete"


def test_round_file_contract(finished):
    run, m, space, _ = finished
    for n in (10, 14, 18):
        body = json.loads((run / f"round_{n:03d}.json").read_text())
        with SearchLedger(run / "ledger.sqlite", m) as led:
            complete = {e["key"] for e in led.complete_configs()}
        assert len(body["selected"]) == 4 and set(body["archive_keys"]) <= complete
        assert len(body["archive_keys"]) == n
        for s in body["selected"]:
            assert s["sources"] and s["key"] == space.key(s["config"])
        assert (run / f"round_{n:03d}.pt").exists()
    import torch

    pt = torch.load(run / "round_018.pt", weights_only=False)
    assert pt["codec"] and pt["latent"] == m["latent"] and pt["round_index"] == 2
    assert pt["sources"] == m["sources"] and "gp_state" in pt and "trust" in pt
    phase = json.loads((run / "worker_phase.json").read_text())
    assert phase["phase"] == "search"


def test_worker_refuses_changed_checkpoint(tmp_path, monkeypatch):
    run, m, space = init_run(tmp_path, monkeypatch)
    ck = tmp_path / "ck.pt"
    data = bytearray(ck.read_bytes())
    data[len(data) // 2] ^= 0xFF
    ck.write_bytes(bytes(data))
    with tls.install():
        with pytest.raises(RuntimeError):
            driver.run_round(run, evaluator=synthetic_receipt)
    err = json.loads((run / "round_010.error.json").read_text())["error"]
    assert "sha256" in err and "checkpoint" in err


def test_worker_refuses_unknown_pruning_protocol(tmp_path):
    space = tiny_space()
    man = {**make_manifest(tmp_path, space), "objective": None,
           "pruning": {"protocol": "x"}}
    (tmp_path / "run").mkdir()
    with tls.install():
        with pytest.raises(RuntimeError):
            driver.guarded_search(tmp_path / "run", man, [], seed=0, round_file="round_010")
    err = json.loads((tmp_path / "run" / "round_010.error.json").read_text())["error"]
    assert "pruning" in err


def test_replay_after_crash_reuses_round_file(tmp_path, monkeypatch):
    run, m, space = init_run(tmp_path, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    objective = json.loads((run / "objective.json").read_text())
    with SearchLedger(run / "ledger.sqlite", manifest) as led:
        complete = led.complete_configs()
    with tls.install():
        driver.guarded_search(
            run, {**manifest, "objective": objective, "q": 4}, complete, seed=3,
            round_file="round_010",
        )
    body = json.loads((run / "round_010.json").read_text())
    with SearchLedger(run / "ledger.sqlite", manifest) as led:
        assert led.pending() == []  # round file exists, nothing reserved
    spawned = []
    monkeypatch.setattr(driver, "guarded_search", lambda *a, **k: spawned.append(1))
    driver.run_round(run)
    assert spawned == []
    with SearchLedger(run / "ledger.sqlite", manifest) as led:
        assert {p["key"] for p in led.pending()} == {s["key"] for s in body["selected"]}


def test_replay_reserves_same_keys(tmp_path, monkeypatch):
    run, m, space = init_run(tmp_path, monkeypatch)
    with tls.install():
        first = driver.run_round(run)  # reserve only
    keys = [s["key"] for s in first["proposal"]["selected"]]
    spawned = []
    monkeypatch.setattr(driver, "guarded_search", lambda *a, **k: spawned.append(1))
    again = driver.run_round(run)  # pending already reserved; nothing re-proposed
    assert spawned == [] and again["status"] == "reserved"
    with SearchLedger(run / "ledger.sqlite", m) as led:
        assert {p["key"] for p in led.pending()} == set(keys)


def test_install_restores_original():
    original = driver._search_worker
    with tls.install():
        assert driver._search_worker is tls.latent_search_worker
    assert driver._search_worker is original


def test_validate_latent_manifest_rules(tmp_path):
    space = tiny_space()
    ok = make_manifest(tmp_path, space)
    tls.validate_latent_manifest(ok)
    live = {**ok, "smoke": False}
    with pytest.raises(ValueError, match="pruning"):  # live arms require staged pruning
        tls.validate_latent_manifest(live)
    with pytest.raises(ValueError, match="budget"):
        tls.validate_latent_manifest({**live_of(ok), "budget": 50, "new_configs": 40})
    with pytest.raises(ValueError, match="q == 4"):
        tls.validate_latent_manifest({**ok, "q": 2})
    tls.validate_latent_manifest({**ok, "q": 2}, running=True)
    frozen = live_of(ok)
    frozen["latent"] = {**ok["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET)}
    tls.validate_latent_manifest(frozen)  # all frozen non-smoke fields satisfied
    no_sur = {**frozen, "latent": {k: v for k, v in frozen["latent"].items() if k != "surrogate"}}
    with pytest.raises(ValueError, match="surrogate"):
        tls.validate_latent_manifest(no_sur)
    sur = ok["latent"]["surrogate"]
    for bad in ({**sur, "kernel": "rbf"}, {**sur, "structure": "joint"}):
        m = {**frozen, "latent": {**frozen["latent"], "surrogate": bad}}
        with pytest.raises(ValueError, match="surrogate"):
            tls.validate_latent_manifest(m)
    with pytest.raises(ValueError, match="search_backend"):
        tls.validate_latent_manifest({**ok, "search_backend": "x"})


# ---- live wiring (goal Step 3): saas-numpyro surrogate, warm start, CPU pinning
NP_SURROGATE = {"kernel": "saas-numpyro", "structure": "independent", "nuts": [16, 8, 4],
                "nuts_cost": None, "cpus": 4, "cpu_offset": 0, "warm_warmup": 8}
NP_BUDGET = {"seconds": 3, "calls": 2000}


def _np_env(tmp, mp):
    """Spawned workers inherit os.environ: a temp compilation cache, no env pinning."""
    mp.setenv("TRIMEM_JAX_CACHE_DIR", str(tmp / "jaxcache"))
    mp.delenv("TRIMEM_JAX_CPUS", raising=False)


@pytest.fixture(scope="module")
def np_finished(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("latent_np")
    mp = pytest.MonkeyPatch()
    try:
        _np_env(tmp, mp)
        run, m, space = init_run(tmp, mp, surrogate=dict(NP_SURROGATE), proposal_budget=NP_BUDGET)
        with tls.install():
            statuses = [driver.run_round(run, evaluator=synthetic_receipt) for _ in range(4)]
    finally:
        mp.undo()
    return run, m, space, statuses, tmp


def _fit(run, n):
    return json.loads((run / f"round_{n:03d}.json").read_text())["surrogate"]


def test_numpyro_rounds_finish_with_three_saas_models(np_finished):
    import torch

    run, m, space, statuses, _ = np_finished
    assert [s.get("complete_configs") for s in statuses[:3]] == [14, 18, 22]
    assert statuses[3]["status"] == "budget_complete"
    state = torch.load(run / "round_014.pt", weights_only=False)["gp_state"]
    assert len({k.split(".")[1] for k in state if k.startswith("models.")}) == 3


def test_numpyro_warm_state_chain_cold_warm_cold(np_finished):
    run, m, _, _, _ = np_finished
    r10, r14, r18 = (_fit(run, n) for n in (10, 14, 18))
    assert r10["kernel"] == "saas-numpyro"
    # round_010: no previous round -> cold; round_014: warm from round_010 (14 <= 1.5 x 10);
    # round_018: aging rule (18 > 1.5 x cold_fit_n 10) -> cold
    assert (r10["fit"]["mode"], r10["fit"]["generation"]) == ("cold", 0)
    assert (r14["fit"]["mode"], r14["fit"]["generation"]) == ("warm", 1)
    assert (r18["fit"]["mode"], r18["fit"]["generation"]) == ("cold", 0)
    assert r10["fit"]["warm_load"]["load"] == "no previous round"
    assert r14["fit"]["warm_load"]["load"] == "loaded"
    assert r14["fit"]["warm_load"]["source"].endswith("latent_warm/round_010.json")
    assert r18["fit"]["warm_load"]["load"] == "loaded" and "aging" in r18["fit"]["warm_reason"]
    for r in (r10, r14, r18):
        f = r["fit"]
        assert f["seconds"] > 0 and set(f["divergences"]) == {"quality", "tokens", "latency"}
        assert f["warm_save"]["written"] is True
    for n in (10, 14, 18):
        doc = json.loads((run / "latent_warm" / f"round_{n:03d}.json").read_text())
        assert doc["meta"]["run_id"] == run.resolve().name
        assert doc["meta"]["checkpoint_sha256"] == m["latent"]["checkpoint_sha256"]


def test_numpyro_round_records_pinned_cpus(np_finished, tmp_path):
    import os

    run, _, _, _, _ = np_finished
    allowed = sorted(os.sched_getaffinity(0))
    for n in (10, 14, 18):
        sur = _fit(run, n)
        assert sur["pinned_cpus"] == allowed[:4]
        assert sur["pin"]["count"] == 4 and sur["pin"]["offset"] == 0 and sur["pin"]["torch_threads"] == 4


def test_matern_round_does_not_pin(finished):
    run, _, _, _ = finished
    sur = json.loads((run / "round_010.json").read_text())["surrogate"]
    assert sur["kernel"] == "matern" and sur["pinned_cpus"] is None
    assert not (run / "latent_warm").exists()


def test_numpyro_worker_rerun_keeps_warm_file_and_refits_identically(np_finished):
    """A worker crash after the warm save but before round_014.json: the re-run loads the same
    round_010 warm state, never rewrites latent_warm/round_014.json and fits bit-identically."""
    run, m, _, _, tmp = np_finished
    body = json.loads((run / "round_014.json").read_text())
    warm = run / "latent_warm" / "round_014.json"
    before = (warm.stat().st_mtime_ns, warm.read_bytes())
    with SearchLedger(run / "ledger.sqlite", m) as led:
        by_key = {e["key"]: e for e in led.complete_configs()}
    entries = [by_key[k] for k in body["archive_keys"]]
    objective = json.loads((run / "objective.json").read_text())
    # crash state: round_010 files + both warm files exist, round_014.json does not. Same
    # directory name as the original run, because run_id (provenance) is the directory name.
    rerun = tmp / "rerun" / run.resolve().name
    (rerun / "latent_warm").mkdir(parents=True)
    for p in run.glob("round_010*"):
        (rerun / p.name).write_bytes(p.read_bytes())
    for n in (10, 14):
        src = run / "latent_warm" / f"round_{n:03d}.json"
        (rerun / "latent_warm" / src.name).write_bytes(src.read_bytes())
    warm_copy = rerun / "latent_warm" / "round_014.json"
    copy_before = warm_copy.stat().st_mtime_ns
    mp = pytest.MonkeyPatch()
    try:
        _np_env(tmp, mp)
        with tls.install():
            driver.guarded_search(rerun, {**m, "objective": objective, "q": 4}, entries,
                                  seed=m["seed"] + 1, round_file="round_014")
    finally:
        mp.undo()
    again = json.loads((rerun / "round_014.json").read_text())
    assert again["surrogate"]["fit"]["mode"] == "warm"
    assert again["surrogate"]["fit"]["warm_save"] == {
        "path": str(warm_copy), "written": False, "existing_identical": True}
    assert warm_copy.stat().st_mtime_ns == copy_before
    assert again["selected"][0]["snapshot"] == body["selected"][0]["snapshot"]  # bit-identical fit
    assert (warm.stat().st_mtime_ns, warm.read_bytes()) == before


def test_numpyro_provenance_mismatch_falls_back_to_cold(tmp_path, monkeypatch):
    _np_env(tmp_path, monkeypatch)
    run, m, space = init_run(tmp_path, monkeypatch, surrogate=dict(NP_SURROGATE),
                             proposal_budget=NP_BUDGET)
    with tls.install():
        driver.run_round(run, evaluator=synthetic_receipt)  # round_010, cold
        path = run / "latent_warm" / "round_010.json"
        doc = json.loads(path.read_text())
        doc["meta"]["objective_sha256"] = "0" * 64  # e.g. objective.json changed
        path.write_text(json.dumps(doc))
        driver.run_round(run, evaluator=synthetic_receipt)  # round_014
    fit = _fit(run, 14)["fit"]
    assert fit["mode"] == "cold" and fit["generation"] == 0
    assert fit["warm_load"]["load"].startswith("rejected")


def test_pin_only_for_saas_numpyro(monkeypatch):
    from trimem.optimizer import latent_numpyro_saas as nps

    import torch

    calls = []
    monkeypatch.setattr(nps, "pin_cpus", lambda cpus, *, offset=0: calls.append((cpus, offset)) or [7])
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)  # keep this process's threads
    assert tls.pin_surrogate_cpus({"kernel": "matern"}) is None
    assert tls.pin_surrogate_cpus({"kernel": "saas"}) is None
    assert calls == []
    monkeypatch.setenv("TRIMEM_JAX_CPUS", "4")
    got = tls.pin_surrogate_cpus(tls.resolve_surrogate({"surrogate": {"kernel": "saas-numpyro"}}))
    assert calls == [(12, 0)]  # manifest defaults: 12 CPUs from offset 0
    assert got["cpus"] == [7] and got["ignored_env"] == "4"
    import os

    assert "TRIMEM_JAX_CPUS" not in os.environ  # the manifest is the only authority
    tls.pin_surrogate_cpus({"kernel": "saas-numpyro", "cpus": 3, "cpu_offset": 5})
    assert calls[-1] == (3, 5)


def test_validate_saas_numpyro_surrogate(tmp_path):
    space = tiny_space()
    ok = make_manifest(tmp_path, space, surrogate=dict(NP_SURROGATE))
    tls.validate_latent_manifest(ok)  # smoke: tiny NUTS allowed
    frozen = live_of(ok)
    live_sur = {"kernel": "saas-numpyro", "structure": "independent", "nuts": [512, 256, 16],
                "nuts_cost": None, "cpus": 12}
    frozen["latent"] = {**ok["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET), "surrogate": live_sur}
    tls.validate_latent_manifest(frozen)

    def with_sur(base, **kw):
        return {**base, "latent": {**base["latent"], "surrogate": {**base["latent"]["surrogate"], **kw}}}

    for bad, match in (
        ({"nuts": [16, 8, 4]}, "512, 256, 16"),
        ({"structure": "coupled-cost"}, "independent"),
        ({"cpus": 0}, "cpus"), ({"cpus": 33}, "cpus"), ({"cpus": "4"}, "cpus"), ({"cpus": True}, "cpus"),
        ({"cpu_offset": -1}, "cpu_offset"), ({"warm_warmup": 64}, "warm_warmup"),
        ({"kernel": "map-saas"}, "surrogate"),
    ):
        with pytest.raises(ValueError, match=match):
            tls.validate_latent_manifest(with_sur(frozen, **bad))
    with pytest.raises(ValueError, match="independent"):
        tls.validate_latent_manifest(with_sur(ok, structure="coupled-cost"))  # also in smoke
    for kernel in ("saas", "matern"):
        tls.validate_latent_manifest(with_sur(frozen, kernel=kernel, nuts=[512, 256, 16]))


def test_validator_requires_16_anchors(tmp_path):
    space = tiny_space()
    ok = make_manifest(tmp_path, space)
    tls.validate_latent_manifest(ok)
    frozen = live_of(ok)
    frozen["latent"] = {**ok["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET)}
    tls.validate_latent_manifest(frozen)
    for man in (ok, frozen):
        with pytest.raises(ValueError, match="anchors"):
            tls.validate_latent_manifest({**man, "latent": {**man["latent"], "anchors": 25}})


# ---- pre-launch fix round
def test_validator_requires_frozen_affine_outside_smoke(tmp_path):
    from trimem.optimizer.latent_plan_codec import load_affine

    space = tiny_space()
    ok = make_manifest(tmp_path, space)
    frozen = live_of(ok)
    frozen["latent"] = {**ok["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET)}
    tls.validate_latent_manifest(frozen)
    aff, meta = load_affine(ok["latent"]["affine"])
    prov = tmp_path / "prov.json"
    save_affine(prov, aff, {**meta, "kind": "provisional-train-v1"})
    lat = {**frozen["latent"], "affine": str(prov), "affine_sha256": file_sha256(prov)}
    with pytest.raises(ValueError, match="frozen-train-init-v1"):
        tls.validate_latent_manifest({**frozen, "latent": lat})
    tls.validate_latent_manifest({**ok, "latent": {**ok["latent"], "affine": str(prov),
                                                   "affine_sha256": file_sha256(prov)}})  # smoke: ok


def test_validator_matern_requires_independent_outside_smoke(tmp_path):
    space = tiny_space()
    ok = make_manifest(tmp_path, space)
    frozen = live_of(ok)
    sur = {**ok["latent"]["surrogate"], "structure": "coupled-cost"}
    frozen["latent"] = {**ok["latent"], "proposal_budget": dict(tls.DEFAULT_BUDGET), "surrogate": sur}
    with pytest.raises(ValueError, match="independent"):
        tls.validate_latent_manifest(frozen)
    tls.validate_latent_manifest({**ok, "latent": {**ok["latent"], "surrogate": sur}})  # smoke: allowed


def test_short_batch_raises_without_round_file(tmp_path, monkeypatch):
    from trimem.optimizer import latent_plan_search as lps

    run, m, space = init_run(tmp_path, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    objective = json.loads((run / "objective.json").read_text())
    with SearchLedger(run / "ledger.sqlite", manifest) as led:
        complete = led.complete_configs()

    def short(*a, **k):
        raise lps.ShortBatch("only 2 legal candidates for q = 4")

    monkeypatch.setattr(lps, "propose_latent_batch", short)
    with pytest.raises(lps.ShortBatch):
        tls.latent_search_worker(run, {**manifest, "objective": objective, "q": 4}, complete, [], 3,
                                 "round_010")
    assert not (run / "round_010.json").exists()
    assert "ShortBatch" in json.loads((run / "round_010.error.json").read_text())["error"]


def test_worker_round_file_records_work_budget(finished):
    run, m, _, _ = finished
    body = json.loads((run / "round_014.json").read_text())
    b = body["budget"]
    assert b["calls"] == b["refinement_calls"] + b["scoring_calls"] <= m["latent"]["proposal_budget"]["calls"]
    assert b["abort_seconds"] == 900 and len(body["selected"]) == 4
    assert body["out_of_box_dropped"] >= 0
