"""Synthetic end-to-end smoke of the latent-MOBO arm (Task 12, spec 12.3), fast variant.

Tiny space, Matern GP, patched codec (every config round-trips, decode = nearest enumerated
config by u, as in test_trace_latent_propose.py), budget 10 + 8 = 2 rounds x q 4. The real
driver ``run_round``, ``guarded_search`` (spawned worker) and ledger run unchanged.
"""

import json

import pytest

from tests.latent_fixtures import (
    SMOKE_HOLDOUT,
    SMOKE_TASKS,
    enumerate_space,
    tiny_model,
    tiny_space,
)
from tools import latent_synthetic_smoke as smoke
from trimem.optimizer.latent_plan_codec import file_sha256, fit_affine, save_affine
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint


def _inputs(tmp):
    space, model = tiny_space(), tiny_model(4)
    ck = tmp / "ck.pt"
    save_checkpoint(ck, model, space, {})
    ap = tmp / "affine.json"
    save_affine(ap, fit_affine(model, SeqGrammar(space), enumerate_space(space)[:60]),
                {"checkpoint_sha256": file_sha256(ck)})
    run = tmp / "space_run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "space": space.manifest(), "tasks": SMOKE_TASKS, "seed": 3,
        "fixed": {"dataset": "tiny"}, "holdout": {"tasks": SMOKE_HOLDOUT},
    }))
    return run, ck, ap


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    run, ck, ap = _inputs(tmp)
    out = tmp / "out"
    report = smoke.main(out, space_run=run, checkpoint=ck, affine=ap, kernel="matern",
                        structure="independent", fast=True)
    return report, out, (run, ck, ap)


def test_new_checks_present(done):
    report, _, _ = done
    for k in ("picks_equal_q", "total_calls_capped", "deterministic_replay"):
        assert report["checks"][k] is True, report["evidence"][k]
    assert report["evidence"]["deterministic_replay"]["equal"] == {"picks": True, "endpoints": True,
                                                                   "calls": True}


def test_every_check_true(done):
    report, _, _ = done
    assert set(report["checks"]) == set(smoke.CHECKS)
    failed = {k: report["evidence"][k] for k, v in report["checks"].items() if not v}
    assert not failed, failed
    assert report["all_pass"] is True


def test_labels_and_file(done):
    report, out, _ = done
    assert report["synthetic"] is True
    assert report["not_evidence_of_improvement"] == "spec §12.3"
    assert json.loads((out / "smoke.json").read_text()) == report


def test_shape_10_plus_8(done):
    report, _, _ = done
    ev = report["evidence"]
    assert ev["batch_sizes"]["observed"] == [4, 4]
    assert ev["unique_configs"]["final_complete"] == 18
    assert report["settings"]["kernel"] == "matern"
    assert ev["gp_three_models"]["surrogate_info"]["classes"] == ["SingleTaskGP"] * 3


def test_restart_reserves_round_file_keys(done):
    report, _, _ = done
    r = report["evidence"]["restart_consistent"]
    assert r["crashed"] and r["pending_after_crash"] == 0
    assert r["reserved_keys"] == r["round_file_keys"] and len(r["reserved_keys"]) == 4
    assert r["files_rewritten"] == []


def test_rounds_record_timings_and_sources(done):
    report, _, _ = done
    rounds = report["rounds"]
    assert [r["file"] for r in rounds] == ["round_010", "round_014"]
    for r in rounds:
        assert r["wall_seconds"] > 0 and r["proposal_seconds"] is not None
        assert r["fit_seconds"] is not None and r["fit_seconds"] >= 0
        assert len(r["picks"]) == 4 and all(p["sources"] for p in r["picks"])
        assert sum(r["anchor_sources"].values()) == 16  # 16 anchors (user 2026-11-04)
        assert set(r["anchor_sources"]) <= {"llm", "front", "newplan", "random", "refill"}
    assert sum(report["source_mix"]["contains"].values()) >= 8
    assert report["source_mix"]["contains"].get("latent", 0) >= 1  # patched codec exercises it


def test_refuses_existing_output(done):
    _, out, (run, ck, ap) = done
    with pytest.raises(FileExistsError):
        smoke.main(out, space_run=run, checkpoint=ck, affine=ap, kernel="matern",
                   structure="independent", fast=True)


def test_receipt_handles_inactive_fields():
    space = tiny_space()
    c = next(c for c in enumerate_space(space) if c["graph_impl"] != "beam")
    m = {"protocol": "p"}
    a, b = smoke.synthetic_receipt(c, "alfworld_1", m), smoke.synthetic_receipt(c, "alfworld_1", m)
    assert a == b and a["synthetic"] is True and a["split"] == "TRAIN"
    for f in ("raw_score", "total_tokens", "latency_seconds"):
        assert isinstance(a[f], (int, float))
    assert smoke.synthetic_receipt({**c, "depth": None, "beam_width": None}, "x", m)["total_tokens"] > 0


def test_binding_call_budget_is_a_hard_cap(tmp_path):
    run, ck, ap = _inputs(tmp_path)
    # pre-launch fix round: ONE shared counter (refinement + projection + pool scoring). With a cap
    # of 1800 the refinement share binds after the scoring reserve; the total stays <= the cap.
    report = smoke.main(tmp_path / "out", space_run=run, checkpoint=ck, affine=ap,
                        kernel="matern", structure="independent", fast=True,
                        proposal_budget={"seconds": 20, "calls": 1800})
    ev = report["evidence"]["total_calls_capped"]
    assert ev["max_calls"] <= 1800, ev
    assert all(r["calls"] == r["refinement"] + r["scoring"] for r in ev["rounds"])
    assert report["checks"]["proposal_budget_respected"] is True
    assert report["checks"]["total_calls_capped"] is True


def test_rescoring_not_verified_when_refit_differs(tmp_path):
    """Task 12 minor: a refit that is not bit-identical must not pass rescoring silently."""
    arm = tmp_path / "arm"
    arm.mkdir()
    body = {"projections": [{"key": "k", "u_canonical": [0.1], "ell_canonical": -1.0}],
            "selected": [{"key": "k", "sources": ["sobol"], "snapshot": "round-snapshot",
                          "ell": -1.0, "config": {}}]}
    (arm / "round_014.json").write_text(json.dumps(body))

    class Snap:
        identity = "a-different-refit"

    ev = smoke._check_rescoring(arm, [{"file": "round_014"}], codec=None,
                                refit={"body": body, "snap": Snap(), "round_file": "round_014"})
    assert ev["pass"] is False
    assert ev["refit_bit_identical"] is False and ev["exact_rescoring"] == "not verified"


def test_fast_smoke_saas_numpyro_warm_refit(tmp_path, monkeypatch):
    """Fast saas-numpyro smoke: round_014 is warm-started from round_010; the main-process refit
    (same warm state, same pinning) is bit-identical, so rescoring is verified exactly."""
    import os

    monkeypatch.setenv("TRIMEM_JAX_CACHE_DIR", str(tmp_path / "jaxcache"))
    monkeypatch.delenv("TRIMEM_JAX_CPUS", raising=False)
    before = os.sched_getaffinity(0)
    run, ck, ap = _inputs(tmp_path)
    report = smoke.main(tmp_path / "out", space_run=run, checkpoint=ck, affine=ap,
                        kernel="saas-numpyro", structure="independent", fast=True,
                        nuts=(16, 8, 4), warm_warmup=8, cpus=4)
    failed = {k: report["evidence"][k] for k, v in report["checks"].items() if not v}
    assert not failed, failed
    ev = report["evidence"]
    assert ev["rescoring_used"]["refit_bit_identical"] is True
    assert ev["gp_three_models"]["surrogate_info"]["kernel"] == "saas-numpyro"
    modes = [r["surrogate_fit"]["mode"] for r in report["rounds"]]
    assert modes == ["cold", "warm"]
    assert report["refit"]["fit"]["mode"] == "warm"
    assert os.sched_getaffinity(0) == before  # the refit pin is undone in this process


def test_fast_smoke_with_staged_pruning(tmp_path):
    """User 2026-11-04: the arm runs the live staged-pruning block (checkpoints every 5 of the 10
    tiny tasks here); the run ends on the driver's pruning budget_complete and every check except
    pruned_at_checkpoint is true. On the tiny space the BO picks are not dominated at 5 tasks, so no
    natural prune is required here (the prune -> partial-panel path is covered deterministically in
    test_trace_latent_pruning.py; the real-space smoke must show one)."""
    run, ck, ap = _inputs(tmp_path)
    (run / "train_panel.json").write_text(json.dumps({"items": [
        {"item_id": t, "stratum": "even" if i % 2 == 0 else "odd"} for i, t in enumerate(SMOKE_TASKS)]}))
    report = smoke.main(tmp_path / "out", space_run=run, checkpoint=ck, affine=ap, kernel="matern",
                        structure="independent", fast=True, pruning=True, stage_step=5)
    failed = {k: report["evidence"][k] for k, v in report["checks"].items()
              if not v and k != "pruned_at_checkpoint"}
    assert not failed, failed
    assert set(report["checks"]) == set(smoke.CHECKS) | {"pruned_at_checkpoint"}
    ev = report["evidence"]
    explored = ev["unique_configs"]["final_complete"] - 10
    assert ev["pruned_at_checkpoint"]["decisions_logged"] >= explored > 0  # every BO config decided at 5
    assert not ev["pruned_at_checkpoint"]["violations"]
    assert report["settings"]["pruning"]["sizes"] == [5, 10]
    assert report["final_status"]["status"] == "budget_complete"
    assert ev["pruned_at_checkpoint"]["episodes_done"] <= 18 * 10  # (10 + 8) x 10 episode budget


def test_rugged_receipt_is_deterministic_bernoulli():
    space = tiny_space()
    c = enumerate_space(space)[0]
    m = {"protocol": "p"}
    rows = [smoke.rugged_receipt(c, f"alfworld_{i}", m) for i in range(50)]
    assert rows[0] == smoke.rugged_receipt(c, "alfworld_0", m)
    assert {r["raw_score"] for r in rows} <= {0, 1} and all(r["synthetic"] for r in rows)
    assert all(r["total_tokens"] > 0 and r["latency_seconds"] > 0 for r in rows)
    assert smoke.LANDSCAPES["structured"] is smoke.synthetic_receipt
