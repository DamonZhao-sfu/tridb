"""Pre-launch fix round (Task 13 adversarial verdict + wiring review): never commit < q, work-based
deterministic stopping with one shared call counter + a raising wall-clock abort, domain check on
every scored candidate, LLM dedup before the cap, SAAS orphan warm-file replace, one anchor
constant."""

import json

import pytest
import torch

from tests.latent_fixtures import enumerate_space, synthetic_receipt, tiny_model, tiny_space
from trimem.experiments import trace_latent_search as tls
from trimem.optimizer import latent_plan_search as lps
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_search import propose_latent_batch
from trimem.optimizer.latent_plan_surrogate import fit_latent_model, latent_snapshot
from trimem.optimizer.latent_region import BatchBudget, TrustState
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, aggregate_panel

torch.set_num_threads(4)
TASKS = ["t0", "t1", "t2", "t3"]
MANIFEST = {"protocol": "p", "evaluation_split": "TRAIN"}


@pytest.fixture(scope="module")
def world():
    space, model = tiny_space(), tiny_model(4)
    configs = enumerate_space(space)
    codec = LatentCodec(space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x")
    colbert = [c for c in configs if c["rank_impl"] == "ColBERT"]
    picks = colbert[:: max(1, len(colbert) // 12)][:12]
    obs = [aggregate_panel(space, c, [synthetic_receipt(c, t, MANIFEST) for t in TASKS], task_ids=TASKS,
                           protocol="p", allow_synthetic=True) for c in picks]
    objective = ObjectiveProtocol.from_initial(obs)
    gp = fit_latent_model(codec, obs, objective, seed=0, maxiter=60, timeout=60, kernel="matern")
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    return space, codec, snap, [o.config for o in obs], configs


@pytest.fixture
def lcodec(world, monkeypatch):
    space, codec, snap, measured, configs = world
    U = codec.encode_many(configs)
    monkeypatch.setattr(codec, "roundtrip", lambda c: True)
    monkeypatch.setattr(codec, "decode", lambda X: [
        configs[int(torch.cdist(x.reshape(1, -1).double(), U).argmin())] for x in torch.as_tensor(X)])
    return codec


def _run(world, codec, *, measured=None, calls=2000, agent=(), q=4, **kw):
    space, _, snap, meas, _ = world
    return propose_latent_batch(space, codec, snap, meas if measured is None else measured, agent=agent,
                                seed=5, round_index=1, q=q, trust=TrustState(),
                                budget=BatchBudget(seconds=None, calls=calls), **kw)


def _picks(out):
    return [(s["key"], s["ell"], tuple(s["sources"])) for s in out["selected"]]


# ---- item 1: never commit < q
def test_all_nonfinite_scores_fill_q_picks(world, lcodec, monkeypatch):
    space = world[0]
    monkeypatch.setattr(type(world[2]), "score", lambda self, cfgs: [float("nan")] * len(cfgs))
    out = _run(world, lcodec)
    assert len(out["selected"]) == 4
    assert all(s["sources"] == ["fill"] and s["ell"] is None for s in out["selected"])
    keys = [s["key"] for s in out["selected"]]
    assert len(set(keys)) == 4 and not set(keys) & {space.key(c) for c in world[3]}
    again = _run(world, lcodec)
    assert _picks(again) == _picks(out)  # deterministic fill (seeded Sobol order)


def test_exhausted_space_raises_short_batch(world, lcodec):
    space, _, _, _, configs = world
    measured = configs[:-2]  # only 2 legal configs left for q = 4
    with pytest.raises(lps.ShortBatch):
        _run(world, lcodec, measured=measured)


# ---- items 2+3: work-based, deterministic, one shared counter, raising abort
def test_bit_identical_picks_under_different_slowdowns(world, lcodec, monkeypatch):
    import time

    real = time.monotonic
    runs = []
    for factor in (1.0, 40.0):
        t0 = real()
        monkeypatch.setattr(lps, "_clock", lambda f=factor, t0=t0: t0 + (real() - t0) * f)
        out = _run(world, lcodec, calls=3000)
        runs.append(out)
    a, b = runs
    assert _picks(a) == _picks(b)
    assert [(e["anchor_key"], e["slot"], e["status"], e["calls"], e["u_end"]) for e in a["endpoints"]] == \
        [(e["anchor_key"], e["slot"], e["status"], e["calls"], e["u_end"]) for e in b["endpoints"]]
    assert a["budget"]["calls"] == b["budget"]["calls"]
    assert all(e["status"] != "timeout" for e in a["endpoints"])


def test_one_shared_call_counter_bounds_all_scoring(world, lcodec):
    out = _run(world, lcodec, calls=3000)
    b = out["budget"]
    assert b["calls"] == b["refinement_calls"] + b["scoring_calls"] <= 3000, b
    assert b["refinement_calls"] > 0 and b["scoring_calls"] > 0 and b["over_cap"] is False
    assert b["calls"] == sum(s["calls"] for s in b["per_slot"])
    # cap below the scoring reserve: refinement stops (0 calls), scoring of the built pool finishes
    tiny = _run(world, lcodec, calls=300)
    tb = tiny["budget"]
    assert tb["refinement_calls"] == 0 and tb["over_cap"] is True and len(tiny["selected"]) == 4
    assert all(e["status"] == "unvisited" for e in tiny["endpoints"])


def test_wall_clock_abort_raises(world, lcodec, monkeypatch):
    ticks = iter(range(0, 10**9, 1000))  # every clock read advances 1000 s
    monkeypatch.setattr(lps, "_clock", lambda: float(next(ticks)))
    with pytest.raises(lps.ProposalAborted):
        _run(world, lcodec)


# ---- item 5: domain check on every scored candidate
def test_out_of_box_candidates_dropped_and_counted(world, lcodec, monkeypatch):
    space = world[0]
    base = _run(world, lcodec)
    bad = base["selected"][0]["key"]
    orig = lcodec.encode
    monkeypatch.setattr(lcodec, "encode", lambda c: [1.5] + list(orig(c))[1:] if lcodec.key(c) == bad else orig(c))
    out = _run(world, lcodec)
    assert bad not in [s["key"] for s in out["selected"]]
    assert out["out_of_box_dropped"] >= 1
    assert base["out_of_box_dropped"] == 0
    assert len(out["selected"]) == 4 and space


# ---- item 7: LLM dedup before the 4-cap; one anchor constant
def test_llm_dedup_before_cap(world, lcodec):
    space, _, _, measured, configs = world
    free = [c for c in configs if space.key(c) not in {space.key(m) for m in measured}]
    agent = [free[0], free[0], free[0], free[1], free[2], free[3]]
    out = _run(world, lcodec, agent=agent)
    llm = [a for a in out["anchors"] if a["source"] == "llm" and a["status"] == "ok"]
    assert len(llm) == 4  # 4 distinct LLM configs despite the repeated first one


def test_single_anchor_constant():
    assert tls.ANCHORS == lps.N_ANCHORS == 16


# ---- item 4: SAAS orphan warm file replaced
class _M:
    def __init__(self, tag):
        self.numpyro_info = {"state": {"objective": None, "n": 1, "tag": tag}}


class _Model:
    def __init__(self, tag):
        self.models = [_M(tag) for _ in range(3)]
        for m, name in zip(self.models, ("quality", "tokens", "latency")):
            m.numpyro_info["state"]["objective"] = name


META = {"checkpoint_sha256": "c", "affine_sha256": "a", "objective_sha256": "o", "run_id": "r",
        "panel_identity": "p"}


def test_orphan_warm_file_replaced_identical_kept(tmp_path):
    first = tls.save_round_warm_state(tmp_path, "round_014", _Model("A"), META)
    assert first["written"] is True
    path = tmp_path / "latent_warm" / "round_014.json"
    same = tls.save_round_warm_state(tmp_path, "round_014", _Model("A"), META)
    assert same == {"path": str(path), "written": False, "existing_identical": True}
    other = tls.save_round_warm_state(tmp_path, "round_014", _Model("B"), META)
    assert other["written"] is True and other["replaced_orphan"] is True
    assert json.loads(path.read_text())["objectives"]["quality"]["tag"] == "B"
