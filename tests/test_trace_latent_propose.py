import pytest
import torch

from tests.latent_fixtures import (
    enumerate_space,
    synthetic_receipt,
    tiny_model,
    tiny_space,
)
from trimem.optimizer import latent_plan_search as lps
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_search import Endpoint, propose_latent_batch
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
    codec = LatentCodec(
        space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x"
    )
    picks = configs[:: max(1, len(configs) // 20)][:20]
    obs = [
        aggregate_panel(
            space, c, [synthetic_receipt(c, t, MANIFEST) for t in TASKS],
            task_ids=TASKS, protocol="p", allow_synthetic=True,
        )
        for c in picks
    ]
    objective = ObjectiveProtocol.from_initial(obs)
    gp = fit_latent_model(codec, obs, objective, seed=0, maxiter=60, timeout=60, kernel="matern")
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    return space, codec, snap, obs, configs, objective, gp


def run(world, *, q=4, measured=None, budget=None, **kw):
    space, codec, snap, obs, *_ = world
    measured = measured if measured is not None else [o.config for o in obs]
    return propose_latent_batch(
        space, codec, snap, measured, seed=5, round_index=1, q=q,
        trust=TrustState(), budget=budget or BatchBudget(seconds=None, calls=2000), **kw,
    )


def test_batch_valid_distinct_novel(world):
    space, codec, snap, obs, *_ = world
    out = run(world)
    keys = [s["key"] for s in out["selected"]]
    assert len(keys) == 4 == len(set(keys))
    assert not set(keys) & {space.key(o.config) for o in obs}
    for s in out["selected"]:
        assert space.canonicalize(s["config"]) == s["config"]
        assert s["sources"] and s["plan"] is not None
    for k in ("starts", "anchors", "endpoints", "projections", "pool_count",
              "provenance", "rejected_agent", "fallbacks", "budget", "trust",
              "codec", "seconds", "seed", "q"):
        assert k in out
    assert out["codec"] == codec.identity


def test_final_batch_q2(world):
    assert len(run(world, q=2)["selected"]) == 2


@pytest.fixture
def lcodec(world, monkeypatch):
    """Codec patched so every config round-trips: decode -> nearest enumerated config by u."""
    space, codec, snap, obs, configs, *_ = world
    U = codec.encode_many(configs)
    monkeypatch.setattr(codec, "roundtrip", lambda c: True)
    monkeypatch.setattr(
        codec, "decode",
        lambda X: [configs[int(torch.cdist(x.reshape(1, -1).double(), U).argmin())]
                   for x in torch.as_tensor(X)],
    )
    return codec


def test_endpoint_score_never_claims_slot(world, lcodec, monkeypatch):
    space, codec, snap, obs, configs, *_ = world
    measured = {space.key(o.config) for o in obs}
    free = [c for c in configs if space.key(c) not in measured]
    scores = snap.score(free)
    X = free[min(range(len(free)), key=lambda i: scores[i])]
    uX = tuple(codec.encode(X))
    Xp = codec.decode(torch.tensor([uX], dtype=torch.double))[0]

    def fake(snapshot, u0, bounds, *, max_calls, seconds, anchor_key, anchor_source, deadline=None):
        return Endpoint(anchor_key, anchor_source, tuple(u0.tolist()), uX, 0.0, 1e9,
                        "converged", 1, 0.0)

    monkeypatch.setattr(lps, "refine_anchor", fake)
    out = run(world, q=1)
    kXp = space.key(Xp)
    assert "latent" in out["provenance"][kXp]
    legal = [(k, c) for k, c in ((space.key(c), c) for c in
             [*free]) if k in out["provenance"]]
    sc = snap.score([c for _, c in legal])
    best = max(range(len(legal)), key=lambda i: (sc[i], legal[i][0]))
    assert out["selected"][0]["key"] == legal[best][0]
    assert legal[best][0] != kXp  # the lowest-scoring endpoint config did not win
    assert out["selected"][0]["ell"] < 1e8


def test_collapsed_endpoints_fill_from_other_sources(world, monkeypatch):
    space, codec, snap, obs, *_ = world
    measured_u = tuple(codec.encode(obs[0].config))

    def fake(snapshot, u0, bounds, *, max_calls, seconds, anchor_key, anchor_source, deadline=None):
        return Endpoint(anchor_key, anchor_source, tuple(u0.tolist()), measured_u, 0.0,
                        0.0, "converged", 1, 0.0)

    monkeypatch.setattr(lps, "refine_anchor", fake)
    out = run(world)
    assert len(out["selected"]) == 4
    assert all("latent" not in s["sources"] for s in out["selected"])


def test_second_pick_uses_pending_snapshot(world):
    snap = world[2]
    out = run(world)
    assert out["selected"][0]["snapshot"] == snap.identity
    assert out["selected"][1]["snapshot"] == snap.with_pending(
        [out["selected"][0]["config"]]).identity


def test_roundtrip_failed_anchor_skipped(world, monkeypatch):
    space, codec, *_ = world
    bad = run(world)["anchors"][0]["key"]
    orig = codec.roundtrip
    monkeypatch.setattr(codec, "roundtrip", lambda c: False if codec.key(c) == bad else orig(c))
    out = run(world)
    rec = [a for a in out["anchors"] if a["key"] == bad]
    assert rec and rec[0]["status"] == "roundtrip_failed"
    assert all(e["anchor_key"] != bad for e in out["endpoints"])


def test_out_of_domain_anchor_skipped(world, monkeypatch):
    space, codec, *_ = world
    # an unmeasured (random) anchor: patching encode of a measured front anchor would also move
    # the posterior that ranks the front anchors (16-anchor sources, user 2026-11-04)
    bad = next(a["key"] for a in run(world)["anchors"] if a["source"] == "random")
    orig = codec.encode
    monkeypatch.setattr(
        codec, "encode",
        lambda c: [1.5] + list(orig(c))[1:] if codec.key(c) == bad else orig(c),
    )
    out = run(world)
    rec = [a for a in out["anchors"] if a["key"] == bad]
    assert rec and rec[0]["status"] == "out_of_domain"
    assert all(e["anchor_key"] != bad for e in out["endpoints"])


def test_budget_exhaustion_marks_unvisited(world):
    out = run(world, budget=BatchBudget(seconds=None, calls=10))
    assert len(out["selected"]) == 4
    assert any(e["status"] == "unvisited" for e in out["endpoints"])


def test_backfill_when_pool_nearly_exhausted(world):
    space, codec, snap, obs, configs, *_ = world
    measured = configs[:-3]
    # pre-launch fix round: never commit < q -- 3 legal configs cannot fill q = 4
    with pytest.raises(lps.ShortBatch):
        run(world, measured=measured)
    out = run(world, measured=measured, q=3)
    mk = {space.key(c) for c in measured}
    assert len(out["selected"]) == 3
    assert not {s["key"] for s in out["selected"]} & mk


def test_backfill_adds_configs(world, monkeypatch):
    from trimem.optimizer.plan_schema import JointSpace

    space, codec, snap, obs, configs, *_ = world
    measured = configs[:-3]
    mk = {space.key(c) for c in measured}
    real = JointSpace.sample

    def sample(self, n, **kw):
        return real(self, n, **kw) if n == 4 and kw.get("max_draws") == 4096 else []

    monkeypatch.setattr(JointSpace, "sample", sample)
    monkeypatch.setattr(lps, "recombine", lambda *a, **k: [])
    mk0 = codec.encode(measured[0])

    def fake(snapshot, u0, bounds, *, max_calls, seconds, anchor_key, anchor_source, deadline=None):
        return Endpoint(anchor_key, anchor_source, tuple(u0.tolist()), tuple(mk0), 0.0,
                        0.0, "converged", 1, 0.0)

    monkeypatch.setattr(lps, "refine_anchor", fake)
    out = run(world, measured=measured, q=3)  # 3 legal configs: q = 4 would raise ShortBatch
    assert any("keys" in f for f in out["fallbacks"])
    assert out["selected"] and not {s["key"] for s in out["selected"]} & mk
    assert all("backfill" in s["sources"] for s in out["selected"])


def test_multi_anchor_refinement(world, lcodec):
    out = run(world, budget=BatchBudget(seconds=None, calls=4000))
    visited = {e["anchor_key"] for e in out["endpoints"] if e["status"] != "unvisited"}
    assert len(visited) >= 2
    assert len(out["selected"]) == 4


def test_latent_pick_has_latent_info(world, lcodec, monkeypatch):
    space, codec, snap, obs, configs, *_ = world
    measured = {space.key(o.config) for o in obs}
    free = [c for c in configs if space.key(c) not in measured]
    sc = snap.score(free)
    F = free[max(range(len(free)), key=lambda i: (sc[i], space.key(free[i])))]
    uF = tuple(codec.encode(F))

    def fake(snapshot, u0, bounds, *, max_calls, seconds, anchor_key, anchor_source, deadline=None):
        return Endpoint(anchor_key, anchor_source, tuple(u0.tolist()), uF, 0.0, 0.0,
                        "converged", 1, 0.0)

    monkeypatch.setattr(lps, "refine_anchor", fake)
    out = run(world, q=1)
    assert len({e["anchor_key"] for e in out["endpoints"]}) >= 2
    pick = out["selected"][0]
    assert pick["key"] == space.key(F) and "latent" in pick["sources"]
    assert pick["latent"] is not None and pick["latent"]["anchor_key"]
    assert pick["latent"]["key"] == pick["key"]


def test_refine_cap_leaves_room_for_start_and_final(world, monkeypatch):
    seen = []
    real = lps.refine_anchor

    def spy(snapshot, u0, bounds, *, max_calls, **kw):
        seen.append(max_calls)
        return real(snapshot, u0, bounds, max_calls=max_calls, **kw)

    monkeypatch.setattr(lps, "refine_anchor", spy)
    # pre-launch fix round: the shared cap first reserves the scoring of all slots, so the cap
    # must exceed that reserve for any refinement to happen
    run(world, budget=BatchBudget(seconds=None, calls=3000))
    # every cap is a share minus the 2 uncounted evaluations
    assert seen and all(0 < m + 2 <= 100 for m in seen)


def test_deterministic_with_call_budget(world):
    a = run(world)
    b = run(world)
    assert [s["key"] for s in a["selected"]] == [s["key"] for s in b["selected"]]
