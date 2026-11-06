"""16 L-BFGS-B anchors from 4 sources (user 2026-11-04): LLM 4 / front 4 / newplan 4 / random 4,
refill random to 16 (latent_plan_search.select_latent_starts + _build_anchors)."""

import pytest
import torch

from tests.latent_fixtures import enumerate_space, synthetic_receipt, tiny_model, tiny_space
from trimem.optimizer import latent_plan_search as lps
from trimem.optimizer.joint_search_controller import frontier_parents
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_search import propose_latent_batch, select_latent_starts
from trimem.optimizer.latent_plan_surrogate import fit_latent_model, latent_snapshot
from trimem.optimizer.latent_region import BatchBudget, TrustState
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, aggregate_panel

torch.set_num_threads(4)
TASKS = ["t0", "t1", "t2", "t3"]
MANIFEST = {"protocol": "p", "evaluation_split": "TRAIN"}
FAMILY = ("filter_order", "graph_impl", "rank_impl")


def _family(c):
    return tuple(c[f] for f in FAMILY)


@pytest.fixture(scope="module")
def world():
    space, model = tiny_space(), tiny_model(4)
    configs = enumerate_space(space)
    codec = LatentCodec(space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x")
    colbert = [c for c in configs if c["rank_impl"] == "ColBERT"]  # leaves 6 families unmeasured
    picks = colbert[:: max(1, len(colbert) // 12)][:12]
    obs = [aggregate_panel(space, c, [synthetic_receipt(c, t, MANIFEST) for t in TASKS], task_ids=TASKS,
                           protocol="p", allow_synthetic=True) for c in picks]
    objective = ObjectiveProtocol.from_initial(obs)
    gp = fit_latent_model(codec, obs, objective, seed=0, maxiter=60, timeout=60, kernel="matern")
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    return space, codec, snap, [o.config for o in obs], configs


def _llm(space, configs, measured, n=6):
    seen = {space.key(c) for c in measured}
    return [c for c in configs if space.key(c) not in seen][-n:]


def test_source_counts_and_order(world):
    space, _, snap, measured, configs = world
    llm = _llm(space, configs, measured)
    starts = select_latent_starts(space, snap, measured, llm=llm, seed=5)
    labels = [s.source for s in starts]
    # order llm, front, newplan, random; each <= 4
    order = [lab for i, lab in enumerate(labels) if lab not in labels[:i]]
    assert order == [s for s in ("llm", "front", "newplan", "random") if s in labels]
    assert labels.count("llm") == 4 and [s.config for s in starts[:4]] == [space.canonicalize(c) for c in llm[:4]]
    for lab in ("front", "newplan", "random"):
        assert 1 <= labels.count(lab) <= 4
    assert len(starts) <= 16
    keys = [space.key(s.config) for s in starts]
    assert len(keys) == len(set(keys))  # dedup across sources


def test_front_at_most_four_and_measured(world):
    space, _, snap, measured, _ = world
    starts = select_latent_starts(space, snap, measured, llm=(), seed=5)
    front = [s.config for s in starts if s.source == "front"]
    parents = frontier_parents(snap, measured)
    assert 1 <= len(front) <= min(4, len(parents))
    assert all(c in parents for c in front)


def test_front_keeps_largest_hv_contributions(world, monkeypatch):
    space, _, snap, measured, _ = world
    five = measured[:5]
    monkeypatch.setattr(lps, "frontier_parents", lambda s, m: list(five))
    front = [s.config for s in select_latent_starts(space, snap, measured, llm=(), seed=5)
             if s.source == "front"]
    assert len(front) == 4 and all(c in five for c in front)
    assert front == lps._front_by_hv(snap, five)[:4]


def test_newplan_only_unmeasured_families(world):
    space, _, snap, measured, _ = world
    known = {_family(c) for c in measured}
    starts = select_latent_starts(space, snap, measured, llm=(), seed=5)
    new = [s.config for s in starts if s.source == "newplan"]
    assert new and len(new) <= 4
    assert all(_family(c) not in known for c in new)
    assert len({_family(c) for c in new}) == len(new)
    assert select_latent_starts(space, snap, measured, llm=(), seed=5) == starts  # deterministic


def test_newplan_empty_when_every_family_measured(world):
    space, _, snap, _, configs = world
    every = list({_family(c): c for c in configs}.values())
    starts = select_latent_starts(space, snap, every, llm=(), seed=5)
    assert "newplan" not in [s.source for s in starts]


def test_random_is_sobol_avoiding_measured(world):
    space, _, snap, measured, _ = world
    starts = select_latent_starts(space, snap, measured, llm=(), seed=5)
    rand = [s.config for s in starts if s.source == "random"]
    keys = {space.key(c) for c in measured}
    want = [c for c in space.sample(4, seed=5 + 200000, avoid=keys)]
    assert rand == [c for c in want if c in rand]  # same draws, dedup may drop earlier-source repeats
    assert not {space.key(c) for c in rand} & keys


def test_dedup_llm_wins_over_later_sources(world):
    space, _, snap, measured, _ = world
    rand = space.sample(4, seed=5 + 200000, avoid={space.key(c) for c in measured})
    starts = select_latent_starts(space, snap, measured, llm=[rand[0]], seed=5)
    hits = [s.source for s in starts if space.key(s.config) == space.key(rand[0])]
    assert hits == ["llm"]


@pytest.fixture
def lcodec(world, monkeypatch):
    space, codec, snap, measured, configs = world
    U = codec.encode_many(configs)
    monkeypatch.setattr(codec, "roundtrip", lambda c: True)
    monkeypatch.setattr(codec, "decode", lambda X: [
        configs[int(torch.cdist(x.reshape(1, -1).double(), U).argmin())] for x in torch.as_tensor(X)])
    return codec


def _propose(world, codec, agent=()):
    space, _, snap, measured, _ = world
    return propose_latent_batch(space, codec, snap, measured, agent=agent, seed=5, round_index=1, q=4,
                                trust=TrustState(), budget=BatchBudget(seconds=None, calls=400))


def test_exactly_16_anchors_with_sources(world, lcodec):
    space, _, _, measured, configs = world
    out = _propose(world, lcodec, agent=_llm(space, configs, measured))
    ok = [a for a in out["anchors"] if a["status"] == "ok"]
    assert lps.N_ANCHORS == 16 and len(ok) == 16
    src = [a["source"] for a in ok]
    assert set(src) <= {"llm", "front", "newplan", "random", "refill"}
    assert src.count("llm") == 4
    assert out["anchor_sources"] == {s: src.count(s) for s in dict.fromkeys(src)}


def test_refill_when_llm_empty(world, lcodec):
    out = _propose(world, lcodec, agent=())
    ok = [a for a in out["anchors"] if a["status"] == "ok"]
    src = [a["source"] for a in ok]
    assert len(ok) == 16 and "llm" not in src and src.count("refill") >= 4
    assert all(s in {"front", "newplan", "random", "refill"} for s in src)
    assert not {a["key"] for a in ok if a["source"] == "refill"} & {lcodec.key(c) for c in world[3]}
