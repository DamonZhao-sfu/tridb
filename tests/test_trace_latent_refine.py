import pytest
import torch

from tests.latent_fixtures import (
    enumerate_space,
    synthetic_receipt,
    tiny_model,
    tiny_space,
)
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_search import project, refine_anchor
from trimem.optimizer.latent_plan_surrogate import fit_latent_model, latent_snapshot
from trimem.optimizer.latent_region import box
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
        space,
        model,
        fit_affine(model, SeqGrammar(space), configs),
        checkpoint_sha256="x",
    )
    picks = configs[:: len(configs) // 12][:12]
    obs = [
        aggregate_panel(
            space,
            c,
            [synthetic_receipt(c, t, MANIFEST) for t in TASKS],
            task_ids=TASKS,
            protocol="p",
            allow_synthetic=True,
        )
        for c in picks
    ]
    objective = ObjectiveProtocol.from_initial(obs)
    gp = fit_latent_model(codec, obs, objective, seed=0, maxiter=60, timeout=60, kernel="matern")
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    a = torch.tensor(codec.encode(picks[0]), dtype=torch.double)
    return codec, snap, a, box(a, 0.4)


def _ref(world, **kw):
    codec, snap, a, b = world
    args = dict(max_calls=100, seconds=None, anchor_key="k", anchor_source="obs")
    args.update(kw)
    return refine_anchor(snap, a, b, **args)


def test_endpoint_stays_in_box(world):
    e = _ref(world)
    b = world[3]
    u = torch.tensor(e.u_end, dtype=torch.double)
    assert (u >= b[0] - 1e-9).all() and (u <= b[1] + 1e-9).all()


def test_refine_does_not_decrease_ell(world):
    e = _ref(world)
    assert e.ell_end >= e.ell_start - 1e-6


def test_calls_counted_and_bounded(world):
    e = _ref(world, max_calls=20)
    assert 0 < e.calls <= 20  # hard cap (Task 12 fix 1): no scipy maxfun overrun


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 8])
def test_call_cap_is_hard_and_returns_best_point(world, cap):
    _codec, snap, a, b = world
    seen = []
    real = snap.acquisition

    class Spy(torch.nn.Module):  # counts every evaluation the optimiser makes
        model = real.model

        def forward(self, X):
            seen.append(1)
            return real(X)

    class S:
        acquisition = Spy()

    e = refine_anchor(S, a, b, max_calls=cap, seconds=None, anchor_key="k", anchor_source="obs")
    assert e.calls <= cap and len(seen) <= cap + 2  # + the uncounted start and final evaluations
    assert e.status in ("maxfun", "converged") and e.u_end is not None
    u = torch.tensor(e.u_end, dtype=torch.double)
    assert (u >= b[0] - 1e-9).all() and (u <= b[1] + 1e-9).all()
    assert e.ell_end >= e.ell_start - 1e-12
    if e.calls == cap:
        assert e.status == "maxfun"


def test_nan_acquisition_reports_nan(world, monkeypatch):
    codec, snap, a, b = world

    class Bad(torch.nn.Module):
        model = snap.acquisition.model

        def forward(self, X):
            return torch.full((X.shape[0],), float("nan"), dtype=torch.double)

    class S:
        acquisition = Bad()

    e = refine_anchor(
        S, a, b, max_calls=20, seconds=None, anchor_key="k", anchor_source="obs"
    )
    assert e.status == "nan" and e.ell_end is None and e.u_end is None


def test_projection_rescores_at_canonical_u(world):
    codec, snap, a, b = world
    e = _ref(world)
    r = project(codec, snap, [e])[0]
    assert r["ell_canonical"] == snap.score([r["config"]])[0]
    assert r["u_canonical"] == tuple(codec.encode(r["config"]))
    assert r["delta"] == pytest.approx(r["ell_canonical"] - r["ell_end"])
    assert r["distance"] >= 0


def test_unusable_endpoints_skipped_and_decode_failure_rejected(world, monkeypatch):
    codec, snap, a, b = world
    e = _ref(world)
    nan = _ref(world, max_calls=20)
    nan = type(nan)(**{**nan.__dict__, "status": "nan", "u_end": None, "ell_end": None})
    assert project(codec, snap, [nan]) == []
    monkeypatch.setattr(
        codec, "decode", lambda U: (_ for _ in ()).throw(RuntimeError("no"))
    )
    r = project(codec, snap, [e])[0]
    assert r["status"] == "rejected" and "no" in r["reason"]
