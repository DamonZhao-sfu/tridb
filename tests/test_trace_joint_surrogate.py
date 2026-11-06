import pytest
import torch

from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_schema import JointSpace
from trimem.optimizer.plan_surrogate import (
    AcquisitionSnapshot,
    ObjectiveProtocol,
    aggregate_panel,
    fit_model,
)


def panel(scores=(-10, 100)):
    return [
        {
            "task_id": str(i),
            "split": "TRAIN",
            "protocol": "test",
            "raw_score": s,
            "total_tokens": 10 + i,
            "latency_seconds": 1 + i,
        }
        for i, s in enumerate(scores)
    ]


def test_raw_negative_and_complete_panel():
    space = JointSpace.live()
    obs = aggregate_panel(
        space, space.default(), panel(), task_ids=["0", "1"], protocol="test"
    )
    assert obs.mean[0] == 45
    assert obs.target[1] == pytest.approx(-torch.log1p(torch.tensor(10.5)).item())
    for rows in [
        panel()[:1],
        panel() + panel()[:1],
        [dict(r, split="TEST") for r in panel()],
        [dict(r, raw_score=None, quality=0.8) for r in panel()],
    ]:
        with pytest.raises(ValueError):
            aggregate_panel(
                space, space.default(), rows, task_ids=["0", "1"], protocol="test"
            )


def test_frozen_transform_and_real_pending_snapshot():
    space = JointSpace.live()
    codec = JointCodec(space)
    configs = space.sample(5, seed=23)
    obs = [
        aggregate_panel(
            space, c, panel((i * 10, 80)), task_ids=["0", "1"], protocol="test"
        )
        for i, c in enumerate(configs)
    ]
    objective = ObjectiveProtocol.from_initial(obs)
    assert objective.span[0] == 20
    model = fit_model(codec, obs, objective, maxiter=2)
    snap = AcquisitionSnapshot(codec, model, obs, objective, seed=3, mc_samples=8)
    candidates = space.sample(3, seed=8)
    scores = snap.score(candidates)
    assert len(scores) == 3 and all(torch.isfinite(torch.tensor(scores)))
    assert snap.score(candidates) == scores
    assert snap.unique_inputs == 3
    pending = snap.with_pending([candidates[0]])
    assert pending.identity != snap.identity and pending.unique_inputs == 0
    assert pending.acquisition.X_pending.shape == (1, 24)
    assert len(pending.score(candidates[1:])) == 2
