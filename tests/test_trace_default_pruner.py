"""Fresh TRACE defaults and immutable historical pruning identities; no episodes."""

from types import SimpleNamespace

import pytest

from trimem.optimizer import pruning_policy as pp
from trimem.optimizer import search_analyzer as sa

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("design", [None, sa.PROTOCOL])
def test_trace_default_and_explicit_overrides(design):
    args = SimpleNamespace(search_design=design)
    block = {"sizes": [50, 100], "z": 1.96}
    assert pp.selected(pp.configure(block, args)) == "syftr-style"
    for name in pp.CHOICES:
        args.pruning_policy = name
        frozen = pp.configure(block, args)
        assert pp.selected(frozen) == name
        assert pp.configure(block, SimpleNamespace(), frozen) == frozen


@pytest.mark.parametrize("design", [sa.LEGACY, *sa.baselines.METHODS])
def test_other_searchers_retain_their_default(design):
    args = SimpleNamespace(search_design=design)
    assert pp.selected(pp.configure({}, args)) == "staged"
    args.pruning_policy = "syftr-style"
    frozen = pp.configure({}, args)
    assert pp.selected(frozen) == "syftr-style"
    assert pp.requires_isolation(frozen, {"protocol": design})


def test_legacy_off_and_missing_metadata_are_not_reinterpreted():
    assert pp.configure({}, SimpleNamespace(staging="none")) == {}
    assert pp.configure({}, SimpleNamespace(), previous={}) == {}
    assert pp.selected({}) == "staged"
    with pytest.raises(ValueError, match="requires --staging pareto"):
        pp.configure({}, SimpleNamespace(staging="none", pruning_policy="syftr-style"))


@pytest.fixture(params=["alfworld", "sciworld", "tau2"])
def case(request, tmp_path, monkeypatch):
    ds = request.param
    if ds == "tau2":
        from trimem.experiments.tau2 import pool_profile as d

        monkeypatch.setattr(d, "OUT", tmp_path / ds)
        monkeypatch.delenv("TAU2_POOL_RUN_TAG", raising=False)
        panel = d.load_panel()

        def parse(*flags):
            return d.build_parser().parse_args(["profile", *flags])

        def load(args):
            return d.load_state(panel, args)
    else:
        from trimem.experiments import bo_pool_driver as d

        monkeypatch.setattr(d, "OUT", tmp_path / ds)
        monkeypatch.delenv(d.RUN_TAG_ENV, raising=False)
        # Loading empty state need not change the process-wide GP's sample map.
        monkeypatch.setattr(d, "apply_sample_kinds", lambda state: None)
        panel = {"pool_hash": "unit", "size": 200}

        def parse(*flags):
            return d.build_parser().parse_args(["profile", "--dataset", ds, *flags])

        def load(args):
            return d.load_state(ds, panel, args)

    def save(path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        d.save(path, state)

    return SimpleNamespace(ds=ds, parse=parse, load=load, save=save)


def test_fresh_driver_default_persisted_without_extra_flags(case):
    state, path = case.load(case.parse())
    assert state["search_design"]["protocol"] == sa.PROTOCOL
    assert state["search_design"]["coverage_sampler"] == "lhs"
    assert state["search_design"]["model"] == sa.MODEL
    block = state["staging"]
    assert pp.selected(block) == "syftr-style"
    assert block["z"] == pp.SYFTR_Z and block["plan_guard"] is False
    assert not pp.requires_isolation(block, state["search_design"])
    case.save(path, state)
    resumed, _ = case.load(case.parse())
    assert resumed["staging"] == block
    with pytest.raises(ValueError, match="fresh isolated"):
        case.load(case.parse("--pruning_policy", "staged"))


def test_existing_three_channel_staged_is_never_silently_switched(case):
    state, path = case.load(case.parse("--pruning_policy", "staged"))
    assert "pruning_policy" not in state["staging"]
    case.save(path, state)
    resumed, _ = case.load(case.parse())
    assert resumed["staging"] == state["staging"]
    assert pp.selected(resumed["staging"]) == "staged"
    with pytest.raises(ValueError, match="fresh isolated"):
        case.load(case.parse("--pruning_policy", "syftr-style"))
    if case.ds != "tau2":
        del state["staging"]  # historical ALF/Sci state predating staging metadata
        case.save(path, state)
        resumed, _ = case.load(case.parse())
        assert pp.selected(resumed["staging"]) == "staged"
