"""RAG-Stack stall rule (TRACE v0.7.1): counter, HV rows, forced picks, worker wrap; q stays 4."""

import json

import pytest
import torch

import tools.prepare_trace_v070_campaign as prep
import tools.prepare_trace_v071_campaign as p71
from tests import v070_fixtures as fx
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.experiments import trace_v070_runtime as rt
from trimem.experiments import trace_v071_runtime as r71
from trimem.experiments import trace_v071_search as ts
from trimem.experiments.trace_v070_ledger import SampledLedger, own_tasks
from trimem.experiments.trace_v070_runtime import v070_patches
from trimem.optimizer.plan_surrogate import ObjectiveProtocol


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return fx.make_shared_init(tmp_path_factory.mktemp("stall"), size=40)


def make_arm(tmp_path, shared_dir, name="arm"):
    old = json.loads((shared_dir / "manifest.json").read_text())
    arm = tmp_path / name
    m = {k: v for k, v in old.items() if k not in ("initial_keys", "objective", "imported_observations")}
    m.update(protocol=f"{old['protocol']}-{name}", initial_configs=10, new_configs=70, budget=80, initialization=None,
             live={**old["live"], "run_dir": str(arm), "environment": {}}, fixed=dict(old["fixed"]))
    saved = prep.STATE["latent"]
    prep.STATE["latent"] = dict(fx.FAKE_LATENT)
    try:
        return arm, p71.arm_v071(prep.arm_import_v070, arm, m, shared_dir, old_protocol_init=True)
    finally:
        prep.STATE["latent"] = saved


class Sim:
    """An arm directory with hand-written round files and controlled configuration means."""

    def __init__(self, arm, m):
        self.arm, self.m = arm, m
        with SampledLedger(arm / "ledger.sqlite", m) as led:
            self.entries = led.complete_configs()
        self.space = driver.get_space(m)
        seen = {e["key"] for e in self.entries}
        self.free = [c for c in fx.enumerate_space(self.space) if self.space.key(c) not in seen]
        obs = ts._eligible(m, self.space, self.entries)
        self.objective = ObjectiveProtocol.from_initial(list(obs.values()))
        self.n = len(self.entries)

    def entry(self, q, tok, lat, pruned=False):
        c = self.free.pop(0)
        key = self.space.key(c)
        tasks = own_tasks(self.m, key)
        rows = [{"task_id": t, "split": "TRAIN", "protocol": self.m["protocol"], "synthetic": True, "agent_steps": 3,
                 "raw_score": q + i % 2, "total_tokens": tok + i % 2, "latency_seconds": lat + i % 2}
                for i, t in enumerate(tasks)]
        return {"key": key, "config": c, "pruned": pruned, "receipts": rows[:20] if pruned else rows}

    def round(self, picks, *, forced=False):
        archive = [e["key"] for e in self.entries]
        self.entries = [*self.entries, *picks]
        body = {"archive_keys": archive, "selected": [{"key": e["key"]} for e in picks], "force_exploration": forced}
        (self.arm / f"round_{self.n:03d}.json").write_text(json.dumps(body))
        self.n += len(picks)

    def state(self, before=None):
        return ts.stall_state(self.arm, self.m, self.entries, self.objective, before=before)

    def bad(self):
        return [self.entry(1.0, 90000, 900) for _ in range(4)]

    def great(self, q):
        return [self.entry(q, 500, 5) for _ in range(4)]


def test_counter_improve_stall_forced_reset(shared, tmp_path):
    arm, m = make_arm(tmp_path, shared)
    with v070_patches(m):
        s = Sim(arm, m)
        assert s.state()["force_exploration"] is False and s.state()["stagnant_rounds"] == 0
        s.round(s.great(500.0))
        st = s.state()
        assert st["stagnant_rounds"] == 0 and st["round_hv"] > st["initial_hv"]
        for k in (1, 2):
            s.round(s.bad())
            st = s.state()
            assert (st["stagnant_rounds"], st["force_exploration"]) == (k, False)
        s.round(s.bad())
        st = s.state()
        assert (st["stagnant_rounds"], st["force_exploration"]) == (3, True)
        s.round(s.bad(), forced=True)  # a forced round resets the counter, even without an improvement
        st = s.state()
        assert (st["stagnant_rounds"], st["force_exploration"]) == (0, False)
        s.round(s.bad())
        assert s.state()["stagnant_rounds"] == 1
        s.round(s.great(5000.0))  # an improvement resets too
        assert s.state()["stagnant_rounds"] == 0
        rows = s.state()["rounds"]
        assert [r["kind"] for r in rows] == ["ordinary"] * 4 + ["forced", "ordinary", "ordinary"]
        path = ts.write_hv_rounds(arm, s.state())
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert lines[0]["round"] == "init" and len(lines) == 8 and all(x["includes_nomemory"] for x in lines)
        assert lines[4]["force_next"] is True and lines[5]["force_next"] is False


def test_pruned_round_counts_as_stalled_and_is_pure(shared, tmp_path):
    arm, m = make_arm(tmp_path, shared, "arm_pruned")
    with v070_patches(m):
        s = Sim(arm, m)
        s.round(s.great(500.0))
        best = s.state()["best_hv"]
        s.round([s.entry(9000.0, 100, 1, pruned=True) for _ in range(4)])  # great partial means, all pruned
        st = s.state()
        assert st["stagnant_rounds"] == 1 and st["best_hv"] == best
        assert st["rounds"][-1]["picks_full_never_pruned"] == 0
        assert s.state() == st  # a pure function of the ledger + round files
        # the state at the start of an earlier round ignores later round files (replay)
        names = sorted(p.stem for p in arm.glob("round_*.json"))
        assert s.state(before=names[-1])["stagnant_rounds"] == 0


def test_nomemory_required(shared, tmp_path):
    arm, m = make_arm(tmp_path, shared, "arm_nomem")
    with v070_patches(m):
        s = Sim(arm, m)
        assert s.state()["initial_hv"] > 0
    with pytest.raises(ValueError, match="complete NoMemory"):
        ts._nomemory_obs(tmp_path / "nowhere", m)


class FakeSnapshot:
    """Variance by config index; ``score`` (the acquisition) must never run in a forced round."""

    identity = "snap"
    calls = 0

    def __init__(self, space, configs, picks=()):
        self.space, self.configs, self.picks = space, configs, list(picks)
        self.var = {space.key(c): 1.0 + i for i, c in enumerate(configs)}
        self.identity = f"snap-{len(self.picks)}"

    def posterior_moments(self, configs):
        v = torch.tensor([[self.var[self.space.key(c)]] * 3 for c in configs], dtype=torch.double)
        return torch.zeros_like(v), v

    def with_pending(self, pending):
        return FakeSnapshot(self.space, self.configs, pending)

    def score(self, configs):
        raise AssertionError("qLogNEHVI must not run in a forced round")


def test_forced_picks_sequential_deterministic_differs_from_greedy():
    space = fx.tiny_space_v070()
    pool = fx.enumerate_space(space)[:12]
    snap = FakeSnapshot(space, pool)
    out = ts.forced_picks(space, snap, pool, pool[:0], 4, {})
    assert [s["key"] for s in out] == [space.key(c) for c in pool[::-1][:4]]  # highest log-volume first
    assert [s["snapshot"] for s in out] == ["snap-0", "snap-1", "snap-2", "snap-3"]  # with_pending per pick
    assert all(s["selection_rule"] == ts.FORCED_RULE and s["ell"] is None for s in out)
    assert out[0]["conditioning"] == "marginal" and out[1]["conditioning"] == "marginal-novelty-only"
    assert ts.forced_picks(space, snap, pool, [], 4, {}) == out  # deterministic on replay
    greedy = pool[0]  # what a stand-in acquisition would rank first
    assert space.key(greedy) not in {s["key"] for s in out}


def test_forced_tie_breaks_novelty_then_key():
    space = fx.tiny_space_v070()
    pool = fx.enumerate_space(space)[:6]
    snap = FakeSnapshot(space, pool)
    snap.var = dict.fromkeys(snap.var, 1.0)
    measured = [pool[0]]
    out = ts.forced_picks(space, snap, pool[1:], measured, 1, {})[0]
    novel = {space.key(c): ts.novelty(space, c, measured) for c in pool[1:]}
    best = max(novel.values())
    assert out["key"] == max(k for k, v in novel.items() if v == best)


def test_conditioning_lowers_a_neighbour_of_an_earlier_pick():
    from botorch.models import ModelListGP, SingleTaskGP

    space = fx.tiny_space_v070()
    pool = fx.enumerate_space(space)
    codec = type("C", (), {"encode": staticmethod(lambda c: ts.plan_vector(space, c))})
    train = torch.tensor([ts.plan_vector(space, c) for c in pool[:6]], dtype=torch.double)
    y = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(*[SingleTaskGP(train, y) for _ in range(3)]).double()
    model.eval()

    class Snap:
        def __init__(self):
            self.codec, self.model = codec, model

        def posterior_moments(self, configs):
            x = torch.tensor([codec.encode(c) for c in configs], dtype=torch.double)
            post = model.posterior(x)
            return post.mean, post.variance

    a = pool[-1]
    cand = [a, pool[-2], pool[20]]
    base, how0 = ts.conditional_log_volume(Snap(), cand, [])
    cond, how1 = ts.conditional_log_volume(Snap(), cand, [a])
    assert (how0, how1) == ("marginal", "schur")
    assert cond[0] < base[0] - 3  # a itself: almost no variance left
    assert (cond <= base + 1e-6).all()  # conditioning never adds variance


def test_worker_wrap_records_flags(shared, tmp_path, monkeypatch):
    arm, m = make_arm(tmp_path, shared, "arm_worker")
    calls = {"stock": 0, "forced": 0}

    def stock_propose(*args):
        calls["stock"] += 1
        return {"selected": [{"key": "g"}]}, "trust", 0

    def forced(*args):
        calls["forced"] += 1
        return {"selected": [{"key": "f", "selection_rule": ts.FORCED_RULE}]}, "trust", 0

    seen = {}

    def stock_worker(directory, manifest, entries, agent, seed, round_file):
        seen["result"] = tls.propose_for_round(directory, round_file, manifest)[0]

    state = {"force_exploration": False, "stagnant_rounds": 1, "round_hv": 0.5, "best_hv": 0.5, "initial_hv": 0.4,
             "patience": 3, "epsilon": 1e-6, "rounds": []}
    monkeypatch.setattr(ts, "_stock_worker", stock_worker)
    monkeypatch.setattr(ts, "stall_state", lambda *a, **k: state)
    monkeypatch.setattr(ts, "forced_proposal", forced)
    monkeypatch.setattr(tls, "propose_for_round", stock_propose)
    objective = {"lo": (0, 0, 0), "span": (1, 1, 1)}
    run = {**m, "objective": objective}
    ts.latent_search_worker_v071(str(arm), run, [], [], 0, "round_010")
    r = seen["result"]
    assert (r["force_exploration"], r["stagnant_rounds"], r["round_hv"]) == (False, 1, 0.5)
    assert r["selection_rule"] == ts.GREEDY_RULE and r["selected"][0]["selection_rule"] == ts.GREEDY_RULE
    state["force_exploration"] = True
    ts.latent_search_worker_v071(str(arm), run, [], [], 0, "round_014")
    r = seen["result"]
    assert r["force_exploration"] is True and r["selection_rule"] == ts.FORCED_RULE
    assert calls == {"stock": 1, "forced": 1}
    assert tls.propose_for_round is stock_propose  # restored
    assert (arm / "hv_rounds.jsonl").exists()


def test_v071_patches_install_the_worker(shared, tmp_path):
    _, m = make_arm(tmp_path, shared, "arm_patch")
    stock = rt.latent_search_worker_v070
    run_round = driver.run_round
    with r71.v071_patches(m), v070_patches(m):
        assert driver._search_worker is ts.latent_search_worker_v071
        assert driver.run_round is not run_round
    assert rt.latent_search_worker_v070 is stock and driver.run_round is run_round
    assert m["q"] == 4 and m["v071"]["stall_rule"] == ts.RULE and ts.RULE["patience"] == 3
