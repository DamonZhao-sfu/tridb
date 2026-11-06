"""SemFilter v2c (TRACE v0.7.1): compact card, TOPN 16, episode reuse, fail-closed gates, worker command, stock path."""

import json
import os
import re
import subprocess

import pytest

from tests.test_v070_nomemory import FakePopen, fake, run_manifest  # noqa: F401 - fixture
from tools import semfilter_v071_measure as measure
from tools import trace_joint_live as live
from tools import trace_v071_episode_worker as ew
from tools import trace_v071_hash_guard as hg71
from trimem.experiments import memory_policy as mp
from trimem.experiments import semantic_rubric as sr
from trimem.experiments import semfilter_v071 as v2c
from trimem.experiments import semfilter_v2 as sf
from trimem.experiments import trace_v071_runtime as r71
from trimem.experiments.session_retrieval import SessionConfigV2, retrieve
from trimem.experiments.trace_v070_runtime import space_patches
from trimem.optimizer.joint_search_ledger import RetryableEpisodeError
from trimem.optimizer.plan_schema_v070 import JointSpaceV070


def card(cid="s1", *, success=False, steps=None, env="alfworld"):
    steps = steps or [
        {"ord": 0, "action": "go to fridge 1", "observation": "The fridge 1 is closed."},
        {"ord": 1, "action": "open fridge 1", "observation": "You open the fridge 1. " + "It holds many things. " * 20},
        {"ord": 2, "action": "look", "observation": "Nothing happens."},
        {"ord": 3, "action": "look", "observation": "Nothing happens."},
        {"ord": 4, "action": "look", "observation": "Nothing happens."},
        {"ord": 5, "action": "take egg 3", "observation": "You pick up the egg 3."},
    ]
    return {"id": cid, "env": env, "success": success, "final_score": 0.0, "n_steps": len(steps),
            "description": "heat some egg and put it in garbagecan.", "memory_policy": mp.SUCCESS_FAILURE, "steps": steps}


def test_compact_card_shorter_keeps_label_goal_actions():
    c = card()
    raw, new = sr.env_card(c), v2c.compact_card(c)
    assert len(new) < len(raw)
    assert new.startswith(mp.FAILED_TAG) and "do not repeat" not in new
    assert v2c.compact_card(card(success=True)).startswith(mp.SUCCESS_TAG)
    assert "Goal: heat some egg" in new and "Outcome: score 0.0 in 6 steps" in new
    for s in c["steps"]:
        assert s["action"] in new
    assert "3-5. look (x3)" in new and "Nothing happens" not in new
    assert max(len(line.split(" -> ")[1]) for line in new.splitlines() if " -> " in line) <= v2c.OBS_CHARS


def test_compact_card_hard_cap_keeps_both_ends():
    steps = [{"ord": i, "action": f"do thing number {i}", "observation": f"result {i} " + "x" * 80} for i in range(200)]
    new = v2c.compact_card(card(steps=steps))
    assert len(new) <= v2c.CARD_CAP and "steps omitted" in new
    assert "do thing number 0" in new and "do thing number 199" in new


@pytest.fixture
def gate(monkeypatch):
    for k, v in v2c.ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    assert v2c.install()
    yield
    v2c.uninstall()


class FakeJudge:
    """httpx.post: scores every PAST EPISODE id; records which ids each request judged."""

    def __init__(self, monkeypatch):
        self.ids, self.users = [], []
        monkeypatch.setattr(sf.httpx, "post", self)

    def __call__(self, url, json=None, timeout=None):
        user = json["messages"][1]["content"]
        ids = re.findall(r"PAST EPISODE (\S+?):?\s*\n", user) or re.findall(r"PAST EPISODE (\S+)", user)
        self.ids.append(ids)
        self.users.append(user)
        content = {"score": 7} if len(ids) == 1 else dict.fromkeys(ids, 7)
        body = {"usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
                "choices": [{"message": {"content": __import__("json").dumps(content)}}]}
        return type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: body})()


def build(threshold=0.6, env=None):
    cfg = sf.settings(env)
    return sf.build("alfworld", "the new task", cfg, base="http://judge", version="semantic-rubric-v2", workers=1,
                    threshold=threshold)


def test_reuse_across_periodic_retrievals(gate, monkeypatch, tmp_path):
    judge = FakeJudge(monkeypatch)
    v2c.LOG["path"] = str(tmp_path / "semfilter_v2c.jsonl")
    sem, view, proto = build(0.6)
    assert proto["protocol"] == "semfilter-v2c" and proto["top_n"] == 16 and proto["judge_input"] == "compact-v2c"
    first = sem([card("a"), card("b"), card("c")])
    assert all(d["useful"] for d in first)  # score 0.7 >= 0.6
    r1 = sf.receipt_block(proto, sem, view, {})
    assert (r1["cache_hits"], r1["judged_new"], r1["tokens_saved_est"]) == (0, 3, 0.0)
    judged_before = [i for ids in judge.ids for i in ids]
    assert sorted(judged_before) == ["a", "b", "c"]

    sem2, view2, proto2 = build(0.8)  # a later retrieval of the same episode, another threshold
    second = sem2([card("b"), card("c"), card("d")])
    assert not any(d["useful"] for d in second)  # the reused score meets each retrieval's own threshold
    assert [d["score"] for d in second] == [0.7, 0.7, 0.7]
    assert [i for ids in judge.ids for i in ids][len(judged_before):] == ["d"]  # only the unseen candidate
    r2 = sf.receipt_block(proto2, sem2, view2, {"retrieval_id": "r2"})
    assert (r2["cache_hits"], r2["judged_new"]) == (2, 1) and r2["tokens_saved_est"] > 0
    assert "Nothing happens" not in judge.users[0] and "3-5. look (x3)" in judge.users[0]  # the compact card is sent
    lines = [json.loads(x) for x in (tmp_path / "semfilter_v2c.jsonl").read_text().splitlines()]
    assert [x["cache_hits"] for x in lines] == [0, 2] and lines[1]["retrieval_id"] == "r2"


def test_reuse_key_has_rubric_and_reuse_can_be_off(gate, monkeypatch):
    judge = FakeJudge(monkeypatch)
    build()[0]([card("a")])
    other = {**os.environ, sf.ENV_RUBRIC: "semantic-rubric-v2"}
    sem, _, _ = build(env=other)
    sem([card("a")])  # another rubric: a miss
    assert sem.v2c["cache_hits"] == 0 and len(judge.ids) == 2
    monkeypatch.setenv(v2c.ENV_REUSE, "off")
    sem, _, _ = build()
    sem([card("a")])
    assert sem.v2c == {"cache_hits": 0, "judged_new": 1, "reuse": "off"} and len(judge.ids) == 3


class Adapter:
    def __init__(self, n=40):
        self.s = [{"id": f"s{i:02d}", "cosine": 1 - i / 100} for i in range(n)]

    def seeds(self, count):
        return self.s[:count]

    def neighbors(self, session, edge, fanout):
        return []

    def allowed(self, session):
        return True

    def hydrate(self, session):
        return {"id": session["id"]}


@pytest.mark.parametrize("plan", ["P1", "P2"])
def test_topn_16_judged_count(gate, plan):
    assert sf.settings()["top_n"] == 16
    judged = []

    def semfilter(cards):
        judged.extend(c["id"] for c in cards)
        return [{"id": c["id"], "useful": True} for c in cards]

    config = SessionConfigV2(plan=plan, seed_count=40, depth=1, fanout=4, top_k=3, edges=("similar",))
    receipt = retrieve(Adapter(), config, semfilter=semfilter, prefilter_top_n=sf.settings()["top_n"])
    assert judged == [f"s{i:02d}" for i in range(16)]  # the 16 highest cosines
    assert len(receipt["semfilter_prefilter"]["dropped_ids"]) == 24


def test_stock_path_unchanged_and_fail_closed(monkeypatch):
    monkeypatch.delenv(v2c.ENV_GATE, raising=False)
    stock = (sf.settings, sf.build, sf.receipt_block)
    assert not v2c.install() and (sf.settings, sf.build, sf.receipt_block) == stock
    env = {sf.ENV_GATE: "1", sf.ENV_JUDGE_INPUT: v2c.JUDGE_INPUT}
    with pytest.raises(ValueError, match=sf.ENV_JUDGE_INPUT):  # a v0.7.1 run without the wrapper stops
        sf.settings(env)
    assert sf.settings({sf.ENV_GATE: "1"})["top_n"] == 32  # stock default unchanged


def test_episode_worker_gates(monkeypatch, tmp_path):
    monkeypatch.delenv(v2c.ENV_GATE, raising=False)
    with pytest.raises(SystemExit, match=v2c.ENV_GATE):
        ew.main()
    f = tmp_path / "input.json"
    f.write_text(json.dumps([{"item_id": "alfworld_1"}]))
    assert ew.check_single_item(["x", "--inference_file", str(f)]) == "alfworld_1"
    f.write_text(json.dumps([{"item_id": "alfworld_1"}, {"item_id": "alfworld_2"}]))
    with pytest.raises(SystemExit, match="one item per episode"):
        ew.check_single_item(["x", "--inference_file", str(f)])


def v071_manifest(tmp_path):
    m = run_manifest(tmp_path)
    m["smoke"] = True
    m["v071"] = r71.block()
    m["fixed"]["semfilter"] = dict(v2c.BLOCK)
    m["live"]["environment"].update(v2c.ENVIRONMENT)
    return m


def test_worker_command_rewritten_only_for_v071(tmp_path, fake):  # noqa: F811
    fake.write, fake.code = False, 1
    config = JointSpaceV070.live().default(rank_impl="none", filter_order="after")
    m = v071_manifest(tmp_path)
    plain = run_manifest(tmp_path)
    saved = live.subprocess
    with space_patches():
        with r71.v071_patches(plain), pytest.raises(RetryableEpisodeError):
            live.evaluate(config, "alfworld_3", plain)
        assert live.subprocess is saved
        with r71.v071_patches(m), pytest.raises(RetryableEpisodeError):
            live.evaluate(config, "alfworld_5", m)
    assert live.subprocess is saved and live.subprocess is subprocess
    (stock, _), (new, env) = fake.seen
    assert stock[2] == "tools.trace_joint_episode_worker" and new[2] == "tools.trace_v071_episode_worker"
    assert stock[3:] != [] and new[3] == stock[3] == "--plan" and env["TRIMEM_SEMFILTER_TOPN"] == "16"
    assert env[v2c.ENV_GATE] == "1"
    assert r71.tag_receipt(lambda *a: {"x": 1})()["semfilter_protocol"] == "semfilter-v2c"
    assert r71.semfilter_mode(m) == "semfilter-v2c" and r71.semfilter_mode(plain) == "semfilter-v2"


def test_validate_v071_manifest(tmp_path):
    m = v071_manifest(tmp_path)
    r71.validate_v071_manifest(m)
    for bad in ({**m, "fixed": {**m["fixed"], "semfilter": {}}},
                {**m, "live": {**m["live"], "environment": {**m["live"]["environment"], "TRIMEM_SEMFILTER_TOPN": "32"}}},
                {**m, "v071": {**m["v071"], "stall_rule": {"patience": 9}}}):
        with pytest.raises(ValueError):
            r71.validate_v071_manifest(bad)


def test_replay_episode_counts():
    recs = [{"semfilter_prefilter": {"judged_ids": [f"c{i}" for i in range(20)]}},
            {"semfilter_prefilter": {"judged_ids": ["c0", "c1", "x", "c19"]}}]
    out = measure.replay_episode(recs, 16)
    assert (out["stock"], out["top"], out["reuse"]) == (24, 20, 18)  # c19 was beyond TOP16 in the first retrieval
    assert (out["periodic_seen"], out["periodic_hits"]) == (4, 2)


def test_v071_hash_guard():
    r = hg71.report()
    assert r["ok"], r["violations"]
    live_init = "results/trace_v070_alfworld_init_20261105"
    assert live_init in r["started_v070_runs"] or not (hg71.hg.ROOT / live_init / "episodes").exists()


def test_v071_guard_flags_a_started_run_mismatch(monkeypatch):
    monkeypatch.setattr(hg71, "started_v070_runs", lambda: ["results/fake"])
    monkeypatch.setattr(hg71.hg, "report", lambda: {"ok": True, "protected_runs": 0, "violations": []})
    monkeypatch.setattr(hg71.hg, "protected", list)
    monkeypatch.setattr(hg71.hg, "mismatches", lambda run: ["/x/hashed.py"])
    monkeypatch.setattr(hg71.hg, "hashed", lambda run: {})
    r = hg71.report()
    assert not r["ok"] and r["violations"] == ["started v0.7.0 run results/fake: /x/hashed.py"]
