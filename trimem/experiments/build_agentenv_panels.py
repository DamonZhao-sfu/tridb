"""Frozen 100-query training panels for the ALFWorld / ScienceWorld BO profiling.

Built once from the AgentGym training lists, stratified (ALFWorld: 6 task types; ScienceWorld:
topics) with proportional largest-remainder allocation and a fixed seed. Every panel item is an
AgentGym server item id whose task is a stored *training* task (split=train, not held out); test
items (the 200 AgentEval items and anything held out) are excluded and listed.

    .venv/bin/python -m trimem.experiments.build_agentenv_panels [--check-server]
writes data/external/panels/{alfworld,sciworld}_train100.json

Extension (user 2026-10-17): ``--extend 500`` writes ``{ds}_train500.json`` = the frozen 100 items (same
order, ids and task keys, first) + 400 new training tasks drawn with the same rules: the candidate pool,
strata and per-stratum shuffles of the 100-panel (seed 20261016; the 100 items are the head of each
stratum's shuffled pool, so the draw is nested), the proportional largest-remainder allocation now
computed for 500 (never below a stratum's count in the 100-panel), the AgentEval test items and any
task whose key equals a test task's key excluded, and (``--check-server``, only when the env servers
are idle) loadability-checked with spare replacement.

    .venv/bin/python -m trimem.experiments.build_agentenv_panels --extend 500 [--check-server]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
AG = ROOT / "data/external/agentgym"
OUT = ROOT / "data/external/panels"
DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_agentenv"
SEED = 20261016
SIZE = 100
SERVERS = {"alfworld": "http://127.0.0.1:36001", "sciworld": "http://127.0.0.1:36002"}


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation (largest remainder), at least 1 per stratum when possible."""
    total = sum(counts.values())
    raw = {k: n * c / total for k, c in counts.items()}
    out = {k: max(1, int(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if sum(out.values()) >= n:
            break
        out[k] += 1
    while sum(out.values()) > n:  # undo the min-1 overshoot from the largest strata
        k = max(out, key=lambda k: out[k])
        out[k] -= 1
    return {k: min(v, counts[k]) for k, v in out.items()}


def candidates(ds: str, tasks: dict[str, dict]) -> tuple[list[dict], dict]:
    test = {x["item_id"] for x in json.load(open(AG / f"{ds}_test.json"))}
    excluded = defaultdict(list)
    out = []
    if ds == "alfworld":
        maps = json.load(open(AG / "mappings_train.json"))
        train_ids = [x["item_id"] for x in json.load(open(AG / "alfworld_train.json"))]
        for idx, m in enumerate(maps):
            item = f"alfworld_{idx}"
            assert train_ids[idx] == f"{m['task_type']}_{m['task_id']}", (
                "train list / mapping order differs"
            )
            key = f"alfworld:train/{m['task_type']}/{m['task_id']}"
            t = tasks.get(key)
            if item in test:
                excluded["test_item"].append(item)
            elif t is None or t["held_out"] or t["split"] != "train":
                excluded["not_a_stored_train_task"].append(item)
            else:
                out.append(
                    {"item_id": item, "task_key": key, "stratum": t["task_type"]}
                )
    else:
        games = json.load(open(AG / "sciworld_games.json"))
        for x in json.load(open(AG / "sciworld_train.json")):
            item = x["item_id"]
            g = games[int(item.split("_")[1])]
            key = f"sciworld:{g['taskName']}/{g['variationIdx']}"
            t = tasks.get(key)
            if item in test:
                excluded["test_item"].append(item)
            elif t is None or t["held_out"] or t["split"] != "train":
                excluded["not_a_stored_train_task"].append(item)
            else:
                out.append({"item_id": item, "task_key": key, "stratum": t["topic"]})
    return out, excluded


def check_server(ds: str, items: list[dict]) -> list[str]:
    import requests

    base, bad = SERVERS[ds], []
    eid = requests.post(f"{base}/create", timeout=120).json()["id"]
    try:
        for it in items:
            idx = int(it["item_id"].split("_")[1])
            body = (
                {"id": eid, "game": idx, "world_type": "Text"}
                if ds == "alfworld"
                else {"id": eid, "data_idx": idx}
            )
            r = requests.post(f"{base}/reset", json=body, timeout=300).json()
            if not isinstance(r, dict) or "error" in r:
                bad.append(it["item_id"])
    finally:
        requests.post(f"{base}/close", json={"id": eid}, timeout=60)
    return bad


def build(ds: str, check: bool) -> dict:
    with psycopg.connect(DSN) as c:
        rows = c.execute(
            "SELECT key, split, held_out, task_type, topic FROM task WHERE env = %s",
            (ds,),
        ).fetchall()
    tasks = {
        r[0]: {"split": r[1], "held_out": r[2], "task_type": r[3], "topic": r[4]}
        for r in rows
    }
    cands, excluded = candidates(ds, tasks)
    rnd = random.Random(SEED)
    by = defaultdict(list)
    for x in cands:
        by[x["stratum"]].append(x)
    quota = allocate({k: len(v) for k, v in by.items()}, SIZE)
    picked, unloadable, spare = [], [], {}
    for k in sorted(by):
        pool = sorted(by[k], key=lambda x: x["item_id"])
        rnd.shuffle(pool)
        picked += pool[: quota[k]]
        spare[k] = pool[quota[k] :]
    if check:  # replace items the env server cannot load with the next spare of the same stratum
        bad = set(check_server(ds, picked))
        while bad:
            unloadable += sorted(bad)
            keep = [x for x in picked if x["item_id"] not in bad]
            repl = []
            for x in picked:
                if x["item_id"] in bad and spare[x["stratum"]]:
                    repl.append(spare[x["stratum"]].pop(0))
            bad = set(check_server(ds, repl))
            picked = keep + repl
    picked.sort(key=lambda x: int(x["item_id"].split("_")[1]))
    ids = [x["item_id"] for x in picked]
    if len(ids) != SIZE or len(set(ids)) != SIZE:
        raise SystemExit(
            f"{ds}: panel has {len(set(ids))} distinct items, need exactly {SIZE}"
        )
    panel = {
        "dataset": ds,
        "split": "train",
        "size": len(picked),
        "seed": SEED,
        "source": f"data/external/agentgym/{ds}_train.json (AgentGym training list)",
        "stratified_by": "task_type" if ds == "alfworld" else "topic",
        "quota": quota,
        "server_checked": check,
        "unloadable_replaced": unloadable,
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "excluded_test_items": sorted(excluded.get("test_item", [])),
        "panel_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        "items": picked,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ds}_train100.json").write_text(json.dumps(panel, indent=1))
    return panel


def shuffled_strata(cands: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """Per-stratum pools in the exact order ``build`` draws them (sorted strata, one Random(seed))."""
    rnd = random.Random(seed)
    by = defaultdict(list)
    for x in cands:
        by[x["stratum"]].append(x)
    out = {}
    for k in sorted(by):
        pool = sorted(by[k], key=lambda x: x["item_id"])
        rnd.shuffle(pool)
        out[k] = pool
    return out


def extend_quota(
    counts: dict[str, int], base: dict[str, int], size: int
) -> dict[str, int]:
    """Totals per stratum for the extended panel: proportional allocation for ``size``, raised to the
    base panel's count where needed (the frozen items stay), then trimmed from the largest strata."""
    q = allocate(counts, size)
    q = {k: max(q.get(k, 0), base.get(k, 0)) for k in counts}
    while sum(q.values()) > size:
        k = max(
            (k for k in q if q[k] > base.get(k, 0)), key=lambda k: q[k] - base.get(k, 0)
        )
        q[k] -= 1
    while sum(q.values()) < size:  # only if a stratum hit its candidate count
        room = [k for k in q if q[k] < counts[k]]
        if not room:
            raise SystemExit("not enough training candidates")
        k = max(room, key=lambda k: counts[k] * size / sum(counts.values()) - q[k])
        q[k] += 1
    return q


def draw_extension(
    cands: list[dict],
    base_items: list[dict],
    size: int,
    test_keys: set[str],
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """New items (``size - len(base)``) drawn with the base panel's rules; pure (no DB / server)."""
    base_ids = {x["item_id"] for x in base_items}
    pools = shuffled_strata(cands, seed)
    base_n = defaultdict(int)
    for x in base_items:
        base_n[x["stratum"]] += 1
    nested = all(
        {x["item_id"] for x in pools.get(k, [])[: base_n[k]]}
        == {x["item_id"] for x in base_items if x["stratum"] == k}
        for k in base_n
    )
    quota = extend_quota({k: len(v) for k, v in pools.items()}, dict(base_n), size)
    new, spare, key_excluded = [], {}, []
    for k, pool in pools.items():
        rest = [x for x in pool if x["item_id"] not in base_ids]
        ok = []
        for x in rest:
            (key_excluded if x["task_key"] in test_keys else ok).append(x)
        need = quota[k] - base_n[k]
        new += ok[:need]
        spare[k] = ok[need:]
    return new, {
        "quota_total": quota,
        "quota_new": {k: quota[k] - base_n[k] for k in quota},
        "nested_prefix_reproduced": nested,
        "spare": spare,
        "excluded_test_task_key": sorted(x["item_id"] for x in key_excluded),
    }


def extend(ds: str, size: int, check: bool) -> dict:
    base_p = OUT / f"{ds}_train{SIZE}.json"
    base = json.load(open(base_p))
    with psycopg.connect(DSN) as c:
        rows = c.execute(
            "SELECT key, split, held_out, task_type, topic FROM task WHERE env = %s",
            (ds,),
        ).fetchall()
        test_ids = [x["item_id"] for x in json.load(open(AG / f"{ds}_test.json"))]
        test_keys = {
            r[0]
            for r in c.execute(
                "SELECT key FROM task WHERE agenteval_item = ANY(%s)", (test_ids,)
            ).fetchall()
        }
    tasks = {
        r[0]: {"split": r[1], "held_out": r[2], "task_type": r[3], "topic": r[4]}
        for r in rows
    }
    cands, excluded = candidates(ds, tasks)
    by_id = {x["item_id"]: x for x in cands}
    for x in base[
        "items"
    ]:  # the frozen items must still be valid candidates with the same key / stratum
        if by_id.get(x["item_id"]) != x:
            raise SystemExit(
                f"{ds}: base item {x['item_id']} is no longer a candidate with the same key"
            )
    new, info = draw_extension(cands, base["items"], size, test_keys)
    unloadable = []
    if check:  # replace items the env server cannot load with the next spare of the same stratum
        bad = set(check_server(ds, new))
        while bad:
            unloadable += sorted(bad)
            repl = [
                info["spare"][x["stratum"]].pop(0)
                for x in new
                if x["item_id"] in bad and info["spare"][x["stratum"]]
            ]
            new = [x for x in new if x["item_id"] not in bad]
            bad = set(check_server(ds, repl))
            new += repl
    new.sort(key=lambda x: int(x["item_id"].split("_")[1]))
    items = list(base["items"]) + new
    ids = [x["item_id"] for x in items]
    test_set = set(test_ids)
    if len(ids) != size or len(set(ids)) != size:
        raise SystemExit(
            f"{ds}: extended panel has {len(set(ids))} distinct items, need exactly {size}"
        )
    if test_set & set(ids) or test_keys & {x["task_key"] for x in items}:
        raise SystemExit(f"{ds}: extended panel contains a test item or test task")
    panel = {
        "dataset": ds,
        "split": "train",
        "size": size,
        "seed": SEED,
        "source": base["source"],
        "stratified_by": base["stratified_by"],
        "quota": info["quota_total"],
        "quota_new": info["quota_new"],
        "server_checked": check,
        "server_check_note": (
            "env servers idle: every new item reset once"
            if check
            else "live loadability check skipped (env servers busy or not requested)"
        ),
        "unloadable_replaced": unloadable,
        "excluded_counts": {
            **{k: len(v) for k, v in excluded.items()},
            "test_task_key": len(info["excluded_test_task_key"]),
        },
        "excluded_test_items": sorted(excluded.get("test_item", [])),
        "excluded_test_task_key_items": info["excluded_test_task_key"],
        "base_panel": {
            "path": str(base_p.relative_to(ROOT)),
            "panel_hash": base["panel_hash"],
            "size": len(base["items"]),
            "position": "items[0:%d]" % len(base["items"]),
        },
        "draw": (
            "same candidate pool, strata and per-stratum Random(seed) shuffles as the base panel; new items "
            "= next unused items of each stratum's shuffled pool, quota = proportional allocation for "
            f"{size} (largest remainder, >= base count)"
        ),
        "nested_prefix_reproduced": info["nested_prefix_reproduced"],
        "built_at": datetime.now(UTC).isoformat(),
        "panel_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        "items": items,
    }
    (OUT / f"{ds}_train{size}.json").write_text(json.dumps(panel, indent=1))
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-server", action="store_true")
    ap.add_argument(
        "--extend",
        type=int,
        default=None,
        help="extend the frozen 100-panels to this size",
    )
    ap.add_argument("--datasets", default="alfworld,sciworld")
    a = ap.parse_args()
    if a.extend:
        for ds in a.datasets.split(","):
            p = extend(ds, a.extend, a.check_server)
            print(
                ds,
                p["size"],
                "hash",
                p["panel_hash"],
                "quota",
                p["quota"],
                "new",
                p["quota_new"],
                "nested",
                p["nested_prefix_reproduced"],
                "excluded",
                p["excluded_counts"],
                "unloadable",
                p["unloadable_replaced"],
            )
        return 0
    for ds in ("alfworld", "sciworld"):
        p = build(ds, a.check_server)
        print(
            ds,
            p["size"],
            "hash",
            p["panel_hash"],
            "quota",
            p["quota"],
            "excluded",
            p["excluded_counts"],
            "unloadable",
            p["unloadable_replaced"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
