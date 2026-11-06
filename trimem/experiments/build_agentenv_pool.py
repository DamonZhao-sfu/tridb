"""Profiling POOL for the ALFWorld / ScienceWorld pool-profiling round (user design 2026-10-19,
docs/trimem_pool_profiling_plan_v0.1.0.md).

Pool = every non-test task of the dataset in the trimem_agentenv task table (``NOT held_out``,
``agenteval_item IS NULL``, key different from every AgentEval test task's key), each mapped to an
AgentGym env-server item id:

* ALFWorld: the AgentGym server indexes only ``mappings_train.json`` (2,420 train games, ids 0..2419)
  and the 200 test games (2420..2619). The other 1,372 train + 140 valid_seen + 134 valid_unseen tasks
  are APPENDED to the server's game list (ids 2620..; ``alfworld_extra_games.json``, loaded by
  ``env_server_managed``; append-only, the official ids are unchanged).
* ScienceWorld: the AgentGym server indexes 4,639 (taskName, variation) games: every variation of
  every task except seven task types it excludes (grow-plant, grow-fruit, the two mendelian-genetics and
  the three inclined-plane tasks). ``--sciworld_scope agentgym`` (default) = the addressable non-test
  tasks (4,144), which cover every task type of the test set; ``all`` appends the 2,568 excluded-type
  tasks too (ids 4639..; ``sciworld_extra_games.json``) for the full 6,712.

Each profiled configuration draws its own stratified random sample from the pool (``draw_sample``:
strata = task_type, proportional largest-remainder allocation with at least one per stratum, per-stratum
``random.Random(seed)`` draw), so samples of different configurations are independent.

    .venv/bin/python -m trimem.experiments.build_agentenv_pool [--datasets alfworld,sciworld] \\
        [--sciworld_scope agentgym|all]
writes data/external/panels/<ds>_pool.json (+ data/external/agentgym/<ds>_extra_games.json when needed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments.build_agentenv_panels import AG, DSN, OUT, allocate

ROOT = Path(__file__).resolve().parents[2]
# Official AgentGym server game-list sizes (train + test mappings / sciworld_games.json).
BASE_COUNT = {"alfworld": 2620, "sciworld": 4639}
STRATUM = "task_type"


def extra_games_path(ds: str) -> Path:
    return AG / f"{ds}_extra_games.json"


def pool_path(ds: str) -> Path:
    return OUT / f"{ds}_pool.json"


def official_index(ds: str) -> dict[str, int]:
    """task key -> AgentGym server item index of the official game list."""
    if ds == "alfworld":
        out = {}
        for name, off in (("mappings_train.json", 0), ("mappings_test.json", None)):
            for i, m in enumerate(json.load(open(AG / name))):
                split = "train" if off is not None else "valid_train"
                idx = i if off is not None else int(m["item_id"])
                out[f"alfworld:{split}/{m['task_type']}/{m['task_id']}"] = idx
        if len(out) != BASE_COUNT[ds]:
            raise SystemExit(
                f"alfworld official game list has {len(out)} entries, expected {BASE_COUNT[ds]}"
            )
        return out
    games = json.load(open(AG / "sciworld_games.json"))
    if len(games) != BASE_COUNT[ds]:
        raise SystemExit(
            f"sciworld_games.json has {len(games)} entries, expected {BASE_COUNT[ds]}"
        )
    return {
        f"sciworld:{g['taskName']}/{g['variationIdx']}": i for i, g in enumerate(games)
    }


def extra_game(ds: str, key: str):
    """Server game entry of an appended task: ALFWorld = path relative to $ALFWORLD_DATA/json_2.1.1;
    ScienceWorld = {taskName, variationIdx}."""
    rel = key.split(":", 1)[1]
    if ds == "alfworld":
        return rel
    name, var = rel.rsplit("/", 1)
    return {"taskName": name, "variationIdx": int(var)}


def build_pool(
    ds: str,
    tasks: list[dict],
    test_items: list[str],
    test_keys: set[str],
    *,
    sciworld_scope: str = "agentgym",
) -> tuple[list[dict], list, dict]:
    """Pure: task rows (key, split, held_out, agenteval_item, task_type) -> (pool items, extra games,
    exclusion counts). Extra games are appended in sorted key order (deterministic ids)."""
    if sciworld_scope not in ("agentgym", "all"):
        raise ValueError("sciworld_scope must be agentgym or all")
    official = official_index(ds)
    test_idx = {int(i.rsplit("_", 1)[1]) for i in test_items}
    excluded: Counter = Counter()
    keep, extra_keys = [], []
    for t in sorted(tasks, key=lambda t: t["key"]):
        if t["held_out"] or t.get("agenteval_item"):
            excluded["held_out_or_test_item"] += 1
            continue
        if t["key"] in test_keys:
            excluded["test_task_key"] += 1
            continue
        if t["key"] in official:
            if official[t["key"]] in test_idx:
                excluded["official_test_index"] += 1
                continue
            keep.append((t, official[t["key"]]))
        elif ds == "alfworld" or sciworld_scope == "all":
            extra_keys.append(t)
        else:
            excluded["not_on_agentgym_server"] += 1
    extra = []
    for j, t in enumerate(extra_keys):
        keep.append((t, BASE_COUNT[ds] + j))
        extra.append(extra_game(ds, t["key"]))
    items = [
        {
            "item_id": f"{ds}_{idx}",
            "task_key": t["key"],
            "stratum": str(t[STRATUM]),
            "split": t["split"],
            "server": "official" if idx < BASE_COUNT[ds] else "appended",
        }
        for t, idx in keep
    ]
    items.sort(key=lambda x: int(x["item_id"].rsplit("_", 1)[1]))
    ids = [x["item_id"] for x in items]
    if len(set(ids)) != len(ids) or set(ids) & set(test_items):
        raise SystemExit(f"{ds}: pool ids collide or contain a test item")
    return items, extra, dict(excluded)


def draw_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified random sample of ``n`` pool items (strata = ``stratum``): proportional largest-remainder
    allocation (>= 1 per stratum), per-stratum draw with one ``random.Random(seed)`` over sorted strata.
    Deterministic in (pool, n, seed); returned in item-id order."""
    by = defaultdict(list)
    for x in pool:
        by[x["stratum"]].append(x)
    if n > len(pool):
        raise ValueError(f"sample of {n} from a pool of {len(pool)}")
    quota = allocate({k: len(v) for k, v in by.items()}, n)
    while (
        sum(quota.values()) < n
    ):  # a stratum hit its size: give the rest to the largest remainders
        room = [k for k in quota if quota[k] < len(by[k])]
        k = max(room, key=lambda k: len(by[k]) * n / len(pool) - quota[k])
        quota[k] += 1
    rnd = random.Random(seed)
    out = []
    for k in sorted(by):
        items = sorted(by[k], key=lambda x: x["item_id"])
        out += rnd.sample(items, quota[k])
    out.sort(key=lambda x: int(x["item_id"].rsplit("_", 1)[1]))
    if len(out) != n:
        raise AssertionError("allocation did not sum to n")
    return out


def sample_seed(experiment: str, label: str, rep: int = 0) -> int:
    """Per-configuration sample seed: deterministic in (experiment, config key / arm label, repeat), so a
    resumed run redraws the identical sample, and different configurations get independent samples."""
    return int(
        hashlib.sha256(f"{experiment}|{label}|{rep}".encode()).hexdigest()[:8], 16
    )


def strata_weights(pool: list[dict]) -> dict[str, float]:
    c = Counter(x["stratum"] for x in pool)
    return {k: v / len(pool) for k, v in sorted(c.items())}


def build(ds: str, sciworld_scope: str = "agentgym") -> dict:
    import psycopg
    from psycopg.rows import dict_row

    test_items = [x["item_id"] for x in json.load(open(AG / f"{ds}_test.json"))]
    with psycopg.connect(DSN, row_factory=dict_row) as c:
        tasks = c.execute(
            "SELECT key, split, held_out, agenteval_item, task_type, topic FROM task WHERE env = %s",
            (ds,),
        ).fetchall()
        test_keys = {
            r["key"]
            for r in c.execute(
                "SELECT key FROM task WHERE agenteval_item = ANY(%s)", (test_items,)
            ).fetchall()
        }
    if len(test_keys) != len(test_items):
        raise SystemExit(
            f"{ds}: {len(test_keys)} test task keys for {len(test_items)} test items"
        )
    items, extra, excluded = build_pool(
        ds, tasks, test_items, test_keys, sciworld_scope=sciworld_scope
    )
    non_test = sum(
        1 for t in tasks if not t["held_out"] and not t.get("agenteval_item")
    )
    if extra:
        eg = {
            "env": ds,
            "base_count": BASE_COUNT[ds],
            "games": extra,
            "note": "appended to the AgentGym server game list by env_server_managed (ids base_count..)",
            "built_at": datetime.now(UTC).isoformat(),
        }
        extra_games_path(ds).write_text(json.dumps(eg, indent=1))
    ids = [x["item_id"] for x in items]
    pool = {
        "dataset": ds,
        "kind": "profiling-pool",
        "size": len(items),
        "stratified_by": STRATUM,
        "sciworld_scope": sciworld_scope if ds == "sciworld" else None,
        "non_test_tasks_in_table": non_test,
        "excluded_counts": excluded,
        "splits": sorted({x["split"] for x in items}),
        "strata": dict(Counter(x["stratum"] for x in items)),
        "appended_games": len(extra),
        "extra_games_file": str(extra_games_path(ds).relative_to(ROOT))
        if extra
        else None,
        "extra_games_sha": hashlib.sha256(json.dumps(extra).encode()).hexdigest()[:16]
        if extra
        else None,
        "test_items_excluded": len(test_items),
        "pool_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        "built_at": datetime.now(UTC).isoformat(),
        "items": items,
    }
    pool_path(ds).write_text(json.dumps(pool, indent=1))
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="alfworld,sciworld")
    ap.add_argument("--sciworld_scope", default="agentgym", choices=("agentgym", "all"))
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        p = build(ds, a.sciworld_scope)
        print(
            ds,
            "pool",
            p["size"],
            "of",
            p["non_test_tasks_in_table"],
            "non-test tasks; appended",
            p["appended_games"],
            "; excluded",
            p["excluded_counts"],
            "; splits",
            p["splits"],
            "; strata",
            len(p["strata"]),
            "; hash",
            p["pool_hash"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
