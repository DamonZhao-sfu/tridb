"""TRAIN-only E2 policies and evidence exports. No episode or model calls.

The ``alfworld_*`` E2 modules are dataset-generic (ALFWorld / ScienceWorld / WebShop); the module names are
kept to avoid churn. Every dataset-specific value lives in ``DATASETS``; ALFWorld keeps its original values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from trimem.optimizer import session_space_agentenv_v2 as space

_SEMFILTER_V2 = {
    "protocol": "semfilter-v2",
    "top_n": 32,
    "batch": 8,
    "max_tokens": 96,
    "rubric": "semantic-rubric-v3",
    "judge_input": "raw",
    "guided": "json_schema strict (ids -> 11-level enum)",
}
# WebShop: identical judge settings, v2b rubric + fill-topk fallback (bo_pool_driver.semfilter_v2b_block).
_SEMFILTER_V2B = {
    **_SEMFILTER_V2,
    "protocol": "semfilter-v2b",
    "sem_fallback": "fill-topk",
    "rubric": "semantic-rubric-v4-webshop",
}
_NOMEMORY_ARGS = {"model": "Qwen/Qwen3.8-27B-FP8", "max_round": 30, "temperature": 0}
# WebShop reason strings are verbatim from results/pool_v5_20261020/run_webshop_v7.sh (user 2026-10-23).
_WEBSHOP_MAX_ROUND_REASON = (
    "user 2026-10-23: WebShop uses AgentGym's own WebShop evaluation turn limit max_round 10 "
    "(vendor/AgentGym/agentenv/examples/basic/base_eval.sh:21) from round 2 on and for every test arm; "
    "rounds 0-1 / NoMemory / noise repeat (30 turns) compared derived-truncation-10"
)
_WEBSHOP_V2B_REASON = (
    "user 2026-10-23: WebShop SemFilter-v2 passed zero cards for most queries (thr 0.8 empty 73-85%; v3 WebShop "
    "conflicts fire under product isolation) -> remove 0.8, fill-topk fallback, rubric v4-webshop, from round 1"
)
DATASETS = {
    "alfworld": {
        "name": "ALFWorld",
        "trace_state": "results/exp2_oneconfig_prepare_20261022/provenance/alfworld_state.json",
        "trace_dir": "results/bo_agentenv_20261016/alfworld_pool-strat200-v7__mem-sf1",
        # TRACE rounds < 6 ran at c128 with SemFilter v1; TRACE display rounds start at source round 6.
        "trace_from_round": 6,
        "trace_old_regime": "old c128 / SemFilter v1 regime",
        "trace_required": True,
        # None: the TRACE state's latency_regime.nomemory.dir (the c16 rerun)
        "nomemory_dir": None,
        "nomemory_size": 200,
        "nomemory_prefix": None,
        "nomemory_run_args": {
            **_NOMEMORY_ARGS,
            "task_name": "alfworld",
            "processes": 8,
            "action_format": "function_calling",
        },
        "max_round": 30,
        "semfilter": _SEMFILTER_V2,
        "semfilter_v2b": False,
        "env": {},
        "evaluation_env": {},
        "identity_extra": {},
        "env_ports": (36001, 36011),
        "validation_splits": ("train", "valid_seen", "valid_unseen"),
    },
    "sciworld": {
        "name": "ScienceWorld",
        "trace_state": "results/bo_agentenv_20261016/sciworld_pool-strat200-v7__mem-sf1/state.json",
        "trace_dir": "results/bo_agentenv_20261016/sciworld_pool-strat200-v7__mem-sf1",
        # controlled-c16-v1 and semfilter-v2 recorded from round 0: every TRACE round is the current regime
        "trace_from_round": 0,
        "trace_old_regime": "pre-c16 / pre-SemFilter-v2 regime",
        "trace_required": False,
        # 200/200 c16 NoMemory episodes (no latency_regime key in the Sci state)
        "nomemory_dir": "results/bo_agentenv_20261016/sciworld_pool-strat200-v7/NoMemory",
        "nomemory_size": 200,
        "nomemory_prefix": None,
        "nomemory_run_args": {
            **_NOMEMORY_ARGS,
            "task_name": "sciworld",
            "processes": 1,
            "action_format": "react",
        },
        "max_round": 30,
        "semfilter": _SEMFILTER_V2,
        "semfilter_v2b": False,
        "env": {},
        "evaluation_env": {},
        "identity_extra": {},
        "env_ports": (36002, 36012),
        # ScienceWorld's own DB splits (its "test" is NOT the AgentEval TEST; held_out false, agenteval_item null)
        "validation_splits": ("dev", "test", "train"),
    },
    "webshop": {
        "name": "WebShop",
        "trace_state": "results/bo_agentenv_20261016/webshop_pool-strat200-v7__mem-sf1/state.json",
        "trace_dir": "results/bo_agentenv_20261016/webshop_pool-strat200-v7__mem-sf1",
        # TRACE: semfilter-v2b from round 1, max-round-v1 (10 turns) from round 2
        "trace_from_round": 2,
        "trace_old_regime": "30-turn / pre-SemFilter-v2b regime",
        "trace_required": False,
        # 189/200 files: the reference is the complete nomemory-prefix-v1 100-task prefix (TRACE v7 baseline),
        # compared in the derived-truncation-10 view (episode_budget, recorded from round 0 here)
        "nomemory_dir": "results/bo_agentenv_20261016/webshop_pool-strat200-v7/NoMemory",
        "nomemory_size": 100,
        "nomemory_prefix": "nomemory-prefix-v1",
        "nomemory_run_args": {
            **_NOMEMORY_ARGS,
            "task_name": "webshop",
            "processes": 8,
            "action_format": "react",
        },
        "max_round": 10,
        "semfilter": _SEMFILTER_V2B,
        "semfilter_v2b": True,
        # Discovery: semfilter-v2 block (default rubric v3) + v2b from round 0 (rubric v4 exported by the driver);
        # TRAIN references keep 30 turns and are compared derived-truncation-10.
        "env": {
            "POOL_EPISODE_MAX_ROUND": "10",
            "POOL_EPISODE_MAX_ROUND_REASON": _WEBSHOP_MAX_ROUND_REASON,
            "POOL_SEMFILTER_V2B": "1",
            "POOL_SEMFILTER_V2B_REASON": _WEBSHOP_V2B_REASON,
        },
        # Evaluation processes (holdout / OneConfig TEST) never run profile(): export v2b's rubric directly.
        "evaluation_env": {"TRIMEM_SEMFILTER_RUBRIC": "semantic-rubric-v4-webshop"},
        "identity_extra": {"max_round": 10, "episode_budget": "max-round-v1"},
        "env_ports": (36003, 36013),
        "validation_splits": ("train",),
    },
}


def spec(dataset):
    if dataset not in DATASETS:
        raise ValueError(
            f"unsupported E2 dataset {dataset!r}; known {sorted(DATASETS)}"
        )
    return DATASETS[dataset]


def dataset_of(document):
    """Manifests frozen before the generalization carry no dataset: they are ALFWorld."""
    return document.get("dataset", "alfworld")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    )
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def policies(catalog, grid):
    """Select once using TRAIN only; keep every infeasible/duplicate SLO index."""
    if any(p.get("split") != "train" or not p.get("eligible") for p in catalog):
        raise ValueError("policy catalog must contain eligible TRAIN-only measurements")
    result = []
    first = {}
    for axis, cost in [("latency", "latency_ms"), ("tokens", "tokens")]:
        for index, limit in enumerate(grid[axis]):
            bounds = {
                "latency_ms": grid["fixed_latency_ms"],
                "tokens": grid["fixed_tokens"],
            }
            bounds[cost] = limit
            feasible = [p for p in catalog if all(p[k] <= v for k, v in bounds.items())]
            best = (
                min(
                    feasible,
                    key=lambda p: (
                        -p["quality"],
                        p["latency_ms"],
                        p["tokens"],
                        p["key"],
                    ),
                )
                if feasible
                else None
            )
            label = f"{axis}-{index:02}"
            key = best["key"] if best else None
            result.append(
                {
                    "policy_id": label,
                    "axis": axis,
                    "index": index,
                    "bounds": bounds,
                    "key": key,
                    "config": best["config"] if best else None,
                    "status": "feasible" if best else "infeasible",
                    "duplicate_of": first.get(key) if key else None,
                    "train_quality": best["quality"] if best else None,
                }
            )
            if key:
                first.setdefault(key, label)
    return result


def test_plan(archives, test_ids, identity):
    """Deduplicate across all policies/archives. Identity never comes from TEST outcomes."""
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("duplicate TEST IDs")
    pairs = {}
    for archive, policy_rows in archives.items():
        for p in policy_rows:
            if p["key"] is None:
                continue
            for task in test_ids:
                pair_id = digest(
                    {"config": p["config"], "task": task, "trial": 0, **identity}
                )
                row = pairs.setdefault(
                    pair_id,
                    {
                        "pair_id": pair_id,
                        "key": p["key"],
                        "config": p["config"],
                        "task": task,
                        "trial": 0,
                        "identity": identity,
                        "policies": [],
                        "status": "pending",
                    },
                )
                row["policies"].append(f"{archive}/{p['policy_id']}")
    return list(pairs.values())


def validate_test_rows(plan, measurements):
    want = {p["pair_id"]: p for p in plan}
    seen = {}
    for row in measurements:
        pair = want.get(row["pair_id"])
        if pair is None or any(
            row.get(k) != pair[k] for k in ("identity", "config", "task", "trial")
        ):
            raise ValueError("TEST reuse identity mismatch")
        if row.get("split") != "test" or row["pair_id"] in seen:
            raise ValueError("duplicate or non-TEST measurement")
        if not all(
            isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
            for k in ("quality", "tokens", "latency_ms")
        ):
            raise ValueError("missing/nonfinite TEST metrics")
        seen[row["pair_id"]] = row
    return seen


def summarize_test(archives, plan, measurements):
    seen = validate_test_rows(plan, measurements)
    result = []
    for archive, items in archives.items():
        for p in items:
            expected = [
                x for x in plan if f"{archive}/{p['policy_id']}" in x["policies"]
            ]
            rows = [seen[x["pair_id"]] for x in expected if x["pair_id"] in seen]
            complete = bool(expected) and len(rows) == len(expected)
            r = {
                "archive": archive,
                **p,
                "expected": len(expected),
                "completed": len(rows),
                "coverage": len(rows) / len(expected) if expected else None,
                "measurement_status": "complete"
                if complete
                else "pending"
                if expected
                else "infeasible",
            }
            for k in ("quality", "tokens", "latency_ms"):
                r[k] = sum(x[k] for x in rows) / len(rows) if complete else None
            r["token_slo_violation"] = (
                r["tokens"] > p["bounds"]["tokens"] if complete else None
            )
            r["latency_slo_violation"] = (
                r["latency_ms"] > p["bounds"]["latency_ms"] if complete else None
            )
            result.append(r)
    return result


def nhv(catalog, normalization):
    """Exact 3D union volume under a fixed reference; no clipping or final-front lookahead."""
    lo, hi = normalization["lower"], normalization["upper"]
    points = []
    for p in catalog:
        raw = [
            p["quality"],
            -math.log(max(p["tokens"], 1)),
            -math.log(max(p["latency_ms"], 1)),
        ]
        point = [(x - a) / (b - a) for x, a, b in zip(raw, lo, hi, strict=True)]
        if all(x > 0 for x in point):
            points.append(point)
    volume = 0.0
    xs = sorted({0.0, *(p[0] for p in points)})
    for x0, x1 in pairwise(xs):
        active = [p for p in points if p[0] >= x1]
        ys = sorted({0.0, *(p[1] for p in active)})
        area = sum(
            (y1 - y0) * max((p[2] for p in active if p[1] >= y1), default=0)
            for y0, y1 in pairwise(ys)
        )
        volume += (x1 - x0) * area
    return volume


def checkpoint_row(archive, checkpoint, normalization):
    rows = checkpoint["catalog"]
    if any(p["split"] != "train" or not p["eligible"] for p in rows):
        raise ValueError("NHV requires completed TRAIN observations")
    return {
        "archive": archive,
        "round": checkpoint["round"],
        "nhv": nhv(rows, normalization),
        "candidate_evaluations": checkpoint.get("candidate_evaluations"),
        "search_seconds": checkpoint.get("search_seconds"),
        "search_tokens": checkpoint.get("search_tokens"),
        "coverage": checkpoint.get("coverage", "complete"),
        "full_configs": len(rows),
        "checkpoint_sha256": digest(checkpoint),
    }


def audited_catalog(state, rows, base, pool, *, historical=False):
    """Require a full exact panel in both opt and episode files, and matching measurement protocol."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import episode_budget as eb

    ds = dataset_of(state)
    s = spec(ds)
    by = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["status"] == "evaluated":
            by[d.row_key(row)].append(row)
    catalog, audit = [], []
    strata, weights, sizes = d.pool_meta(pool)
    bo = d.bo_module()
    for rr in state["rounds"]:
        for c in rr["configs"]:
            key = space.key(c["config"])
            expected = c["sample"]["ids"]
            directory = Path(base) / "configs" / d.cfg_name(c["config"])
            selected = [r for r in by[key] if r["query_id"] in expected]
            actual = {r["query_id"] for r in selected}
            files = d.drv.item_files(directory, ds)
            reasons = []
            if historical and rr["round"] < s["trace_from_round"]:
                reasons.append(s["trace_old_regime"])
            if (
                actual != set(expected)
                or len(selected) != len(expected)
                or not actual <= files.keys()
            ):
                reasons.append("incomplete or duplicate episode/opt evidence")
            if c.get("staging", {}).get("pruned_at") is not None:
                reasons.append("pruned")
            if c.get("staging", {}).get("stage_reached", 0) < len(expected):
                reasons.append("full stage not committed")
            if any(
                r["protocol"].get("sample_hash") != c["sample"]["sample_hash"]
                or r["protocol"].get("pool_hash") != pool["pool_hash"]
                or r["protocol"].get("memory_policy") != "success+failure-v1"
                for r in selected
            ):
                reasons.append("sample/pool/memory protocol mismatch")
            # Rows without protocol.max_round ran at the native 30 turns (episode_budget.row_max_round).
            if any(eb.row_max_round(r["protocol"]) != s["max_round"] for r in selected):
                reasons.append("episode budget (max_round) mismatch")
            if c["config"]["plan"] in space.SEMFILTER_PLANS and any(
                any(
                    r["protocol"].get("semfilter", {}).get(k) != v
                    for k, v in s["semfilter"].items()
                )
                for r in selected
            ):
                reasons.append("SemFilter identity mismatch")
            args = (
                read_json(directory / "run_args.json")
                if (directory / "run_args.json").exists()
                else {}
            )
            if (
                args.get("model") != "Qwen/Qwen3.8-27B-FP8"
                or args.get("temperature") != 0
                or args.get("max_round") != s["max_round"]
            ):
                reasons.append("missing/mismatched evaluator model or sampling")
            runtime = state.get("runtime_profile", {})
            if runtime.get("name") != "controlled-c16-v1" or rr["round"] < runtime.get(
                "from_round", 0
            ):
                reasons.append("runtime not eligible")
            a = {
                "key": key,
                "source_round": rr["round"],
                "display_round": rr["round"] - s["trace_from_round"]
                if historical
                else rr["round"],
                "expected": len(expected),
                "completed_opt": len(actual),
                "completed_files": len(set(expected) & files.keys()),
                "sample_hash": c["sample"]["sample_hash"],
                "runtime": runtime.get("name"),
                "eligible": not reasons,
                "reason": "; ".join(reasons),
                "path": str(directory),
                "config": c["config"],
                "split": "train",
            }
            if not reasons:
                estimates = bo.config_estimates(
                    selected,
                    samples={key: expected},
                    strata=strata,
                    weights=weights,
                    sizes=sizes,
                    experiment=state["experiment"],
                )
                e = estimates.get(key)
                if e is None:
                    a.update(eligible=False, reason="missing aggregate")
                else:
                    a.update(
                        quality=e["quality"].mean,
                        tokens=e["tokens"].mean,
                        latency_ms=e["latency_ms"].mean,
                    )
                    catalog.append(a)
            audit.append(a)
    return catalog, audit


def plot_exports(out, checkpoints, test_results):
    """Plot only supplied measured points; no placeholder curves in prepare artifacts."""
    if not checkpoints and not any(
        r["measurement_status"] == "complete" for r in test_results
    ):
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    for axis in ("candidate_evaluations", "search_seconds", "search_tokens"):
        groups = defaultdict(list)
        for r in checkpoints:
            if r.get(axis) is not None:
                groups[r["archive"]].append(r)
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, rows in groups.items():
            ax.step(
                [r[axis] for r in rows],
                [r["nhv"] for r in rows],
                where="post",
                label=name,
                linestyle="--" if "historical" in name else "-",
            )
        # Common support only; each observation carried forward within measured range.
        methods = defaultdict(list)
        for name, rows in groups.items():
            if "historical" not in name:
                methods[name.rsplit("-", 1)[0]].append(rows)
        for method, runs in methods.items():
            if len(runs) < 2:
                continue
            lo = max(rs[0][axis] for rs in runs)
            hi = min(rs[-1][axis] for rs in runs)
            grid = sorted({r[axis] for rs in runs for r in rs if lo <= r[axis] <= hi})
            ys = [
                sum(next(r["nhv"] for r in reversed(rs) if r[axis] <= x) for rs in runs)
                / len(runs)
                for x in grid
            ]
            ax.step(
                grid, ys, where="post", lw=3, label=f"{method} mean (n={len(runs)})"
            )
        ax.set(xlabel=axis, ylabel="TRAIN NHV (fixed normalization)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            p = out / f"nhv_{axis}.{ext}"
            fig.savefig(p)
            generated.append(str(p))
        plt.close(fig)
    complete = [r for r in test_results if r["measurement_status"] == "complete"]
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, axis, cost in zip(
            axes, ("latency", "tokens"), ("latency_ms", "tokens"), strict=True
        ):
            for name in sorted({r["archive"] for r in complete}):
                rows = sorted(
                    [
                        r
                        for r in test_results
                        if r["archive"] == name and r["axis"] == axis
                    ],
                    key=lambda r: r["index"],
                )
                ax.plot(
                    [r[cost] if r[cost] is not None else math.nan for r in rows],
                    [
                        r["quality"] if r["quality"] is not None else math.nan
                        for r in rows
                    ],
                    marker="o",
                    label=name,
                )
                bad = [
                    r
                    for r in rows
                    if r["token_slo_violation"] or r["latency_slo_violation"]
                ]
                ax.scatter(
                    [r[cost] for r in bad],
                    [r["quality"] for r in bad],
                    marker="x",
                    c="red",
                )
            ax.set(xlabel=cost, ylabel="TEST success")
            ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "svg", "png"):
            p = out / f"oneconfig_test.{ext}"
            fig.savefig(p)
            generated.append(str(p))
        plt.close(fig)
    return generated


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render E2 measured exports only")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = read_json(args.data)
    print(json.dumps(plot_exports(args.out, data["checkpoints"], data["test_results"])))


if __name__ == "__main__":
    main()
