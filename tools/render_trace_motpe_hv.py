"""TRAIN HV vs search budget: historical TRACE (ALFWorld / ScienceWorld / WebShop) and live E2 MO-TPE (ALFWorld).

Read-only on experiment files and PostgreSQL; launches no episodes, analyzer calls or model fits.
User 2026-10-24: plot only measured runs; MO-TPE exists only as the first E2 ALFWorld seed.

Y axis: HV of the configurations fully evaluated by each checkpoint (measured archive, never predicted
HVI), under E2's prespecified normalization ``predefined-v1`` -- (Q, -ln tokens, -ln latency_ms) with
fixed bounds and reference shared by every method, dataset and checkpoint. Pruned prefixes cost budget
but never enter the front.

X axes (cumulative, origin = search start):
  tokens    recorded agent + semantic-judge + noise-repeat + analyzer tokens; shared NoMemory reference
            excluded for both methods (E2 reuses it). Missing usage receipts make it a lower bound.
  hours     TRACE: observed elapsed time to the round barrier (includes pauses / restarts).
            MO-TPE: the E2 ledger's active search seconds (worker downtime excluded).
  episodes  fresh completed candidate task episodes, pruned prefixes included.

ALFWorld TRACE starts at its round-6 proposal (controlled-c16 / SemFilter v2 regime); its earlier c128
regime is not charged, but its observations warm-started the GP. Not an equal-budget controlled comparison.

    PYTHONPATH=. .venv-optimizer/bin/python tools/render_trace_motpe_hv.py --collect
    PYTHONPATH=. .venv/bin/python tools/render_trace_motpe_hv.py --render
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/trace_motpe_hv_20261024"
DATA = OUT / "trace_motpe_hv.json"
MOTPE = (
    ROOT
    / "results/exp2_alfworld_campaign_20261024_run/runs"
    / "alfworld_pool-strat200-v7-e2t61022f795c3__mem-sf1"
)
_GPLHS = ROOT / "results/exp2_alfworld_campaign_20261024_gplhs2/runs"
# Live E2 ALFWorld searches, seed 20261022, identical 10-config init / samples / stage orders.
E2_RUNS = {
    "MO-TPE": MOTPE,
    "GP+qLogNEHVI": _GPLHS / "alfworld_pool-strat200-v7-e2g6102211501c__mem-sf1",
    "LHS": _GPLHS / "alfworld_pool-strat200-v7-e2l6102211501c__mem-sf1",
}
DATASETS = (("alfworld", 6), ("sciworld", None), ("webshop", None))
NAMES = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld", "webshop": "WebShop"}
# Copied from alfworld_e2_campaign.prepare (frozen E2 manifest); natural logs.
NORMALIZATION = {
    "source": "predefined-v1",
    "objectives": ["quality", "-ln(tokens)", "-ln(latency_ms)"],
    "lower": [0, -math.log(1_000_000), -math.log(3_600_000)],
    "upper": [1, -math.log(1000), -math.log(1000)],
    "reference": [0, 0, 0],
}
AXES = (
    ("tokens", "Cumulative profiling tokens (M)", 1e-6),
    ("hours", "Cumulative search time (h)", 1.0),
    ("episodes", "Cumulative task episodes (k)", 1e-3),
)
COLORS = {
    "TRACE": "#2766a2",
    "MO-TPE": "#b86418",
    "GP+qLogNEHVI": "#009E73",
    "LHS": "#CC79A7",
}


def nhv(points: list[dict]) -> float:
    from trimem.experiments.alfworld_e2_evidence import nhv as e2_nhv

    return e2_nhv(points, NORMALIZATION)


def out_of_box(points: list[dict]) -> int:
    """Points that the fixed box drops (any normalized coordinate <= 0)."""
    lo, hi = NORMALIZATION["lower"], NORMALIZATION["upper"]
    return sum(
        any(
            (x - a) / (b - a) <= 0
            for x, a, b in zip(
                (p["quality"], -math.log(p["tokens"]), -math.log(p["latency_ms"])),
                lo,
                hi,
                strict=True,
            )
        )
        for p in points
    )


def collect_trace(ds: str, origin_round: int | None) -> dict:
    run = collect_pool(ds, origin_round=origin_round)
    configs = [
        {
            "round": r["round"],
            "quality": r["q"],
            "tokens": r["tok"],
            "latency_ms": r["lat"] * 1000,
        }
        for r in run["configurations"]
        if r["full"]
    ]
    checkpoints = []
    for cp in run["checkpoints"]:
        front = [p for p in configs if p["round"] <= cp["round"]]
        spent = cp["token_breakdown"]
        checkpoints.append(
            {
                "round": cp["round"],
                "full_configs": len(front),
                "nhv": nhv(front),
                "tokens": cp["recorded_tokens"] - spent.get("nomemory_reference", 0),
                "hours": cp["elapsed_hours"],
                "episodes": cp["tasks"],
                "token_breakdown": spent,
                "checkpoint_utc": cp["checkpoint_utc"],
            }
        )
    return {
        "method": "TRACE",
        "dataset": ds,
        "source": run["source"],
        "experiment": run["experiment"],
        "cost_origin": f"round {origin_round} proposal"
        if origin_round is not None
        else "run creation",
        "state_sha256": run["state_sha256"],
        "configs_out_of_box": out_of_box(configs),
        "checkpoints": checkpoints,
    }


def e2_tokens(base: Path, barrier: float, rows: list[dict]) -> dict:
    """Agent/noise call receipts and evaluated judge receipts up to ``barrier`` (unix seconds)."""
    spent = {"agent": 0, "noise_repeat": 0, "semantic": 0}
    for role, root in (("agent", "configs"), ("noise_repeat", "noise_repeat_VGR")):
        for path in (base / root).rglob("llm_calls.jsonl"):
            if {"test", "_orphans"} & set(path.relative_to(base).parts):
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    call = json.loads(line)
                    if call["t"] <= barrier:
                        spent[role] += call["prompt_tokens"] + call["completion_tokens"]
    spent["semantic"] = sum(
        int(r["judge_tokens"] or 0)
        for r in rows
        if datetime.fromisoformat(r["updated_at"]).timestamp() <= barrier
    )
    return spent


def collect_e2(method: str, base: Path) -> dict:
    from trimem.experiments.alfworld_e2_campaign import read_rows

    state = json.loads((base / "state.json").read_text())
    rows = read_rows(state["experiment"])
    checkpoints = []
    for path in sorted((base / "e2_checkpoints").glob("*.json")):
        cp = json.loads(path.read_text())
        # Checkpoints are written once, right after their round commits (never rewritten on replay).
        barrier = path.stat().st_mtime
        spent = e2_tokens(base, barrier, rows)
        checkpoints.append(
            {
                "round": cp["round"],
                "full_configs": len(cp["catalog"]),
                "nhv": nhv(cp["catalog"]),
                "tokens": sum(spent.values()),
                "hours": cp["search_seconds"] / 3600,
                "episodes": cp["candidate_evaluations"],
                "token_breakdown": spent,
                "checkpoint_utc": datetime.fromtimestamp(barrier, UTC).isoformat(),
            }
        )
    return {
        "method": method,
        "dataset": "alfworld",
        "source": str(base.relative_to(ROOT)),
        "experiment": state["experiment"],
        "cost_origin": "E2 campaign start (paired 10-config initialization)",
        "configs_out_of_box": out_of_box(
            cp["catalog"]
        ),  # last checkpoint = full archive
        "checkpoints": checkpoints,
    }


def curve_value(cps: list[dict], axis: str, x: float) -> float:
    """Step HV at budget ``x``: best measured archive whose checkpoint cost <= x (0 before the first)."""
    return max((c["nhv"] for c in cps if c[axis] <= x), default=0.0)


def summary(series: list[dict]) -> dict:
    """Equal-budget summaries over the common budget range of the methods on one dataset."""
    out = {}
    for axis, _, _ in AXES:
        budget = min(s["checkpoints"][-1][axis] for s in series)
        row = {"common_budget": budget}
        for s in series:
            cps = s["checkpoints"]
            xs = sorted({0.0, budget, *(c[axis] for c in cps if c[axis] <= budget)})
            auc = sum(curve_value(cps, axis, a) * (b - a) for a, b in pairwise(xs))
            row[s["method"]] = {
                "hv_at_budget": curve_value(cps, axis, budget),
                "normalized_auc": auc / budget,
            }
        out[axis] = row
    return out


def collect() -> dict:
    runs = [collect_trace(ds, origin) for ds, origin in DATASETS] + [
        collect_e2(method, base) for method, base in E2_RUNS.items()
    ]
    by = {}
    for r in runs:
        by.setdefault(r["dataset"], []).append(r)
    data = {
        "schema": "trace-motpe-hv-v1",
        "split": "TRAIN",
        "generated_at": datetime.now(UTC).isoformat(),
        "normalization": NORMALIZATION,
        "runs": runs,
        "equal_budget": {ds: summary(s) for ds, s in by.items() if len(s) > 1},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


def render(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [ds for ds, _ in DATASETS]
    fig, axes = plt.subplots(
        len(datasets), len(AXES), figsize=(12, 3.0 * len(datasets)), squeeze=False
    )
    for i, ds in enumerate(datasets):
        for j, (axis, label, scale) in enumerate(AXES):
            ax = axes[i][j]
            for run in (r for r in data["runs"] if r["dataset"] == ds):
                cps = run["checkpoints"]
                # No archive before the first checkpoint: the curve starts at its first measurement.
                xs = [c[axis] * scale for c in cps]
                ys = [c["nhv"] for c in cps]
                ax.step(
                    xs,
                    ys,
                    where="post",
                    color=COLORS[run["method"]],
                    lw=1.8,
                    label=f"{run['method']} ({cps[-1]['full_configs']} configs)",
                )
                ax.plot(xs, ys, "o", ms=3.5, color=COLORS[run["method"]])
            ax.set_xlim(left=0)
            ax.grid(alpha=0.3, lw=0.6)
            if i == len(datasets) - 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"{NAMES[ds]}\nTRAIN HV (predefined-v1)")
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Measured TRAIN HV vs search budget: historical TRACE vs live E2 MO-TPE / GP+qLogNEHVI / LHS (ALFWorld)\n"
        "E2 runs share seed, 10-config init, task samples and 50→100 pruning; TRACE is not an equal-budget control",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"trace_motpe_hv.{ext}", dpi=180)


# --- Local copy of render_search_efficiency_nhv.collect_pool with ``origin_round`` and a per-role token
# split (agent / noise_repeat / nomemory_reference). Kept here, outside trimem/, because trimem/**/*.py is
# the frozen execution surface of running E2 holdout evaluations (alfworld_e2_campaign.source_hashes).
import hashlib  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

from trimem.experiments import render_profiling_budget_hv as history  # noqa: E402
from trimem.experiments.render_search_efficiency_nhv import (  # noqa: E402
    NAMES as SE_NAMES,
    completed_rounds,
    round_barriers,
    training_call_paths,
    volumes,
    webshop_note,
)


def collect_pool(ds, origin_round=None):
    """``origin_round``: charge cost only from that round's proposal (default: run creation)."""
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import bo_pool_driver as driver

    base = ROOT / "results/bo_agentenv_20261016" / f"{ds}_pool-strat200-v7__mem-sf1"
    raw = (base / "state.json").read_bytes()
    state = json.loads(raw)
    complete_path = base / "profile_complete.json"
    complete = history.load(complete_path) if complete_path.exists() else None
    barriers = round_barriers(state, complete)
    rounds = [r for r in completed_rounds(state) if r["round"] in barriers]
    first = state.get("catalog_regime", {}).get("from_round", 0)
    if not rounds or rounds[-1]["round"] < first:
        raise ValueError(f"{ds}: no completed eligible round")
    with psycopg.connect(
        history.DSN,
        row_factory=dict_row,
        options="-c default_transaction_read_only=on -c statement_timeout=60000",
    ) as conn:
        rows = conn.execute(
            "SELECT retrieval_id,query_id,experiment_id,split,status,config,protocol,created_at,updated_at,"
            "quality,total_tokens,e2e_ms,generation,retrieval_ms "
            "FROM opt.session_retrieval WHERE experiment_id=%s AND split='train'",
            (state["experiment"],),
        ).fetchall()
    groups = defaultdict(list)
    for row in rows:
        groups[history.key(row["protocol"]["space_config"])].append(row)

    samples = driver.observed_samples(state)
    # The production stratified estimator preserves each task-type's workload weight.
    if driver.eb.block(state):
        # max-round-v1 (WebShop): NHV points in the post-profiling view the catalog uses (30-turn rounds
        # derived-truncation-n, bo_pool_driver.estimates); the budget axes stay as measured (the run's real cost)
        estimates = driver.estimates(
            ds, driver.load_pool(ds), state, state["experiment"]
        )
    else:
        estimates = driver.estimates_on(
            [r for r in rows if r["status"] == "evaluated"],
            samples,
            driver.load_pool(ds),
            state["experiment"],
        )
    full = driver.full_keys(state)
    v2b = (state.get("semfilter") or {}).get(driver.SEMFILTER_V2B_KEY)
    if v2b:  # semfilter-v2b (WebShop): old-regime SemFilter configurations are not in the catalog / NHV
        full -= {
            history.key(c["config"])
            for rd in state["rounds"]
            if rd["round"] < v2b["from_round"]
            for c in rd["configs"]
            if c["config"]["plan"] in driver.space.SEMFILTER_PLANS
        }
    records = []
    for rd in rounds:
        for c in rd["configs"]:
            k = history.key(c["config"])
            observed = set(samples.get(k, []))
            rs = [
                r
                for r in groups[k]
                if str(r["query_id"]) in observed and r["status"] == "evaluated"
            ]
            if len(rs) != len(observed) or len({r["query_id"] for r in rs}) != len(rs):
                raise ValueError(
                    f"{ds}: incomplete/duplicate observed outcomes at round {rd['round']}"
                )
            rec = {
                "key": k,
                "round": rd["round"],
                "n": len(rs),
                "full": k in full and rd["round"] >= first,
                "completion": max(
                    [history.stamp(rd["proposed_at"])]
                    + [history.stamp(r["updated_at"]) for r in rs]
                ),
            }
            if rec["full"]:
                e = estimates[k]
                rec.update(
                    q=e["quality"].mean,
                    tok=e["tokens"].mean,
                    lat=e["latency_ms"].mean / 1000,
                )
            records.append(rec)
    eligible = [r for r in records if r["full"]]
    reference = {
        "tokens": 1.1 * max(r["tok"] for r in eligible),
        "latency_seconds": 1.1 * max(r["lat"] for r in eligible),
    }
    start = history.stamp(
        state["created_at"]
        if origin_round is None
        else next(r for r in state["rounds"] if r["round"] == origin_round)[
            "proposed_at"
        ]
    )

    # Read call receipts, including archived failed/superseded configurations.
    # Exclude TEST/router files and deduplicate symlinks and copied call records.
    candidates = [base / "configs"] + [
        p
        for p in base.iterdir()
        if p.is_dir()
        and p.name.startswith(("_superseded_", "_discarded_", "_round0_prefix"))
    ]
    references = [driver.nomemory_dir(ds, "stratified", regime=False)]
    nm = state.get("latency_regime", {}).get("nomemory", {}).get("dir")
    if nm:
        references.append(Path(nm))
    role_of = {}
    for role, roots in (
        ("nomemory_reference", references),
        ("noise_repeat", [base / "noise_repeat_VGR"]),
        ("agent", candidates),
    ):
        for path in training_call_paths(roots):
            role_of.setdefault(path, role)
    paths = role_of.keys()
    events, seen_calls, ignored_trailing = [], set(), 0
    for path in sorted(paths):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                ignored_trailing += (
                    1  # An active writer can leave an unfinished last line.
                )
                continue
            identity = history.key(call)
            if identity in seen_calls:
                continue
            seen_calls.add(identity)
            at = float(call["t"])
            if at < start:
                continue  # Inherited baseline: incremental run cost, not charged again.
            events.append(
                {
                    "at": at,
                    "tokens": int(
                        call.get("prompt_tokens", 0) + call.get("completion_tokens", 0)
                    ),
                    "role": role_of[path],
                }
            )
    for row in rows:
        g = row["generation"] or {}
        events.append(
            {
                "at": history.stamp(row["updated_at"]),
                "tokens": int(g.get("semantic_tokens") or 0),
                "role": "semantic",
            }
        )
    for path in (base / "analyst").glob("round_*.json"):
        a = history.load(path)
        if a.get("finished_at"):
            events.append(
                {
                    "at": history.stamp(a["finished_at"]),
                    "tokens": int(a.get("input_tokens") or 0)
                    + int(a.get("output_tokens") or 0),
                    "role": "analyzer",
                }
            )

    checkpoints = []
    for rd in rounds:
        if rd["round"] < first:
            continue
        upto = [r for r in records if r["round"] <= rd["round"]]
        end = barriers[rd["round"]]
        if end < max(r["completion"] for r in upto):
            raise ValueError(
                f"{ds}: outcomes were rewritten after the recorded round barrier"
            )
        admitted = [r for r in upto if r["full"]]
        # A fresh evaluated SQL row proves one completed episode, but cannot expose
        # every overwritten retry. Thus this axis is a recorded lower bound.
        finished = [
            r
            for r in rows
            if r["status"] == "evaluated"
            and start <= history.stamp(r["updated_at"]) <= end
        ]
        reused = sum(bool((r["generation"] or {}).get("reused_from")) for r in finished)
        spent = Counter()
        for e in events:
            if start <= e["at"] <= end:
                spent[e["role"]] += e["tokens"]
        checkpoints.append(
            {
                "round": rd["round"],
                "evaluations": len(upto),
                "full_configs": len(admitted),
                "tasks": len(finished) - reused,
                "reused_episodes": reused,
                "elapsed_hours": (end - start) / 3600,
                "recorded_tokens": sum(spent.values()),
                "token_breakdown": dict(spent),
                "checkpoint_utc": history.iso(end),
                "boundary": "next proposal / final profile completion (includes side arms)",
                **volumes(admitted, reference),
            }
        )
    note = (
        "Current v7 catalog: rounds 6–21, controlled-c16 / SemFilter v2; earlier rounds are excluded "
        "from NHV, while their recorded search costs remain on the axes. This is a continuation, not a fresh search."
        if ds == "alfworld"
        else "Current v7 run; only completed round barriers are shown. Initialization is a single measured "
        "point until another round completes. Superseded SemFilter observations are excluded from NHV."
        + (webshop_note() if ds == "webshop" else "")
    )
    if raw != (base / "state.json").read_bytes():
        raise ValueError(
            f"{ds}: state changed during collection; retry a stable snapshot"
        )
    return {
        "dataset": ds,
        "name": SE_NAMES[ds],
        "source": str(base.relative_to(ROOT)),
        "experiment": state["experiment"],
        "reference": reference,
        "start_utc": history.iso(start),
        "checkpoints": checkpoints,
        "configurations": records,
        "note": note,
        "catalog_from_round": first,
        "method": state.get("search_design", {}),
        "state_sha256": hashlib.sha256(raw).hexdigest(),
        "state_changed_during_read": False,
        "audit": {
            "call_log_files": len(paths),
            "ignored_trailing_records": ignored_trailing,
            "latest_proposed_round": state["rounds"][-1]["round"],
            "last_complete_round": rounds[-1]["round"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    data = collect() if args.collect else json.loads(DATA.read_text())
    if args.render:
        render(data)
    print(json.dumps(data["equal_budget"], indent=2))
    for r in data["runs"]:
        last = r["checkpoints"][-1]
        print(
            r["dataset"],
            r["method"],
            len(r["checkpoints"]),
            "checkpoints; last:",
            {
                k: last[k]
                for k in ("round", "full_configs", "nhv", "tokens", "hours", "episodes")
            },
            "out-of-box configs:",
            r["configs_out_of_box"],
        )


if __name__ == "__main__":
    main()
