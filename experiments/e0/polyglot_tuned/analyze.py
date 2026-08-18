"""Compute E0's three headline numbers from the recorded plan-space sweep.

Reads results/e0/plan_space/raw.jsonl (one row per query x plan) and applies the rules frozen
in configs/e0/plan_space_v0.1.yaml BEFORE the sweep ran:

  plan spread            max/min median-latency WITHIN a query's quality-equivalent plan set.
                         The equivalence set is what makes the ratio mean anything -- without
                         it the fastest plan is always the one that answers nothing.
  default suboptimality  latency(the hardcoded GraphRAG-style plan) / latency(that query's
                         fastest quality-equivalent plan).
  shape distribution     which shape wins each query, and how often the winner changes.

Every number is also reported split by annotation_status, because 30 of the 40 queries are
auto-derived rather than hand-audited: if the two strata disagree, the derivation biased the
workload and the headline is not trustworthy.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def equivalence_set(rows: list[dict[str, Any]], eps_hit: float, eps_mrr: float):
    """Plans within eps of the BEST quality achieved on this query.

    A query whose best plan still answers nothing has no equivalence set: every plan is
    equally useless and a latency ratio over them would be meaningless. Those queries are
    reported as unanswered, never folded into the spread distribution.
    """
    live = [r for r in rows if not r["infeasible"] and not r["error"]]
    if not live:
        return []
    best_hit = max(r["hit@1"] for r in live)
    best_mrr = max(r["mrr"] for r in live)
    if best_mrr <= 0:
        return []
    return [
        r
        for r in live
        if abs(r["hit@1"] - best_hit) <= eps_hit and abs(r["mrr"] - best_mrr) <= eps_mrr
    ]


def analyze(raw: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    eps_hit = float(config["quality"]["eps_hit"])
    eps_mrr = float(config["quality"]["eps_mrr"])
    default = config["default_plan"]
    threshold = float(config["stop_conditions"]["plan_spread_median_below"])

    rows = load_rows(raw)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)

    per_query: list[dict[str, Any]] = []
    unanswered: list[str] = []
    for query_id, plans in sorted(by_query.items()):
        equivalent = equivalence_set(plans, eps_hit, eps_mrr)
        if not equivalent:
            unanswered.append(query_id)
            continue
        latencies = [r["latency_ms"] for r in equivalent]
        fastest = min(equivalent, key=lambda r: r["latency_ms"])
        default_rows = [
            r
            for r in plans
            if r["shape"] == default["shape"]
            and r["k"] == default["k"]
            and r["hops"] == default["hops"]
            and r["predicate_placement"] == default["predicate_placement"]
        ]
        default_row = default_rows[0] if default_rows else None
        per_query.append(
            {
                "query_id": query_id,
                "annotation_status": plans[0]["annotation_status"],
                "template": plans[0]["template"],
                "query_hop_limit": plans[0]["query_hop_limit"],
                "equivalent_plans": len(equivalent),
                "total_plans": len(plans),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "plan_spread": max(latencies) / min(latencies),
                "best_plan": fastest["plan_tag"],
                "best_shape": fastest["shape"],
                # The hardcoded plan's quality matters as much as its cost: if it does not
                # even reach the equivalence set, its "suboptimality ratio" would compare a
                # right answer against a wrong one.
                "default_reaches_quality": bool(
                    default_row
                    and abs(default_row["mrr"] - max(r["mrr"] for r in equivalent))
                    <= eps_mrr
                ),
                "default_ms": default_row["latency_ms"] if default_row else None,
                "default_suboptimality": (
                    default_row["latency_ms"] / fastest["latency_ms"]
                    if default_row
                    else None
                ),
            }
        )

    spreads = [q["plan_spread"] for q in per_query]
    shape_wins = Counter(q["best_shape"] for q in per_query)
    feasible_by_shape = Counter()
    for query_id, plans in by_query.items():
        equivalent = equivalence_set(plans, eps_hit, eps_mrr)
        for shape in {r["shape"] for r in equivalent}:
            feasible_by_shape[shape] += 1

    def summarize(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": round(ordered[0], 2),
            "median": round(statistics.median(ordered), 2),
            "p90": round(ordered[int(len(ordered) * 0.9)], 2),
            "max": round(ordered[-1], 2),
        }

    strata: dict[str, Any] = {}
    for status in sorted({q["annotation_status"] for q in per_query}):
        subset = [q for q in per_query if q["annotation_status"] == status]
        strata[status] = {
            "plan_spread": summarize([q["plan_spread"] for q in subset]),
            "shape_wins": dict(Counter(q["best_shape"] for q in subset)),
        }

    default_ratios = [
        q["default_suboptimality"]
        for q in per_query
        if q["default_suboptimality"] is not None and q["default_reaches_quality"]
    ]
    default_wrong = sum(1 for q in per_query if not q["default_reaches_quality"])

    median_spread = statistics.median(spreads) if spreads else float("nan")
    return {
        "schema_version": "e0-plan-space-analysis-v0.1.0",
        "raw": str(raw),
        "config": str(config_path),
        "queries_total": len(by_query),
        "queries_answered": len(per_query),
        "queries_unanswered": unanswered,
        "plan_spread": summarize(spreads),
        "default_plan": dict(default),
        "default_suboptimality_when_correct": summarize(default_ratios),
        "default_plan_misses_quality_on": default_wrong,
        "shape_wins": dict(shape_wins),
        "shape_reaches_quality_on_queries": dict(feasible_by_shape),
        "shape_invariant": len(shape_wins) == 1,
        "strata": strata,
        "stop_condition": {
            "plan_spread_median": round(median_spread, 3),
            "threshold": threshold,
            "proposition_O_survives": bool(median_spread >= threshold),
        },
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--raw", type=Path, default=Path("results/e0/plan_space/raw.jsonl")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e0/plan_space_v0.1.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/e0/plan_space/analysis.json")
    )
    args = parser.parse_args(argv)

    result = analyze(args.raw, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"queries: {result['queries_answered']}/{result['queries_total']} answered "
        f"({len(result['queries_unanswered'])} unanswered by EVERY plan)"
    )
    print(f"plan spread: {result['plan_spread']}")
    print(f"shape wins: {result['shape_wins']}")
    print(
        f"shape reaches quality on N queries: {result['shape_reaches_quality_on_queries']}"
    )
    print(f"default plan {result['default_plan']}")
    print(f"  misses quality on {result['default_plan_misses_quality_on']} queries")
    print(
        f"  suboptimality when correct: {result['default_suboptimality_when_correct']}"
    )
    for status, values in result["strata"].items():
        print(
            f"stratum {status}: spread={values['plan_spread']} wins={values['shape_wins']}"
        )
    stop = result["stop_condition"]
    print(
        f"STOP CONDITION: median spread {stop['plan_spread_median']}x vs threshold "
        f"{stop['threshold']}x -> proposition O survives = {stop['proposition_O_survives']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
