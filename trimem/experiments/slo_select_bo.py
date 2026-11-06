"""SLO-constrained BO configuration selection from the TRAIN profiling (user requests 2026-10-19).

v2 (percentile SLOs). Candidates: ALL complete configurations of the training panel (ALFWorld /
ScienceWorld ``<ds>:train100:bo3obj-v1``, τ²-bench ``tau2:train178:bo3obj-v1``), with the page's per-query
metrics (``render_profiling_points.train_panel``: quality = panel mean, tokens = agent in + out + SemFilter,
latency = e2e retrieval + episode, seconds). Thresholds are percentiles over ALL those configurations of
the dataset (numpy linear interpolation):

  SLO-1: latency <= p25(latency)                          -> max quality
  SLO-2: latency <= median(latency) and tokens <= median(tokens) -> max quality
  SLO-3: quality >= Q3[ds]                                -> min latency
         Q3 = 0.90 ALFWorld, 0.80 ScienceWorld, τ² = median config quality rounded to 2 decimals.
Deterministic tie-breaks (exact ties are common): max-quality SLOs -> lower latency, then lower tokens;
min-latency SLO -> higher quality, then lower tokens.

A selected configuration counts as tested when an identical configuration already has a complete test
arm (BO Q-best / Knee, plan-default fixed arms, or a completed v1 SLO arm); otherwise it is reported as
not tested (no run is started here).

Writes results/slo_selection_20261019/selections_v2.{json,md} (v1 selections.json is left untouched: the
queued v1 SLO test run reads it). Read-only on opt; runs no episode.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.slo_select_bo
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trimem.experiments import render_atlas_results as ra
from trimem.experiments import render_profiling_points as rp
from trimem.optimizer import session_space_agentenv as space

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/slo_selection_20261019"
V1 = OUT / "selections.json"
CTRL = ROOT / "results/eval_controlled_20261015"
# (dataset, panel, BO experiment, BO run dir)
PANELS = [
    ("alfworld", "train100", "alfworld:train100:bo3obj-v1", "alfworld"),
    ("sciworld", "train100", "sciworld:train100:bo3obj-v1", "sciworld"),
    ("tau2", "train178", "tau2:train178:bo3obj-v1", "tau2"),
]
Q3 = {
    "alfworld": 0.90,
    "sciworld": 0.80,
    "tau2": None,
    "webshop": None,  # user 2026-10-23 (WebShop pool run): train percentile bound = median config quality
}  # None -> median config quality, 2 decimals
SLOS = {
    "SLO-1": {
        "lat": "p25",
        "objective": "max_quality",
        "text": "mean latency <= p25 of all configs -> highest quality",
    },
    "SLO-2": {
        "lat": "p50",
        "tok": "p50",
        "objective": "max_quality",
        "text": "latency <= median and tokens <= median of all configs -> highest quality",
    },
    "SLO-3": {
        "q": "q3",
        "objective": "min_latency",
        "text": "quality >= dataset threshold -> lowest latency",
    },
}
FIXED = {"VGR": "VGR", "VGR+Rank": "VGR+Rank", "P2": "P2", "P2+Rank": "P2+Rank"}
N_TEST = {"alfworld": 200, "sciworld": 200, "tau2": 100}


def bounds(ds: str, pts: list[dict]) -> dict[str, dict]:
    lat, tok, q = (np.array([p[k] for p in pts]) for k in ("lat", "tok", "q"))
    v = {"p25": float(np.percentile(lat, 25)), "p50": float(np.median(lat))}
    q3 = Q3[ds] if Q3[ds] is not None else round(float(np.median(q)), 2)
    return {
        "SLO-1": {"lat": v["p25"]},
        "SLO-2": {"lat": v["p50"], "tok": float(np.median(tok))},
        "SLO-3": {"q": q3},
    }


def feasible(p: dict, b: dict) -> bool:
    return all(p["q"] >= x if k == "q" else p[k] <= x for k, x in b.items())


def choose(cands: list[dict], b: dict, objective: str) -> dict | None:
    ok = [p for p in cands if feasible(p, b)]
    if not ok:
        return None
    if objective == "max_quality":
        return min(ok, key=lambda p: (-p["q"], p["lat"], p["tok"]))
    return min(ok, key=lambda p: (p["lat"], -p["q"], p["tok"]))


def n_done(ds: str, arm: str) -> int:
    if ds == "tau2":
        return len(ra.tau2(arm))
    return len(list((CTRL / f"{ds}_{arm}").glob(f"{ds}_*.json")))


def tested_arms(ds: str, run: str) -> dict[str, str]:
    """config key -> existing complete test arm (success-only-v0 controlled arms)."""
    cand: dict[str, str] = {}
    sel = json.load(open(rp.BO / run / "selection.json"))
    for name, k in (("BO-Qbest", "q_best"), ("BO-Knee", "knee")):
        cand.setdefault(space.key(sel[k]["config"]), name)
    for name, plan in FIXED.items():
        cand.setdefault(space.key({"plan": plan, **space.plan_defaults()}), name)
    if (
        V1.exists() and ds in json.load(open(V1))["datasets"]
    ):  # completed v1 SLO arms (e.g. sciworld BO-SLO1)
        for s in json.load(open(V1))["datasets"][ds]["selections"].values():
            if s and s["action"] == "new test run":
                cand.setdefault(space.key(s["config"]), s["test_arm"])
    return {k: v for k, v in cand.items() if n_done(ds, v) >= N_TEST[ds]}


def main() -> int:
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "version": "v2-percentile",
        "candidates": "all complete configurations of the training panel",
        "slos": {k: v["text"] for k, v in SLOS.items()},
        "datasets": {},
    }
    for ds, panel, exp, run in PANELS:
        P = rp.train_panel(ds, panel, exp, run)
        pts = P["points"]
        B = bounds(ds, pts)
        tested = tested_arms(ds, run)
        sels = {}
        for name, s in SLOS.items():
            p = choose(pts, B[name], s["objective"])
            if p is None:
                sels[name] = None
                continue
            arm = tested.get(space.key(p["cfg"]))
            sels[name] = {
                "label": p["label"],
                "config": p["cfg"],
                "train_quality": p["q"],
                "train_quality_se": p["q_se"],
                "train_latency_s": p["lat"],
                "train_tokens": p["tok"],
                "qbest": p["qbest"],
                "knee": p["knee"],
                "fixed_default": p["fixed"],
                "front3": p["front3"],
                "test_arm": arm,
                "tested": arm is not None,
                "action": "reuse existing test arm"
                if arm
                else "not tested (no run started)",
                "n_feasible": sum(feasible(q, B[name]) for q in pts),
            }
        res["datasets"][ds] = {
            "experiment": exp,
            "panel": panel,
            "n_queries": P["n_queries"],
            "n_configs": P["n_configs"],
            "bounds": B,
            "nomemory_train": P["nomem"],
            "selections": sels,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "selections_v2.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    tmp.replace(OUT / "selections_v2.json")
    md = [
        "# BO configurations selected under percentile SLOs (v2, training profiling)",
        "",
        f"Generated {res['generated']}. Candidates = ALL complete configurations of the training panel; "
        "thresholds = percentiles over those configurations. Metrics per query: quality = panel mean, tokens = "
        "agent in+out + SemFilter, latency = retrieval + episode (e2e).",
        "Tie-breaks: max-quality SLOs -> lower latency, then lower tokens; SLO-3 -> higher quality, then lower "
        "tokens. Untested selections are not run here.",
        "",
    ]
    for ds, D in res["datasets"].items():
        B = D["bounds"]
        md += [
            f"## {rp.DS_NAME[ds]} ({D['experiment']}, {D['n_queries']} queries, {D['n_configs']} configs)",
            "",
            f"- SLO-1: latency <= {B['SLO-1']['lat']:.2f} s (p25) -> max quality",
            f"- SLO-2: latency <= {B['SLO-2']['lat']:.2f} s (median) and tokens <= {B['SLO-2']['tok']:,.0f} "
            "(median) -> max quality",
            f"- SLO-3: quality >= {B['SLO-3']['q']:.2f}"
            f"{' (median config quality, rounded)' if Q3[ds] is None else ''} -> min latency",
            "",
            "| SLO | config | train quality | latency s | tokens | feasible configs | test arm |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in D["selections"].items():
            if s is None:
                md.append(
                    f"| {name} | none (no configuration satisfies the SLO) | | | | 0 | |"
                )
                continue
            tag = " / ".join(
                t
                for t, f in (
                    ("Opt-OneConfig", s["qbest"]),
                    ("Knee", s["knee"]),
                    ("plan default", s["fixed_default"]),
                )
                if f
            )
            md.append(
                f"| {name} | {s['label']}{' (' + tag + ')' if tag else ''} | "
                f"{s['train_quality']:.4f} ± {s['train_quality_se']:.3f} | {s['train_latency_s']:.2f} | "
                f"{s['train_tokens']:,.0f} | {s['n_feasible']} | "
                f"{s['test_arm'] + ' (reused)' if s['tested'] else 'not tested'} |"
            )
        md.append("")
    (OUT / "selections_v2.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
