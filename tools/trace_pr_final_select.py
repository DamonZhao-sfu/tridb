"""Final configuration selection by GP posterior mean (shrunk TRAIN estimate).

The TRAIN sample mean of the best of 50 configurations is biased upward (winner's curse).
This tool refits the run's own surrogate (``plan_surrogate.fit_model``: per-config fixed
noise = squared standard error, the manifest's frozen objective scaling and seed) on every
complete TRAIN configuration and reads the posterior mean at each observed configuration.
The posterior mean pools evidence from similar configurations, so a lucky panel mean is
pulled toward its neighbours.

Outputs ``<run>/final_selection.json``:
* ``one_config``      : argmax posterior mean quality (the deployed single configuration).
* ``gp_pareto``       : Pareto set over posterior means (quality max, tokens / latency min).
* per-config sample mean, posterior mean and posterior sd in original units.
* ``test_missing``    : GP-selected keys without a complete TEST row (need a TEST run).
If the TEST ledger exists, it also reports TEST means of the selected keys (read last,
after every choice is fixed). Reads only; never writes a ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def fit_posterior(run: Path, *, allow_partial: bool = False) -> tuple[list[dict], dict]:
    from trimem.experiments.trace_joint_plan_search import get_space, observations
    from trimem.optimizer.joint_search_ledger import SearchLedger
    from trimem.optimizer.plan_codec import JointCodec
    from trimem.optimizer.plan_surrogate import ObjectiveProtocol, fit_model

    m = json.loads((run / "manifest.json").read_text())
    space = get_space(m)
    with SearchLedger(run / "ledger.sqlite", m) as led:
        if m.get("pruning"):
            from trimem.experiments.trace_joint_plan_search import staged_batch_size

            finished = not led.pending() and staged_batch_size(led, m) == 0
        else:
            finished = not led.pending() and len(led.complete_configs()) == m["budget"]
        if not allow_partial and not finished:
            raise ValueError("Final selection requires the complete TRAIN search")
        obs = observations(led, space, m)
        pruned = [e["pruned"] for e in led.complete_configs()]  # same order as observations()
    codec = JointCodec(space)
    # The objective scaling is frozen in objective.json after initialization.
    frozen = m.get("objective") or json.loads((run / "objective.json").read_text())
    objective = ObjectiveProtocol(**frozen)
    torch.set_num_threads(4)
    model = fit_model(codec, obs, objective, seed=m["seed"], maxiter=m.get("gp_maxiter", 150))
    x = torch.tensor([codec.encode(o.config) for o in obs], dtype=torch.double)
    with torch.no_grad():
        post = model.posterior(x)
        mean, var = post.mean, post.variance
    lo = torch.tensor(objective.lo, dtype=torch.double)
    span = torch.tensor(objective.span, dtype=torch.double)
    target = mean * span + lo  # (quality, -log1p tokens, -log1p latency)
    sd = var.sqrt() * span
    rows = []
    for i, o in enumerate(obs):
        rows.append(
            {
                "key": space.key(o.config),
                "config": o.config,
                "train_mean": {"quality": o.mean[0], "tokens": o.mean[1], "latency": o.mean[2]},
                "posterior_mean": {
                    "quality": float(target[i, 0]),
                    "tokens": math.expm1(-float(target[i, 1])),
                    "latency": math.expm1(-float(target[i, 2])),
                },
                "posterior_sd_quality": float(sd[i, 0]),
                "train_se_quality": math.sqrt(o.variance[0]),
                # Staged pruning: a pruned config has a partial panel; it informs the GP but is
                # never a deployment or TEST candidate.
                "pruned": bool(pruned[i]),
            }
        )
    meta = {"protocol": m["protocol"], "seed": m["seed"], "objective": frozen, "configs": len(obs)}
    return rows, meta


def pareto(points: list[tuple[float, float, float]]) -> list[int]:
    """Indices of non-dominated points; every coordinate is maximized."""
    return [
        i
        for i, p in enumerate(points)
        if not any(
            all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p)) for q in points
        )
    ]


def test_means(run: Path, keys: list[str]) -> dict[str, dict]:
    path = run / "holdout" / "ledger.sqlite"
    if not path.exists():
        return {}
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for key in keys:
        rs = [
            json.loads(r)
            for (r,) in db.execute(
                "SELECT receipt FROM episodes WHERE config = ? AND state = 'done'", (key,)
            )
        ]
        if len(rs) == 200:
            out[key] = {
                f: sum(r[f] for r in rs) / 200
                for f in ("raw_score", "total_tokens", "latency_seconds", "agent_steps")
            }
    return out


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    rows, meta = fit_posterior(run)
    # Every choice below uses TRAIN only.
    full = [i for i, r in enumerate(rows) if not r["pruned"]]
    one_gp = max(full, key=lambda i: rows[i]["posterior_mean"]["quality"])
    one_raw = max(full, key=lambda i: rows[i]["train_mean"]["quality"])
    gp_front = [full[j] for j in pareto([
        (rows[i]["posterior_mean"]["quality"], -rows[i]["posterior_mean"]["tokens"], -rows[i]["posterior_mean"]["latency"])
        for i in full
    ])]
    raw_front = [full[j] for j in pareto([
        (rows[i]["train_mean"]["quality"], -rows[i]["train_mean"]["tokens"], -rows[i]["train_mean"]["latency"])
        for i in full
    ])]
    selected = sorted(set(gp_front) | {one_gp})
    test = test_means(run, [r["key"] for r in rows])  # TEST read last
    report = {
        **meta,
        "rule": "argmax GP posterior mean quality over complete TRAIN configs",
        "one_config": rows[one_gp]["key"],
        "one_config_sample_mean": rows[one_raw]["key"],
        "gp_pareto": [rows[i]["key"] for i in gp_front],
        "sample_mean_pareto": [rows[i]["key"] for i in raw_front],
        "test_missing": [rows[i]["key"] for i in selected if rows[i]["key"] not in test],
        "configs": [{**r, "test": test.get(r["key"])} for r in rows],
    }
    tested = [r for r in rows if r["key"] in test]
    if len(tested) >= 3:
        y = [test[r["key"]]["raw_score"] for r in tested]
        report["spearman_vs_test"] = {
            "n": len(tested),
            "train_mean": spearman([r["train_mean"]["quality"] for r in tested], y),
            "posterior_mean": spearman([r["posterior_mean"]["quality"] for r in tested], y),
        }
    out = run / "final_selection.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out}")
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["posterior_mean"]["quality"])[:10]
    print("rank key        train  ±se   post  ±sd   test   tokens  latency  pareto")
    for rank, i in enumerate(order, 1):
        r = rows[i]
        t = test.get(r["key"])
        print(
            f"{rank:>4} {r['key'][:8]}  {r['train_mean']['quality']:6.2f} {r['train_se_quality']:4.1f}"
            f"  {r['posterior_mean']['quality']:6.2f} {r['posterior_sd_quality']:4.1f}"
            f"  {t['raw_score'] if t else float('nan'):6.2f}"
            f"  {r['posterior_mean']['tokens'] / 1e3:6.1f}k {r['posterior_mean']['latency']:6.1f}s"
            f"  {'GP' if i in gp_front else '  '}{' S' if i in raw_front else ''}"
        )
    print("OneConfig (GP):", rows[one_gp]["key"][:8], " OneConfig (sample):", rows[one_raw]["key"][:8])
    print("selected keys without TEST rows:", [k[:8] for k in report["test_missing"]])
    if "spearman_vs_test" in report:
        print("Spearman vs TEST:", report["spearman_vs_test"])


if __name__ == "__main__":
    main()
