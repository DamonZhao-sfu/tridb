"""report.md of a pool-profiling run evaluated with outer 5-fold CV (user 2026-10-21; ``bo_pool_driver report``).

Sections: profiling status (incl. a user pause), the primary CV test table (mean ± std across folds, SE_tasks),
the reused controlled reference arms (NoMemory / VGR), the deployed full-train arms, the train-selected
configurations per SLO per fold, the router candidates (BRANE and LimeQO, per fold and deployed), and a
BRANE vs LimeQO vs Opt-OneConfig comparison with per-fold paired differences (both routers and every
Opt-OneConfig fold arm share the same folds and the same test tasks).

Pure: takes the report dict and the CV fold records; never reads the DB or runs episodes.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from trimem.experiments import router_outer_cv as rcv
from trimem.optimizer.query_router import outer_cv as ocv

METRICS = (
    ("quality", ".3f"),
    ("tokens", ",.0f"),
    ("latency_s", ".1f"),
    ("steps", ".2f"),
    ("paired_gain_vs_nomemory", "+.3f"),
)


def _f(v, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def paired_diff(per_fold: dict[str, dict], a: str, b: str) -> dict[str, dict]:
    """Per metric: mean ± std across folds of (arm ``a`` - arm ``b``) on the same fold and test tasks."""
    out = {}
    for m, _ in METRICS:
        d = [
            f[a][m] - f[b][m]
            for f in per_fold.values()
            if a in f and b in f and f[a].get(m) is not None and f[b].get(m) is not None
        ]
        out[m] = ocv.mean_std(d)
    return out


def _ms(x: dict, fmt: str) -> str:
    if x["mean"] is None:
        return "–"
    s = format(x["mean"], fmt)
    return s + (
        f" ± {format(x['std'], fmt.lstrip('+'))}" if x["std"] is not None else ""
    )


def reference_table(refs: dict[str, dict], qlabel: str) -> list[str]:
    lines = [
        f"| reference arm | {qlabel} | tokens | latency s | steps | steps (succ.) | gain vs NoMemory | "
        f"SE_tasks {qlabel} | SE_tasks steps | n | protocol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n, r in refs.items():
        lines.append(
            f"| {n} | {_f(r.get('quality'), '.3f')} | {_f(r.get('tokens'), ',.0f')} | "
            f"{_f(r.get('latency_s'), '.1f')} | {_f(r.get('steps'), '.2f')} | {_f(r.get('steps_success'), '.2f')} | "
            f"{_f(r.get('paired_gain_vs_nomemory'), '+.3f')} | {_f(r.get('quality_se'), '.3f')} | "
            f"{_f(r.get('steps_se'), '.2f')} | {r.get('covered', r.get('n'))} | {r.get('protocol', '')} |"
        )
    return lines


def deployed_table(arms: dict[str, dict], qlabel: str) -> list[str]:
    lines = [
        f"| arm (full-train fit) | {qlabel} | tokens | latency s | steps | steps (succ.) | gain vs NoMemory | "
        "SE_tasks | SE_tasks steps | configs | covered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in arms.items():
        lines.append(
            f"| {n} | {_f(v.get('quality'), '.3f')} | {_f(v.get('tokens'), ',.0f')} | "
            f"{_f(v.get('latency_s'), '.1f')} | {_f(v.get('steps'), '.2f')} | {_f(v.get('steps_success'), '.2f')} | "
            f"{_f(v.get('paired_gain_vs_nomemory'), '+.3f')} | {_f(v.get('quality_se'), '.3f')} | "
            f"{_f(v.get('steps_se'), '.2f')} | {v.get('distinct_configs')} | {v.get('covered')}/{v.get('n')} |"
        )
    return lines


def selections_table(folds: list[dict], label_of: Callable[[dict], str]) -> list[str]:
    """Train-selected Opt-OneConfig configuration per SLO per fold (+ train estimates)."""
    slos = sorted({n for f in folds for n in f["oneconfig"]})
    lines = [
        "| fold | " + " | ".join(slos) + " |",
        "|---|" + "---|" * len(slos),
    ]
    for f in folds:
        cells = []
        for n in slos:
            v = f["oneconfig"].get(n)
            if v is None:
                cells.append("infeasible")
                continue
            t = v["train"]
            cells.append(
                f"{label_of(v['cfg'])} (train q {t['q']:.3f}, tok {t['tok']:,.0f}, lat {t['lat']:.1f}s)"
            )
        lines.append(f"| {f['fold']} | " + " | ".join(cells) + " |")
    return lines


def slo_weights_table(folds: dict[str, list[dict]]) -> list[str]:
    """Opt-PerQuery SLO weight choices per fold (train OOF replay; train-only, not test arms)."""
    lines = ["| router | fold | SLO-1 | SLO-2 | SLO-3 |", "|---|---|---|---|---|"]
    for name, recs in folds.items():
        for f in recs:
            cells = [
                "infeasible"
                if (v := f["perquery_slo"].get(n)) is None
                else "("
                + ", ".join(f"{x:g}" for x in v["weights"])
                + f") replay q {v['train_replay']['q']:.3f}"
                for n in ("SLO-1", "SLO-2", "SLO-3")
            ]
            lines.append(f"| {name} | {f['fold']} | " + " | ".join(cells) + " |")
    return lines


def candidates_table(
    folds: dict[str, list[dict]],
    deployed: dict[str, list[dict]],
    label_of: Callable[[dict], str],
) -> list[str]:
    lines = ["| router | fit | # candidates | candidates |", "|---|---|---:|---|"]
    for name, recs in folds.items():
        for f in recs:
            c = f["router"]["candidate_configs"]
            lines.append(
                f"| {name} | fold {f['fold']} | {len(c)} | {', '.join(map(label_of, c))} |"
            )
    for name, c in deployed.items():
        lines.append(
            f"| {name} | full train | {len(c)} | {', '.join(map(label_of, c))} |"
        )
    return lines


def comparison(ts: dict, weights: list[str], slos: list[str]) -> list[str]:
    """BRANE vs LimeQO (same weights) and each router vs Opt-OneConfig (per SLO), per-fold paired differences."""
    per = ts.get("per_fold", {})
    summ = ts.get("summary", {})
    lines = [
        "Per-fold paired differences (same fold, same 200 test tasks); mean ± std across folds.",
        "",
        "| comparison | Δ success | Δ tokens | Δ latency s | Δ steps | Δ gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def row(a: str, b: str) -> None:
        if a in summ and b in summ:
            d = paired_diff(per, a, b)
            lines.append(
                f"| {a} − {b} | {_ms(d['quality'], '+.3f')} | {_ms(d['tokens'], '+,.0f')} | "
                f"{_ms(d['latency_s'], '+.1f')} | {_ms(d['steps'], '+.2f')} | "
                f"{_ms(d['paired_gain_vs_nomemory'], '+.3f')} |"
            )

    for w in weights:
        row(f"Opt-PerQuery-{rcv.LIMEQO_PREFIX}{w}", f"Opt-PerQuery-{w}")
    for s in slos:
        for w in weights:
            row(f"Opt-PerQuery-{w}", f"Opt-OneConfig-{s}")
            row(f"Opt-PerQuery-{rcv.LIMEQO_PREFIX}{w}", f"Opt-OneConfig-{s}")
    return lines


def render(
    *,
    ds: str,
    rep: dict,
    profile: dict,
    brane_folds: list[dict],
    limeqo_folds: list[dict],
    deployed_candidates: dict[str, list[dict]],
    label_of: Callable[[dict], str],
    refs: dict[str, dict],
    weights: list[str],
    slos: list[str],
    n_test: int,
) -> str:
    qlabel = "success rate" if ds == "alfworld" else "score/100"
    cv = rep.get("outer_cv") or {}
    ts = cv.get("test") or {}
    lines = [f"# {ds} pool profiling — test report (outer 5-fold CV)", ""]
    if profile.get("paused_by_user"):
        lines += [
            f"**Profiling paused by the user** after round {profile['last_round']} ({profile.get('pause_protocol')}): "
            f"{profile.get('pause_reason')}. Rounds {profile['rounds']}, {profile['evaluated_configs']} evaluated "
            f"configurations ({profile['full_configs']} full, {profile['pruned_configs']} pruned), "
            f"{profile['budget_used']:,} / {profile['budget_episodes']:,} budget episodes. Resume: {profile['resume']}.",
            "",
        ]
    lines += [
        f"Experiment `{rep['experiment']}`; {rep['pairs']} evaluated (config, task) test pairs. Each distinct "
        "(configuration, task) pair ran ONCE and is shared by every arm routing to it (pair reuse); quality reuse "
        "of identical-memory episodes applies.",
        "",
    ]
    lines += rcv.markdown(ts, qlabel, n_test)
    if cv.get("incomplete_test_arms"):
        lines += [
            "",
            f"Incomplete CV test arms (excluded): {cv['incomplete_test_arms']}",
        ]
    lines += [
        "",
        "## Reference arms (reused controlled-v1 test arms; single run, no folds)",
        "",
        *reference_table(refs, qlabel),
        "",
        "Not like-for-like on cost: the controlled-v1 arms ran at test concurrency "
        f"{rep.get('reused_arms_concurrency')} (2 × 8) under memory policy success-only-v0 and the earlier serving "
        f"setup; the arms above ran at test concurrency {rep.get('test_concurrency')} under success+failure-v1. "
        "Success rates / paired gains are comparable (same 200 tasks, same official evaluator); tokens and "
        "especially latency are not.",
        "",
        "## Deployed arms (router / selection fitted on the full training set; secondary)",
        "",
        *deployed_table(rep.get("arms", {}), qlabel),
        "",
        "## Train-selected Opt-OneConfig configuration per SLO per fold",
        "",
        *selections_table(brane_folds, label_of),
        "",
        "## Opt-PerQuery SLO weight choices per fold (train OOF replay; train-only, not test arms)",
        "",
        *slo_weights_table({"BRANE": brane_folds, "LimeQO": limeqo_folds}),
        "",
        "## Router candidates",
        "",
        *candidates_table(
            {"BRANE": brane_folds, "LimeQO": limeqo_folds},
            deployed_candidates,
            label_of,
        ),
        "",
        "## BRANE vs LimeQO vs Opt-OneConfig",
        "",
        *comparison(ts, weights, slos),
        "",
    ]
    return "\n".join(lines)


def load_folds(base, cv_dir: str) -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted(
            (base / cv_dir).glob("fold*.json"), key=lambda p: int(p.stem[4:])
        )
    ]
