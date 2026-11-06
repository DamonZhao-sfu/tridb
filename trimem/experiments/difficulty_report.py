"""Reporting-only ExpRAG task-type strata. Never used for routing or test selection.

Refresh existing results and the shared HTML page:
    .venv/bin/python -m trimem.experiments.difficulty_report
"""

from __future__ import annotations

import fcntl
import html
import json
import math
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFINITION = json.loads((ROOT / "trimem/data/exprag_difficulty_v1.json").read_text())
OUT = ROOT / "results/difficulty_20261019"
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
START, END = "<!-- difficulty-results:start -->", "<!-- difficulty-results:end -->"


def load_task_types(ds: str) -> dict[str, str]:
    import psycopg

    with psycopg.connect(
        "host=/tmp/trimem-traj port=55450 dbname=trimem_agentenv"
    ) as c:
        c.execute("SET TRANSACTION READ ONLY")
        rows = c.execute(
            "SELECT agenteval_item, task_type FROM task WHERE env=%s "
            "AND agenteval_item IS NOT NULL",
            (ds,),
        ).fetchall()
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"duplicate AgentEval item metadata: {ds}")
    return result


def _mean(v: list[float]) -> float | None:
    return st.mean(v) if v else None


def _se(v: list[float]) -> float | None:
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None


def summarize(
    ds: str,
    expected: list[str],
    types: dict[str, str],
    episodes: dict[str, dict],
    baseline: dict[str, float],
    *,
    success_threshold: float = 1.0,
) -> dict:
    """Equal-weight task means and paired differences on the same observed IDs.

    Episode fields: quality in [0,1], optional tokens / latency_s. Missing episodes
    are not failures. Unknown task types remain visible. SE is undefined for n < 2.
    """
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected query IDs")
    labels = {t: g for g in ("easy", "hard") for t in DEFINITION[ds][g]}
    groups = {
        g: [q for q in expected if labels.get(types.get(q), "unknown") == g]
        for g in ("easy", "hard", "unknown")
    }
    groups = {"all": expected, **groups}
    out = {}
    for group, ids in groups.items():
        have = [q for q in ids if q in episodes]
        vals = [episodes[q]["quality"] for q in have]
        paired = [q for q in have if q in baseline]
        delta = [episodes[q]["quality"] - baseline[q] for q in paired]
        out[group] = {
            "expected": len(ids),
            "n": len(have),
            "missing": sorted(set(ids) - set(have)),
            "complete": len(have) == len(ids),
            "quality": _mean(vals),
            "quality_se": _se(vals),
            "success_rate": _mean([float(v >= success_threshold) for v in vals]),
            "paired_n": len(paired),
            "paired_gain": _mean(delta),
            "paired_gain_se": _se(delta),
            "paired_baseline_quality": _mean([baseline[q] for q in paired]),
            "wins": sum(d > 0 for d in delta),
            "losses": sum(d < 0 for d in delta),
            "ties": sum(d == 0 for d in delta),
            "task_types": sorted({types.get(q, "<missing>") for q in ids}),
        }
        for field in ("tokens", "latency_s"):
            costs = [
                episodes[q][field] for q in have if episodes[q].get(field) is not None
            ]
            out[group][field] = _mean(costs)
            out[group][field + "_n"] = len(costs)
    return out


ALF_REFERENCES = ("NoMemory", "VGR")


def alf_v7_pool(ds: str, expected: list[str], types: dict, base: dict) -> list[dict]:
    """ALFWorld v7 re-test (alf_pool_page.test_table: deployed arms, the primary results; user 2026-10-22)."""
    from trimem.experiments import alf_pool_page as ap

    t = ap.test_table()
    if t.get("run") != "v7" or t.get("status") != "complete":
        return []
    arms = {
        a["label"]: summarize(
            ds,
            expected,
            types,
            {
                q: {"quality": x["q"], "tokens": x["tok"], "latency_s": x["lat"]}
                for q, x in a["items"].items()
            },
            base,
        )
        for a in t["arms"]
    }
    return [
        {
            "path": f"{t['dir']}/test_v7c16 (deployed arms)",
            "baseline": {"label": "controlled-v1 NoMemory"},
            "notes": [],
            "difficulty": {
                "baseline": summarize(
                    ds,
                    expected,
                    types,
                    {q: {"quality": v} for q, v in base.items()},
                    base,
                ),
                "arms": arms,
            },
        }
    ]


def fixed_plans_pool(
    ds: str, expected: list[str], types: dict, base: dict
) -> list[dict]:
    """Fixed-plan reference arms on the test set (user 2026-10-23; ``fixed_plans_page``): one row group per plan
    default; an empty "pending" table until the fixed-plan report exists."""
    from trimem.experiments import fixed_plans_page as fp

    t = fp.table(ds)
    head = {
        "baseline": {"label": "controlled-v1 NoMemory"},
        "notes": [fp.status_line(t)],
    }
    if t.get("status") != "complete":
        return [
            {
                **head,
                "path": f"fixed-plan defaults ({t.get('status')})",
                "difficulty": {"baseline": {}, "arms": {}},
            }
        ]
    arms = {
        a["label"]: summarize(
            ds,
            expected,
            types,
            {
                q: {"quality": x["q"], "tokens": x["tok"], "latency_s": x["lat"]}
                for q, x in a["items"].items()
            },
            base,
        )
        for a in t["arms"]
    }
    return [
        {
            **head,
            "path": f"{t['dir']} (fixed-plan defaults)",
            "difficulty": {
                "baseline": summarize(
                    ds,
                    expected,
                    types,
                    {q: {"quality": v} for q, v in base.items()},
                    base,
                ),
                "arms": arms,
            },
        }
    ]


def collect() -> dict:
    from trimem.experiments import render_atlas_results as ra
    from trimem.experiments import render_profiling_points as rp

    report = {
        "generated": datetime.now(UTC).isoformat(),
        "definition": DEFINITION,
        "datasets": {},
    }
    for ds in ("alfworld", "sciworld"):
        types = load_task_types(ds)
        expected = [
            r["item_id"]
            for part in ("a", "b")
            for r in json.loads((ra.CTRL / "shards" / f"{ds}_{part}.json").read_text())
        ]
        base = {
            f"{ds}_{q}": r["q"]
            for q, r in ra.agentenv(ds, "NoMemory", tolerant=True).items()
        }
        arms = {}
        for path in sorted(ra.CTRL.glob(f"{ds}_*")):
            if not path.is_dir() or not any(path.glob(f"{ds}_[0-9]*.json")):
                continue  # nested lambda-fill pair shards are not independent full-panel arms
            arm = path.name[len(ds) + 1 :]
            if ds == "alfworld" and arm not in ALF_REFERENCES:
                continue  # user 2026-10-22: ALFWorld = the v7 re-test + its No-Memory / VGR references only
            episodes = {
                f"{ds}_{q}": {
                    "quality": r["q"],
                    "tokens": r["tin"] + r["tout"] + r["tsem"]
                    if r["cost_complete"]
                    else None,
                    "latency_s": r["ret"] + r["gen"] if r["cost_complete"] else None,
                }
                for q, r in ra.agentenv(ds, arm, tolerant=True).items()
            }
            arms[arm] = summarize(ds, expected, types, episodes, base)
        pool = []
        if ds == "alfworld":
            pool = alf_v7_pool(ds, expected, types, base)
        for path in sorted(
            (ROOT / "results/bo_agentenv_20261016").glob(
                f"{ds}_pool-strat200*/test_report.json"
            )
            if ds != "alfworld"
            else []
        ):
            r = json.loads(path.read_text())
            pool.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "baseline": r["baseline"],
                    "notes": r.get("notes", []),
                    "difficulty": r.get("difficulty"),
                }
            )
        pool += fixed_plans_pool(ds, expected, types, base)
        sweep = None if ds == "alfworld" else rp.test_sweep(ds)
        report["datasets"][ds] = {
            "controlled": arms,
            "lambda_sweep": sweep,
            "pool": pool,
        }
    return report


def tables(report: dict) -> list[tuple[str, list[list[str]]]]:
    fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    result = []
    for ds, data in report["datasets"].items():
        collections = [(f"{ds} · controlled-v1", data["controlled"])]
        sweep = data.get("lambda_sweep")
        if sweep:
            collections.append(
                (
                    f"{ds} · controlled-v1 Opt-PerQuery λ replay（复用相同配置/题目）",
                    {
                        f"lambda={p['lambda']}": p["difficulty"]
                        for p in sweep["points"]
                        if "difficulty" in p
                    },
                )
            )
        for p in data["pool"]:
            d = p.get("difficulty")
            if d:
                collections.append(
                    (
                        f"{ds} · {p['path']} · baseline: {p['baseline']['label']}",
                        {"NoMemory": d["baseline"], **d["arms"]},
                    )
                )
        for title, arms in collections:
            rows = []
            for name, groups in arms.items():
                for group, r in groups.items():
                    if group == "unknown" and not r["expected"]:
                        continue
                    rows.append(
                        [
                            name,
                            group,
                            f"{r['n']}/{r['expected']}",
                            fmt(r["quality"]),
                            fmt(r["quality_se"]),
                            fmt(r["success_rate"]),
                            fmt(r["paired_gain"]),
                            fmt(r["paired_gain_se"]),
                            str(r["paired_n"]),
                            fmt(r["tokens"]),
                            fmt(r["latency_s"]),
                        ]
                    )
            result.append((title, rows))
    return result


def write_report() -> dict:
    # Serialize collection as well as publication so an older snapshot cannot win.
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "refresh.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _write_report()


def _write_report() -> dict:
    report = collect()
    headers = [
        "Arm",
        "分组",
        "已完成/计划",
        "Quality",
        "SE",
        "成功率",
        "Δ NoMemory",
        "Δ SE",
        "配对数",
        "Token",
        "延迟 s",
    ]
    note = (
        "ExpRAG v2 Appendix B.2 的任务类型划分，不是官方难度等级。Quality = ALFWorld 成功率 / "
        "ScienceWorld score÷100，成功率 = reward 达到满分。均值按任务等权，Δ 为同题配对增益（0–1尺度）。"
        "未完成组仅为已完成样本的临时统计。SE 在少于2题时不估计。"
        "controlled-v1 延迟为检索+模型调用时间，pool 延迟为端到端+router开销，两者不直接比较。"
        "pool 若沿用旧 NoMemory，其服务配置和负载不同。不同 λ 的复用 episode 不是独立重复实验。"
        "排队实验完成并刷新页面后自动补齐下表，分组不参与 BO / SLO 选择。"
    )
    md = [
        "# ALFWorld / ScienceWorld easy-hard metrics",
        "",
        report["generated"],
        "",
        note,
        "",
        f"Source: {DEFINITION['source']}",
        "",
    ]
    sections = [
        START,
        '<section class="panel section" id="difficulty-results"><h2>ALFWorld / ScienceWorld · easy / hard</h2>',
        f"<p>{html.escape(note)}</p><p>{html.escape(report['generated'])} · "
        f'<a href="{DEFINITION["source"]}">ExpRAG 分类依据</a></p>',
    ]
    for title, rows in tables(report):
        md += [
            "## " + title,
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
        ]
        md += ["| " + " | ".join(row) + " |" for row in rows] + [""]
        sections += [
            f'<details><summary>{html.escape(title)}</summary><div style="overflow:auto"><table><thead><tr>',
            "".join(f"<th>{html.escape(h)}</th>" for h in headers),
            "</tr></thead><tbody>",
        ]
        sections += [
            "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>"
            for row in rows
        ]
        sections.append("</tbody></table></div></details>")
    sections += ["</section>", END]
    sec = "\n".join(sections)
    OUT.mkdir(parents=True, exist_ok=True)
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        page = PAGE.read_text()
        if START in page:
            page = page[: page.index(START)] + sec + page[page.index(END) + len(END) :]
        else:
            anchor = "<!-- profiling-points:end -->"
            if anchor not in page:
                anchor = "<!-- four-dataset-results:end -->"
            if anchor not in page:
                raise ValueError("results section missing in HTML")
            page = page.replace(anchor, anchor + "\n" + sec, 1)
        for path, content in [
            (
                OUT / "report.json",
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            ),
            (OUT / "report.md", "\n".join(md)),
            (PAGE, page),
        ]:
            tmp = path.with_suffix(path.suffix + ".difficulty.tmp")
            from trimem.experiments.atlas_presentation import prepare_html

            tmp.write_text(prepare_html(content) if path == PAGE else content)
            tmp.replace(path)
    return report


if __name__ == "__main__":
    write_report()
    print(OUT / "report.md")
