"""The final tables: quality per arm, retrieval latency per system, and the
diagnostics that say whether either can be read.

    python3 -m bench.agent_memory.gem_oe.report --matrix bench/out/oe/all7_...

Separate from `analyze.py`, which reduces ONE matrix. This assembles the finished
comparison across arms and across systems, and it is the thing that must refuse to
print a number the data does not support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import psycopg

DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
#: Fractions of a task's improvable headroom. The primary metric is the iteration at
#: which each is first reached.
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)
#: Below this fraction of (corpus_best - seed), a task cannot discriminate between
#: arms: the whole spread is inside the evaluator's own cross-version drift.
HEADROOM_FLOOR = 0.30


def references(dsn: str) -> dict[str, dict[str, float]]:
    conn = psycopg.connect(dsn)
    rows = conn.execute(
        "SELECT b.task_uid, r.seed_fit, b.corpus_best FROM"
        " (SELECT task_uid, max(fitness) corpus_best FROM gem_eg_node"
        "  WHERE is_valid AND fitness IS NOT NULL GROUP BY 1) b"
        " JOIN (SELECT task_uid, max(fitness) seed_fit FROM gem_eg_node"
        "       WHERE parent_node_uid IS NULL AND is_valid AND fitness IS NOT NULL"
        "       GROUP BY 1) r USING (task_uid)"
    ).fetchall()
    out = {}
    for task, seed, best in rows:
        seed, best = float(seed), float(best)
        out[task] = {"seed": seed, "best": best, "gap": best - seed,
                     "headroom": (best - seed) / best if best else 0.0}
    return out


def read_cell(cell: Path, ref: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    receipt = cell / "run_receipt.json"
    trace = cell / "evolution_trace.jsonl"
    if not receipt.is_file() or not trace.is_file():
        return None
    meta = json.loads(receipt.read_text())
    task = meta["task_uid"]
    r = ref.get(task, {})

    best = float("-inf")
    curve: list[float] = []
    hashes: set[str] = set()
    copied = zero = scored = 0
    for line in trace.open():
        row = json.loads(line)
        code = (row.get("child_code") or "").strip()
        if code:
            hashes.add(hashlib.sha1(code.encode(), usedforsecurity=False).hexdigest())
            prompt = row.get("prompt") or {}
            text = (f"{prompt.get('system', '')}\n{prompt.get('user', '')}"
                    if isinstance(prompt, dict) else str(prompt))
            if len(code) > 200 and code in text:
                copied += 1
        score = (row.get("child_metrics") or {}).get("combined_score")
        if score is not None:
            scored += 1
            score = float(score)
            if score == 0.0:
                zero += 1
            best = max(best, score)
        curve.append(best if best > float("-inf") else 0.0)

    reached: dict[str, int | None] = {}
    for alpha in THRESHOLDS:
        target = r.get("seed", 0.0) + alpha * r.get("gap", 0.0)
        reached[f"t{int(alpha * 100)}"] = next(
            (i for i, v in enumerate(curve) if v >= target), None
        ) if r.get("gap", 0) > 0 else None

    tel = cell / "retrieval_telemetry.jsonl"
    lat: dict[str, float] = {}
    if tel.is_file():
        rows = [json.loads(x) for x in tel.open()]
        for key in ("first_row_ms", "total_ms", "ann_ms", "traverse_ms", "filter_rank_ms"):
            vals = [x[key] for x in rows if x.get(key) is not None]
            if vals:
                lat[key] = round(statistics.median(vals), 1)

    rate = meta.get("injection_rate_requested")
    arm = meta["arm"]
    label = arm if arm in ("nocontext", "none") or rate is None else f"{arm}@{rate:.0%}"
    return {
        "task": task.split(":")[-1], "arm": label, "base_arm": arm,
        "iters": len(curve), "best": None if best == float("-inf") else best,
        "curve": curve, "zero": zero, "scored": scored,
        "distinct": len(hashes), "copied": copied,
        "copy_rate": round(copied / len(curve), 4) if curve else None,
        "headroom": r.get("headroom"), "corpus_best": r.get("best"),
        "seed": r.get("seed"), **reached, **lat,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--min-iters", type=int, default=36)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    ref = references(args.dsn)
    cells = [c for c in sorted(args.matrix.rglob("math_*")) if c.is_dir()]
    rows = [r for r in (read_cell(c, ref) for c in cells) if r]
    short = [r for r in rows if r["iters"] < args.min_iters]
    rows = [r for r in rows if r["iters"] >= args.min_iters]
    if short:
        print(f"排除 {len(short)} 个未跑满的 cell(< {args.min_iters} 轮)\n")
    if not rows:
        raise SystemExit("no cell reached the minimum iteration count")

    arm_order = {"nocontext": 0, "none": 1, "gem@10%": 2, "gem@50%": 3,
                 "gem@100%": 4, "polyglot@100%": 5}
    rows.sort(key=lambda r: (-(r["headroom"] or 0), arm_order.get(r["arm"], 9)))

    print("表 1 — Quality(按可改进余量降序)")
    print(f"{'task':<22}{'arm':<14}{'余量':>7}{'best':>9}{'语料最佳':>10}"
          f"{'已达':>7}{'t25':>5}{'t50':>5}{'t75':>5}{'t90':>5}{'0分':>5}{'不同':>5}{'抄%':>6}")
    for r in rows:
        gap = ref.get(f"math:{r['task']}", {}).get("gap") or 1.0
        pct = (r["best"] - r["seed"]) / gap if r["best"] is not None else None
        low = "*" if (r["headroom"] or 0) < HEADROOM_FLOOR else " "
        print(f"{r['task']:<22}{r['arm']:<14}{(r['headroom'] or 0):>6.3f}{low}"
              f"{(r['best'] if r['best'] is not None else 0):>9.4f}"
              f"{(r['corpus_best'] or 0):>10.4f}{(f'{pct:.0%}' if pct is not None else '--'):>7}"
              + "".join(f"{(r[k] if r[k] is not None else '--'):>5}"
                        for k in ("t25", "t50", "t75", "t90"))
              + f"{r['zero']:>5}{r['distinct']:>5}{r['copy_rate'] * 100:>5.0f}%")
    print(f"\n  * 余量 < {HEADROOM_FLOOR:.0%}：该题上任何 arm 差异都落在 evaluator "
          "自身的库版本漂移内，不可解读")

    lat_rows = [r for r in rows if r.get("first_row_ms")]
    if lat_rows:
        print("\n表 2 — 检索延迟(仅有检索的 arm)")
        print(f"{'task':<22}{'arm':<14}{'first_row':>11}{'total':>9}"
              f"{'ann':>8}{'traverse':>10}{'filter':>9}")
        for r in lat_rows:
            print(f"{r['task']:<22}{r['arm']:<14}{r['first_row_ms']:>10.1f}ms"
                  f"{r.get('total_ms', 0):>8.1f}ms"
                  + "".join(f"{r.get(k, 0):>7.1f}ms"
                            for k in ("ann_ms", "traverse_ms", "filter_rank_ms")))
        print("\n  polyglot 的 first_row == total：三段串行，第三段不结束无法返回任何行。")

    print("\n表 3 — 按 arm 汇总(仅余量 >= 门槛的 task)")
    usable = [r for r in rows if (r["headroom"] or 0) >= HEADROOM_FLOOR]
    print(f"{'arm':<14}{'cells':>6}{'平均已达':>10}{'t50中位':>9}{'平均抄%':>9}{'不同/轮':>9}")
    for arm in sorted({r["arm"] for r in usable}, key=lambda a: arm_order.get(a, 9)):
        sub = [r for r in usable if r["arm"] == arm]
        gaps = [(r["best"] - r["seed"]) / (ref[f"math:{r['task']}"]["gap"] or 1)
                for r in sub if r["best"] is not None]
        t50 = [r["t50"] for r in sub if r["t50"] is not None]
        print(f"{arm:<14}{len(sub):>6}{statistics.mean(gaps):>9.1%}"
              f"{(statistics.median(t50) if t50 else float('nan')):>9.1f}"
              f"{statistics.mean(r['copy_rate'] for r in sub) * 100:>8.1f}%"
              f"{statistics.mean(r['distinct'] / r['iters'] for r in sub):>9.3f}")

    worst = max((r["copy_rate"] or 0) for r in rows)
    if worst > 0.05:
        print(f"\n  警告：某 cell 有 {worst:.0%} 的轮次原样复现了自己的 prompt。"
              "其曲线反映的是照抄而非搜索。")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
