"""ALFWorld TRAIN search curves: TRACE-TC (task-complementary-v1, run tc27tq) versus TRACE (trace-hv, run hl27trace).

Both arms share the 8 initial configurations, TRAIN samples, NoMemory reference and 20/40/60/80/100 staging; TRACE ran
3 rounds (24 proposals), TRACE-TC 6 rounds (48). Metric: the page's fixed-normalization 3-objective HV (quality,
tokens, latency) over the eligible (fully evaluated) archive. Dense points re-evaluate the archive at every full
completion (plot_sci_hv_hours.dense_curve); round ends equal the e2 checkpoints. Hours are active search time rebased to
the round-0 checkpoint. Writes PNG/PDF/CSV/summary under results/trace_tc_alfworld_20261027_tcprune/train_compare,
one xlsx sheet, and adds a "TRACE-TC (1 seed)" run to the ALFWorld block of the page's hcp-data (Search efficiency).
"""

from __future__ import annotations

import csv
import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper/tools"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "paper/sigmod/revision_20261027_sci_hv_hours"))
import nhv_alfworld_additions as nhv  # noqa: E402
import plot_sci_hv_hours as dense  # noqa: E402
import render_search_cost_preview as preview  # noqa: E402

PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
TC = ROOT / "results/trace_tc_alfworld_20261027_tcprune/campaign/runs/alfworld_pool-strat200-v7-tc27tq__mem-sf1"
OUT = ROOT / "results/trace_tc_alfworld_20261027_tcprune/train_compare"
WORKBOOK = ROOT / "triMEM_optimizer/trimem_experiments.xlsx"
SHEET = "alf_trace_tc_train_hv"
ARMS = [("TRACE", nhv.TRACE, "#24856c", "D"), ("TRACE-TC", TC, "#d55e00", "o")]


def read(path):
    return json.loads(Path(path).read_text())


def round_stats(base):
    out = []
    for rr in read(base / "state.json")["rounds"]:
        pruned = [e for e in rr["configs"] if e["staging"].get("pruned_at") is not None]
        out.append({"round": rr["round"], "proposed": len(rr["configs"]), "pruned": len(pruned)})
    return out


def main():
    from trimem.experiments import alfworld_e2_campaign as c
    from trimem.experiments import bo_pool_driver as d

    normalization = read(ROOT / "results/exp2_alfworld_prepare_20261024_run/manifest.json")["normalization"]

    def key_of(row):
        return d.space.key(d.space.validate(row["protocol"]["space_config"]))

    rows, summary, curves = [], {}, {}
    for label, base, *_ in ARMS:
        cps = nhv.checkpoints(base, normalization)
        pts = dense.dense_curve({"directory": str(base)}, normalization, c.read_rows, key_of)
        h0 = cps[0]["hours"]
        for cp, p in zip(cps, [p for p in pts if p[4] == "checkpoint"], strict=True):
            assert abs(cp["hv3"] - p[1]) < 1e-12 and abs(cp["hours"] - h0 - p[0]) < 1e-6, (label, cp["round"])
        curves[label] = (cps, pts)
        stats = round_stats(base)
        for p in pts:
            rows.append({"arm": label, "record": p[4], "round": p[3], "hours": p[0], "hv3": p[1], "full_configs": p[2]})
        summary[label] = {
            "rounds": len(cps),
            "round_end": [
                {"round": cp["round"], "hours": round(cp["hours"] - h0, 3), "hv3": round(cp["hv3"], 4),
                 "hv_tokens": round(cp["hv_tokens"], 4), "hv_latency": round(cp["hv_latency"], 4),
                 "full_configs": cp["configs"], "episodes": cp["episodes"], "tokens": cp["tokens"],
                 **{k: s[k] for k in ("proposed", "pruned")}}
                for cp, s in zip(cps, stats, strict=True)
            ],
        }
        summary[label]["gain_hv3"] = round(cps[-1]["hv3"] - cps[0]["hv3"], 4)
    # Matched budget: TRACE-TC at TRACE's last round and at TRACE's final active hours.
    tr_cps, _ = curves["TRACE"]
    tc_cps, tc_pts = curves["TRACE-TC"]
    tr_hours = tr_cps[-1]["hours"] - tr_cps[0]["hours"]
    summary["matched"] = {
        "trace_final": {"round": tr_cps[-1]["round"], "hours": round(tr_hours, 3), "hv3": round(tr_cps[-1]["hv3"], 4)},
        "tc_same_round": round(tc_cps[len(tr_cps) - 1]["hv3"], 4),
        "tc_at_trace_hours": round(max((p for p in tc_pts if p[0] <= tr_hours), key=lambda p: p[0])[1], 4),
    }

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    xend = max(pts[-1][0] for _, pts in curves.values())
    for label, _, color, marker in ARMS:
        _, pts = curves[label]
        xs = [p[0] for p in pts] + [xend]
        ys = [p[1] for p in pts] + [pts[-1][1]]
        ax.step(xs, ys, where="post", color=color, lw=1.4, label=label)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=marker, ms=3.2, mfc="white", mec=color)
        for p in pts:
            if p[4] == "checkpoint" and p[3] > 0:
                ax.annotate(f"r{p[3]}", (p[0], p[1]), textcoords="offset points", xytext=(2, -9), fontsize=6,
                            color=color)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Normalized HV (3-objective)")
    ax.set_title("ALFWorld TRAIN: TRACE-TC vs TRACE", fontsize=8)
    ax.grid(color="#d9dcdf", lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "train_hv_hours.png", dpi=200)
    fig.savefig(OUT / "train_hv_hours.pdf")
    with (OUT / "train_hv_hours.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import openpyxl

    wb = openpyxl.load_workbook(WORKBOOK)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)
    ws.append(list(rows[0]))
    for r in rows:
        ws.append(list(r.values()))
    wb.save(WORKBOOK)

    # Search-efficiency block of the page: add TRACE-TC beside TRACE (1 seed), same normalization and rebasing.
    pattern = re.compile(r'(<script[^>]*id="hcp-data"[^>]*>)(.*?)(</script>)', re.DOTALL)
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = PAGE.read_text()
        m = pattern.search(text)
        data = json.loads(m[2])
        if data["normalization"] != normalization:
            raise ValueError("normalization mismatch")
        alf = next(x for x in data["datasets"] if x["key"] == "alfworld")
        run = {
            "id": "trace-tc-1seed",
            "name": "TRACE-TC (1 seed)",
            "color": "#d55e00",
            "historical": False,
            "source": str(TC.relative_to(ROOT)),
            "seed": 20261023,
            "checkpoints": nhv.checkpoints(TC, normalization),
            "note": "task-complementary-v1 (DLS slot scored by per-query HV gain A_TC; tc-complementary-v1 pruning "
            "for TC picks), same 8 initial configs, samples and staging as TRACE (1 seed); rounds 0-1 shared with "
            "the original tc27tc arm, rounds 2-5 run tc27tq. One seed.",
            "budget_scope": "Candidate agent + semantic tokens (complete receipts); shared NoMemory/noise and analyzer "
            "usage excluded. Time = active search segments.",
        }
        preview.rebase_checkpoints({"datasets": [{"runs": [run]}]})
        alf["runs"] = [r for r in alf["runs"] if r["id"] != run["id"]] + [run]
        data.setdefault("additions", []).append({"at": "2026-10-28", "what": "ALFWorld TRACE-TC (1 seed) TRAIN curve"})
        text = text[: m.start(2)] + json.dumps(data).replace("</", "<\\/") + text[m.end(2):]
        tmp = PAGE.with_suffix(".tctrain.tmp")
        tmp.write_text(text)
        tmp.replace(PAGE)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
