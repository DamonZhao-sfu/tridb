"""Export the W1 measurements collected so far into one workbook.

    python3 tools/evotrace/export_xlsx.py --out results/e2/w1/W1_results.xlsx

Every sheet names its source run and its status, because the runs are NOT
interchangeable: the first full sweep used an arm-dependent nDCG ideal, which flattered
arms that reached less, and its quality columns are superseded. Its parity, recall and
latency columns are unaffected and still stand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
SUB_FILL = PatternFill("solid", fgColor="D9E2F3")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="FCE4E4")
GOOD_FILL = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _head(ws: Any, row: int, values: list[str], *, width: dict[int, int] | None = None) -> None:
    for col, value in enumerate(values, 1):
        cell = ws.cell(row=row, column=col, value=value)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[row].height = 30
    for col, size in (width or {}).items():
        ws.column_dimensions[get_column_letter(col)].width = size


def _title(ws: Any, text: str, note: str = "") -> int:
    ws["A1"] = text
    ws["A1"].font = Font(name=FONT, bold=True, size=13)
    row = 2
    if note:
        ws["A2"] = note
        ws["A2"].font = Font(name=FONT, italic=True, size=9, color="7F7F7F")
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[2].height = 42
        row = 3
    return row + 1


def _row(ws: Any, row: int, values: list[Any], *, fills: dict[int, PatternFill] | None = None,
         fmts: dict[int, str] | None = None, bold: bool = False) -> None:
    for col, value in enumerate(values, 1):
        cell = ws.cell(row=row, column=col, value=value)
        cell.font = Font(name=FONT, size=10, bold=bold)
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center" if col > 1 else "left", vertical="center")
        if fmts and col in fmts:
            cell.number_format = fmts[col]
        if fills and col in fills:
            cell.fill = fills[col]


# ---------------------------------------------------------------------------


def sheet_readme(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("00_README")
    row = _title(
        ws,
        "EvoTrace -> GEM/TriDB Experience Graph: W1 measurements",
        "One row per sheet. Read the STATUS column before quoting any number: the first "
        "full sweep's quality columns are superseded by a methodology fix (see 05).",
    )
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 52
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 46
    _head(ws, row, ["Sheet", "Contents", "Status", "Source"])
    rows = [
        ("01_Corpus", "Corpus counts vs the paper's own claims", "FINAL",
         "data/evotrace/normalized/integrity_report.json"),
        ("02_Workload", "The four W1 queries: shape, seed, population", "FINAL",
         "experiments/e2/w1_runner.py SHAPES + groundtruth.py"),
        ("03_Quality_full", "Quality + latency, all 18,291 points x 5 arms",
         "QUALITY SUPERSEDED", "results/e2/w1/full/w1_report.json (v1)"),
        ("04_Quality_fixed", "Quality with the arm-independent ideal", "PRELIMINARY n=16",
         "smoke run after the fix; full v2 sweep in flight"),
        ("05_Corrections", "Every defect found and what it changed", "FINAL", "this session"),
        ("06_Latency_legs", "ann / graph / filter / vector breakdown", "PRELIMINARY n=32",
         "results/e2/w1/latency (w1_latency.py)"),
        ("07_Materialisation", "Intermediate-result volume per stage", "PRELIMINARY n=32",
         "same run as 06"),
        ("08_W1a_focus", "W1.a only - the comparison target", "MIXED",
         "assembled from 03/04/06/07"),
    ]
    for offset, values in enumerate(rows):
        fill = {3: GOOD_FILL} if values[2] == "FINAL" else {3: WARN_FILL}
        if "SUPERSEDED" in values[2]:
            fill = {3: BAD_FILL}
        _row(ws, row + 1 + offset, list(values), fills=fill)


def sheet_corpus(wb: Workbook, data: dict[str, Any]) -> None:
    integrity = data["integrity"]
    ws = wb.create_sheet("01_Corpus")
    row = _title(
        ws,
        "Corpus, against the source paper's own counts",
        "Dataset ZIB-IOL/EvoTrace @ 349117b0 (CC-BY-4.0). Six of seven counts reproduce "
        "the paper exactly, which is the strongest evidence the normalizer is correct. "
        "Task count is the one disagreement and is reported, not adopted.",
    )
    _head(ws, row, ["Metric", "Observed", "Paper claim", "Agrees"],
          width={1: 26, 2: 14, 3: 14, 4: 10})
    r = row + 1
    for key, comp in integrity["paper_comparison"].items():
        agrees = comp["agrees"]
        _row(ws, r, [key, comp["observed"], comp["paper_claim"], "yes" if agrees else "NO"],
             fills={4: GOOD_FILL if agrees else BAD_FILL}, fmts={2: "#,##0", 3: "#,##0"})
        r += 1
    r += 1
    _head(ws, r, ["Derived", "Value", "", ""])
    r += 1
    obs = integrity["observed"]
    derived = [
        ("base entities (task+session+node)", integrity["base_entities"]),
        ("canonical logical edges", integrity["canonical_logical_edges"]),
        ("context edges", obs["context_edges"]),
        ("unique artifacts", obs["unique_artifacts"]),
        ("artifact references", obs["artifact_references"]),
        ("artifacts shared across sessions", integrity["artifacts_shared_across_sessions"]),
        ("prompts", obs["prompts"]),
        ("state events", obs["state_events"]),
        ("core-graph-ready runs", integrity["core_graph_ready_runs"]),
        ("runs with prompt history", integrity["runs_with_prompts"]),
    ]
    for label, value in derived:
        _row(ws, r, [label, value, "", ""], fmts={2: "#,##0"})
        r += 1
    r += 1
    _row(ws, r, ["domain split", str(integrity["domain_split"]), "", ""], bold=True)
    _row(ws, r + 1, ["backend split", str(integrity["backend_split"]), "", ""], bold=True)


WORKLOAD = [
    ("W1.a", "cross-session reuse", "every real parent->child step", 10479,
     "target Task description (18 vectors)", "eg_hier", 2,
     "reward-graded relevance (5 levels)", "18/18 tasks, 4/4 backends"),
    ("W1.b", "stuck-state escape", "child did NOT beat its parent", 6525,
     "the stuck node's code (10,672 vectors)", "eg_lineage", 3,
     "retrieved fitness > stuck point", "4/4 backends (52-84%)"),
    ("W1.b-fail", "true failure repair", "executor rejected it (TLE/compile/crash)", 668,
     "failed node code + error signature", "eg_lineage", 2,
     "sibling or child is clean", "3/4 backends, 46/121 sessions"),
    ("W1.d", "improvement-path reconstruction", "node set a new best-so-far", 619,
     "breakthrough node code", "eg_child_of (inverse)", 4,
     "dataset's own best_so_far_lineages", "95/121 runs have precomputed truth"),
]


def sheet_workload(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("02_Workload")
    row = _title(
        ws,
        "The W1 query family",
        "Each query is one or two tjs_open calls: a seedless ANN entry, then a "
        "filter-first bounded typed traversal. The predicate is identical across all "
        "four: kind='node' AND is_valid AND fitness IS NOT NULL AND session_uid<>target.",
    )
    _head(ws, row, ["Query", "Name", "Fires when", "Decision points", "Seed vector",
                    "Edge type", "Hops", "Ground truth", "Coverage"],
          width={1: 11, 2: 30, 3: 36, 4: 14, 5: 34, 6: 20, 7: 7, 8: 32, 9: 30})
    for offset, values in enumerate(WORKLOAD):
        fill = {4: SUB_FILL} if values[0] == "W1.a" else None
        _row(ws, row + 1 + offset, list(values), fills=fill, fmts={4: "#,##0"},
             bold=values[0] == "W1.a")
    r = row + len(WORKLOAD) + 2
    _row(ws, r, ["TOTAL", "", "", sum(w[3] for w in WORKLOAD), "", "", "", "", ""],
         bold=True, fmts={4: "#,##0"})
    r += 2
    ws.cell(row=r, column=1, value="Arms (all inside TriDB, no external baseline)").font = Font(
        name=FONT, bold=True, size=11)
    r += 1
    _head(ws, r, ["Arm", "vector", "graph", "relational", "Ranking", "Represents", "", "", ""])
    arms = [
        ("fused", "yes", "yes", "yes", "similarity", "the full fused query"),
        ("no_graph", "yes", "no", "yes", "similarity", "a vector store + predicate"),
        ("no_vector", "no", "yes", "yes", "reward", "a graph store + predicate"),
        ("filter_only", "no", "no", "yes", "reward", "a relational scan"),
        ("reward_rank", "yes", "yes", "yes", "reward",
         "NOT an ablation: the G4 axis (tjs_open can only rank by vector)"),
    ]
    for offset, values in enumerate(arms):
        _row(ws, r + 1 + offset, list(values) + ["", "", ""])


def sheet_quality(wb: Workbook, data: dict[str, Any], key: str, name: str,
                  title: str, note: str, superseded: bool) -> None:
    report = data[key]
    ws = wb.create_sheet(name)
    row = _title(ws, title, note)
    _head(ws, row, ["Query", "Arm", "n", "n scored", "empty ideal", "parity (exact)",
                    "parity (tie-eq)", "oracle recall", "nDCG@10", "ceiling", "rank%",
                    "harmful@10", "trivial hit", "first row p50 ms", "first row p95 ms",
                    "total p50 ms", "examined p50"],
          width={1: 11, 2: 13, 3: 9, 4: 9, 5: 11, 6: 11, 7: 11, 8: 11, 9: 10, 10: 10,
                 11: 9, 12: 11, 13: 10, 14: 13, 15: 13, 16: 12, 17: 12})
    r = row + 1
    for cell_key in sorted(report["cells"]):
        query_id, arm = cell_key.split("|")
        s = report["cells"][cell_key]
        highlight = {1: SUB_FILL, 2: SUB_FILL} if query_id == "W1.a" else None
        _row(
            ws, r,
            [query_id, arm, s["n"], s["n_scored"], s["n_empty_ideal"],
             s["parity_exact_rate"], s.get("parity_tie_equivalent_rate"),
             s["oracle_recall"], s["ndcg"], s.get("ndcg_ceiling"), s.get("rank_efficiency"),
             s["harmful"], s["trivial_hit_rate"], s["first_row_ms_p50"],
             s["first_row_ms_p95"], s["total_ms_p50"], s["examined_p50"]],
            fills=highlight,
            fmts={3: "#,##0", 4: "#,##0", 5: "#,##0", 6: "0.000", 7: "0.000", 8: "0.0000",
                  9: "0.0000", 10: "0.0000", 11: "0.00", 12: "0.000", 13: "0.0000",
                  14: "0.00", 15: "0.00", 16: "0.00", 17: "#,##0"},
        )
        r += 1
    if superseded:
        r += 1
        cell = ws.cell(row=r, column=1,
                       value="SUPERSEDED (quality columns only): nDCG here used an "
                             "arm-dependent ideal, which drops a decision point from "
                             "scoring when THAT ARM's own pool held nothing good - "
                             "flattering arms that reach less. Parity, recall and "
                             "latency columns are unaffected.")
        cell.font = Font(name=FONT, bold=True, size=10, color="9C0006")
        cell.fill = BAD_FILL
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=17)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[r].height = 46


CORRECTIONS = [
    ("data", "combined_score = -2^63 sentinel", 193,
     "163 of them are status=accepted and 50 carry no error signature, so they read as "
     "healthy nodes with an astronomically bad reward; drags a task's reward minimum to "
     "-9.2e18 and wrecks every percentile the selectivity sweep needs",
     "detected by EXACT equality (nearest real reward is -4.25e7, 11 orders away) -> fitness=None + recorded"),
    ("data", "NaN / Inf reward", 38,
     "NaN > x is False for every x: one NaN silently turns every reward comparison in "
     "the ground truth into 'did not improve' while raising nothing",
     "fitness=None + `nonfinite_metrics` recorded; json.dumps(allow_nan=False) as a fail-closed net"),
    ("data", "NUL (U+0000) inside metrics text", "several sessions",
     "PostgreSQL text/jsonb cannot represent it; the load aborted mid-run",
     "stripped and recorded per field"),
    ("data", "error_signature labelled search-rejection as execution failure", 698,
     "would have put 698 non-defects into the repair population",
     "prefix now names WHO rejected it; execution failures went 266 -> 1,056 and span 3/4 backends"),
    ("data", "duplicate (src,dst) context edges", 13,
     "silently eaten by the loader's primary key three stages later",
     "deduped in the normalizer and counted"),
    ("harness", "stage2 used knobs.hops, oracle used spec.hops", "W1.b, W1.d",
     "W1.a coincidentally had both = 2 so it looked fine; W1.b (3 vs 2) and W1.d (4 vs 2) "
     "silently ran a different query than the oracle scored",
     "QuerySpec now owns WHAT the answer is; Knobs owns only HOW hard to look. W1.d recall 0.79 -> 1.00"),
    ("harness", "TIE_EPS = 1e-9, tighter than float32 storage", "all queries",
     "engine and oracle differ by 5.96e-08 (2^-24) on genuinely tied pairs; 10,672 nodes "
     "share only 9,733 artifacts so ties are structural",
     "TIE_EPS = 1e-6: three orders above the noise, three below the ~1e-3 signal"),
    ("harness", "ef_search sweep had no effect", "40/100/400/1000",
     "the residual was NOT HNSW approximation - relaxed_order scans exhaustively, so "
     "stage 1 was already exact; the residual was ill-posed entry selection at ties",
     "entry selection made total by (distance, uid). W1.b 0.875 -> 0.969, W1.d 0.906 -> PASS"),
    ("harness", "graph_only arm still selected entries by ANN", "all queries",
     "so 'fused vs graph_only' compared two RANKING functions while both used the vector "
     "leg - it could not answer the question its name implied",
     "arms redefined so each name's removed leg is actually removed; ranking is now its own axis"),
    ("metric", "nDCG ideal was arm-dependent", "all queries",
     "an arm that reaches less gets an easier ideal, and a point where its pool held "
     "nothing good was dropped from scoring entirely (36.6% of W1.a's for no_graph, "
     "14.0% of W1.b's for fused) - the arms were averaged over different denominators",
     "ideal now describes the TASK, not the retriever. Conclusion FLIPPED: no_vector wins all four"),
]


def sheet_corrections(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("05_Corrections")
    row = _title(
        ws,
        "Defects found and what each one changed",
        "Recorded because several of them produced plausible-looking but wrong numbers "
        "that only a parity gate against an independent oracle caught.",
    )
    _head(ws, row, ["Layer", "Defect", "Scale", "Why it mattered", "Fix / effect"],
          width={1: 10, 2: 44, 3: 16, 4: 60, 5: 60})
    for offset, values in enumerate(CORRECTIONS):
        r = row + 1 + offset
        _row(ws, r, list(values))
        for col in (4, 5):
            ws.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 58
        fill = BAD_FILL if values[0] == "metric" else (WARN_FILL if values[0] == "harness" else None)
        if fill:
            ws.cell(row=r, column=1).fill = fill


def sheet_latency(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("06_Latency_legs")
    row = _title(
        ws,
        "Per-leg latency (ms) - what each modality costs, and what fusing saves",
        "tjs_open interleaves the three legs inside one C call and exposes no internal "
        "timers, so the split is measured by INCREMENTAL COMPOSITION: each leg executed "
        "separately. ann+graph+filter+vector is exactly what an unfused pipeline pays. "
        "Measured concurrently with another sweep, so absolute values are inflated.",
    )
    _head(ws, row, ["m_seeds", "Query", "n", "ann", "graph", "filter", "vector",
                    "ANN share of fused", "unfused expand", "fused expand",
                    "expand saved", "expand speedup", "fused total", "fused first row"],
          width={1: 9, 2: 11, 3: 8, 4: 10, 5: 9, 6: 9, 7: 9, 8: 16, 9: 14, 10: 13,
                 11: 12, 12: 13, 13: 12, 14: 14})
    r = row + 1
    for label, key in (("4", "lat_m4"), ("1", "lat_m1")):
        for query_id, s in data[key]["per_query"].items():
            legs = s["legs_ms"]
            first = s.get("fused_first_row_ms")
            _row(
                ws, r,
                [label, query_id, s["n"], legs["ann"]["p50"], legs["graph"]["p50"],
                 legs["filter"]["p50"], legs["vector"]["p50"], s["ann_share_of_fused"],
                 s["unfused_expand_ms"]["p50"], s["fused_expand_ms"]["p50"],
                 s["expand_saving_ms"]["p50"], s["expand_speedup"],
                 s["fused_ms"]["p50"], first["p50"] if first else None],
                fills={2: SUB_FILL} if query_id == "W1.a" else None,
                fmts={3: "#,##0", 4: "0.00", 5: "0.00", 6: "0.00", 7: "0.00",
                      8: "0.0%", 9: "0.00", 10: "0.00", 11: "0.00", 12: "0.00",
                      13: "0.00", 14: "0.00"},
                bold=query_id == "W1.a",
            )
            r += 1
    r += 1
    for text in (
        "Finding: the workload is ANN-BOUND. For the three node-seeded queries the ANN "
        "entry is 92-97% of total latency and graph/filter/vector are all sub-millisecond.",
        "Finding: fusion is net negative here by a CONSTANT ~0.5 ms per tjs_open call "
        "(server-side cursor setup). Dropping m_seeds 4 -> 1 leaves the gap unchanged, so "
        "it is fixed per-call overhead, not a scaling disadvantage.",
        "W1.a is the only query where the non-ANN legs are material (graph 3.58 + filter "
        "8.67 + vector 15.75 ms against a 35.84 ms ANN).",
    ):
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = Font(name=FONT, size=10, italic=True)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=14)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[r].height = 30
        r += 1


def sheet_materialisation(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("07_Materialisation")
    row = _title(
        ws,
        "Intermediate-result volume and work done per query",
        "The rows an UNFUSED pipeline has to materialise at each stage boundary. This is "
        "the quantity a fused operator never creates, and the reason W1.a is the only "
        "query in this family where fusion has anything to save.",
    )
    _head(ws, row, ["Query", "rows after graph", "rows after filter", "rows returned",
                    "reduction graph->answer", "graph edge-steps", "distance computations"],
          width={1: 12, 2: 17, 3: 17, 4: 14, 5: 22, 6: 17, 7: 20})
    r = row + 1
    for query_id, s in data["lat_m4"]["per_query"].items():
        m, w = s["rows_materialised"], s["work"]
        ws.cell(row=r, column=1)  # placeholder so _row writes cleanly
        _row(ws, r,
             [query_id, m["after_graph_p50"], m["after_filter_p50"], m["returned_p50"],
              None, w["graph_examined_p50"], w["candidates_examined_p50"]],
             fills={1: SUB_FILL} if query_id == "W1.a" else None,
             fmts={2: "#,##0", 3: "#,##0", 4: "#,##0", 5: "0.0\"x\"", 6: "#,##0", 7: "#,##0"},
             bold=query_id == "W1.a")
        # Formula, not a Python-computed constant, so the sheet recalculates.
        ws.cell(row=r, column=5, value=f"=IF(D{r}=0,\"\",B{r}/D{r})")
        ws.cell(row=r, column=5).number_format = '0.0"x"'
        ws.cell(row=r, column=5).font = Font(name=FONT, size=10)
        ws.cell(row=r, column=5).border = BORDER
        ws.cell(row=r, column=5).alignment = Alignment(horizontal="center")
        r += 1
    r += 1
    cell = ws.cell(row=r, column=1,
                   value="W1.a materialises 2,751 rows to return 10. The three lineage "
                         "queries reach only 6-12 rows, so their graph leg does almost "
                         "nothing - which is why their fused ceiling is 0.29-0.42 despite "
                         "a rank efficiency of 0.66-0.91.")
    cell.font = Font(name=FONT, size=10, italic=True)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[r].height = 44


def sheet_w1a(wb: Workbook, data: dict[str, Any]) -> None:
    ws = wb.create_sheet("08_W1a_focus")
    row = _title(
        ws,
        "W1.a only - the comparison target for Mem0 / Cognee / MemOS",
        "10,479 decision points. Seeded by a Task DESCRIPTION vector (18 of them) and "
        "expanded Task -> Session -> Node over the eg_hier rollup type at 2 hops.",
    )
    _head(ws, row, ["Metric", "fused", "no_graph", "no_vector", "filter_only", "reward_rank",
                    "Source"],
          width={1: 32, 2: 12, 3: 12, 4: 12, 5: 12, 6: 13, 7: 30})
    arms = ["fused", "no_graph", "no_vector", "filter_only", "reward_rank"]
    r = row + 1
    metrics = [
        ("n (decision points)", "n", "#,##0", "03 v1 full"),
        ("parity, tie-equivalent", "parity_tie_equivalent_rate", "0.000", "03 v1 full"),
        ("oracle recall", "oracle_recall", "0.0000", "03 v1 full"),
        ("first row p50 (ms)", "first_row_ms_p50", "0.00", "03 v1 full"),
        ("first row p95 (ms)", "first_row_ms_p95", "0.00", "03 v1 full"),
        ("candidates examined p50", "examined_p50", "#,##0", "03 v1 full"),
    ]
    for label, key, fmt, source in metrics:
        values = [data["v1_full"]["cells"][f"W1.a|{a}"].get(key) for a in arms]
        _row(ws, r, [label, *values, source], fmts={i: fmt for i in range(2, 7)})
        r += 1
    r += 1
    _row(ws, r, ["QUALITY - arm-independent ideal (preliminary, n=16)", "", "", "", "", "", ""],
         bold=True, fills={1: WARN_FILL})
    r += 1
    for label, key, fmt in (
        ("nDCG@10", "ndcg", "0.0000"),
        ("ceiling (reachability)", "ndcg_ceiling", "0.0000"),
        ("rank efficiency", "rank_efficiency", "0.00"),
        ("harmful@10", "harmful", "0.000"),
    ):
        values = [data["smoke_corrected"]["cells"][f"W1.a|{a}"].get(key) for a in arms]
        _row(ws, r, [label, *values, "04 fixed-ideal smoke"],
             fmts={i: fmt for i in range(2, 7)})
        r += 1
    r += 2
    ws.cell(row=r, column=1, value="Per-leg latency, W1.a (ms, m_seeds=4)").font = Font(
        name=FONT, bold=True, size=11)
    r += 1
    _head(ws, r, ["ann", "graph", "filter", "vector", "unfused total", "fused total",
                  "ANN share", ""])
    legs = data["lat_m4"]["per_query"]["W1.a"]
    r += 1
    _row(ws, r, [legs["legs_ms"]["ann"]["p50"], legs["legs_ms"]["graph"]["p50"],
                 legs["legs_ms"]["filter"]["p50"], legs["legs_ms"]["vector"]["p50"],
                 legs["unfused_ms"]["p50"], legs["fused_ms"]["p50"],
                 legs["ann_share_of_fused"], ""],
         fmts={1: "0.00", 2: "0.00", 3: "0.00", 4: "0.00", 5: "0.00", 6: "0.00", 7: "0.0%"})
    r += 2
    for text in (
        "Comparison contract for Mem0 / Cognee / MemOS: the ground truth is "
        "system-independent, so nDCG@10, ceiling, rank efficiency and harmful@10 are "
        "directly comparable across systems once each system's returned ids are mapped "
        "back to node_uid.",
        "Latency is comparable only for the search headline. first_row_ms has no analogue "
        "in a system that returns a materialised list - report it as N/A, never as a number.",
        "Embedding model must be identical across systems (Qwen3-Embedding-0.6B, 1024d, "
        "cosine) or the experiment measures embedding quality, not system quality.",
    ):
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = Font(name=FONT, size=10, italic=True)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[r].height = 32
        r += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/e2/w1/W1_results.xlsx"))
    args = parser.parse_args(argv)

    data = json.loads(args.bundle.read_text())
    wb = Workbook()
    wb.remove(wb.active)

    sheet_readme(wb, data)
    sheet_corpus(wb, data)
    sheet_workload(wb, data)
    sheet_quality(
        wb, data, "v1_full", "03_Quality_full",
        "Full sweep: 18,291 decision points x 5 arms (v1)",
        "Parity, oracle recall and latency are FINAL. The nDCG / harmful columns used an "
        "arm-dependent ideal and are SUPERSEDED - see sheet 05, row 'nDCG ideal was "
        "arm-dependent'. A corrected full sweep (v2) is in flight.",
        superseded=True,
    )
    sheet_quality(
        wb, data, "smoke_corrected", "04_Quality_fixed",
        "Quality with the arm-independent ideal (preliminary, n=16 per cell)",
        "Same code path as the v2 full sweep. Small n: good enough to establish direction "
        "and the ceiling/rank decomposition, not enough for confidence intervals.",
        superseded=False,
    )
    sheet_corrections(wb, data)
    sheet_latency(wb, data)
    sheet_materialisation(wb, data)
    sheet_w1a(wb, data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"wrote {args.out}  ({len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
