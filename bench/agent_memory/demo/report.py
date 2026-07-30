"""Render the GEM Wikipedia demo evidence as JSON plus a compact Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, sort_keys=True, default=str)}`"
    return str(value)


def _table(rows: list[tuple[str, Any]]) -> list[str]:
    out = ["| Metric | Value |", "| --- | ---: |"]
    out.extend(f"| {name} | {_fmt(value)} |" for name, value in rows)
    return out


def render(results: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    acts = results.get("acts") or {}
    lines = [
        "# GEM Wikipedia demo",
        "",
        f"Scope: `{results.get('scope_id')}`",
        "",
        "This report is condition-by-condition evidence. It does not make a "
        "wholesale conformance claim.",
        "",
        "## Operating labels",
        "",
    ]
    lines.extend(f"- {label}" for label in manifest.get("labels") or ())

    coverage = results.get("coverage") or {}
    lines += ["", "## HotpotQA coverage", ""]
    lines += _table(
        [
            ("questions considered", coverage.get("questions_considered")),
            ("fully resolved", coverage.get("fully_resolved")),
            (
                "fully resolved fraction",
                coverage.get("fully_resolved_fraction"),
            ),
            ("questions executed", results.get("questions_run")),
            ("candidate-pool note", coverage.get("note")),
        ]
    )

    ingest = acts.get("ingest")
    if ingest:
        delta = (ingest.get("transition") or {}).get("delta") or {}
        cost = (ingest.get("transition") or {}).get("cost") or {}
        lines += ["", "## Act 1 — deterministic ingest", ""]
        lines += _table(
            [
                ("strategy", ingest.get("strategy")),
                ("variant", ingest.get("strategy_variant")),
                ("events", ingest.get("events")),
                ("units created", delta.get("units_created")),
                ("fields appended", delta.get("fields_appended")),
                ("edges created", delta.get("edges_created")),
                ("seconds", cost.get("seconds")),
                ("embed calls", cost.get("embed_calls")),
                ("embed sequences", cost.get("embed_sequences")),
                ("DB statements", cost.get("db_statements")),
            ]
        )

    retrieve = acts.get("retrieve")
    if retrieve:
        lines += [
            "",
            "## Act 2 — retrieval",
            "",
            "VECTOR and FUSED are separate operating points. Their HNSW order "
            "settings are not pooled. Every per-query engine probe is preserved "
            "verbatim in `results.json`.",
        ]
        points = retrieve.get("operating_points") or {}
        for name in ("vector", "fused"):
            point = points.get(name)
            if not point:
                continue
            metrics = point.get("metrics") or {}
            cost = point.get("cost") or {}
            lines += [
                "",
                f"### {name.upper()} — `{point.get('hnsw_iterative_scan')}`",
                "",
            ]
            lines += _table(
                [
                    ("questions", metrics.get("questions")),
                    (
                        "joint evidence recall@k",
                        metrics.get("joint_evidence_recall_at_k"),
                    ),
                    (
                        "mean evidence recall@k",
                        metrics.get("mean_evidence_recall_at_k"),
                    ),
                    ("mean returned units", metrics.get("mean_returned_units")),
                    ("short-result queries", metrics.get("short_result_queries")),
                    ("failed queries", metrics.get("failed_queries")),
                    ("termination reasons", metrics.get("termination_reasons")),
                    ("budget-capped queries", metrics.get("budget_capped_queries")),
                    ("graph-censored queries", metrics.get("graph_censored_queries")),
                    (
                        "strict C6 salience increases",
                        metrics.get("strict_salience_increases"),
                    ),
                    ("salience pairs", metrics.get("salience_pairs")),
                    ("seconds", cost.get("seconds")),
                    ("embed calls", cost.get("embed_calls")),
                    ("DB statements", cost.get("db_statements")),
                ]
            )

    revise = acts.get("revise")
    if revise:
        c1 = revise.get("c1_example") or {}
        c2 = revise.get("c2_policy_probe") or {}
        c3 = revise.get("c3_evidence") or {}
        replay_delta = (revise.get("replay_transition") or {}).get("delta") or {}
        lines += ["", "## Act 3 — real Wikidata revisions", ""]
        lines += _table(
            [
                ("revision rows", revise.get("revision_rows")),
                ("parsed edits", revise.get("parsed_edits")),
                ("parse fraction", revise.get("parse_fraction")),
                ("observed supersessions", revise.get("observed_supersessions")),
                ("changed units", revise.get("changed_units")),
                ("changed unit kinds", revise.get("changed_unit_kinds")),
                ("values superseded", replay_delta.get("values_superseded")),
                ("extension units propagated", c3.get("extension_units_propagated")),
            ]
        )
        lines += ["", "### C1 example", ""]
        lines += _table(
            [
                ("holds", c1.get("holds")),
                ("QID / field", f"{c1.get('qid')} / {c1.get('field')}"),
                ("old value", c1.get("old_value")),
                ("new value", c1.get("new_value")),
                ("default values", c1.get("default_values")),
                ("as_of", c1.get("as_of")),
                ("as_of values", c1.get("as_of_values")),
            ]
        )
        lines += ["", "### C2 rollback probe", ""]
        lines += _table(
            [
                ("holds", c2.get("holds")),
                ("transition committed", c2.get("committed")),
                ("state byte-identical", c2.get("state_byte_identical")),
                ("aborted reason", c2.get("aborted_reason")),
            ]
        )
        lines += ["", "### C3 typed propagation", ""]
        lines += _table(
            [
                ("holds", c3.get("holds")),
                ("extension units propagated", c3.get("extension_units_propagated")),
                (
                    "association edges from changed units",
                    c3.get("association_edges_from_changed_units"),
                ),
                (
                    "association-only neighbours not propagated",
                    c3.get("association_only_neighbours_not_propagated"),
                ),
            ]
        )

    forget = acts.get("forget")
    if forget:
        lines += ["", "## Act 4 — forgetting tick", ""]
        lines += _table(
            [
                ("holds", forget.get("holds")),
                ("units before", forget.get("units_before")),
                ("units after", forget.get("units_after")),
                ("retrieved units", forget.get("retrieved_units_before_tick")),
                (
                    "retrieved units not archived",
                    forget.get("retrieved_units_not_archived"),
                ),
                (
                    "never-retrieved units",
                    forget.get("never_retrieved_units_before_tick"),
                ),
                (
                    "never-retrieved units archived",
                    forget.get("never_retrieved_units_archived"),
                ),
                ("states after", forget.get("states_after")),
                ("explicit archived lookup", forget.get("archived_explicit_lookup")),
                ("no rows deleted", forget.get("no_rows_deleted")),
            ]
        )

    conf = results.get("conformance") or {}
    lines += ["", "## Act 5 — C1–C6 condition report", ""]
    lines += [
        "| Condition | Holds | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in conf.get("results") or ():
        evidence = json.dumps(row.get("evidence") or {}, sort_keys=True, default=str)
        lines.append(f"| {row.get('condition')} | {row.get('holds')} | `{evidence}` |")
    lines += [
        "",
        f"Satisfied: `{json.dumps(conf.get('satisfied') or [])}`  ",
        f"Violated: `{json.dumps(conf.get('violated') or [])}`  ",
        f"Unchecked: `{json.dumps(conf.get('unchecked') or [])}`",
        "",
        "Graph topology evidence is commit-visible, not snapshot-isolated. No "
        "claim here depends on repeatable-read topology.",
        "",
    ]
    return "\n".join(lines)


def write(
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    manifest_path = output / "manifest.json"
    report_path = output / "report.md"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render(results, manifest), encoding="utf-8")
    return {
        "results": results_path,
        "manifest": manifest_path,
        "report": report_path,
    }
