from __future__ import annotations

import json

from experiments.e1.composition.report_dataset_level import _normalize
from experiments.e1.composition.report_staged import (
    ANALYSES,
    ARMS,
    LABELS,
    _bootstrap_ci,
    _median,
    summarize_composition_templates,
)


def test_query_bootstrap_is_seeded_and_does_not_expand_the_input_unit():
    values = [1.2, 1.5, 2.0, 2.5]
    first = _bootstrap_ci(values, _median, seed=42, resamples=500)
    second = _bootstrap_ci(values, _median, seed=42, resamples=500)
    assert first == second
    assert first is not None
    assert first[0] <= _median(values) <= first[1]


def test_template_summary_keeps_single_query_strata_descriptive():
    rows = []
    for analysis in ANALYSES:
        for label in LABELS:
            rows.extend(
                [
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "template": "common",
                        "quality_matched": True,
                        "polyglot_over_tridb_p50": 1.5,
                    },
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "template": "common",
                        "quality_matched": True,
                        "polyglot_over_tridb_p50": 2.5,
                    },
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "template": "rare",
                        "quality_matched": True,
                        "polyglot_over_tridb_p50": 1.25,
                    },
                ]
            )

    summary = summarize_composition_templates(rows, seed=7)
    assert len(summary) == len(ANALYSES) * len(LABELS) * 2
    rare = next(
        row
        for row in summary
        if row["analysis"] == "iso_plan"
        and row["operating_label"] == "fast"
        and row["template"] == "rare"
    )
    assert rare["queries_quality_matched"] == 1
    assert rare["median_speedup_p50"] == 1.25
    assert rare["speedup_median_ci95_low"] is None
    assert rare["speedup_median_ci95_high"] is None


def test_dataset_level_normalization_keeps_datasets_separate(tmp_path):
    paths = []
    for name, query_count, full_mrr in (("prime", 29, 0.8), ("mag", 30, 0.6)):
        summary = {
            "evaluation_audit": {"evaluation_queries": query_count},
            "composition": [{"analysis": "iso_plan", "median_speedup_p50": 2.0}],
            "modality": [
                {
                    "arm": arm,
                    "mean_mrr": full_mrr if arm == "vector_graph_relational" else 0.1,
                    "mean_recall_at_20": 0.5,
                    "mean_full_constraint_validity_fraction": 1.0,
                }
                for arm in ARMS
            ],
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append((name, path))

    datasets, composition, modality_delta = _normalize(paths)

    assert [row["evaluation_queries"] for row in datasets] == [29, 30]
    assert [row["dataset"] for row in composition] == ["prime", "mag"]
    prime_vector_mrr = next(
        row
        for row in modality_delta
        if row["dataset"] == "prime"
        and row["arm"] == "vector_only"
        and row["metric"] == "mean_mrr"
    )
    assert round(prime_vector_mrr["drop_from_all_three"], 1) == 0.7
