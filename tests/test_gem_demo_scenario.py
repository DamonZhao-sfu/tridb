"""Offline tests for G2/G3 orchestration and report separation."""

from __future__ import annotations

import json

from bench.agent_memory.demo import report, scenario


def test_retrieval_metrics_grade_unique_unit_titles_and_short_results():
    questions = [
        {"gold_titles": ["A", "B"]},
        {"gold_titles": ["C", "D"]},
    ]
    observations = [
        {
            "committed": True,
            # Multiple fields from A must still count as one returned unit.
            "returned_titles": ["A", "A", "B"],
            "returned_units": 2,
            "probes": {
                "termination_reason": "filter_first",
                "budget_capped": False,
                "graph_censored": False,
            },
            "strict_salience_increases": 3,
            "salience_pairs": 3,
        },
        {
            "committed": True,
            "returned_titles": ["C"],
            "returned_units": 1,
            "probes": {
                "termination_reason": "budget",
                "budget_capped": True,
                "graph_censored": True,
            },
            "strict_salience_increases": 1,
            "salience_pairs": 1,
        },
    ]
    metrics = scenario._retrieval_metrics(questions, observations, k=2)
    assert metrics["joint_evidence_recall_at_k"] == 0.5
    assert metrics["mean_evidence_recall_at_k"] == 0.75
    assert metrics["short_result_queries"] == 1
    assert metrics["termination_reasons"] == {"budget": 1, "filter_first": 1}
    assert metrics["budget_capped_queries"] == 1
    assert metrics["graph_censored_queries"] == 1


def test_report_keeps_strict_and_relaxed_operating_points_in_separate_tables():
    results = {
        "scope_id": "s",
        "coverage": {},
        "questions_run": 1,
        "acts": {
            "retrieve": {
                "operating_points": {
                    "vector": {
                        "hnsw_iterative_scan": "strict_order",
                        "metrics": {},
                        "cost": {},
                    },
                    "fused": {
                        "hnsw_iterative_scan": "relaxed_order",
                        "metrics": {},
                        "cost": {},
                    },
                }
            }
        },
        "conformance": {"results": []},
    }
    rendered = report.render(results, {"labels": []})
    assert "### VECTOR — `strict_order`" in rendered
    assert "### FUSED — `relaxed_order`" in rendered
    assert "not pooled" in rendered


def test_write_emits_the_three_plan_artifacts(tmp_path):
    paths = report.write(
        {"scope_id": "s", "acts": {}, "conformance": {"results": []}},
        {"labels": ["stock PostgreSQL"]},
        tmp_path,
    )
    assert set(paths) == {"results", "manifest", "report"}
    assert json.loads(paths["results"].read_text())["scope_id"] == "s"
    assert json.loads(paths["manifest"].read_text())["labels"]
    assert paths["report"].read_text().startswith("# GEM Wikipedia demo")


def test_cli_defaults_to_all_and_accepts_every_documented_phase():
    parser = __import__(
        "bench.agent_memory.demo.__main__", fromlist=["build_parser"]
    ).build_parser()
    assert parser.parse_args([]).phase == "all"
    for phase in ("all", "ingest", "retrieve", "revise", "forget"):
        assert parser.parse_args(["--phase", phase]).phase == phase
