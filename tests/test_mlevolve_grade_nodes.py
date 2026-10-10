"""Tests for post-hoc held-out grading of MLEvolve nodes.

Covers the three things that would silently corrupt a result: grading CV instead
of held-out, mis-attributing a score to the wrong node when MLEvolve's
submission-path isolation fails, and re-invoking the grader on identical files.
"""

from __future__ import annotations

import json
import stat

import pytest

from bench.agent_memory.traceml.mlevolve_ab.grade_nodes import grade_run


def _node(nid, *, ctime, code="print(1)", metric=None, buggy=False, stage="improve"):
    return {
        "id": nid,
        "parent": None,
        "ctime": ctime,
        "code": code,
        "stage": stage,
        "is_buggy": buggy,
        "metric": {"value": metric, "maximize": True} if metric is not None else None,
    }


@pytest.fixture
def run(tmp_path):
    """A run dir with 4 nodes: two distinct submissions, one duplicate, one missing."""
    logs = tmp_path / "logs"
    ws = tmp_path / "workspace" / "submission"
    logs.mkdir(parents=True)
    ws.mkdir(parents=True)

    journal = {
        "nodes": [
            _node("n1", ctime=1, code="A", metric=0.30),
            _node("n2", ctime=2, code="B", metric=0.90),  # CV best
            _node("n3", ctime=3, code="C", metric=0.50),
            _node("n4", ctime=4, code="D", metric=None, buggy=True),
        ],
        "node2parent": {"n2": "n1", "n3": "n1", "n4": "n3"},
    }
    (logs / "journal.json").write_text(json.dumps(journal))

    (ws / "submission_n1.csv").write_text("id,y\n1,0.1\n")
    (ws / "submission_n2.csv").write_text("id,y\n1,0.2\n")
    (ws / "submission_n3.csv").write_text("id,y\n1,0.1\n")  # byte-identical to n1
    # n4 writes nothing
    return tmp_path


def _fake_mlebench(tmp_path, scores: dict[str, float], *, lower_better=False):
    """A stub grader that scores by submission content, and counts its invocations."""
    calls = tmp_path / "grader_calls.txt"
    script = tmp_path / "fake_mlebench"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json, pathlib\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        f"scores = {scores!r}\n"
        "sub = pathlib.Path(sys.argv[2]).read_text()\n"
        "with open(calls, 'a') as f: f.write(sys.argv[2] + '\\n')\n"
        "print('some log line')\n"
        # The real CLI logs this marker before the JSON (mlebench/cli.py grade-sample);
        # grade_nodes parses via the shared, marker-anchored parse_competition_report.
        "print('[INFO] Competition report:')\n"
        "print(json.dumps({'competition_id': sys.argv[3], 'score': scores[sub],\n"
        "                  'valid_submission': True, 'any_medal': False,\n"
        f"                  'above_median': True, 'is_lower_better': {lower_better}}}, indent=2))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, calls


def test_grades_every_node_and_reports_both_numbers(run, tmp_path):
    grader, calls = _fake_mlebench(
        tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75}
    )
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)

    assert s["n_nodes"] == 4
    assert s["n_with_submission"] == 3
    # n1 and n3 are byte-identical -> two distinct files, two grader calls
    assert s["n_distinct_submissions"] == 2
    assert len(calls.read_text().splitlines()) == 2

    # best held-out over all nodes (paper's headline metric)
    assert s["best_heldout"] == 0.75
    assert s["best_heldout_node"] == "n2"

    # and what MLEvolve's own CV rule would have shipped
    assert s["selected_final_node"] == "n2"
    assert s["selected_final_heldout"] == 0.75
    assert s["selected_final_cv"] == 0.90


def test_cv_best_and_heldout_best_can_disagree(run, tmp_path):
    """The whole reason both numbers are required."""
    grader, _ = _fake_mlebench(tmp_path, {"id,y\n1,0.1\n": 0.95, "id,y\n1,0.2\n": 0.10})
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)

    assert s["selected_final_node"] == "n2"  # CV says n2 (0.90)
    assert s["selected_final_heldout"] == 0.10  # ...which is held-out terrible
    assert s["best_heldout"] == 0.95  # n1/n3 were actually better
    assert s["best_heldout_node"] in {"n1", "n3"}


def test_lower_is_better_direction_is_honoured(run, tmp_path):
    grader, _ = _fake_mlebench(
        tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75}, lower_better=True
    )
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)
    assert s["is_lower_better"] is True
    assert s["best_heldout"] == 0.40


def test_identical_submissions_are_graded_once_and_share_the_score(run, tmp_path):
    grader, calls = _fake_mlebench(
        tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75}
    )
    grade_run(run, "slug", mlebench=grader, cache=tmp_path)
    import csv

    rows = {
        r["node_id"]: r
        for r in csv.DictReader(open(run / "grading" / "node_grades.csv"))
    }
    assert rows["n1"]["heldout_score"] == "0.4"
    assert rows["n3"]["heldout_score"] == "0.4"
    assert rows["n3"]["graded_from_cache_of"] == "n1"
    assert len(calls.read_text().splitlines()) == 2


def test_overwrite_by_differing_code_is_flagged(run, tmp_path):
    """n1 and n3 carry different code but wrote the same bytes.

    That is the signature of executor.isolate_submission_path failing to rewrite
    the output path, which silently attributes a score to the wrong node.
    """
    grader, _ = _fake_mlebench(tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75})
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)

    assert s["suspicious_overwrites"], (
        "identical bytes from different code must be flagged"
    )
    hit = s["suspicious_overwrites"][0]
    assert sorted(hit["node_ids"]) == ["n1", "n3"]
    assert hit["distinct_code_variants"] == 2
    assert "isolate_submission_path" in s["WARNING"]


def test_identical_code_writing_identical_bytes_is_not_flagged(run, tmp_path):
    journal = json.loads((run / "logs" / "journal.json").read_text())
    for n in journal["nodes"]:
        if n["id"] == "n3":
            n["code"] = "A"  # same code as n1 -> duplicate output is expected
    (run / "logs" / "journal.json").write_text(json.dumps(journal))

    grader, _ = _fake_mlebench(tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75})
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)
    assert s["sha_collisions"], "the collision is still recorded"
    assert not s["suspicious_overwrites"], "but it is not suspicious"


def test_missing_submission_is_recorded_not_crashed(run, tmp_path):
    grader, _ = _fake_mlebench(tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75})
    grade_run(run, "slug", mlebench=grader, cache=tmp_path)
    import csv

    rows = {
        r["node_id"]: r
        for r in csv.DictReader(open(run / "grading" / "node_grades.csv"))
    }
    assert rows["n4"]["submission_path"] == ""
    assert rows["n4"]["grader_error"] == "no submission file"


def test_grader_failure_is_surfaced_not_swallowed(run, tmp_path):
    bad = tmp_path / "broken_mlebench"
    bad.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)

    s = grade_run(run, "slug", mlebench=bad, cache=tmp_path)
    assert s["n_grader_errors"] == 3
    assert s["best_heldout"] is None


def test_undetermined_direction_refuses_to_pick_a_cv_best(run, tmp_path):
    """If determine_metric_direction failed, guessing would corrupt the result."""
    journal = json.loads((run / "logs" / "journal.json").read_text())
    for n in journal["nodes"]:
        if n["metric"]:
            n["metric"]["maximize"] = None
    (run / "logs" / "journal.json").write_text(json.dumps(journal))

    grader, _ = _fake_mlebench(tmp_path, {"id,y\n1,0.1\n": 0.40, "id,y\n1,0.2\n": 0.75})
    s = grade_run(run, "slug", mlebench=grader, cache=tmp_path)
    assert s["selected_final_node"] is None
    assert s["best_heldout"] == 0.75  # held-out best still computable
