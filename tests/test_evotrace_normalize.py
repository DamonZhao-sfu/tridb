"""Behavioural gates for the EvoTrace normalizer.

Fixture-driven, no network and no database: these pin the rules that a silent
regression would otherwise turn into a wrong headline count — retained failures,
attempt-grained nodes, fail-closed parents, and lineage kept apart from context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.evotrace.ids import artifact_uid, node_uid, session_uid
from tools.evotrace.normalize import error_signature, normalize, normalize_session

pytestmark = pytest.mark.unit

REV = "349117b04832b681ccf69ea2518c31a579853b3e"


def _program(
    pid: str,
    *,
    iteration: int,
    parent: str | None = None,
    status: str = "accepted",
    score: float | None = 1.0,
    sha: str = "a" * 64,
    prompts: str | None = None,
    context: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = dict(metrics or {})
    if score is not None:
        body.setdefault("combined_score", score)
    return {
        "id": pid,
        "language": "cpp",
        "metrics": body,
        "iteration_found": iteration,
        "parent_id": parent,
        "other_context_ids": context or [],
        "timestamp": None,
        "metadata": {"changes": "Full rewrite"},
        "generation": 0,
        "status": status,
        "source": "checkpoint.programs",
        "solution_sha256": sha,
        "prompts_sha256": prompts,
        "artifacts": None,
    }


def _write_run(root: Path, run_rel: str, programs: list[dict[str, Any]], **extra: Any) -> Path:
    base = root / run_rel
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "meta.json").write_text(json.dumps({"backend": run_rel.split("/")[0], "counts": {}}))
    (base / "programs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in programs))
    (base / "iterations.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in extra.get(
                "iterations",
                [
                    {
                        "iteration": 1,
                        "role": "population_member",
                        "slot_key": programs[0]["id"],
                        "program_id": programs[0]["id"],
                        "value": None,
                    }
                ],
            )
        )
    )
    (base / "iter_scalars.jsonl").write_text(
        json.dumps({"iteration": 1, "key": "best_program_id", "value": programs[0]["id"]}) + "\n"
    )
    (base / "logs/llm_calls.jsonl").write_text(
        json.dumps({"ts": None, "model": "dpsk-reasoner", "prompt_tokens": 10}) + "\n"
    )
    return base


RUN_A = "evox/heilbronn_triangle_dpsk-reasoner_100_aaaaaa"
RUN_B = "gepa_native/heilbronn_triangle_dpsk-reasoner_100_bbbbbb"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write_run(
        raw,
        RUN_A,
        [
            _program("p0", iteration=0, sha="a" * 64),
            _program("p1", iteration=1, parent="p0", sha="b" * 64, prompts="c" * 64, score=2.0),
            # Same code as p0, a SECOND attempt: one artifact, two nodes.
            _program("p2", iteration=2, parent="p1", sha="a" * 64, score=0.5),
            # A failure. Retained: it is the Repair negative.
            _program(
                "p3",
                iteration=3,
                parent="p1",
                status="rejected",
                score=None,
                sha="d" * 64,
                metrics={"error": "SyntaxError: unexpected EOF"},
            ),
            # Context/influence, which must NOT become a lineage edge.
            _program("p4", iteration=4, parent="p1", sha="e" * 64, context=["p0", "p2"]),
            # A parent that does not exist in this session: fail closed.
            _program("p5", iteration=5, parent="ghost", sha="f" * 64),
        ],
    )
    _write_run(raw, RUN_B, [_program("q0", iteration=0, sha="a" * 64)])
    return tmp_path


def test_failures_are_retained(corpus: Path) -> None:
    report = normalize(corpus, REV)
    assert report["observed"]["rejected"] == 1
    nodes = [
        json.loads(line)
        for line in (corpus / "normalized" / "nodes.jsonl").read_text().splitlines()
    ]
    rejected = [n for n in nodes if n["status"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["is_valid"] is False
    assert rejected[0]["error_signature"] == "error:SyntaxError: unexpected EOF"


def test_nodes_are_attempts_artifacts_dedup(corpus: Path) -> None:
    """Identical code in two attempts and two sessions: 3 references, 1 artifact."""
    report = normalize(corpus, REV)
    assert report["observed"]["nodes"] == 7
    artifacts = {
        json.loads(line)["artifact_uid"]: json.loads(line)
        for line in (corpus / "normalized" / "artifacts.jsonl").read_text().splitlines()
    }
    shared = artifacts[artifact_uid("a" * 64)]
    assert shared["reference_count"] == 3
    assert len(shared["sessions"]) == 2
    # Content shared across sessions is a leakage signal the split audit needs.
    assert report["artifacts_shared_across_sessions"] == 1


def test_unresolved_parent_fails_closed(corpus: Path) -> None:
    report = normalize(corpus, REV)
    assert report["reject_counts"]["unresolved_parent_id"] == 1
    lineage = [
        json.loads(line)
        for line in (corpus / "normalized" / "lineage_edges.jsonl").read_text().splitlines()
    ]
    # p1,p2,p3,p4 have resolvable parents; p5's ghost parent produced no edge.
    assert len(lineage) == 4
    assert all("ghost" not in edge["src_node_uid"] for edge in lineage)


def test_context_edges_are_not_lineage(corpus: Path) -> None:
    normalize(corpus, REV)
    context = [
        json.loads(line)
        for line in (corpus / "normalized" / "context_edges.jsonl").read_text().splitlines()
    ]
    assert len(context) == 2
    assert {edge["relation"] for edge in context} == {"context_for"}
    session = session_uid(REV, RUN_A)
    assert {edge["dst_node_uid"] for edge in context} == {node_uid(session, "p4")}


def test_state_events_are_not_nodes(corpus: Path) -> None:
    report = normalize(corpus, REV)
    # 2 runs x (1 membership + 1 scalar)
    assert report["observed"]["state_events"] == 4
    assert report["observed"]["nodes"] == 7
    assert report["base_entities"] == (
        report["observed"]["tasks"]
        + report["observed"]["sessions"]
        + report["observed"]["nodes"]
    )


def test_prompt_completeness_is_reported_not_faked(corpus: Path) -> None:
    acc = normalize_session(corpus / "raw", RUN_A, REV)
    assert len(acc.prompts) == 1  # only p1 carries prompts_sha256
    assert len(acc.nodes) == 6
    # The reference exists; the blob does not. Those are different facts.
    assert acc.prompts[0]["blob_present"] is False


def test_wall_clock_is_declared_unavailable(corpus: Path) -> None:
    normalize(corpus, REV)
    sessions = [
        json.loads(line)
        for line in (corpus / "normalized" / "sessions.jsonl").read_text().splitlines()
    ]
    assert all(s["wall_clock_available"] is False for s in sessions)


def test_paper_comparison_reports_disagreement(corpus: Path) -> None:
    """The report must state observed vs paper, never silently adopt the paper."""
    report = normalize(corpus, REV)
    comparison = report["paper_comparison"]["sessions"]
    assert comparison["observed"] == 2
    assert comparison["paper_claim"] == 121
    assert comparison["agrees"] is False


def test_error_signature_separates_executor_from_search() -> None:
    """Who rejected the attempt is part of the label, not an implementation detail."""
    # Clean and accepted.
    assert error_signature({"combined_score": 1.0}, "accepted") is None

    # The EXECUTOR rejected it -> a repair target.
    assert error_signature({"judge_result": "TIME_LIMIT_EXCEEDED"}, "rejected") == (
        "judge:TIME_LIMIT_EXCEEDED"
    )
    assert error_signature({"judge_result": "COMPILATION_ERROR"}, "accepted") == (
        "judge:COMPILATION_ERROR"
    )
    assert error_signature({"error": "boom\ntraceback line"}, "rejected") == "error:boom"

    # The SEARCH rejected it: the program ran clean, it just did not improve.
    # 698 of the corpus's 1,708 rejected nodes look exactly like this, and calling
    # them a failure of type ACCEPTED would put non-defects in the repair population.
    assert error_signature({"judge_result": "ACCEPTED"}, "rejected") == "rejected:not_improved"
    assert error_signature({"judge_result": "ACCEPTED", "message": ""}, "rejected") == (
        "rejected:not_improved"
    )

    # No signal at all from either side.
    assert error_signature({}, "failed") == "rejected:failed"


def test_nonfinite_rewards_are_neutralised_not_propagated(tmp_path: Path) -> None:
    """NaN/Inf must never reach `fitness` or the emitted JSON.

    `NaN > x` is False for every x, so a non-finite reward silently turns every
    comparison in the ground truth into "did not improve" while raising nothing. And
    bare NaN is not JSON, so it also breaks the database load. 38 nodes at the pinned
    revision carry a non-finite combined_score.
    """
    raw = tmp_path / "raw"
    _write_run(
        raw,
        RUN_A,
        [
            _program("p0", iteration=0, sha="a" * 64),
            _program("p1", iteration=1, parent="p0", sha="b" * 64, score=float("nan")),
            _program(
                "p2",
                iteration=2,
                parent="p0",
                sha="c" * 64,
                score=1.5,
                metrics={"loss": float("-inf"), "c2": float("nan")},
            ),
        ],
    )
    normalize(tmp_path, REV)
    text = (tmp_path / "normalized" / "nodes.jsonl").read_text()
    # The staging file must be parseable by a strict JSON reader, not just by Python.
    assert "NaN" not in text and "Infinity" not in text
    nodes = {json.loads(line)["program_id"]: json.loads(line) for line in text.splitlines()}

    assert nodes["p1"]["fitness"] is None
    assert nodes["p1"]["nonfinite_metrics"] == ["combined_score"]

    assert nodes["p2"]["fitness"] == 1.5
    assert nodes["p2"]["metrics"]["loss"] is None
    assert nodes["p2"]["metrics"]["c2"] is None
    assert sorted(nodes["p2"]["nonfinite_metrics"]) == ["c2", "loss"]

    assert nodes["p0"]["nonfinite_metrics"] == []


def test_nul_bytes_are_stripped_and_recorded(tmp_path: Path) -> None:
    """U+0000 cannot exist in PostgreSQL text; the corpus has it inside captured stderr."""
    raw = tmp_path / "raw"
    _write_run(
        raw,
        RUN_A,
        [
            _program(
                "p0",
                iteration=0,
                sha="a" * 64,
                status="rejected",
                score=None,
                metrics={"judge_result": "WRONG_ANSWER", "message": "Out of range: \x00\x00bad"},
            )
        ],
    )
    normalize(tmp_path, REV)
    text = (tmp_path / "normalized" / "nodes.jsonl").read_text()
    assert "\u0000" not in text and "\x00" not in text
    node = json.loads(text.splitlines()[0])
    assert node["metrics"]["message"] == "Out of range: bad"
    assert node["nul_stripped_metrics"] == ["message"]
    # The executor rejected it, so it stays a repair target.
    assert node["error_signature"] == "judge:WRONG_ANSWER"
