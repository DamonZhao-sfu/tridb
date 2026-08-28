"""Identity gates for the EvoTrace normalizer.

Every count in the E2 track (base entities, lineage edges, Task cardinality) is only
reproducible if these parses are stable, so they are pinned here against the run names
actually present at revision 349117b.
"""

from __future__ import annotations

import pytest

from tools.evotrace.ids import (
    RunNameError,
    artifact_uid,
    node_uid,
    parse_run_name,
    prompt_uid,
    session_uid,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "task_uid"),
    [
        # The same ALE problem written two ways in the tree.
        ("ale_bench_ahc015_claude-haiku-4-5_100_dcafe1", "ale:ahc015"),
        ("ahc015_gflash-low_100_91da9e", "ale:ahc015"),
        ("ahc015_deepseek-reasoner_nodiff_100_91da9e", "ale:ahc015"),
        ("ale_bench_ahc008_deepseek-deepseek-reasoner_100_abc123", "ale:ahc008"),
        # A model token containing an '@' and a URL-ish tail.
        (
            "ale_bench_ahc025_local-deepseek-deepseek-reasoner@https---model-host-v1_100_1269aa",
            "ale:ahc025",
        ),
        # Math tasks, including the two spellings of one problem.
        ("heilbronn_triangle_dpsk-chat_t0.7_100_7063ff", "math:heilbronn_triangle"),
        ("heilbronn_convex-13_deepseek-deepseek-reasoner_100_aaaaaa", "math:heilbronn_convex_13"),
        ("heilbronn_convex_13_dpsk-reasoner_100_bbbbbb", "math:heilbronn_convex_13"),
        ("second_autocorr_ineq_dpsk-reasoner_nodiff_100_228833", "math:second_autocorr_ineq"),
        ("third_autocorr_ineq_dpsk-reasoner_100_cccccc", "math:third_autocorr_ineq"),
        ("uncertainty_ineq_deepseek-deepseek-reasoner_100_dddddd", "math:uncertainty_ineq"),
        ("circle_packing_dpsk-reasoner_100_eeeeee", "math:circle_packing"),
        ("signal_processing_deepseek-deepseek-reasoner_100_ffffff", "math:signal_processing"),
        ("first_autocorr_ineq_dpsk-reasoner_100_012345", "math:first_autocorr_ineq"),
    ],
)
def test_task_identity_is_stable(name: str, task_uid: str) -> None:
    assert parse_run_name(name).task_uid == task_uid


def test_longest_task_prefix_wins() -> None:
    """`first_/second_/third_autocorr_ineq` must not collapse into one Task."""
    keys = {
        parse_run_name(f"{stem}_dpsk-reasoner_100_abcdef").task_key
        for stem in ("first_autocorr_ineq", "second_autocorr_ineq", "third_autocorr_ineq")
    }
    assert len(keys) == 3


def test_config_parse() -> None:
    run = parse_run_name("heilbronn_triangle_dpsk-chat_t1.0_100_c4aef9")
    assert run.model == "dpsk-chat"
    assert run.temperature == 1.0
    assert run.mode == "diff"
    assert run.configured_iterations == 100
    assert run.run_hash == "c4aef9"

    nodiff = parse_run_name("ahc015_deepseek-reasoner_nodiff_100_91da9e")
    assert nodiff.mode == "nodiff"
    assert nodiff.model == "deepseek-reasoner"
    assert nodiff.temperature is None


def test_unregistered_task_fails_closed() -> None:
    """An unknown stem must raise, never land in a catch-all Task."""
    with pytest.raises(RunNameError):
        parse_run_name("brand_new_problem_dpsk-reasoner_100_abcdef")


def test_missing_suffix_fails_closed() -> None:
    with pytest.raises(RunNameError):
        parse_run_name("heilbronn_triangle_dpsk-chat")


def test_group_makes_sessions_distinct() -> None:
    """Identical run names under different groups are DIFFERENT sessions."""
    rev = "349117b04832b681ccf69ea2518c31a579853b3e"
    a = session_uid(rev, "gepa_native/strong_seed/second_autocorr_ineq_dpsk-reasoner_100_c76047")
    b = session_uid(rev, "shinkaevolve/strong_seed/second_autocorr_ineq_dpsk-reasoner_100_c76047")
    assert a != b
    assert a.startswith("349117b0:")


def test_node_identity_is_attempt_grained() -> None:
    """Same code in two sessions -> two nodes, one artifact."""
    rev = "349117b04832b681ccf69ea2518c31a579853b3e"
    s1 = session_uid(rev, "evox/run_a_100_aaaaaa")
    s2 = session_uid(rev, "evox/run_b_100_bbbbbb")
    assert node_uid(s1, "prog-1") != node_uid(s2, "prog-1")
    assert artifact_uid("deadbeef") == artifact_uid("deadbeef")
    assert prompt_uid(node_uid(s1, "prog-1"), "cafe") == f"{s1}#prog-1@cafe"
