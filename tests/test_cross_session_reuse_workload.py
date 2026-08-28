from experiments.cross_session_reuse.workload.registry import (
    all_variants,
    profile_variants,
    seeds_for,
)
from experiments.cross_session_reuse.workload.tasks.circle_packing_evaluator import (
    evaluate as evaluate_circle,
)
from experiments.cross_session_reuse.workload.tasks.code_repair_evaluator import (
    evaluate as evaluate_repair,
)

TASKS = (
    __import__(
        "experiments.cross_session_reuse.workload.tasks", fromlist=["__file__"]
    )
)


def test_coverage_profile_has_every_family_once():
    variants = profile_variants("coverage")
    assert len(variants) == 7
    assert len({variant.family for variant in variants}) == 7


def test_focus_profile_prioritizes_three_families():
    variants = all_variants()
    focused = [variant for variant in variants if variant.priority == "focus"]
    assert {variant.family for variant in focused} == {
        "circle_packing",
        "code_repair",
        "gpu_kernel_optimization",
    }
    assert all(seeds_for(variant, "focus") == (101, 102) for variant in focused)
    assert sum(len(seeds_for(variant, "focus")) for variant in variants) == 22


def test_registered_files_exist():
    for variant in all_variants():
        assert variant.initial_program.is_file(), variant.task_id
        assert variant.evaluator.is_file(), variant.task_id


def test_circle_initial_program_is_valid_for_focused_variants(monkeypatch):
    initial = TASKS.__path__[0] + "/circle_packing_initial.py"
    for count in (12, 18, 26):
        monkeypatch.setenv("CSR_CIRCLE_N", str(count))
        result = evaluate_circle(initial)
        assert result["status"] == "valid"
        assert result["combined_score"] > 0


def test_repair_initial_program_exposes_each_seeded_bug(monkeypatch):
    initial = TASKS.__path__[0] + "/code_repair_initial.py"
    for variant in ("chunk", "path", "record"):
        monkeypatch.setenv("CSR_REPAIR_VARIANT", variant)
        result = evaluate_repair(initial)
        assert result["is_buggy"] == 1.0
        assert result["combined_score"] < 1.0
