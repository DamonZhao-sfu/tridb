"""Pure helpers of the TRACE / MO-TPE HV-vs-budget renderer (no DB, no files)."""

import math

import pytest

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "render_trace_motpe_hv",
    Path(__file__).resolve().parents[1] / "tools/render_trace_motpe_hv.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def run(method, points):
    return {
        "method": method,
        "checkpoints": [
            {"tokens": t, "hours": t, "episodes": t, "nhv": v} for t, v in points
        ],
    }


@pytest.mark.unit
def test_curve_value_is_step_and_zero_before_first_checkpoint():
    cps = run("A", [(10, 0.2), (20, 0.3)])["checkpoints"]
    assert m.curve_value(cps, "tokens", 5) == 0.0
    assert m.curve_value(cps, "tokens", 10) == 0.2
    assert m.curve_value(cps, "tokens", 19.9) == 0.2
    assert m.curve_value(cps, "tokens", 25) == 0.3


@pytest.mark.unit
def test_summary_uses_common_budget():
    s = m.summary([run("A", [(10, 0.2), (20, 0.3)]), run("B", [(5, 0.25), (15, 0.26)])])
    row = s["tokens"]
    assert row["common_budget"] == 15
    assert row["A"]["hv_at_budget"] == 0.2
    assert row["B"]["hv_at_budget"] == 0.26
    # A: 0 on [0,10), 0.2 on [10,15] -> 1.0 / 15
    assert math.isclose(row["A"]["normalized_auc"], 1.0 / 15)
    assert math.isclose(row["B"]["normalized_auc"], (0.25 * 10 + 0.26 * 0) / 15 + 0.0)


@pytest.mark.unit
def test_nhv_fixed_box():
    best = {"quality": 1.0, "tokens": 1000, "latency_ms": 1000}
    assert math.isclose(m.nhv([best]), 1.0)
    assert (
        m.out_of_box([best, {"quality": 0.5, "tokens": 2e6, "latency_ms": 1000}]) == 1
    )
