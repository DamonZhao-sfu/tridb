"""Admission must stay closed while other GPU experiments have work."""

from copy import deepcopy

import pytest

from trimem.exg.when_idle import idle


def sample():
    return dict(
        gpus=[dict(utilization=0)],
        blockers=[],
        unknown_gpu_pids=[],
        cpu_percent=10,
        available_gb=100,
        services={"8000": dict(status="ready", running=0, waiting=0, completed=100)},
    )


@pytest.mark.parametrize(
    "reason",
    [
        "queued_job",
        "gpu_busy",
        "unknown_process",
        "api_running",
        "api_waiting",
        "unknown_api",
        "cpu_busy",
        "low_ram",
    ],
)
def test_do_not_start_while_busy(reason):
    state = deepcopy(sample())
    if reason == "queued_job":
        state["blockers"] = [dict(pid=123, module="trimem.experiments.oe_matrix")]
    elif reason == "gpu_busy":
        state["gpus"][0]["utilization"] = 90
    elif reason == "unknown_process":
        state["unknown_gpu_pids"] = [123]
    elif reason == "api_running":
        state["services"]["8000"]["running"] = 1
    elif reason == "api_waiting":
        state["services"]["8000"]["waiting"] = 1
    elif reason == "unknown_api":
        state["services"]["8000"] = dict(status="unknown")
    elif reason == "cpu_busy":
        state["cpu_percent"] = 70
    elif reason == "low_ram":
        state["available_gb"] = 8
    assert not idle(state)


def test_idle_resident_model_or_absent_model_can_be_used():
    state = sample()
    assert idle(state)
    state["services"]["8000"] = dict(status="absent")
    assert idle(state)
