"""Fail-closed verification of a process's dedicated cgroup before dispatch."""

from pathlib import Path

from trimem.experiments.scienceworld_trace_ablation import validate_system_limits


def verify_current_cgroup(
    limits, *, cgroup_base=Path("/sys/fs/cgroup"), proc_file=Path("/proc/self/cgroup")
):
    validate_system_limits(limits)
    lines = Path(proc_file).read_text().splitlines()
    rows = [line[3:] for line in lines if line.startswith("0::/")]
    if len(rows) != 1:
        raise ValueError("cannot identify the process cgroup-v2 path")
    base = Path(cgroup_base).resolve()
    group = (base / rows[0].lstrip("/")).resolve()
    if group == base or not group.is_relative_to(base):
        raise ValueError("ablation requires a dedicated cgroup leaf")
    expected_cpu = f"{max(1000, int(float(limits['cpu_quota_cores']) * 100000))} 100000"
    if (group / "cpu.max").read_text().strip() != expected_cpu or (
        group / "memory.max"
    ).read_text().strip() != str(limits["memory_max_bytes"]):
        raise ValueError("active cgroup CPU or memory limit differs from manifest")
    return {
        "cgroup": str(group),
        "cpu_max": expected_cpu,
        "memory_max_bytes": limits["memory_max_bytes"],
    }
