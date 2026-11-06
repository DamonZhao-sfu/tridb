"""Interchangeable profiling allocators; fresh TRACE defaults to syftr-style.

The syftr-style adapter uses optimistic bounds, z=1.645 and strict dominance,
following datarobot/syftr at 7174a26e4171a534f13f2f805a1e1745807af39b
(syftr/pruning.py and custom_metrics.py). Adaptations: three mean objectives,
our stratified SEs/checkpoints, and only full-panel comparator configurations.
It is not a verbatim reproduction of their two-objective RAG experiment.
"""

from __future__ import annotations

from trimem.optimizer import staged_pruning as sp

CHOICES = ("staged", "syftr-style", "none")
SYFTR_Z = 1.645


def add_arguments(parser):
    parser.add_argument(
        "--pruning_policy",
        choices=CHOICES,
        default=None,
        help="fresh TRACE: syftr-style (default); legacy/baselines: staged; "
        "recorded policy retained on resume; explicit staged/syftr-style/none overrides",
    )


def selected(staging):
    # Missing policy metadata identifies a historical staged run, not a new run.
    return staging.get("pruning_policy", {}).get("name", "staged")


def fresh_default(args):
    from trimem.optimizer.search_analyzer import PROTOCOL

    design = getattr(args, "search_design", None) or PROTOCOL
    return (
        "syftr-style"
        if design == PROTOCOL and getattr(args, "staging", None) != "none"
        else "staged"
    )


def requires_isolation(staging, search_design):
    from trimem.optimizer.search_analyzer import PROTOCOL

    name = selected(staging)
    return name != "staged" and not (
        name == "syftr-style" and search_design["protocol"] == PROTOCOL
    )


def configure(block, args, previous=None):
    """None means a fresh run; even an empty previous block retains legacy defaults."""
    old = selected(previous or {})
    name = getattr(args, "pruning_policy", None) or (
        old if previous is not None else fresh_default(args)
    )
    if name not in CHOICES:
        raise ValueError(f"unknown pruning policy: {name}")
    if previous is not None and name != old:
        raise ValueError("pruning policy changed; use a fresh isolated run")
    if name == "staged":
        return block  # Preserve historical state schemas byte-for-byte.
    if getattr(args, "staging", None) == "none":
        raise ValueError("--pruning_policy requires --staging pareto")
    desc = (
        "3-D optimistic corner, z=1.645, strictly dominated by full-panel point estimates; "
        "no stage comparators and no plan guard; stratified means/SEs and fixed checkpoints"
        if name == "syftr-style"
        else "no early stopping; common checkpoints and budget"
    )
    return {
        **block,
        "pruning_policy": {
            "protocol": "profiling-pruning-policy-v1",
            "name": name,
            "description": desc,
            "upstream_commit": "7174a26e4171a534f13f2f805a1e1745807af39b"
            if name == "syftr-style"
            else None,
        },
        "z": SYFTR_Z if name == "syftr-style" else 0.0,
        "plan_guard": False,
        **({"rule": f"profiling-pruning-policy-v1:{name}"} if "rule" in block else {}),
        **(
            {"prune_objectives": desc}
            if "prune_objectives" in block
            else {"comparators": desc}
        ),
    }


def decide(o, full, *, policy="staged", stage, last, protected=False, partial=None):
    if policy == "staged":
        return sp.decide(
            o, full, stage=stage, last=last, protected=protected, partial=partial
        )
    if policy not in CHOICES:
        raise ValueError(f"unknown pruning policy: {policy}")
    corner = (
        sp.optimistic_corner(o, SYFTR_Z) if policy == "syftr-style" else sp.point(o)
    )
    if policy == "none" or last or protected:
        reason = (
            "pruning disabled"
            if policy == "none"
            else "last stage"
            if last
            else "protected"
        )
        return sp.Decision(False, stage, corner, None, reason)
    doms = tuple(
        (key, sp.FULL_POINT, sp.point(full[key]))
        for key in sorted(full)
        if sp.weakly_dominates(sp.point(full[key]), corner)
        and sp.point(full[key]) != corner
    )
    return sp.Decision(
        bool(doms),
        stage,
        corner,
        doms[0][0] if doms else None,
        "syftr-style: optimistic corner strictly dominated by a full-panel point"
        if doms
        else "not dominated",
        doms,
        len(full),
    )


def guard(decisions, objs, plan_of, covered, *, policy):
    return (
        sp.guard_plans(decisions, objs, plan_of, covered)
        if policy == "staged"
        else decisions
    )
