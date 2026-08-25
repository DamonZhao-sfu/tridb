"""Identity and canonicalization for the EvoTrace corpus.

Every downstream count depends on these functions agreeing with themselves across
runs, so they are pure, deterministic and separately testable. The rule everywhere is
**fail closed**: an unparseable run name raises rather than landing in a catch-all
bucket, because a silent bucket turns a source change into a wrong Task count instead
of a failed gate.

Identity scheme (EvoTraceDoc.md §3.2):

    task_uid     = "<domain>:<task_key>"
    session_uid  = "<dataset revision[:8]>:<backend>[/<group>]/<run name>"
    node_uid     = "<session_uid>#<program id>"
    artifact_uid = "sha256:<solution_sha256>"
    prompt_uid   = "<node_uid>@<prompts_sha256>"

A node is an ATTEMPT, not a program text: the same code appearing in two iterations is
two nodes sharing one artifact. Deduplicating nodes by content would erase exactly the
sibling/repeat structure the Experience Graph exists to record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Trailing `_<configured iterations>_<6-hex run id>` on every run directory name.
_RUN_SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<iters>\d+)_(?P<runhash>[0-9a-f]{6})$")

#: ALE (AtCoder Heuristic Contest, C++) tasks. Two spellings occur in the tree —
#: `ahc015_...` and `ale_bench_ahc015_...` — for the same underlying problem.
_ALE_TASK = re.compile(r"^(?:ale_bench_)?(?P<task>ahc\d{3})_(?P<rest>.+)$")

#: Python math-discovery tasks, matched as a prefix of the stem. Registered rather than
#: inferred: an unregistered stem must fail the gate, not invent a Task.
MATH_TASKS: tuple[str, ...] = (
    "circle_packing",
    "first_autocorr_ineq",
    "second_autocorr_ineq",
    "third_autocorr_ineq",
    "uncertainty_ineq",
    "heilbronn_triangle",
    "heilbronn_convex-13",
    "heilbronn_convex_13",
    "signal_processing",
)

#: `heilbronn_convex-13` and `heilbronn_convex_13` are one problem written two ways.
_TASK_ALIASES = {"heilbronn_convex-13": "heilbronn_convex_13"}

#: Trailing config flags carried in the run name after the model token.
_MODE_FLAGS = ("nodiff",)
_TEMPERATURE = re.compile(r"^t(?P<value>\d+\.\d+)$")


class RunNameError(ValueError):
    """A run directory name did not match any registered Task. Fail closed."""


@dataclass(frozen=True)
class RunName:
    """The parse of one EvoTrace run directory name."""

    raw: str
    domain: str
    task_key: str
    model: str
    mode: str
    temperature: float | None
    configured_iterations: int
    run_hash: str

    @property
    def task_uid(self) -> str:
        return f"{self.domain}:{self.task_key}"


def parse_run_name(name: str) -> RunName:
    """Split a run directory name into Task identity and session configuration.

    >>> parse_run_name("ale_bench_ahc015_claude-haiku-4-5_100_dcafe1").task_uid
    'ale:ahc015'
    >>> parse_run_name("ahc015_gflash-low_100_dcafe1").task_uid
    'ale:ahc015'
    >>> r = parse_run_name("heilbronn_triangle_dpsk-chat_t0.7_100_dcafe1")
    >>> (r.task_uid, r.model, r.temperature)
    ('math:heilbronn_triangle', 'dpsk-chat', 0.7)
    """
    suffix = _RUN_SUFFIX.match(name)
    if suffix is None:
        raise RunNameError(f"no `_<iters>_<runhash>` suffix: {name!r}")
    stem = suffix.group("stem")
    iterations = int(suffix.group("iters"))
    run_hash = suffix.group("runhash")

    ale = _ALE_TASK.match(stem)
    if ale is not None:
        domain, task_key, rest = "ale", ale.group("task"), ale.group("rest")
    else:
        # Longest registered prefix wins, so `second_autocorr_ineq` is never shadowed
        # by a shorter registration.
        matched = sorted(
            (t for t in MATH_TASKS if stem == t or stem.startswith(t + "_")),
            key=len,
            reverse=True,
        )
        if not matched:
            raise RunNameError(f"no registered Task matches stem {stem!r} (run {name!r})")
        raw_key = matched[0]
        domain = "math"
        task_key = _TASK_ALIASES.get(raw_key, raw_key)
        rest = stem[len(raw_key) :].lstrip("_")

    model, mode, temperature = _parse_config(rest)
    return RunName(
        raw=name,
        domain=domain,
        task_key=task_key,
        model=model,
        mode=mode,
        temperature=temperature,
        configured_iterations=iterations,
        run_hash=run_hash,
    )


def _parse_config(rest: str) -> tuple[str, str, float | None]:
    """`<model>[_nodiff][_t<temp>]` -> (model, mode, temperature).

    ``mode`` is the paper-relevant one: `diff` (the default patch-based edit) versus
    `nodiff` (full rewrite). It changes what an edit *is*, so it is a session property
    and not cosmetic metadata.
    """
    parts = rest.split("_") if rest else []
    mode = "diff"
    temperature: float | None = None
    model_parts: list[str] = []
    for part in parts:
        if part in _MODE_FLAGS:
            mode = part
            continue
        temp = _TEMPERATURE.match(part)
        if temp is not None:
            temperature = float(temp.group("value"))
            continue
        model_parts.append(part)
    return ("_".join(model_parts) or "unknown", mode, temperature)


def session_uid(revision: str, run_rel: str) -> str:
    """Stable session identity: the pinned revision plus the run's path in the tree.

    The path carries the group (`ablation`, `empty_seed`, ...), which is load-bearing:
    the same run name appears under different groups and those are different sessions.
    """
    return f"{revision[:8]}:{run_rel}"


def node_uid(session: str, program_id: str) -> str:
    """One ATTEMPT. Program ids are only assumed unique WITHIN a session."""
    return f"{session}#{program_id}"


def artifact_uid(solution_sha256: str) -> str:
    return f"sha256:{solution_sha256}"


def prompt_uid(node: str, prompts_sha256: str) -> str:
    return f"{node}@{prompts_sha256}"
