"""Which stored sessions trajectory memory may inject, and how their outcome is shown (versioned).

User decision 2026-10-18: trajectory memory also uses FAILED sessions, clearly labelled as wrong. This
replaces the success-only policy in every plan (VGR, VGR+Rank, P2, P2+Rank and the BO / router configs
built on them) for all agent-memory datasets. There is no separate budget or quota for failed sessions
("有的话就注入，没的话就不注入"): the candidate pool and the selection (vector seeds + native graph
traversal + SemFilter / Rank / top-k) are unchanged, they simply range over successful AND failed
sessions, under the same memory cap (``llm_budget.MEMORY_TOKENS``).

Policies
- ``success-only-v0``: the old behaviour (only ``session.success IS TRUE`` is retrievable; cards and
  headers byte-identical to the pre-2026-10-18 runs; SemFilter / SemRank rubric ``semantic-rubric-v1``).
  Experiment ids and result directories are NOT tagged, so old runs stay reproducible and resumable.
- ``success+failure-v1`` (default): ``success IS TRUE`` or ``success IS FALSE`` is retrievable; a session
  whose outcome is unknown (``success IS NULL``, e.g. ALFWorld AgentTraj-L games that could not be
  replay-verified) is neither and stays excluded. Every card carries an explicit SUCCESSFUL / FAILED
  label, the memory header explains both kinds, the semantic rubric is ``semantic-rubric-v2``. Experiment
  ids get ``:mem=success+failure-v1`` and memory-arm result directories ``__mem-sf1``.

Selection: ``TRIMEM_MEMORY_POLICY`` (environment; drivers pass it to their runner subprocesses) or each
runner's ``--memory_policy``. Isolation (L1 / product / scenario / entity / held-out) is unchanged and
applies to failed sessions exactly as to successful ones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SUCCESS_ONLY = "success-only-v0"
SUCCESS_FAILURE = "success+failure-v1"
POLICIES = (SUCCESS_ONLY, SUCCESS_FAILURE)
DEFAULT = SUCCESS_FAILURE  # user decision 2026-10-18: ON by default
ENV_VAR = "TRIMEM_MEMORY_POLICY"
EXPERIMENT_TAG = {SUCCESS_ONLY: "", SUCCESS_FAILURE: ":mem=" + SUCCESS_FAILURE}
DIR_TAG = {
    SUCCESS_ONLY: "",
    SUCCESS_FAILURE: "__mem-sf1",
}  # no "memory": AppWorld rejects such paths
RUBRIC = {SUCCESS_ONLY: "semantic-rubric-v1", SUCCESS_FAILURE: "semantic-rubric-v2"}
MARKER = ".memory_policy"  # not *.json: result scanners glob *.json in run directories

FAILED_TAG = "FAILED ATTEMPT"
SUCCESS_TAG = "SUCCESSFUL ATTEMPT"
HEADER_NOTE = (
    "Each past episode is labelled with its outcome. A SUCCESSFUL ATTEMPT solved its task: reuse its "
    "plan when the goal matches yours. A FAILED ATTEMPT did NOT solve its task: never follow its plan; "
    "use it only to recognise and avoid the same mistake."
)


def validate(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(
            f"unknown memory policy {policy!r} (one of {', '.join(POLICIES)})"
        )
    return policy


def current() -> str:
    """The process-wide policy: ``$TRIMEM_MEMORY_POLICY`` if set, else ``DEFAULT``."""
    return validate(os.environ.get(ENV_VAR) or DEFAULT)


def admits(policy: str, success: bool | None) -> bool:
    """Outcome part of admissibility (isolation is the adapter's job). ``None`` = unknown outcome."""
    validate(policy)
    if success is None:
        return False
    return bool(success) or policy == SUCCESS_FAILURE


def rubric(policy: str) -> str:
    return RUBRIC[validate(policy)]


def tag(experiment: str, policy: str | None = None) -> str:
    """Experiment id under ``policy`` (idempotent; success-only ids are unchanged)."""
    policy = validate(policy or current())
    t = EXPERIMENT_TAG[policy]
    if any(v and v in experiment for v in EXPERIMENT_TAG.values() if v != t):
        raise ValueError(
            f"experiment id {experiment!r} already carries another memory policy"
        )
    return (
        experiment
        if (not t or experiment.endswith(t) or t + ":" in experiment)
        else experiment + t
    )


def tag_dir(path: Path, policy: str | None = None) -> Path:
    """Memory-arm result directory under ``policy`` (success-only directories are unchanged)."""
    policy = validate(policy or current())
    t = DIR_TAG[policy]
    path = Path(path)
    return path if (not t or path.name.endswith(t)) else path.with_name(path.name + t)


def card_stamp(policy: str) -> dict:
    """Fields an adapter adds to a hydrated card: none under success-only (receipts stay byte-identical to
    the pre-2026-10-18 runs), ``memory_policy`` otherwise (the renderers then label the card)."""
    return {} if validate(policy) == SUCCESS_ONLY else {"memory_policy": policy}


def labelled(card: dict) -> bool:
    """Cards hydrated under a labelling policy carry ``memory_policy``; older cards render unchanged."""
    return card.get("memory_policy") == SUCCESS_FAILURE


def label(success: bool, outcome: str) -> str:
    if success:
        return f"{SUCCESS_TAG} — this approach solved the task (outcome: {outcome})."
    return (
        f"{FAILED_TAG} — this approach did NOT solve the task (outcome: {outcome}); do not repeat it; "
        "use it to avoid the same mistake."
    )


def rank_view(render):
    """Card renderer for the non-LLM Rank operator (ColBERT): the card WITHOUT its outcome label, so the
    ranking is the same content similarity as under success-only (labels would perturb the scores even
    when no failed session is in the pool). SemFilter / SemRank judges see the labelled card."""

    def unlabelled(card: dict) -> str:
        return render({k: v for k, v in card.items() if k != "memory_policy"})

    return unlabelled


def card_outcomes(cards: list[dict]) -> dict:
    """Successful vs failed cards (recorded per episode in the result file and the opt receipt)."""
    ok = sum(1 for c in cards if c.get("success") is True)
    bad = sum(1 for c in cards if c.get("success") is False)
    return {"success": ok, "failed": bad, "total": len(cards)}


def header(base: str, policy: str) -> str:
    """``base`` (a dataset's success-only header) under ``policy``; unchanged for success-only."""
    if validate(policy) == SUCCESS_ONLY:
        return base
    stripped = base.rstrip("\n")
    return stripped + " " + HEADER_NOTE + base[len(stripped) :]


def protocol(policy: str) -> dict:
    return {"memory_policy": validate(policy)}


def claim_dir(out: Path, policy: str) -> None:
    """Result directories never mix policies. A success+failure run writes ``.memory_policy`` and refuses
    a directory that already holds results without it (those are success-only runs); a success-only run
    refuses a directory marked for another policy. Success-only never writes the marker (old dirs untouched)."""
    validate(policy)
    out = Path(out)
    marker = out / MARKER
    if marker.exists():
        found = json.loads(marker.read_text()).get("memory_policy")
        if found != policy:
            raise SystemExit(
                f"{out} holds results of memory policy {found!r}, not {policy!r}"
            )
        return
    if policy == SUCCESS_ONLY:
        return
    if out.exists() and (any(out.glob("*.json")) or (out / "memory").exists()):
        raise SystemExit(
            f"{out} already holds results without a memory-policy marker (success-only runs); "
            f"use a separate directory for {policy!r}"
        )
    out.mkdir(parents=True, exist_ok=True)
    # Two halves of one arm may race. The marker must appear atomically WITH its content: an O_EXCL create
    # followed by a write let the other half read an empty marker (JSONDecodeError, the half died at start;
    # ALF c16 rerun 2026-10-21). Write a private temp file, then hard-link it into place (fails if it exists).
    tmp = out / f"{MARKER}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"memory_policy": policy}))
    try:
        os.link(tmp, marker)
    except FileExistsError:
        claim_dir(out, policy)
    finally:
        tmp.unlink(missing_ok=True)


def policy_of_experiment(experiment: str) -> str:
    """Memory policy an experiment id was run under (untagged ids are success-only)."""
    return (
        SUCCESS_FAILURE
        if EXPERIMENT_TAG[SUCCESS_FAILURE] in experiment
        else SUCCESS_ONLY
    )


def check_router(artifact: dict, policy: str) -> None:
    """A query router is trained on one BO experiment's opt rows; it is deployed only under that policy."""
    exp = ((artifact or {}).get("trained_on") or {}).get("experiment")
    if exp is not None and policy_of_experiment(exp) != validate(policy):
        raise SystemExit(
            f"router trained on {exp!r} (memory policy {policy_of_experiment(exp)}); "
            f"this run uses {policy}"
        )
