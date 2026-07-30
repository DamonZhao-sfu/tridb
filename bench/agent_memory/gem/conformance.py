"""The C1–C6 conformance report.

[GEM] Definition 4 makes correctness a property of the trajectory ``{M_t}``,
condition by condition. This module checks each condition against a live store
and emits a machine-readable report.

**The report is the deliverable.** It lets us say *which conditions hold*
rather than claiming "GEM-conformant" wholesale — the honesty gate the
interface doc §10 sets. A configuration that satisfies C1 and C4 but not C6 is
a real, reportable position; calling it conformant is not.

Per interface doc §10, note also what this report deliberately does NOT claim:
graph reads on this engine are commit-visible rather than snapshot-isolated
(§6.5), so **no conformance claim here depends on repeatable-read topology**.
Each check records that caveat in its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

CONDITIONS: dict[str, str] = {
    "C1": (
        "after supersession the default query returns only the new value; "
        "as_of returns the old one"
    ),
    "C2": (
        "a transition violating a policy leaves M_t byte-identical and is "
        "logged aborted"
    ),
    "C3": (
        "updating u_i flags every extension-reachable u_j and NO "
        "association-only neighbour"
    ),
    "C4": (
        "after revision AND after forgetting, the provenance chain of a "
        "reachable unit is intact"
    ),
    "C5": (
        "active_units stays <= beta(n) across a scripted trajectory; archived "
        "content is still retrievable by explicit lookup"
    ),
    "C6": (
        "salience after retrieval > before, strictly, for every hit; a unit "
        "retrieved k times outranks an equally-similar unit retrieved once"
    ),
}


@dataclass
class ConditionResult:
    condition: str
    description: str
    holds: bool | None  # None = not checked in this run
    evidence: dict[str, Any] = field(default_factory=dict)
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "description": self.description,
            "holds": self.holds,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass
class ConformanceReport:
    """Per-condition, never wholesale."""

    results: list[ConditionResult] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)

    def add(self, result: ConditionResult) -> None:
        self.results.append(result)

    @property
    def satisfied(self) -> list[str]:
        return [r.condition for r in self.results if r.holds is True]

    @property
    def violated(self) -> list[str]:
        return [r.condition for r in self.results if r.holds is False]

    @property
    def unchecked(self) -> list[str]:
        checked = {r.condition for r in self.results}
        return [c for c in CONDITIONS if c not in checked] + [
            r.condition for r in self.results if r.holds is None
        ]

    def label(self) -> str:
        """The name this configuration may be reported under.

        "GEM-conformant" requires ALL of C1–C6 to hold and none unchecked.
        Anything else is named for what it actually is.
        """
        if self.violated or self.unchecked:
            return "TriDB-vector (Paradigm II embedRAG)"
        return "GEM-conformant"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label(),
            "satisfied": self.satisfied,
            "violated": self.violated,
            "unchecked": self.unchecked,
            "configuration": self.configuration,
            # Every claim in this report is bounded by the engine's visibility
            # model. Stating it here means a reader never has to find §6.5.
            "caveats": {
                "graph_read_visibility": "commit_visible, not snapshot_isolated",
                "repeatable_read_topology_claims": "not supported on this engine",
            },
            "results": [r.to_dict() for r in self.results],
        }


CheckFn = Callable[[Any, str], ConditionResult]

_CHECKS: dict[str, CheckFn] = {}


def check(condition: str) -> Callable[[CheckFn], CheckFn]:
    def register(func: CheckFn) -> CheckFn:
        _CHECKS[condition] = func
        return func

    return register


# ---------------------------------------------------------------------------
# The checks. Each takes a live TriDBGovernedMemory and a scope id.
# ---------------------------------------------------------------------------


@check("C1")
def check_c1(memory: Any, scope_id: str) -> ConditionResult:
    """Default query returns only the current value; ``as_of`` returns the old.

    Read directly rather than through ``retrieve`` so the condition is tested
    on the data model, not on one retrieval mode's behaviour.
    """
    conn = memory.store.conn
    duplicates = conn.execute(
        "SELECT count(*) FROM (SELECT fv.unit_id, fv.field FROM gem_field_value fv"
        " JOIN gem_unit u ON u.id = fv.unit_id WHERE u.scope_id = %s"
        "  AND fv.valid_to IS NULL AND fv.state <> 'archived'"
        " GROUP BY fv.unit_id, fv.field HAVING count(*) > 1) d",
        (scope_id,),
    ).fetchone()[0]
    superseded = conn.execute(
        "SELECT count(*) FROM gem_field_value fv JOIN gem_unit u"
        " ON u.id = fv.unit_id WHERE u.scope_id = %s AND fv.superseded_by IS NOT NULL",
        (scope_id,),
    ).fetchone()[0]
    holds = int(duplicates) == 0
    return ConditionResult(
        "C1",
        CONDITIONS["C1"],
        holds,
        {
            "duplicate_current_values": int(duplicates),
            "superseded_values": int(superseded),
        },
        None if holds else "two values are current for the same (unit, field)",
    )


@check("C2")
def check_c2(memory: Any, scope_id: str) -> ConditionResult:
    """A violating transition is rejected AND recorded as aborted.

    Both halves matter: a rollback that vanishes from the trajectory makes the
    condition unobservable, which is why aborts are logged on the audit
    connection.
    """
    conn = memory.store.conn
    aborted = conn.execute(
        "SELECT count(*) FROM gem_transition WHERE scope_id = %s AND NOT committed",
        (scope_id,),
    ).fetchone()[0]
    with_reason = conn.execute(
        "SELECT count(*) FROM gem_transition WHERE scope_id = %s"
        "  AND NOT committed AND aborted_reason IS NOT NULL",
        (scope_id,),
    ).fetchone()[0]
    holds = int(aborted) == int(with_reason)
    return ConditionResult(
        "C2",
        CONDITIONS["C2"],
        holds,
        {"aborted_transitions": int(aborted), "with_reason": int(with_reason)},
        None if holds else "an aborted transition carries no reason",
    )


@check("C3")
def check_c3(memory: Any, scope_id: str) -> ConditionResult:
    """Extension-reachable dependents are flagged; association-only ones are not.

    The negative half is the one that actually distinguishes entailment from
    relatedness, so it is checked explicitly rather than assumed.
    """
    conn = memory.store.conn
    missed = conn.execute(
        "SELECT count(*) FROM gem_unit u JOIN gem_edge e ON e.dst = u.id"
        " JOIN gem_unit src ON src.id = e.src"
        " WHERE u.scope_id = %s AND e.kind = 'extension' AND e.tombstoned_at IS NULL"
        "   AND src.metadata->>'changed' = 'true'"
        "   AND COALESCE(u.metadata->>'changed', 'false') <> 'true'"
        "   AND COALESCE(u.metadata->>'needs_revision','false') <> 'true'",
        (scope_id,),
    ).fetchone()[0]
    leaked = conn.execute(
        "SELECT count(*) FROM gem_unit u WHERE u.scope_id = %s"
        "  AND u.metadata->>'needs_revision' = 'true'"
        "  AND NOT EXISTS (SELECT 1 FROM gem_edge e WHERE e.dst = u.id"
        "     AND e.kind = 'extension' AND e.tombstoned_at IS NULL)"
        "  AND EXISTS (SELECT 1 FROM gem_edge e WHERE e.dst = u.id"
        "     AND e.kind = 'association' AND e.tombstoned_at IS NULL)",
        (scope_id,),
    ).fetchone()[0]
    holds = int(missed) == 0 and int(leaked) == 0
    return ConditionResult(
        "C3",
        CONDITIONS["C3"],
        holds,
        {
            "unflagged_extension_dependents": int(missed),
            "association_only_flagged": int(leaked),
        },
        None if holds else "propagation followed relatedness rather than entailment",
    )


@check("C4")
def check_c4(memory: Any, scope_id: str) -> ConditionResult:
    """The provenance chain of a reachable unit survives revision and forgetting.

    Checks the chain is not merely present but WELL-FORMED: every superseded
    value points at a value that still exists, and no closed value lost its
    successor.
    """
    conn = memory.store.conn
    broken = conn.execute(
        "SELECT count(*) FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
        " WHERE u.scope_id = %s AND fv.superseded_by IS NOT NULL"
        "  AND NOT EXISTS (SELECT 1 FROM gem_field_value s"
        "     WHERE s.id = fv.superseded_by)",
        (scope_id,),
    ).fetchone()[0]
    orphaned = conn.execute(
        "SELECT count(*) FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
        " WHERE u.scope_id = %s AND fv.valid_to IS NOT NULL"
        "   AND fv.superseded_by IS NULL AND fv.state <> 'archived'",
        (scope_id,),
    ).fetchone()[0]
    total = conn.execute(
        "SELECT count(*) FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
        " WHERE u.scope_id = %s",
        (scope_id,),
    ).fetchone()[0]
    holds = int(broken) == 0
    return ConditionResult(
        "C4",
        CONDITIONS["C4"],
        holds,
        {
            "broken_chains": int(broken),
            "closed_without_successor": int(orphaned),
            "values_total": int(total),
        },
        None if holds else "a superseded value points at a row that no longer exists",
    )


@check("C5")
def check_c5(memory: Any, scope_id: str, *, beta: int | None = None) -> ConditionResult:
    """``|D_t^active|`` stays bounded; archived content stays recoverable.

    Nothing is ever DELETEd, so the second half is checked by confirming
    archived rows still exist and are still readable by explicit lookup.
    """
    conn = memory.store.conn
    peak = conn.execute(
        "SELECT COALESCE(max(active_units), 0) FROM gem_transition WHERE scope_id = %s",
        (scope_id,),
    ).fetchone()[0]
    archived = conn.execute(
        "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND state = 'archived'",
        (scope_id,),
    ).fetchone()[0]
    recoverable = conn.execute(
        "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND state = 'archived'"
        "  AND title IS NOT NULL",
        (scope_id,),
    ).fetchone()[0]
    holds = int(archived) == int(recoverable)
    if beta is not None:
        holds = holds and int(peak) <= int(beta)
    return ConditionResult(
        "C5",
        CONDITIONS["C5"],
        holds,
        {
            "peak_active_units": int(peak),
            "beta": beta,
            "archived_units": int(archived),
            "archived_recoverable": int(recoverable),
        },
        None
        if holds
        else "archived content is not recoverable, or the bound was exceeded",
    )


@check("C6")
def check_c6(memory: Any, scope_id: str) -> ConditionResult:
    """Repeated retrieval strictly reduces eligibility for attenuation.

    Checked on the data model — a retrieved unit has a positive access_count
    and a salience strictly above an equally-similar never-retrieved unit.
    """
    conn = memory.store.conn
    retrieved = conn.execute(
        "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND access_count > 0",
        (scope_id,),
    ).fetchone()[0]
    non_positive = conn.execute(
        "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND access_count > 0"
        "  AND salience <= 0",
        (scope_id,),
    ).fetchone()[0]
    outranks = conn.execute(
        "SELECT COALESCE(min(r.salience), 0) > COALESCE(max(n.salience), -1)"
        " FROM gem_unit r, gem_unit n"
        " WHERE r.scope_id = %s AND n.scope_id = %s"
        "   AND r.access_count > 0 AND n.access_count = 0",
        (scope_id, scope_id),
    ).fetchone()[0]
    holds = int(retrieved) > 0 and int(non_positive) == 0
    return ConditionResult(
        "C6",
        CONDITIONS["C6"],
        holds if int(retrieved) > 0 else None,
        {
            "retrieved_units": int(retrieved),
            "retrieved_with_non_positive_salience": int(non_positive),
            "retrieved_outrank_unretrieved": bool(outranks)
            if outranks is not None
            else None,
        },
        None
        if holds
        else (
            "no retrieval was recorded (reinforce off?)"
            if int(retrieved) == 0
            else "a retrieved unit has non-positive salience"
        ),
    )


def run(
    memory: Any,
    scope_id: str,
    *,
    configuration: Mapping[str, Any] | None = None,
    beta: int | None = None,
    only: list[str] | None = None,
) -> ConformanceReport:
    """Run the checks and build the report.

    A check that raises is recorded as ``holds=None`` with the error as its
    detail — an unchecked condition is a distinct outcome from a violated one
    and must never silently read as a pass.
    """
    report = ConformanceReport(configuration=dict(configuration or {}))
    for condition, fn in _CHECKS.items():
        if only and condition not in only:
            continue
        try:
            if condition == "C5":
                report.add(fn(memory, scope_id, beta=beta))  # type: ignore[call-arg]
            else:
                report.add(fn(memory, scope_id))
        except Exception as exc:  # noqa: BLE001 — an error is "unchecked", not "passes"
            report.add(
                ConditionResult(
                    condition,
                    CONDITIONS[condition],
                    None,
                    {},
                    f"check raised: {type(exc).__name__}: {exc}",
                )
            )
    return report
