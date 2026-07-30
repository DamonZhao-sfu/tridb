"""``P_t`` — policy evaluation as a commit postcondition.

[GEM] Definition 3 policies are ``<event, condition, action>`` rules living
inside ``M_t``. A general policy LANGUAGE is the paper's own research
direction and is explicitly **out of scope** here — a closed, named registry
with JSONB parameters gets the correctness conditions enforced without
inventing a DSL. Adding a policy means adding a registry entry: a code change,
but a ten-line one, while the RULES stay data.

**Two evaluation points:**

1. **Postcondition, before commit.** For every enabled policy matching the
   operator's event, evaluate ``condition`` against the *proposed* ``M_{t+1}``
   — which, inside the transaction, is simply what this transaction can see.
   Any failure raises :class:`~bench.agent_memory.gem.store.PolicyViolation`,
   the transaction rolls back, and the audit connection logs it with
   ``aborted_reason``. **This is what lifts C2 to a data-model guarantee**:
   rollback is the enforcement mechanism, not an application-level check.

2. **Action triggering, inside the operator.** e.g. ``propagate-on-change``
   flags dependent units after a field update.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from bench.agent_memory.gem.store import PolicyViolation, Tx
from bench.agent_memory.gem.types import Policy, PolicyEvent

Condition = Callable[[Tx, Mapping[str, Any]], bool]
Action = Callable[[Tx, Mapping[str, Any]], None]

CONDITIONS: dict[str, Condition] = {}
ACTIONS: dict[str, Action] = {}


def condition(name: str) -> Callable[[Condition], Condition]:
    def register(func: Condition) -> Condition:
        CONDITIONS[name] = func
        return func

    return register


def action(name: str) -> Callable[[Action], Action]:
    def register(func: Action) -> Action:
        ACTIONS[name] = func
        return func

    return register


# ---------------------------------------------------------------------------
# Conditions — each returns True when the proposed M_{t+1} is ACCEPTABLE
# ---------------------------------------------------------------------------


@condition("no_duplicate_current")
def _no_duplicate_current(tx: Tx, params: Mapping[str, Any]) -> bool:
    """C1/C2: at most one current value per ``(unit, field)``.

    Belt and braces. The partial unique index already makes a violating plan
    abort at commit, so this can only fire if that index is missing — which is
    exactly the regression worth catching loudly, since [GEM] Observation 3a
    identifies it as the whole engine-level mechanism append-only stores lack.
    """
    row = tx.execute(
        "SELECT count(*) FROM (SELECT fv.unit_id, fv.field FROM gem_field_value fv"
        " JOIN gem_unit u ON u.id = fv.unit_id"
        " WHERE u.scope_id = %s AND fv.valid_to IS NULL AND fv.state <> 'archived'"
        " GROUP BY fv.unit_id, fv.field HAVING count(*) > 1) dupes",
        (tx.scope_id,),
    ).fetchone()
    return row is None or int(row[0]) == 0


@condition("dependents_flagged")
def _dependents_flagged(tx: Tx, params: Mapping[str, Any]) -> bool:
    """C3: every extension-reachable dependent of a changed unit is flagged.

    Checks one hop out from the units this transition CHANGED
    (``tx.changed_units``) — deliberately not from ``delta.propagated_units``,
    which holds the dependents that got flagged. Reading the latter would demand
    that the flagged units' own dependents also be flagged, one hop further than
    ingest ever flags, so any ``A -ext-> B -ext-> C`` chain would abort every
    ingest under the default ``propagate-on-change`` policy. The multi-hop walk
    is ``revise``'s job and is bounded by ``max_hops``.
    """
    changed = list(tx.changed_units)
    if not changed:
        return True
    row = tx.execute(
        "SELECT count(*) FROM gem_unit u JOIN gem_edge e ON e.dst = u.id"
        " WHERE e.src = ANY(%s) AND e.kind = 'extension'"
        "   AND e.tombstoned_at IS NULL"
        # A dependent that is itself direct revision evidence is already being
        # re-evaluated in this transition; requiring it to remain flagged would
        # make two changed classes connected by an extension edge fail C3.
        "   AND NOT (u.id = ANY(%s))"
        "   AND COALESCE(u.metadata->>'needs_revision', 'false') <> 'true'",
        (changed, changed),
    ).fetchone()
    return row is None or int(row[0]) == 0


@condition("active_below_bound")
def _active_below_bound(tx: Tx, params: Mapping[str, Any]) -> bool:
    """C5: ``|D_t^active| <= beta(n)``.

    ``beta`` is supplied as a plain bound in the policy's JSONB parameters
    rather than as an expression — a policy language is out of scope, and a
    constant bound is enough to make the condition testable.
    """
    bound = params.get("max_active_units")
    if bound is None:
        return True
    active = tx.delta.active_units
    return active is None or active <= int(bound)


@condition("salience_monotone")
def _salience_monotone(tx: Tx, params: Mapping[str, Any]) -> bool:
    """C6: no retrieval may lower a unit's salience.

    Only meaningful on the retrieve path, where ``salience_updates`` is
    non-zero; other operators pass trivially. Negative salience is the
    observable signature of a decay applied on the read path, which the design
    forbids (decay is lazy, at forget tick time).
    """
    if tx.operator != "retrieve" or tx.delta.salience_updates == 0:
        return True
    row = tx.execute(
        "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND salience < 0",
        (tx.scope_id,),
    ).fetchone()
    return row is None or int(row[0]) == 0


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@action("flag_for_revision")
def _flag_for_revision(tx: Tx, params: Mapping[str, Any]) -> None:
    units = [int(u) for u in (params.get("units") or tx.delta.propagated_units)]
    if not units:
        return
    tx.execute(
        "UPDATE gem_unit SET metadata = jsonb_set(metadata, '{needs_revision}',"
        " 'true'::jsonb) WHERE id = ANY(%s)",
        (units,),
    )


@action("promote_edge_candidate")
def _promote_edge_candidate(tx: Tx, params: Mapping[str, Any]) -> None:
    """Flag an association edge as an extension candidate — flag only.

    Actual promotion grants propagation rights and widens the C3 frontier, so
    it stays a ``revise`` decision.
    """
    threshold = int(params.get("co_access_threshold", 5))
    tx.execute(
        "UPDATE gem_unit SET metadata = jsonb_set(metadata,"
        " '{extension_candidate}', 'true'::jsonb) WHERE id IN ("
        "  SELECT src FROM gem_edge WHERE kind = 'association'"
        "   AND tombstoned_at IS NULL AND co_access_count >= %s)",
        (threshold,),
    )


@action("attenuate")
def _attenuate(tx: Tx, params: Mapping[str, Any]) -> None:
    """Move units below a salience threshold one rung down the ladder.

    Never a DELETE: C4 requires the provenance chain of anything reachable to
    survive and C5 requires archived content to stay recoverable.
    """
    threshold = float(params.get("threshold", 0.0))
    state = str(params.get("state", "hidden"))
    if state not in ("compressed", "hidden", "archived"):
        raise ValueError(f"attenuate: {state!r} is not a ladder rung")
    tx.execute(
        "UPDATE gem_unit SET state = %s WHERE scope_id = %s AND salience < %s"
        "  AND state = 'active'",
        (state, tx.scope_id, threshold),
    )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

#: The operator each policy event corresponds to, for postcondition matching.
_EVENT_OPERATORS: dict[PolicyEvent, tuple[str, ...]] = {
    PolicyEvent.UNIT_INGESTED: ("ingest",),
    PolicyEvent.FIELD_UPDATED: ("ingest", "revise"),
    PolicyEvent.RETRIEVAL: ("retrieve",),
    PolicyEvent.TICK: ("forget",),
}


class PolicyEngine:
    """Evaluates ``P_t`` as a commit postcondition and triggers actions.

    Policies are loaded from ``gem_policy`` — they live inside the state, per
    Definition 3 — with :attr:`extra` available for tests and for seeding
    before the table exists.
    """

    def __init__(self, *, extra: list[Policy] | None = None) -> None:
        self.extra = list(extra or [])

    def load(self, tx: Tx) -> list[Policy]:
        from bench.agent_memory.gem.store import TriDBMemoryView

        view = TriDBMemoryView(tx.store, tx=tx)
        try:
            stored = view.policies(tx.scope_id)
        except Exception:  # noqa: BLE001 — gem_policy may not exist yet
            stored = []
        return [*stored, *self.extra]

    def evaluate(self, tx: Tx) -> None:
        """Postcondition. Raises :class:`PolicyViolation` to reject the transition."""
        for policy in self.load(tx):
            if not policy.enabled:
                continue
            if tx.operator not in _EVENT_OPERATORS.get(policy.event, ()):
                continue
            name = policy.condition.get("name")
            if not name:
                continue
            check = CONDITIONS.get(str(name))
            if check is None:
                raise PolicyViolation(
                    policy.name,
                    f"condition {name!r} is not in the registry — the registry is "
                    "closed by design (a policy language is out of scope)",
                )
            tx.policies_evaluated.append(policy.name)
            params = policy.condition.get("params") or {}
            if not check(tx, params):
                raise PolicyViolation(policy.name, f"condition {name!r} does not hold")

    def trigger(self, tx: Tx, event: PolicyEvent, **params: Any) -> list[str]:
        """Run the ACTIONS of policies matching ``event``. Returns their names."""
        fired: list[str] = []
        for policy in self.load(tx):
            if not policy.enabled or policy.event is not event:
                continue
            name = policy.action.get("name")
            if not name:
                continue
            act = ACTIONS.get(str(name))
            if act is None:
                raise PolicyViolation(
                    policy.name, f"action {name!r} is not in the registry"
                )
            act(tx, {**(policy.action.get("params") or {}), **params})
            fired.append(policy.name)
        return fired


# ---------------------------------------------------------------------------
# Seed policies — [GEM] Listing 1 and the correctness conditions
# ---------------------------------------------------------------------------


def seed_policies(*, max_active_units: int | None = None) -> list[Policy]:
    """The three policies shipped by default."""
    policies = [
        Policy(
            name="propagate-on-change",
            event=PolicyEvent.FIELD_UPDATED,
            condition={"name": "dependents_flagged"},
            action={"name": "flag_for_revision"},
        ),
        Policy(
            name="reinforce-on-read",
            event=PolicyEvent.RETRIEVAL,
            condition={"name": "salience_monotone"},
            action={"name": "promote_edge_candidate", "params": {}},
        ),
    ]
    if max_active_units is not None:
        policies.append(
            Policy(
                name="bound-active-state",
                event=PolicyEvent.TICK,
                condition={
                    "name": "active_below_bound",
                    "params": {"max_active_units": max_active_units},
                },
                action={"name": "attenuate", "params": {"threshold": 0.0}},
            )
        )
    return policies
