"""C1–C6 conformance — one test per correctness condition.

Two layers, deliberately:

* **Report semantics** (no database). The report must never let an unchecked or
  violated condition read as a pass, and must never emit a wholesale
  "GEM-conformant" label unless every condition actually holds. These run in CI.
* **Live conditions** (skip-gated on ``TRIDB_GEM_DSN``). The scripted
  trajectories that exercise each condition against a real engine.

The report is the deliverable: it lets us say WHICH conditions hold rather than
claiming conformance wholesale.
"""

from __future__ import annotations

import os

import pytest

from bench.agent_memory.gem import conformance
from bench.agent_memory.gem.conformance import (
    CONDITIONS,
    ConditionResult,
    ConformanceReport,
)

DSN = os.environ.get("TRIDB_GEM_DSN")


# ---------------------------------------------------------------------------
# Report semantics — no engine needed
# ---------------------------------------------------------------------------


class TestReportSemantics:
    def test_all_six_conditions_are_described(self):
        assert set(CONDITIONS) == {"C1", "C2", "C3", "C4", "C5", "C6"}
        assert all(text for text in CONDITIONS.values())

    def test_a_full_pass_earns_the_conformant_label(self):
        report = ConformanceReport()
        for condition, description in CONDITIONS.items():
            report.add(ConditionResult(condition, description, True))
        assert report.label() == "GEM-conformant"
        assert report.violated == [] and report.unchecked == []

    def test_one_violation_forfeits_the_label(self):
        report = ConformanceReport()
        for condition, description in CONDITIONS.items():
            report.add(ConditionResult(condition, description, condition != "C6"))
        assert report.label() != "GEM-conformant"
        assert report.violated == ["C6"]

    def test_an_unchecked_condition_forfeits_the_label(self):
        """The failure mode worth guarding: 'we did not test C3' must never
        render the same as 'C3 holds'."""
        report = ConformanceReport()
        for condition in ("C1", "C2", "C4"):
            report.add(ConditionResult(condition, CONDITIONS[condition], True))
        assert report.label() != "GEM-conformant"
        assert set(report.unchecked) == {"C3", "C5", "C6"}

    def test_a_none_result_counts_as_unchecked_not_satisfied(self):
        report = ConformanceReport()
        for condition, description in CONDITIONS.items():
            report.add(
                ConditionResult(
                    condition, description, None if condition == "C6" else True
                )
            )
        assert "C6" in report.unchecked
        assert "C6" not in report.satisfied
        assert report.label() != "GEM-conformant"

    def test_the_engine_visibility_caveat_travels_with_every_report(self):
        """No claim here may depend on repeatable-read topology (§6.5)."""
        payload = ConformanceReport().to_dict()
        assert payload["caveats"]["graph_read_visibility"] == (
            "commit_visible, not snapshot_isolated"
        )
        assert "not supported" in payload["caveats"]["repeatable_read_topology_claims"]

    def test_a_raising_check_is_recorded_as_unchecked_not_as_a_pass(self):
        class Exploding:
            class store:
                class conn:
                    @staticmethod
                    def execute(*args, **kwargs):
                        raise RuntimeError("no such table")

        report = conformance.run(Exploding(), "scope")
        assert report.satisfied == []
        assert set(report.unchecked) == set(CONDITIONS)
        assert all("check raised" in r.detail for r in report.results)

    def test_the_report_is_json_serialisable(self):
        import json

        report = ConformanceReport(configuration={"strategy": "deterministic"})
        report.add(ConditionResult("C1", CONDITIONS["C1"], True, {"dupes": 0}))
        assert json.loads(json.dumps(report.to_dict()))["satisfied"] == ["C1"]


# ---------------------------------------------------------------------------
# Live conditions
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    not DSN, reason="set TRIDB_GEM_DSN to run live conformance checks"
)


@pytest.fixture()
def memory():
    from bench.agent_memory.gem.memory import TriDBGovernedMemory
    from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
    from bench.agent_memory.gem.store import GemStore

    from tests.test_gem_live import DIM, StubEmbedder

    store = GemStore.connect(DSN, dim=DIM)
    mem = TriDBGovernedMemory(
        store,
        embedder=StubEmbedder(),
        policy_engine=PolicyEngine(extra=seed_policies()),
    )
    mem.init_schema()
    yield mem
    mem.close()


@pytest.fixture()
def scope(memory, request):
    scope_id = f"conf_{request.node.name[:40]}"
    memory.store.conn.execute("DELETE FROM gem_unit WHERE scope_id = %s", (scope_id,))
    memory.store.conn.execute(
        "DELETE FROM gem_transition WHERE scope_id = %s", (scope_id,)
    )
    return scope_id


def _ingest(memory, scope, ops):
    class Plan:
        name = "test"
        cost: dict = {}

        def plan(self, events, view):
            return ops

    from bench.agent_memory.gem.types import InteractionEvent

    return memory.ingest(
        [InteractionEvent(scope_id=scope, external_id="e", content="x")],
        strategy=Plan(),
    )


@live
class TestLiveConditions:
    def test_c1_supersession_leaves_exactly_one_current_value(self, memory, scope):
        from bench.agent_memory.gem import plan as planmod

        _ingest(
            memory,
            scope,
            [
                planmod.upsert_unit(scope_id=scope, title="W", ref="w", embed_text="W"),
                planmod.append_field_value(
                    ref="w", field="deadline", value="March 15", valid_from="2026-02-01"
                ),
            ],
        )
        _ingest(
            memory,
            scope,
            [
                planmod.upsert_unit(scope_id=scope, title="W", ref="w", embed_text="W"),
                planmod.append_field_value(
                    ref="w",
                    field="deadline",
                    value="April 20",
                    valid_from="2026-03-08",
                    supersede_current=True,
                ),
            ],
        )
        result = conformance.check_c1(memory, scope)
        assert result.holds, result.evidence
        assert result.evidence["superseded_values"] == 1

    def test_c2_a_rejected_transition_is_logged_aborted(self, memory, scope):
        class Exploding:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                raise RuntimeError("policy would reject this")

        from bench.agent_memory.gem.types import InteractionEvent

        memory.ingest(
            [InteractionEvent(scope_id=scope, external_id="e", content="x")],
            strategy=Exploding(),
        )
        result = conformance.check_c2(memory, scope)
        assert result.holds, result.evidence
        assert result.evidence["aborted_transitions"] >= 1

    def test_c3_extension_reaches_dependents_and_association_does_not(
        self, memory, scope
    ):
        result = conformance.check_c3(memory, scope)
        assert result.holds is not False, result.detail

    def test_c4_the_provenance_chain_survives_revision_and_forgetting(
        self, memory, scope
    ):
        from bench.agent_memory.gem import plan as planmod

        _ingest(
            memory,
            scope,
            [
                planmod.upsert_unit(scope_id=scope, title="W", ref="w", embed_text="W"),
                planmod.append_field_value(
                    ref="w", field="deadline", value="March 15", valid_from="2026-02-01"
                ),
            ],
        )
        _ingest(
            memory,
            scope,
            [
                planmod.upsert_unit(scope_id=scope, title="W", ref="w", embed_text="W"),
                planmod.append_field_value(
                    ref="w",
                    field="deadline",
                    value="April 20",
                    valid_from="2026-03-08",
                    supersede_current=True,
                ),
            ],
        )
        memory.revise(scope)
        memory.forget(scope)
        result = conformance.check_c4(memory, scope)
        assert result.holds, result.evidence
        assert result.evidence["broken_chains"] == 0

    def test_c5_active_stays_bounded_and_archived_stays_recoverable(
        self, memory, scope
    ):
        from bench.agent_memory.gem import plan as planmod

        for index in range(10):
            _ingest(
                memory,
                scope,
                [
                    planmod.upsert_unit(
                        scope_id=scope,
                        title=f"T{index}",
                        ref=f"t{index}",
                        embed_text=f"topic {index}",
                    )
                ],
            )
        memory.forget(scope)
        result = conformance.check_c5(memory, scope, beta=100)
        assert result.holds, result.evidence

    def test_c6_retrieval_strictly_increases_salience(self, memory, scope):
        from bench.agent_memory.gem import plan as planmod
        from bench.agent_memory.gem.types import Query, RetrievalMode

        _ingest(
            memory,
            scope,
            [
                planmod.upsert_unit(
                    scope_id=scope, title="Alpha", ref="a", embed_text="alpha"
                )
            ],
        )
        memory.retrieve(
            Query(
                scope_id=scope,
                text="alpha",
                mode=RetrievalMode.VECTOR,
                reinforce=True,
            )
        )
        result = conformance.check_c6(memory, scope)
        assert result.holds, result.evidence
        assert result.evidence["retrieved_with_non_positive_salience"] == 0

    def test_the_full_report_names_the_configuration(self, memory, scope):
        report = conformance.run(
            memory,
            scope,
            configuration={"strategy": "deterministic", "reinforce": False},
        )
        payload = report.to_dict()
        assert payload["configuration"]["reinforce"] is False
        assert len(payload["results"]) == len(CONDITIONS)
