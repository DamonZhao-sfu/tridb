"""Unit tests for the LLM-mediated and agentic ingest strategies (no network).

The model is a scripted fake, so these cover the parts that are ours: the
validate() gate matrix, the two embedding regimes that make [AM] §4.3's
bimodality measurable, the mandatory caps of Paradigm IV, and the accounting
that turns a weak construction model into a FAILED CONFIGURATION rather than a
low-accuracy datapoint ([AM] §4.4).
"""

from __future__ import annotations

import json

import pytest

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.ingest import validate_plan
from bench.agent_memory.gem.strategies.agentic import AgenticIngestStrategy
from bench.agent_memory.gem.strategies.llm_mediated import (
    MODE_BATCH,
    MODE_SEQUENTIAL,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    LLMMediatedIngestStrategy,
)
from bench.agent_memory.gem.types import InteractionEvent, SemanticUnit


class ScriptedClient:
    """Returns queued responses in order; records the prompts it was given."""

    def __init__(self, responses):
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.prompts: list[tuple[str, str]] = []
        self.index = 0

    def complete(self, system, user):
        self.prompts.append((system, user))
        if self.index < len(self.responses):
            text = self.responses[self.index]
            self.index += 1
        else:
            text = json.dumps({"done": True})
        return text, {"prompt_tokens": 10, "completion_tokens": 5}


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeChunker:
    def __init__(self, n=1):
        self.n = n

    def chunk(self, text):
        return [f"{text}::{i}" for i in range(self.n)]


class FakeView:
    def __init__(self, units=None):
        self._units = list(units or [])

    def units(self, scope_id, *, limit=100):
        return self._units[:limit]

    def unit(self, unit_id):
        return next((u for u in self._units if u.id == unit_id), None)

    def find_similar(self, scope_id, embedding, *, k=10):
        return self._units[:k]

    def edges(self, unit_id):
        return []

    def policies(self, scope_id=None):
        return []


def _event(content="the deadline is April 20"):
    return InteractionEvent(
        scope_id="s1", external_id="e1", content=content, event_time="2026-03-08"
    )


def _extraction(**overrides):
    base = {
        "host": {"new_title": "Website Redesign", "summary": "the redesign project"},
        "facts": [
            {
                "field": "deadline",
                "value": "April 20",
                "valid_from": "2026-03-08",
                "confidence": 0.9,
            }
        ],
        "edges": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate() — one case per gate
# ---------------------------------------------------------------------------


class TestValidateGates:
    def _strategy(self):
        return LLMMediatedIngestStrategy(
            client=ScriptedClient([]), model="m", chunker=FakeChunker()
        )

    def test_a_well_formed_extraction_passes(self):
        assert self._strategy().validate(_extraction()) == (True, None)

    def test_non_object_is_rejected(self):
        ok, reason = self._strategy().validate(["not", "an", "object"])
        assert not ok and reason.startswith("schema")

    def test_missing_host_is_rejected(self):
        ok, reason = self._strategy().validate({"facts": []})
        assert not ok and "host" in reason

    def test_host_without_id_or_title_is_rejected(self):
        ok, reason = self._strategy().validate(_extraction(host={"summary": "x"}))
        assert not ok and "neither" in reason

    def test_non_integer_unit_id_is_rejected(self):
        ok, reason = self._strategy().validate(_extraction(host={"unit_id": "12"}))
        assert not ok and "integer" in reason

    @pytest.mark.parametrize("missing", ["field", "value", "valid_from"])
    def test_a_fact_missing_a_required_key_is_rejected(self, missing):
        fact = {"field": "deadline", "value": "April 20", "valid_from": "2026-03-08"}
        del fact[missing]
        ok, reason = self._strategy().validate(_extraction(facts=[fact]))
        assert not ok and missing in reason

    def test_two_values_for_one_field_without_supersession_is_a_conflict(self):
        """Would leave two current values, which the partial unique index would
        abort — naming it here beats surfacing an integrity error."""
        facts = [
            {"field": "deadline", "value": "A", "valid_from": "2026-01-01"},
            {"field": "deadline", "value": "B", "valid_from": "2026-02-01"},
        ]
        ok, reason = self._strategy().validate(_extraction(facts=facts))
        assert not ok and reason.startswith("conflict")

    def test_the_same_field_twice_WITH_supersession_is_allowed(self):
        facts = [
            {"field": "deadline", "value": "A", "valid_from": "2026-01-01"},
            {
                "field": "deadline",
                "value": "B",
                "valid_from": "2026-02-01",
                "supersede": True,
            },
        ]
        assert self._strategy().validate(_extraction(facts=facts))[0]

    def test_an_edge_kind_outside_the_two_native_kinds_is_rejected(self):
        edges = [{"dst_title": "Milestones", "kind": "mentions", "rel": "r"}]
        ok, reason = self._strategy().validate(_extraction(edges=edges))
        assert not ok and reason.startswith("enum")

    def test_an_edge_without_a_target_is_rejected(self):
        ok, reason = self._strategy().validate(
            _extraction(edges=[{"kind": "extension"}])
        )
        assert not ok and reason.startswith("referential")


# ---------------------------------------------------------------------------
# LLM-mediated planning
# ---------------------------------------------------------------------------


class TestLLMMediatedPlanning:
    def test_emits_a_valid_plan(self):
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction()]),
            model="m",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView())
        assert validate_plan(ops) == []
        assert any(o["kind"] == planmod.UPSERT_UNIT for o in ops)
        assert any(
            o["field"] == "deadline"
            for o in ops
            if o["kind"] == planmod.APPEND_FIELD_VALUE
        )

    def test_provenance_records_the_pinned_prompt_and_model(self):
        """C4: a stored fact must always be traceable to the exact extractor."""
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction()]),
            model="qwen3-8b",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView())
        fact = next(o for o in ops if o["kind"] == planmod.APPEND_FIELD_VALUE)
        assert fact["provenance"]["extractor_model"] == "qwen3-8b"
        assert fact["provenance"]["prompt_version"] == PROMPT_VERSION
        assert fact["provenance"]["confidence"] == 0.9

    def test_the_unit_embedding_source_is_title_summary_not_chunk_text(self):
        """Different from DeterministicIngest — never compare retrieval quality
        across two embedding_source values without saying so."""
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction()]),
            model="m",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        unit = next(
            o
            for o in strategy.plan([_event()], FakeView())
            if o["kind"] == planmod.UPSERT_UNIT
        )
        assert unit["metadata"]["embedding_source"] == "title_summary"
        assert unit["metadata"]["schema_version"] == SCHEMA_VERSION

    def test_a_malformed_extraction_is_rejected_and_counted_not_raised(self):
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient(["this is not json at all"]),
            model="m",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView())
        assert ops == []
        assert strategy.rejections[0]["gate"] == "json"

    def test_markdown_fences_are_tolerated_but_nothing_else_is_repaired(self):
        """A model that cannot follow the output contract at all IS the
        capability-floor finding — repairing its output would hide the signal
        being measured."""
        fenced = "```json\n" + json.dumps(_extraction()) + "\n```"
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([fenced]),
            model="m",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        assert strategy.plan([_event()], FakeView())

    def test_an_edge_target_that_does_not_exist_is_created_in_the_same_plan(self):
        """Otherwise the link would be a dangling vertex."""
        extraction = _extraction(
            edges=[{"dst_title": "Milestones", "kind": "extension", "rel": "schedules"}]
        )
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([extraction]),
            model="m",
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView())
        assert validate_plan(ops) == []
        titles = {o["title"] for o in ops if o["kind"] == planmod.UPSERT_UNIT}
        assert "Milestones" in titles

    def test_llm_cost_is_accumulated_for_the_transition_meter(self):
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction(), _extraction()]),
            model="m",
            chunker=FakeChunker(2),
            embedder=FakeEmbedder(),
        )
        strategy.plan([_event()], FakeView())
        assert strategy.cost["llm_calls"] == 2
        assert strategy.cost["prompt_tokens"] == 20


class TestEmbeddingRegimes:
    """[AM] §4.3: embedding traffic is bimodal by paradigm, and the two regimes
    stress a serving stack differently. The difference must be structural."""

    def test_batch_leaves_embedding_to_the_operator_one_call_for_the_plan(self):
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction(), _extraction()]),
            model="m",
            mode=MODE_BATCH,
            chunker=FakeChunker(2),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView())
        units = [o for o in ops if o["kind"] == planmod.UPSERT_UNIT]
        # left unresolved -> the operator batches them into ONE embed call
        assert all(u["embedding"] is None and u["embed_text"] for u in units)

    def test_sequential_resolves_each_vector_itself_giving_1_to_1(self):
        embedder = FakeEmbedder()
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction()]),
            model="m",
            mode=MODE_SEQUENTIAL,
            chunker=FakeChunker(),
            embedder=embedder,
        )
        ops = strategy.plan([_event()], FakeView())
        units = [o for o in ops if o["kind"] == planmod.UPSERT_UNIT]
        assert all(u["embedding"] is not None for u in units)
        assert strategy.cost["embed_calls"] >= len(units)

    def test_sequential_without_an_embedder_is_refused(self):
        """It would silently degrade into III.a and mislabel the run."""
        with pytest.raises(ValueError, match="paradigm signature"):
            LLMMediatedIngestStrategy(
                client=ScriptedClient([]), model="m", mode=MODE_SEQUENTIAL
            )

    def test_sequential_resolves_add_vs_update_against_existing_state(self):
        """III.b's write loop: similarity search resolves ADD/UPDATE/DELETE."""
        existing = SemanticUnit(id=12, scope_id="s1", title="Website", summary="")
        existing.fields = {}
        strategy = LLMMediatedIngestStrategy(
            client=ScriptedClient([_extraction(host={"unit_id": 12})]),
            model="m",
            mode=MODE_SEQUENTIAL,
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
        )
        ops = strategy.plan([_event()], FakeView([existing]))
        fact = next(o for o in ops if o["kind"] == planmod.APPEND_FIELD_VALUE)
        # no current value yet -> ADD, not UPDATE
        assert fact["supersede_current"] is False
        assert fact["unit_id"] == 12

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode must be"):
            LLMMediatedIngestStrategy(
                client=ScriptedClient([]), model="m", mode="freestyle"
            )


# ---------------------------------------------------------------------------
# Agentic — Paradigm IV
# ---------------------------------------------------------------------------


class TestAgenticCaps:
    def test_caps_are_mandatory_constructor_arguments(self):
        """[AM] Recommendation 10: LLM-bounded phases need EXTERNAL iteration
        caps. A default is how a cap becomes someone else's problem."""
        with pytest.raises(TypeError):
            AgenticIngestStrategy(client=ScriptedClient([]), model="m")

    @pytest.mark.parametrize("rounds,calls", [(0, 5), (5, 0), (-1, 5)])
    def test_non_positive_caps_are_refused(self, rounds, calls):
        with pytest.raises(ValueError, match="must be positive"):
            AgenticIngestStrategy(
                client=ScriptedClient([]),
                model="m",
                max_rounds=rounds,
                max_tool_calls=calls,
            )

    def test_exhausting_max_rounds_records_capped(self):
        never_done = [{"tool": "search_memory", "args": {}} for _ in range(20)]
        strategy = AgenticIngestStrategy(
            client=ScriptedClient(never_done),
            model="m",
            max_rounds=3,
            max_tool_calls=100,
            chunker=FakeChunker(),
        )
        strategy.plan([_event()], FakeView())
        assert strategy.capped is True

    def test_exhausting_max_tool_calls_records_capped_and_keeps_what_exists(self):
        writes = [
            {
                "tool": "write_field",
                "args": {
                    "new_title": f"T{i}",
                    "field": "f",
                    "value": "v",
                    "valid_from": "2026-01-01",
                },
            }
            for i in range(20)
        ]
        strategy = AgenticIngestStrategy(
            client=ScriptedClient(writes),
            model="m",
            max_rounds=100,
            max_tool_calls=2,
            chunker=FakeChunker(),
        )
        ops = strategy.plan([_event()], FakeView())
        assert strategy.capped is True
        assert ops, "a capped run must still commit what exists"

    def test_stopping_on_the_budget_is_capped_even_if_done_would_be_next(self):
        """The budget check fires BEFORE the agent's next turn, so we never
        learn it was about to finish. Recording that as capped is the honest
        reading: the loop was stopped, not concluded."""
        strategy = AgenticIngestStrategy(
            client=ScriptedClient(
                [
                    {"tool": "search_memory", "args": {}},
                    {"tool": "search_memory", "args": {}},
                    {"done": True},  # never reached — budget already spent
                ]
            ),
            model="m",
            max_rounds=10,
            max_tool_calls=2,
            chunker=FakeChunker(1),
        )
        strategy.plan([_event()], FakeView())
        assert strategy.capped is True

    def test_finishing_under_budget_is_not_capped(self):
        strategy = AgenticIngestStrategy(
            client=ScriptedClient(
                [{"tool": "search_memory", "args": {}}, {"done": True}]
            ),
            model="m",
            max_rounds=10,
            max_tool_calls=5,
            chunker=FakeChunker(1),
        )
        strategy.plan([_event()], FakeView())
        assert strategy.capped is False

    def test_budget_exhausted_with_chunks_left_is_capped(self):
        strategy = AgenticIngestStrategy(
            client=ScriptedClient([{"tool": "search_memory", "args": {}}] * 20),
            model="m",
            max_rounds=10,
            max_tool_calls=2,
            chunker=FakeChunker(5),  # five chunks, budget for ~one
        )
        strategy.plan([_event()], FakeView())
        assert strategy.capped is True

    def test_a_clean_finish_is_not_capped(self):
        strategy = AgenticIngestStrategy(
            client=ScriptedClient([{"done": True}]),
            model="m",
            max_rounds=5,
            max_tool_calls=5,
            chunker=FakeChunker(),
        )
        strategy.plan([_event()], FakeView())
        assert strategy.capped is False


class TestAgenticTools:
    def _strategy(self, script, **kwargs):
        return AgenticIngestStrategy(
            client=ScriptedClient(script),
            model="m",
            max_rounds=kwargs.pop("max_rounds", 10),
            max_tool_calls=kwargs.pop("max_tool_calls", 10),
            chunker=FakeChunker(),
            **kwargs,
        )

    def test_write_field_creates_the_unit_and_the_fact(self):
        strategy = self._strategy(
            [
                {
                    "tool": "write_field",
                    "args": {
                        "new_title": "Website",
                        "field": "deadline",
                        "value": "April 20",
                        "valid_from": "2026-03-08",
                    },
                },
                {"done": True},
            ]
        )
        ops = strategy.plan([_event()], FakeView())
        assert validate_plan(ops) == []
        assert any(o["kind"] == planmod.UPSERT_UNIT for o in ops)
        assert any(o["kind"] == planmod.APPEND_FIELD_VALUE for o in ops)

    def test_link_creates_both_endpoints_so_nothing_dangles(self):
        strategy = self._strategy(
            [
                {
                    "tool": "link",
                    "args": {
                        "src_title": "A",
                        "dst_title": "B",
                        "kind": "extension",
                        "rel": "entails",
                    },
                },
                {"done": True},
            ]
        )
        ops = strategy.plan([_event()], FakeView())
        assert validate_plan(ops) == []
        titles = {o["title"] for o in ops if o["kind"] == planmod.UPSERT_UNIT}
        assert titles == {"A", "B"}

    def test_an_invalid_edge_kind_is_refused_by_the_tool_not_the_schema(self):
        strategy = self._strategy(
            [
                {
                    "tool": "link",
                    "args": {"src_title": "A", "dst_title": "B", "kind": "mentions"},
                },
                {"done": True},
            ]
        )
        strategy.plan([_event()], FakeView())
        assert any(
            "extension|association" in str(t["result"]) for t in strategy.transcript
        )

    def test_read_unit_sees_state_through_the_view(self):
        """Inside the operator's transaction this view is bound to the Tx, so
        the agent's later reads see its own earlier writes — which is what
        makes the loop coherent without per-round commits."""
        unit = SemanticUnit(id=7, scope_id="s1", title="Website", summary="s")
        strategy = self._strategy(
            [{"tool": "read_unit", "args": {"unit_id": 7}}, {"done": True}]
        )
        strategy.plan([_event()], FakeView([unit]))
        result = strategy.transcript[0]["result"]
        assert result["unit_id"] == 7 and result["title"] == "Website"

    def test_an_unknown_tool_stops_the_loop_and_is_recorded(self):
        strategy = self._strategy([{"tool": "drop_database", "args": {}}])
        strategy.plan([_event()], FakeView())
        assert strategy.rejections[0]["gate"] == "enum"

    def test_malformed_json_stops_the_loop_and_is_recorded(self):
        strategy = self._strategy(["not json"])
        strategy.plan([_event()], FakeView())
        assert strategy.rejections[0]["gate"] == "json"

    def test_the_tool_surface_is_exactly_the_specified_five(self):
        from bench.agent_memory.gem.strategies.agentic import TOOLS

        assert set(TOOLS) == {
            "search_memory",
            "read_unit",
            "write_field",
            "link",
            "split_topic",
        }
