"""Small live G2/G3 gate. Skipped unless ``TRIDB_GEM_DEMO_DSN`` is set."""

from __future__ import annotations

import os

import pytest

from bench.agent_memory.demo import adapter, scenario
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies

DSN = os.environ.get("TRIDB_GEM_DEMO_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TRIDB_GEM_DEMO_DSN to run the GEM wiki demo live gate"
)

DIM = 384


class StubEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * DIM
            for index, char in enumerate(text.encode("utf-8")):
                vector[index % DIM] += (char % 31) / 31.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture()
def memory():
    mem = TriDBGovernedMemory.connect(
        DSN,
        dim=DIM,
        embedder=StubEmbedder(),
        policy_engine=PolicyEngine(extra=seed_policies()),
    )
    mem.init_schema()
    yield mem
    mem.close()


@pytest.fixture()
def scope(memory, request):
    value = f"demo_live_{request.node.name[:35]}"
    scenario.reset_scope(memory, value)
    return value


def _wiki() -> adapter.WikiSlice:
    return adapter.WikiSlice(
        articles=(
            adapter.WikiArticle(
                "Article A",
                1,
                "Q1",
                "Article A is evidence about alpha.",
                label="Article A",
                classes=("QClass",),
            ),
            adapter.WikiArticle(
                "Article B",
                2,
                "Q2",
                "Article B is evidence about beta.",
                label="Article B",
            ),
        ),
        classes=(adapter.WikiClass("QClass", "example class"),),
        links=(("Article A", "Article B"),),
        revisions=(
            adapter.WikiRevision(
                "Q1", 1, "2026-01-01T00:00:00Z", "u", "/* wbsetlabel-set:1|en */ old A"
            ),
            adapter.WikiRevision(
                "Q1", 2, "2026-02-01T00:00:00Z", "u", "/* wbsetlabel-set:1|en */ new A"
            ),
            adapter.WikiRevision(
                "QClass",
                3,
                "2026-01-01T00:00:00Z",
                "u",
                "/* wbsetlabel-set:1|en */ old class",
            ),
            adapter.WikiRevision(
                "QClass",
                4,
                "2026-02-01T00:00:00Z",
                "u",
                "/* wbsetlabel-set:1|en */ new class",
            ),
        ),
        manifest={"counts": {"articles": 2, "classes": 1}},
    )


def test_all_four_operators_and_typed_evidence_live(memory, scope):
    wiki = _wiki()
    ingest = scenario.run_ingest(memory, wiki, scope)
    assert ingest["transition"]["committed"]

    retrieval = scenario.run_retrieve(
        memory,
        [
            {
                "id": "q",
                "question": "alpha beta",
                "gold_titles": ["Article A", "Article B"],
            }
        ],
        scope,
        k=2,
        term_cond=32,
    )
    points = retrieval["operating_points"]
    assert points["vector"]["hnsw_iterative_scan"] == "strict_order"
    assert points["fused"]["hnsw_iterative_scan"] == "relaxed_order"
    assert points["vector"]["queries"][0]["probes"]["hnsw_iterative_scan"] == (
        "strict_order"
    )
    assert points["fused"]["queries"][0]["probes"]["hnsw_iterative_scan"] == (
        "relaxed_order"
    )

    revised = scenario.run_revise(memory, wiki, scope)
    assert revised["c1_example"]["holds"]
    assert revised["c2_policy_probe"]["holds"]
    assert revised["c3_evidence"]["holds"]
    assert revised["c3_evidence"]["extension_units_propagated"] >= 1
    assert revised["c3_evidence"]["association_only_neighbours_not_propagated"] >= 1

    forgotten = scenario.run_forget(memory, scope)
    assert forgotten["holds"]
    assert forgotten["no_rows_deleted"]
    assert forgotten["archived_explicit_lookup"]["state"] == "archived"
