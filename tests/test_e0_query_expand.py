"""Tests for the E0 query-set expansion — synthetic graph, no STARK download, no network.

The contract under test is the one the derived annotations stake everything on: an emitted
query's official answers must ALL lie inside the emitted typed-hop envelope, and the anchor
must be an entity the query text actually names. A derivation that silently relaxes either
turns E0's plan-spread numbers into measurements of a made-up workload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.e0.query_expand import (  # noqa: E402
    AUTO_STATUS,
    Graph,
    _min_type_cover,
    _template,
    derive_one,
)


def build_graph(arcs: list[tuple[int, int, str]]) -> Graph:
    """CSR from (src, dst, edge_type). Arcs are given in BOTH directions by the caller,
    mirroring STARK's storage (verified: every arc has its same-type reverse)."""
    types = sorted({t for _, _, t in arcs})
    lookup = {t: i for i, t in enumerate(types)}
    src = np.array([a[0] for a in arcs], dtype=np.int64)
    dst = np.array([a[1] for a in arcs], dtype=np.int64)
    et = np.array([lookup[a[2]] for a in arcs], dtype=np.int16)
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    n = int(max(src.max(), dst.max())) + 1
    indptr = np.searchsorted(src_s, np.arange(n + 1), side="left")
    return Graph(indptr, dst[order], et[order], types)


def both(a: int, b: int, t: str) -> list[tuple[int, int, str]]:
    return [(a, b, t), (b, a, t)]


# 0=Drug "Travoprostine", 1=Gene "KinaseAlpha", 2=Component "MitoMembrane",
# 3=unrelated Gene "NoiseGene", 4=second component reachable only via a 2nd type
ARCS = (
    both(0, 1, "target")
    + both(1, 2, "interacts with")
    + both(0, 3, "enzyme")
    + both(1, 4, "expression present")
)
NAMES = {
    0: "Travoprostine",
    1: "KinaseAlpha",
    2: "MitoMembrane",
    3: "NoiseGene",
    4: "OtherComponent",
}
TYPES = {
    0: "drug",
    1: "gene/protein",
    2: "cellular_component",
    3: "gene/protein",
    4: "cellular_component",
}
BY_NAME = {v.casefold(): [k] for k, v in NAMES.items()}


@pytest.fixture
def graph() -> Graph:
    return build_graph(ARCS)


def test_typed_reachable_matches_pilot_semantics(graph):
    # Edge-type set applies at EVERY hop; the anchor is never a result; already-seen nodes
    # are not re-collected. This mirrors stark_prime_prepare._typed_reachable exactly.
    tid = graph.type_id
    assert graph.typed_reachable(0, [tid["target"]], 1) == {1}
    assert graph.typed_reachable(0, [tid["target"]], 2) == {1}  # nothing new at hop 2
    two = graph.typed_reachable(0, [tid["target"], tid["interacts with"]], 2)
    assert two == {1, 2}
    assert 0 not in two


def test_derive_finds_two_hop_chain(graph):
    query = {
        "source_query_id": 1,
        "query_text": "Which cell structures interact with genes influenced by Travoprostine?",
        "answer_ids": [2],
    }
    got = derive_one(query, graph, TYPES, NAMES, BY_NAME, max_anchor_candidates=8)
    assert got is not None
    assert got["anchor_names"] == ["Travoprostine"]
    assert got["hop_limit"] == 2
    assert set(got["edge_types"]) == {"target", "interacts with"}
    assert got["target_entity_type"] == "cellular_component"
    assert got["template"] == "typed_chain"
    # The stated envelope really does cover the answers — the whole point of the exercise.
    ids = [graph.type_id[t] for t in got["edge_types"]]
    assert set(got["answer_ids"]) <= graph.typed_reachable(
        got["anchor_ids"][0], ids, got["hop_limit"]
    )


def test_anchor_must_be_named_in_the_query_text(graph):
    # Graph-wise, Travoprostine is a perfectly good anchor for answer 2. But if the text
    # never names it, accepting it would invent a workload the query does not describe.
    query = {
        "source_query_id": 2,
        "query_text": "Which cell structures are involved in this interaction?",
        "answer_ids": [2],
    }
    assert (
        derive_one(query, graph, TYPES, NAMES, BY_NAME, max_anchor_candidates=8) is None
    )


def test_mixed_answer_types_are_refused(graph):
    # The pilot resolver requires one answer type; derived rows must mean the same thing.
    query = {
        "source_query_id": 3,
        "query_text": "Travoprostine effects?",
        "answer_ids": [1, 2],  # gene/protein + cellular_component
    }
    assert (
        derive_one(query, graph, TYPES, NAMES, BY_NAME, max_anchor_candidates=8) is None
    )


def test_unreachable_answers_yield_nothing(graph):
    isolated = build_graph(ARCS + both(9, 10, "ppi"))
    names = {**NAMES, 9: "Lonely", 10: "Unreachable"}
    types = {**TYPES, 9: "drug", 10: "cellular_component"}
    by_name = {v.casefold(): [k] for k, v in names.items()}
    query = {
        "source_query_id": 4,
        "query_text": "What does Travoprostine reach?",
        "answer_ids": [10],  # in another component entirely
    }
    assert (
        derive_one(query, isolated, types, names, by_name, max_anchor_candidates=8)
        is None
    )


def test_min_type_cover_prefers_the_smaller_set(graph):
    tid = graph.type_id
    budget = set(graph.incident_types(0))
    cover = _min_type_cover(graph, 0, [1], 1, budget)
    assert cover == [tid["target"]]  # size 1 wins over any pair


def test_min_type_cover_caps_at_two_types(graph):
    # Answer 4 needs target + expression present at 2 hops: exactly two, so it is found.
    tid = graph.type_id
    budget = set(range(len(graph.type_names)))
    cover = _min_type_cover(graph, 0, [4], 2, budget)
    assert cover is not None and len(cover) == 2
    assert set(cover) == {tid["target"], tid["expression present"]}


def test_template_naming():
    assert _template(1, 1, False) == "typed_neighbors"
    assert _template(1, 2, False) == "typed_chain"
    assert _template(2, 1, True) == "neighbor_intersection"
    assert _template(2, 2, True) == "shared_neighbor_constraint"


def test_emitted_rows_are_flagged_auto_derived(tmp_path, graph):
    # Derived rows must never claim the audited status; downstream reports stratify on it.
    query = {
        "source_query_id": 5,
        "query_text": "Genes targeted by Travoprostine?",
        "answer_ids": [1],
    }
    got = derive_one(query, graph, TYPES, NAMES, BY_NAME, max_anchor_candidates=8)
    assert got is not None
    row = {**got, "annotation_status": AUTO_STATUS}
    assert row["annotation_status"] != "manually_audited_v0.1"
    assert json.loads(json.dumps(row))["annotation_status"] == AUTO_STATUS
