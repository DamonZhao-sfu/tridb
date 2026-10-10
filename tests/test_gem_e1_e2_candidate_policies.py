from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import DecisionPoint
from bench.agent_memory.gem_eg.oracle import (
    Oracle,
    Predicate,
    QuerySpec,
    candidate_policy_hash,
)
from experiments.gem_eg_e1_e2.protocol import balanced_smoke_sample


def _node(uid: str, *, valid: bool = True) -> dict:
    return {
        "_kind": "node",
        "node_uid": uid,
        "task_uid": "task",
        "session_uid": "history",
        "is_valid": valid,
        "fitness": 1.0 if valid else None,
        "failure_class": "clean" if valid else "execution",
    }


def _fixture() -> tuple[Corpus, QuerySpec]:
    corpus = Corpus()
    corpus.tasks["task"] = {"task_uid": "task", "domain": "math"}
    corpus.nodes = {
        "seed": _node("seed", valid=False),
        "bridge": _node("bridge"),
        "near-a": _node("near-a"),
        "near-b": _node("near-b"),
        "global-a": _node("global-a"),
        "global-b": _node("global-b"),
    }
    corpus.children = defaultdict(list)
    # BFS order is the injection contract; the low-similarity bridge is deliberately
    # first while cosine reranking will place it last.
    corpus.children["seed"] = ["bridge", "near-a", "near-b"]
    vectors = np.asarray(
        [
            [1.0, 0.0],  # seed / query
            [0.0, 1.0],  # bridge: vector-far
            [0.80, 0.20],
            [0.75, 0.25],
            [0.99, 0.01],
            [0.98, 0.02],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    uids = ["seed", "bridge", "near-a", "near-b", "global-a", "global-b"]
    corpus.vectors = vectors
    corpus.vector_row = {uid: i for i, uid in enumerate(uids)}
    spec = QuerySpec(
        query_id="W1.b",
        decision_point="dp",
        seed_uid="seed",
        relation="lineage",
        hops=1,
        predicate=Predicate(kind="node", require_valid=True, require_fitness=True),
        k=2,
        ann_entry_kind="node",
        ann_m_seeds=1,
    )
    return corpus, spec


def test_three_candidate_policies_are_semantically_distinct() -> None:
    corpus, spec = _fixture()
    oracle = Oracle(corpus)

    vector = oracle.run_candidate_policy(spec, "vector_only")
    rerank = oracle.run_candidate_policy(spec, "graph_rerank")
    inject = oracle.run_candidate_policy(spec, "graph_inject")

    assert vector.topk == ("global-a", "global-b")
    assert rerank.topk == ("near-a", "near-b")
    assert inject.topk == ("bridge", "near-a")
    assert inject.injected == inject.topk
    assert inject.distances[0] > inject.distances[1]
    assert not rerank.injected


def test_multi_source_bfs_deduplicates_in_seed_and_frontier_order() -> None:
    corpus, _ = _fixture()
    corpus.children["seed-2"] = ["bridge", "z"]
    assert corpus.traverse_many(
        ["seed", "seed-2", "seed"],
        relation="lineage",
        hops=1,
        include_seeds=True,
    ) == ["seed", "seed-2", "bridge", "near-a", "near-b", "z"]


def test_candidate_policy_is_part_of_the_arm_hash() -> None:
    _, spec = _fixture()
    hashes = {
        candidate_policy_hash(spec, policy)
        for policy in ("vector_only", "graph_rerank", "graph_inject")
    }
    assert len(hashes) == 3


def test_smoke_sample_has_exactly_fifteen_queries_per_shape() -> None:
    points = []
    shapes = ("W1.a", "W1.b", "W1.b-fail", "W1.d")
    for shape in shapes:
        for index in range(40):
            points.append(
                DecisionPoint(
                    dp_id=f"{shape}:{index}",
                    query_id=shape,
                    session_uid=f"session:{index}",
                    task_uid=f"task:{index % 4}",
                    iteration=index,
                    seed_uid=f"node:{index}",
                    parent_fitness=0.0,
                    best_so_far=0.0,
                    outcome_uid=f"outcome:{index}",
                    outcome_fitness=1.0,
                    backend=f"backend:{index % 4}",
                    domain=f"domain:{index % 2}",
                )
            )
    sample = balanced_smoke_sample(points)
    assert len(sample) == 60
    assert {
        shape: sum(point.query_id == shape for point in sample) for shape in shapes
    } == {shape: 15 for shape in shapes}
