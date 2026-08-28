"""Leakage-safe structural graphs for CrossEp-Know.

Semantic experience edges are deliberately absent: EvoMemBench does not release them.
Runtime extractors may add versioned SUPPORTS, CONTRADICTS, or DERIVED_FROM edges, but
those edges must not be treated as benchmark ground truth.

The online graph and evaluator graph are intentionally different objects.  Rubrics
are withheld labels and must never enter GEM's queryable state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from bench.agent_memory.evomembench.dataset import EvoContext


@dataclass(frozen=True)
class ExperienceNode:
    uid: str
    kind: str
    text: str


@dataclass(frozen=True)
class ExperienceEdge:
    source_uid: str
    target_uid: str
    kind: str


@dataclass(frozen=True)
class StructuralExperienceGraph:
    nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]
    visibility: str = "online"


def _rubric_uid(rubric: str) -> str:
    digest = hashlib.sha256(rubric.encode()).hexdigest()
    return f"evomembench:rubric:{digest}"


def build_structural_graph(context: EvoContext) -> StructuralExperienceGraph:
    """Build only the chronology available to an online agent."""
    nodes = [ExperienceNode(context.context_uid, "context", context.category)]
    edges: list[ExperienceEdge] = []
    previous_uid: str | None = None
    for episode in context.episodes:
        nodes.append(
            ExperienceNode(episode.episode_uid, "episode", episode.retrieval_query)
        )
        edges.append(
            ExperienceEdge(episode.episode_uid, context.context_uid, "BELONGS_TO")
        )
        if previous_uid is not None:
            edges.append(ExperienceEdge(previous_uid, episode.episode_uid, "PRECEDES"))
        previous_uid = episode.episode_uid
    return StructuralExperienceGraph(tuple(nodes), tuple(edges), "online")


def build_evaluation_audit_graph(context: EvoContext) -> StructuralExperienceGraph:
    """Build the withheld rubric graph for scoring, never for GEM ingestion."""
    nodes: list[ExperienceNode] = []
    edges: list[ExperienceEdge] = []
    seen_rubrics: set[str] = set()
    for episode in context.episodes:
        nodes.append(
            ExperienceNode(episode.episode_uid, "episode_ref", episode.source_task_id)
        )
        for rubric in episode.rubrics:
            rubric_uid = _rubric_uid(rubric)
            if rubric_uid not in seen_rubrics:
                seen_rubrics.add(rubric_uid)
                nodes.append(ExperienceNode(rubric_uid, "rubric", rubric))
            edges.append(
                ExperienceEdge(episode.episode_uid, rubric_uid, "EVALUATED_BY")
            )
    return StructuralExperienceGraph(tuple(nodes), tuple(edges), "evaluation_only")
