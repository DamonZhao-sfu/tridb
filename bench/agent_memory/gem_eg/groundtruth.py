"""Decision points and graded relevance, derived from what the search actually did.

The Experience Graph paper's reuse claim needs an answer to "should this historical
experience have been retrieved here?". EvoTrace can answer it without human labels and
without an LLM, because it records the reward every attempt actually earned. That makes
the labels deterministic, auditable and reproducible — which is the whole reason this
is the primary quality evidence rather than the live agent study.

RELEVANCE (EvoTraceDoc.md §B.6.2)
---------------------------------
For a decision point in session `s` about to take step `i` from parent `p`, and a
candidate node `x` from some other session:

    3  x is valid and r(x) > best-so-far(s, i)   -- better than anything s has reached
    2  x is valid and r(p) < r(x) <= best-so-far -- improves on the current parent
    1  x is valid and r(x) <= r(p);  or x failed the way s later fails  -- marginal
    0  everything else valid                     -- irrelevant
   -1  x roots a dead end: no descendant ever beat it   -- HARMFUL to retrieve

Level -1 is why `harmful@k` exists: an experience store that confidently returns known
dead ends is worse than one that returns nothing.

WHAT THIS MODULE REFUSES TO DO
------------------------------
* Compare rewards across evaluator contracts. `combined_score` is not commensurable
  between tasks, so grades 3 and 2 are only assigned WITHIN a task. Cross-task queries
  get {1, 0, -1} only, and say so.
* Guess a reward. 81 of 10,672 nodes have no usable fitness (43 never had one, 38 were
  non-finite); they can never earn grade 3/2.
* Use wall-clock order. There is none. Everything is keyed on `iteration`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from bench.agent_memory.gem_eg.corpus import Corpus

GRADE_BEST = 3
GRADE_IMPROVES = 2
GRADE_MARGINAL = 1
GRADE_IRRELEVANT = 0
GRADE_HARMFUL = -1

QUERY_REUSE = "W1.a"
QUERY_STUCK = "W1.b"
QUERY_REPAIR = "W1.b-fail"
QUERY_PATH = "W1.d"


@dataclass(frozen=True)
class DecisionPoint:
    """One real moment in the trace where the agent chose what to do next."""

    dp_id: str
    query_id: str
    session_uid: str
    task_uid: str
    iteration: int
    #: The node the query is seeded from (the agent's current state).
    seed_uid: str
    parent_fitness: float | None
    best_so_far: float | None
    #: What actually happened next — never fed to the retriever, only used to label.
    outcome_uid: str
    outcome_fitness: float | None
    backend: str
    domain: str
    #: True when this decision point's session shares code with another session, so a
    #: cross-session hit may be trivially "correct". Reported, not silently dropped.
    leakage_suspect: bool = False
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


class GroundTruth:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._dead_end: dict[str, bool] = {}
        self.leaked_artifacts: set[str] = set()
        self._pool_cache: dict[tuple[str, str], list[float]] = {}
        self._shared_artifact_sessions = self._sessions_sharing_artifacts()

    # -- leakage ---------------------------------------------------------

    def _sessions_sharing_artifacts(self) -> set[str]:
        """Sessions holding DERIVED code byte-identical to code in another session.

        21 artifacts are referenced from more than one session, but flagging on that
        alone marks all 121 sessions and is therefore useless. The reason is benign:
        the biggest offenders span 17-20 sessions each because they are the *seed
        program* every session of a task starts from. Sharing a starting point is not
        leakage — everyone has it.

        Excluding root nodes leaves **7 artifacts across 18 sessions**: cases where one
        session independently produced code another session also produced. Those are
        the ones a cross-session result can be trivially "right" about.
        """
        roots = {uid for uid in self.corpus.nodes if uid not in self.corpus.parent}
        by_artifact: dict[str, set[str]] = defaultdict(set)
        for uid, node in self.corpus.nodes.items():
            if uid in roots or not node.get("artifact_uid"):
                continue
            by_artifact[node["artifact_uid"]].add(node["session_uid"])
        self.leaked_artifacts = {
            artifact for artifact, sessions in by_artifact.items() if len(sessions) > 1
        }
        flagged: set[str] = set()
        for artifact in self.leaked_artifacts:
            flagged |= by_artifact[artifact]
        return flagged

    def is_trivial_hit(self, candidate_uid: str, session_uid: str) -> bool:
        """Does this candidate's code already exist, byte-identical, in the target session?

        The exact per-result check, used at scoring time. Retrieving code the target
        session independently wrote is not evidence that retrieval generalises, so such
        hits are counted and reported apart from the headline.
        """
        artifact = self.corpus.nodes[candidate_uid].get("artifact_uid")
        if not artifact or artifact not in self.leaked_artifacts:
            return False
        return any(
            self.corpus.nodes[uid].get("artifact_uid") == artifact
            for uid in self.corpus.session_nodes.get(session_uid, [])
        )

    # -- dead ends -------------------------------------------------------

    def is_dead_end(self, uid: str) -> bool:
        """No descendant ever beat this node's own reward.

        Memoised over the lineage forest, so the whole corpus costs one pass rather
        than one traversal per candidate per decision point.
        """
        cached = self._dead_end.get(uid)
        if cached is not None:
            return cached
        node = self.corpus.nodes[uid]
        base = node.get("fitness")
        stack = list(self.corpus.children.get(uid, []))
        if not stack:
            # A leaf is a dead end only if it is not itself a good outcome; a valid
            # leaf that scored well is a fine thing to retrieve.
            result = not node["is_valid"] or base is None
            self._dead_end[uid] = result
            return result
        result = True
        seen: set[str] = set()
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            cnode = self.corpus.nodes[child]
            cfit = cnode.get("fitness")
            if cnode["is_valid"] and cfit is not None and (base is None or cfit > base):
                result = False
                break
            stack.extend(self.corpus.children.get(child, []))
        self._dead_end[uid] = result
        return result

    # -- grading ---------------------------------------------------------

    def grade(self, dp: DecisionPoint, candidate_uid: str, *, same_task: bool) -> int:
        node = self.corpus.nodes[candidate_uid]
        if self.is_dead_end(candidate_uid):
            return GRADE_HARMFUL
        if not node["is_valid"]:
            # A failure is marginally useful only when it failed the way this decision
            # point is about to fail — that is avoid-this evidence, not noise.
            outcome = self.corpus.nodes[dp.outcome_uid]
            if (
                node.get("error_signature")
                and node["failure_class"] == "execution"
                and node.get("error_signature") == outcome.get("error_signature")
            ):
                return GRADE_MARGINAL
            return GRADE_IRRELEVANT

        fitness = node.get("fitness")
        if fitness is None:
            return GRADE_IRRELEVANT
        if not same_task:
            # Rewards from a different evaluator contract are not comparable, so the
            # top grades are simply unavailable. Saying "1" here is honest; saying "3"
            # would be ranking two different units against each other.
            return GRADE_MARGINAL
        if dp.best_so_far is not None and fitness > dp.best_so_far:
            return GRADE_BEST
        if dp.parent_fitness is not None and fitness > dp.parent_fitness:
            return GRADE_IMPROVES
        return GRADE_MARGINAL

    def grades(self, dp: DecisionPoint, candidates: list[str]) -> dict[str, int]:
        return {
            uid: self.grade(dp, uid, same_task=self.corpus.nodes[uid]["task_uid"] == dp.task_uid)
            for uid in candidates
        }

    # -- arm-independent ideal --------------------------------------------

    def _task_pool(self, task_uid: str, exclude_session: str) -> list[float]:
        """Sorted rewards of every retrievable non-dead-end node of a task.

        Cached per (task, excluded session) — 121 such pairs exist, so the whole corpus
        costs one pass instead of one scan per decision point.
        """
        key = (task_uid, exclude_session)
        cached = self._pool_cache.get(key)
        if cached is not None:
            return cached
        rewards = sorted(
            node["fitness"]
            for session in self.corpus.task_sessions.get(task_uid, [])
            if session != exclude_session
            for node in (self.corpus.nodes[u] for u in self.corpus.session_nodes.get(session, []))
            if node["is_valid"]
            and node["fitness"] is not None
            and not self.is_dead_end(node["node_uid"])
        )
        self._pool_cache[key] = rewards
        return rewards

    def ideal_grades(self, dp: DecisionPoint, k: int) -> list[int]:
        """The best k grades ANY retrieval could have returned at this decision point.

        Arm-independent, and that is the whole point. Deriving the ideal from what an
        arm actually reached rewards an arm for reaching less: a small or poor candidate
        pool yields a low ideal, and a pool with nothing good gets the decision point
        dropped from scoring entirely. Measured on the first full run, that dropped
        36.6% of W1.a's points for `no_graph` and 14.0% of W1.b's for `fused`, so the
        arms were being averaged over different denominators and their nDCG could not
        be compared. The ideal must describe the TASK, never the retriever.
        """
        import bisect

        rewards = self._task_pool(dp.task_uid, dp.session_uid)
        n_best = 0
        if dp.best_so_far is not None:
            n_best = len(rewards) - bisect.bisect_right(rewards, dp.best_so_far)
        n_improve = 0
        if dp.parent_fitness is not None:
            upper = (
                bisect.bisect_right(rewards, dp.best_so_far)
                if dp.best_so_far is not None
                else len(rewards)
            )
            n_improve = max(0, upper - bisect.bisect_right(rewards, dp.parent_fitness))
        n_marginal = max(0, len(rewards) - n_best - n_improve)

        grades: list[int] = []
        for grade, count in ((GRADE_BEST, n_best), (GRADE_IMPROVES, n_improve), (GRADE_MARGINAL, n_marginal)):
            take = min(count, k - len(grades))
            if take > 0:
                grades.extend([grade] * take)
            if len(grades) >= k:
                break
        return grades

    # -- decision-point enumeration --------------------------------------

    def reuse_points(self) -> Iterator[DecisionPoint]:
        """W1.a — every real parent->child step. 10,479 of them."""
        for child_uid, parent_uid in self.corpus.parent.items():
            child = self.corpus.nodes[child_uid]
            parent = self.corpus.nodes[parent_uid]
            iteration = child.get("iteration") or 0
            yield DecisionPoint(
                dp_id=f"{QUERY_REUSE}:{child_uid}",
                query_id=QUERY_REUSE,
                session_uid=child["session_uid"],
                task_uid=child["task_uid"],
                iteration=iteration,
                seed_uid=child["task_uid"],  # the Task vector: the paper's ANN entry
                parent_fitness=parent.get("fitness"),
                best_so_far=self.corpus.best_at(child["session_uid"], iteration),
                outcome_uid=child_uid,
                outcome_fitness=child.get("fitness"),
                backend=self._backend(child),
                domain=self.corpus.tasks[child["task_uid"]]["domain"],
                leakage_suspect=child["session_uid"] in self._shared_artifact_sessions,
                meta={"parent_uid": parent_uid},
            )

    def stuck_points(self) -> Iterator[DecisionPoint]:
        """W1.b — the agent produced a child that did NOT beat its parent.

        6,786 of 10,400 comparable steps, and present in all four backends at 52-84%.
        This replaced the originally-planned failure-repair query, whose population
        turned out to be exactly zero: every rejected node in this corpus is a leaf.
        """
        for child_uid, parent_uid in self.corpus.parent.items():
            child = self.corpus.nodes[child_uid]
            parent = self.corpus.nodes[parent_uid]
            cf, pf = child.get("fitness"), parent.get("fitness")
            if cf is None or pf is None or cf > pf:
                continue
            iteration = child.get("iteration") or 0
            yield DecisionPoint(
                dp_id=f"{QUERY_STUCK}:{child_uid}",
                query_id=QUERY_STUCK,
                session_uid=child["session_uid"],
                task_uid=child["task_uid"],
                iteration=iteration,
                seed_uid=child_uid,  # the stuck code itself
                parent_fitness=pf,
                best_so_far=self.corpus.best_at(child["session_uid"], iteration),
                outcome_uid=child_uid,
                outcome_fitness=cf,
                backend=self._backend(child),
                domain=self.corpus.tasks[child["task_uid"]]["domain"],
                leakage_suspect=child["session_uid"] in self._shared_artifact_sessions,
                meta={"parent_uid": parent_uid, "delta": cf - pf},
            )

    def repair_points(self) -> Iterator[DecisionPoint]:
        """W1.b-fail — an EXECUTION failure that was demonstrably repaired.

        Only nodes whose failure came from the executor (timeout / compile / crash),
        never from the search discarding a clean program. 1,056 execution failures
        exist across 3 of 4 backends and 46 of 121 sessions; those with a clean sibling
        or a clean child carry the repair ground truth.
        """
        for uid, node in self.corpus.nodes.items():
            if node["failure_class"] != "execution":
                continue
            repairs = [
                c for c in self.corpus.siblings(uid) if self.corpus.nodes[c]["failure_class"] == "clean"
            ] + [
                c
                for c in self.corpus.children.get(uid, [])
                if self.corpus.nodes[c]["failure_class"] == "clean"
            ]
            if not repairs:
                continue
            iteration = node.get("iteration") or 0
            parent_uid = self.corpus.parent.get(uid)
            yield DecisionPoint(
                dp_id=f"{QUERY_REPAIR}:{uid}",
                query_id=QUERY_REPAIR,
                session_uid=node["session_uid"],
                task_uid=node["task_uid"],
                iteration=iteration,
                seed_uid=uid,
                parent_fitness=(
                    self.corpus.nodes[parent_uid].get("fitness") if parent_uid else None
                ),
                best_so_far=self.corpus.best_at(node["session_uid"], iteration),
                outcome_uid=sorted(repairs)[0],
                outcome_fitness=self.corpus.nodes[sorted(repairs)[0]].get("fitness"),
                backend=self._backend(node),
                domain=self.corpus.tasks[node["task_uid"]]["domain"],
                leakage_suspect=node["session_uid"] in self._shared_artifact_sessions,
                meta={
                    "error_signature": node.get("error_signature"),
                    "repairs": sorted(repairs),
                },
            )

    def path_points(self) -> Iterator[DecisionPoint]:
        """W1.d — every breakthrough: a node that set a new best-so-far. 619 of them.

        Ground truth is the ancestor chain that produced it, which the dataset also
        ships precomputed under `analysis/best_so_far_lineages/` for 95 of 121 runs.
        """
        for session_uid in sorted(self.corpus.sessions):
            for iteration, uid, fitness in self.corpus.best_so_far(session_uid):
                node = self.corpus.nodes[uid]
                yield DecisionPoint(
                    dp_id=f"{QUERY_PATH}:{uid}",
                    query_id=QUERY_PATH,
                    session_uid=session_uid,
                    task_uid=node["task_uid"],
                    iteration=iteration,
                    seed_uid=uid,
                    parent_fitness=None,
                    best_so_far=fitness,
                    outcome_uid=uid,
                    outcome_fitness=fitness,
                    backend=self._backend(node),
                    domain=self.corpus.tasks[node["task_uid"]]["domain"],
                    leakage_suspect=session_uid in self._shared_artifact_sessions,
                    meta={"ancestors": self.corpus.ancestors(uid)},
                )

    def all_points(self) -> list[DecisionPoint]:
        return [
            *self.reuse_points(),
            *self.stuck_points(),
            *self.repair_points(),
            *self.path_points(),
        ]

    @staticmethod
    def _backend(node: dict[str, Any]) -> str:
        return node["session_uid"].split(":", 1)[-1].split("/", 1)[0]


def write(points: list[DecisionPoint], path: Path) -> dict[str, Any]:
    from collections import Counter

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for dp in points:
            handle.write(
                json.dumps(
                    {
                        "dp_id": dp.dp_id,
                        "query_id": dp.query_id,
                        "session_uid": dp.session_uid,
                        "task_uid": dp.task_uid,
                        "iteration": dp.iteration,
                        "seed_uid": dp.seed_uid,
                        "parent_fitness": dp.parent_fitness,
                        "best_so_far": dp.best_so_far,
                        "outcome_uid": dp.outcome_uid,
                        "outcome_fitness": dp.outcome_fitness,
                        "backend": dp.backend,
                        "domain": dp.domain,
                        "leakage_suspect": dp.leakage_suspect,
                        "meta": dp.meta,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    per_query = Counter(dp.query_id for dp in points)
    return {
        "total": len(points),
        "per_query": dict(per_query),
        "per_backend": dict(Counter(dp.backend for dp in points)),
        "leakage_suspect": sum(1 for dp in points if dp.leakage_suspect),
        "path": str(path),
    }
