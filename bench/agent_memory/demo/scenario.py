"""The five-act GEM Wikipedia scenario. Reads pinned files; never the network.

The fetcher and the scenario are deliberately separate. This module accepts a
``WikiSlice`` and the resolved HotpotQA question file, drives the four GEM
operators against one live PostgreSQL process, and returns plain JSON-ready
evidence for :mod:`bench.agent_memory.demo.report`.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.demo import adapter
from bench.agent_memory.demo.strategies import (
    WikiSliceIngestStrategy,
    WikidataRevisionStrategy,
    revision_events,
)
from bench.agent_memory.gem import conformance
from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.store import TriDBMemoryView
from bench.agent_memory.gem.types import (
    InteractionEvent,
    Policy,
    PolicyEvent,
    Query,
    RetrievalMode,
)

DEFAULT_DSN = "postgresql://hza214@127.0.0.1:55432/gem_demo"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SLICE = Path("data/wiki_demo")
DEFAULT_OUTPUT = Path("bench/out/gem_wiki_demo")
INITIAL_VALID_FROM = "2026-07-30T00:00:00Z"


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so psycopg timestamps and enums become strings."""
    return json.loads(json.dumps(value, default=str))


def _transition_payload(result: Any) -> dict[str, Any]:
    return _jsonable(asdict(result))


def _require_committed(result: Any, act: str) -> None:
    if not result.committed:
        raise RuntimeError(f"{act} aborted: {result.aborted_reason}")


def load_resolved_questions(
    path: Path | str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("coverage") or {}), list(payload.get("questions") or [])


def reset_scope(memory: TriDBGovernedMemory, scope_id: str) -> dict[str, int]:
    """Drop one demo-owned scope while preserving every other scope.

    The native graph has no vertex delete. Active native edges are tombstoned
    before the relational rows cascade away; vertices remain allocated and are
    harmless because a fresh ingest receives fresh dense vids.
    """
    conn = memory.store.conn
    edges = conn.execute(
        "SELECT DISTINCT e.src, e.dst, e.edge_type FROM gem_edge e"
        " JOIN gem_unit u ON u.id = e.src OR u.id = e.dst"
        " WHERE u.scope_id = %s AND e.tombstoned_at IS NULL",
        (scope_id,),
    ).fetchall()
    with conn.transaction():
        for src, dst, edge_type in edges:
            conn.execute(
                "SELECT graph_store.gph_tombstone_edge(%s, %s, %s)",
                (int(src), int(dst), int(edge_type)),
            )
        deleted = conn.execute(
            "DELETE FROM gem_unit WHERE scope_id = %s", (scope_id,)
        ).rowcount
        transitions = conn.execute(
            "DELETE FROM gem_transition WHERE scope_id = %s", (scope_id,)
        ).rowcount
        policies = conn.execute(
            "DELETE FROM gem_policy WHERE scope_id = %s", (scope_id,)
        ).rowcount
    return {
        "units": int(deleted),
        "native_edges_tombstoned": len(edges),
        "transitions": int(transitions),
        "policies": int(policies),
    }


def _scope_fingerprint(memory: TriDBGovernedMemory, scope_id: str) -> str:
    """Stable digest of D_t + S_t for the C2 rollback probe."""
    conn = memory.store.conn
    rows = {
        "units": conn.execute(
            "SELECT id, title, summary, state, salience, access_count, last_access,"
            " metadata FROM gem_unit WHERE scope_id = %s ORDER BY id",
            (scope_id,),
        ).fetchall(),
        "fields": conn.execute(
            "SELECT fv.id, fv.unit_id, fv.field, fv.value, fv.valid_from,"
            " fv.valid_to, fv.superseded_by, fv.source_external_ids,"
            " fv.transition_id, fv.salience, fv.state"
            " FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
            " WHERE u.scope_id = %s ORDER BY fv.id",
            (scope_id,),
        ).fetchall(),
        "edges": conn.execute(
            "SELECT e.src, e.dst, e.edge_type, e.kind, e.rel, e.weight,"
            " e.co_access_count, e.tombstoned_at FROM gem_edge e"
            " JOIN gem_unit u ON u.id = e.src"
            " WHERE u.scope_id = %s ORDER BY e.src, e.dst, e.edge_type",
            (scope_id,),
        ).fetchall(),
    }
    body = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _cost_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "seconds",
        "llm_calls",
        "embed_calls",
        "embed_sequences",
        "prompt_tokens",
        "completion_tokens",
        "embed_input_tokens",
        "db_statements",
    )
    return {
        key: sum(float(row.get(key, 0) or 0) for row in rows)
        if key == "seconds"
        else sum(int(row.get(key, 0) or 0) for row in rows)
        for key in keys
    }


def run_ingest(
    memory: TriDBGovernedMemory, wiki: adapter.WikiSlice, scope_id: str
) -> dict[str, Any]:
    events = adapter.slice_events(wiki, scope_id=scope_id)
    strategy = WikiSliceIngestStrategy(valid_from=INITIAL_VALID_FROM)
    result = memory.ingest(events, strategy=strategy)
    _require_committed(result, "act 1 ingest")
    return {
        "strategy": strategy.name,
        "strategy_variant": strategy.variant,
        "events": len(events),
        "source_counts": dict(wiki.manifest.get("counts") or {}),
        "transition": _transition_payload(result),
    }


def _retrieval_metrics(
    question_rows: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    joint = 0
    evidence_sum = 0.0
    returned_sum = 0
    short = 0
    strict_c6 = 0
    c6_hits = 0
    failures = 0
    reasons: Counter[str] = Counter()
    capped = 0
    censored = 0

    for question, observation in zip(question_rows, observations, strict=True):
        if not observation.get("committed"):
            failures += 1
            continue
        titles = set(observation.get("returned_titles") or ())
        gold = set(question.get("gold_titles") or ())
        matched = len(gold & titles)
        joint += int(bool(gold) and matched == len(gold))
        evidence_sum += matched / len(gold) if gold else 0.0
        returned = int(observation.get("returned_units", 0))
        returned_sum += returned
        short += int(returned < k)
        probe = observation.get("probes") or {}
        reasons[str(probe.get("termination_reason"))] += 1
        capped += int(bool(probe.get("budget_capped")))
        censored += int(bool(probe.get("graph_censored")))
        c6_hits += int(observation.get("salience_pairs", 0))
        strict_c6 += int(observation.get("strict_salience_increases", 0))

    n = len(question_rows)
    return {
        "questions": n,
        "k": k,
        "joint_evidence_recall_at_k": joint / n if n else 0.0,
        "mean_evidence_recall_at_k": evidence_sum / n if n else 0.0,
        "mean_returned_units": returned_sum / n if n else 0.0,
        "short_result_queries": short,
        "failed_queries": failures,
        "termination_reasons": dict(sorted(reasons.items())),
        "budget_capped_queries": capped,
        "graph_censored_queries": censored,
        "strict_salience_increases": strict_c6,
        "salience_pairs": c6_hits,
    }


def run_retrieve(
    memory: TriDBGovernedMemory,
    question_rows: Sequence[Mapping[str, Any]],
    scope_id: str,
    *,
    k: int,
    term_cond: int,
) -> dict[str, Any]:
    operating_points: dict[str, Any] = {}
    for mode in (RetrievalMode.VECTOR, RetrievalMode.FUSED):
        observations: list[dict[str, Any]] = []
        costs: list[dict[str, Any]] = []
        for row in question_rows:
            result = memory.retrieve(
                Query(
                    scope_id=scope_id,
                    text=str(row["question"]),
                    k=k,
                    mode=mode,
                    reinforce=True,
                    term_cond=term_cond,
                )
            )
            unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
            titles = list(dict.fromkeys(hit.title for hit in result.hits))
            pairs = [
                (hit.salience_before, hit.salience_after)
                for hit in result.hits
                if hit.salience_before is not None and hit.salience_after is not None
            ]
            observations.append(
                {
                    "question_id": row["id"],
                    "committed": result.committed,
                    "aborted_reason": result.aborted_reason,
                    "returned_units": len(unit_ids),
                    "returned_unit_ids": unit_ids,
                    "returned_titles": titles,
                    # Engine probes are copied without renaming or aggregation.
                    "probes": _jsonable(dict(result.probes)),
                    "strict_salience_increases": sum(a > b for b, a in pairs),
                    "salience_pairs": len(pairs),
                }
            )
            costs.append(_jsonable(asdict(result.cost)))

        order = "strict_order" if mode is RetrievalMode.VECTOR else "relaxed_order"
        metrics = _retrieval_metrics(question_rows, observations, k=k)
        operating_points[mode.value] = {
            "mode": mode.value,
            "hnsw_iterative_scan": order,
            "reinforce": True,
            "metrics": metrics,
            "cost": _cost_totals(costs),
            "queries": observations,
        }
    return {"operating_points": operating_points}


def _c1_example(memory: TriDBGovernedMemory, scope_id: str) -> dict[str, Any]:
    conn = memory.store.conn
    candidate = conn.execute(
        "SELECT u.id, COALESCE(u.metadata->>'qid', parent.metadata->>'qid'),"
        " fv.field, count(*) AS writes"
        " FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
        " LEFT JOIN gem_unit parent"
        " ON parent.id = CASE WHEN u.metadata->>'split_from' ~ '^[0-9]+$'"
        " THEN (u.metadata->>'split_from')::bigint ELSE NULL END"
        " WHERE u.scope_id = %s AND fv.source_external_ids IS NOT NULL"
        " GROUP BY u.id, COALESCE(u.metadata->>'qid', parent.metadata->>'qid'),"
        " fv.field HAVING count(*) > 1 AND count(DISTINCT fv.value) > 1"
        " ORDER BY writes DESC, u.id, fv.field LIMIT 1",
        (scope_id,),
    ).fetchone()
    if candidate is None:
        return {"observed": False}
    unit_id, qid, field_name, writes = candidate
    history = conn.execute(
        "SELECT value, valid_from, valid_to, superseded_by"
        " FROM gem_field_value WHERE unit_id = %s AND field = %s"
        " ORDER BY valid_from, id",
        (int(unit_id), field_name),
    ).fetchall()
    newest = history[-1]
    old = next(row for row in reversed(history[:-1]) if row[0] != newest[0])
    current = conn.execute(
        "SELECT value FROM gem_field_value WHERE unit_id = %s AND field = %s"
        " AND valid_to IS NULL",
        (int(unit_id), field_name),
    ).fetchall()
    as_of = conn.execute(
        "SELECT value FROM gem_field_value WHERE unit_id = %s AND field = %s"
        " AND valid_from <= %s AND (valid_to IS NULL OR valid_to > %s)",
        (int(unit_id), field_name, old[1], old[1]),
    ).fetchall()
    return {
        "observed": True,
        "unit_id": int(unit_id),
        "qid": qid,
        "field": field_name,
        "writes": int(writes),
        "old_value": old[0],
        "new_value": newest[0],
        "old_valid_to": str(old[2]),
        "old_superseded_by": old[3],
        "default_values": [row[0] for row in current],
        "as_of": str(old[1]),
        "as_of_values": [row[0] for row in as_of],
        "holds": (
            len(current) == 1
            and current[0][0] == newest[0]
            and [row[0] for row in as_of] == [old[0]]
        ),
    }


def _c2_policy_probe(
    memory: TriDBGovernedMemory, scope_id: str, unit_id: int
) -> dict[str, Any]:
    name = f"gem-wiki-c2-{scope_id}"
    policy = Policy(
        name=name,
        scope_id=scope_id,
        event=PolicyEvent.FIELD_UPDATED,
        condition={"name": "active_below_bound", "params": {"max_active_units": 0}},
        action={},
    )
    memory.put_policy(policy)
    before = _scope_fingerprint(memory, scope_id)

    class ViolatingPlan:
        name = "deterministic"
        variant = "c2_policy_probe"

        def plan(
            self, events: Sequence[InteractionEvent], view: Any
        ) -> list[dict[str, Any]]:
            return [
                planmod.append_field_value(
                    unit_id=unit_id,
                    field="c2_probe",
                    value="this write must roll back",
                    valid_from=INITIAL_VALID_FROM,
                    provenance={"operator": "ingest"},
                )
            ]

    event = InteractionEvent(
        scope_id=scope_id,
        external_id="c2-policy-probe",
        content="policy rejection probe",
    )
    result = memory.ingest([event], strategy=ViolatingPlan())
    after = _scope_fingerprint(memory, scope_id)
    # Leave the policy as trajectory provenance but disable it so later acts do
    # not inherit a deliberate impossible bound.
    memory.put_policy(
        Policy(
            name=name,
            scope_id=scope_id,
            event=PolicyEvent.FIELD_UPDATED,
            condition=policy.condition,
            action={},
            enabled=False,
        )
    )
    return {
        "committed": result.committed,
        "aborted_reason": result.aborted_reason,
        "fingerprint_before": before,
        "fingerprint_after": after,
        "state_byte_identical": before == after,
        "holds": not result.committed and before == after,
    }


def run_revise(
    memory: TriDBGovernedMemory, wiki: adapter.WikiSlice, scope_id: str
) -> dict[str, Any]:
    edits = adapter.parse_revisions(wiki.revisions)
    observed_supersessions = adapter.supersessions(edits)
    strategy = WikidataRevisionStrategy()
    replay = memory.ingest(
        revision_events(edits, scope_id=scope_id),
        strategy=strategy,
    )
    _require_committed(replay, "act 3 real revision replay")
    changed_ids = sorted(set(int(unit_id) for unit_id in replay.units))
    revision = memory.revise(
        scope_id,
        evidence=[{"unit_id": unit_id} for unit_id in changed_ids],
        max_hops=3,
    )
    _require_committed(revision, "act 3 revise")

    conn = memory.store.conn
    kinds = dict(
        conn.execute(
            "SELECT metadata->>'kind', count(*) FROM gem_unit"
            " WHERE scope_id = %s AND id = ANY(%s)"
            " GROUP BY metadata->>'kind'",
            (scope_id, changed_ids),
        ).fetchall()
    )
    propagated = set(int(unit_id) for unit_id in revision.delta.propagated_units)
    association_rows = conn.execute(
        "SELECT DISTINCT e.src, e.dst FROM gem_edge e"
        " WHERE e.src = ANY(%s) AND e.kind = 'association'"
        " AND e.tombstoned_at IS NULL ORDER BY e.src, e.dst",
        (changed_ids,),
    ).fetchall()
    association_only = [
        {"src": int(src), "dst": int(dst)}
        for src, dst in association_rows
        if int(dst) not in propagated
    ]

    c1 = _c1_example(memory, scope_id)
    c2 = _c2_policy_probe(memory, scope_id, changed_ids[0])
    return {
        "revision_rows": len(wiki.revisions),
        "parsed_edits": len(edits),
        "parse_fraction": len(edits) / len(wiki.revisions) if wiki.revisions else 0.0,
        "observed_supersessions": len(observed_supersessions),
        "entities_with_supersessions": len({row[0] for row in observed_supersessions}),
        "unresolved_entities": sorted(set(strategy.unresolved)),
        "changed_units": len(changed_ids),
        "changed_unit_kinds": {str(k): int(v) for k, v in kinds.items()},
        "replay_transition": _transition_payload(replay),
        "revise_transition": _transition_payload(revision),
        "c1_example": c1,
        "c2_policy_probe": c2,
        "c3_evidence": {
            "extension_units_propagated": len(propagated),
            "propagated_unit_ids": sorted(propagated),
            "association_edges_from_changed_units": len(association_rows),
            "association_only_neighbours_not_propagated": len(association_only),
            "association_only_examples": association_only[:20],
            "holds": bool(propagated) and bool(association_only),
        },
    }


def run_forget(memory: TriDBGovernedMemory, scope_id: str) -> dict[str, Any]:
    conn = memory.store.conn
    before_total = int(
        conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope_id,)
        ).fetchone()[0]
    )
    retrieved_ids = [
        int(row[0])
        for row in conn.execute(
            "SELECT id FROM gem_unit WHERE scope_id = %s AND access_count > 0"
            " ORDER BY id",
            (scope_id,),
        ).fetchall()
    ]
    never_ids = [
        int(row[0])
        for row in conn.execute(
            "SELECT id FROM gem_unit WHERE scope_id = %s AND access_count = 0"
            " ORDER BY id",
            (scope_id,),
        ).fetchall()
    ]
    result = memory.forget(scope_id)
    _require_committed(result, "act 4 forget")
    after_total = int(
        conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope_id,)
        ).fetchone()[0]
    )
    retrieved_surviving = int(
        conn.execute(
            "SELECT count(*) FROM gem_unit WHERE id = ANY(%s) AND state <> 'archived'",
            (retrieved_ids or [-1],),
        ).fetchone()[0]
    )
    never_archived = int(
        conn.execute(
            "SELECT count(*) FROM gem_unit WHERE id = ANY(%s) AND state = 'archived'",
            (never_ids or [-1],),
        ).fetchone()[0]
    )
    archived_row = conn.execute(
        "SELECT id FROM gem_unit WHERE scope_id = %s AND state = 'archived'"
        " ORDER BY id LIMIT 1",
        (scope_id,),
    ).fetchone()
    recovered = (
        None
        if archived_row is None
        else TriDBMemoryView(memory.store).unit(int(archived_row[0]))
    )
    states = {
        str(state): int(count)
        for state, count in conn.execute(
            "SELECT state, count(*) FROM gem_unit WHERE scope_id = %s"
            " GROUP BY state ORDER BY state",
            (scope_id,),
        ).fetchall()
    }
    return {
        "transition": _transition_payload(result),
        "units_before": before_total,
        "units_after": after_total,
        "retrieved_units_before_tick": len(retrieved_ids),
        "retrieved_units_not_archived": retrieved_surviving,
        "never_retrieved_units_before_tick": len(never_ids),
        "never_retrieved_units_archived": never_archived,
        "states_after": states,
        "archived_explicit_lookup": None
        if recovered is None
        else {
            "unit_id": recovered.id,
            "title": recovered.title,
            "state": recovered.state.value,
        },
        "no_rows_deleted": before_total == after_total,
        "holds": (
            before_total == after_total
            and retrieved_surviving == len(retrieved_ids)
            and never_archived == len(never_ids)
            and recovered is not None
        ),
    }


def _condition_payload(report: conformance.ConformanceReport) -> list[dict[str, Any]]:
    return [_jsonable(result.to_dict()) for result in report.results]


def environment_manifest(
    memory: TriDBGovernedMemory,
    *,
    source_manifest: Mapping[str, Any],
    scope_id: str,
    model_name: str,
) -> dict[str, Any]:
    conn = memory.store.conn
    extensions = {
        str(name): str(version)
        for name, version in conn.execute(
            "SELECT extname, extversion FROM pg_extension"
            " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY extname"
        ).fetchall()
    }
    points = {
        mode: memory.manifest(
            strategy_name="deterministic",
            mode=mode,
            route="topic",
            reinforce=True,
            revise_enabled=True,
            forget_enabled=True,
            embedding_model=model_name,
        )
        for mode in ("vector", "fused")
    }
    return {
        "scope_id": scope_id,
        "database": conn.execute("SELECT current_database()").fetchone()[0],
        "postgres_version": conn.execute("SHOW server_version").fetchone()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "extensions": extensions,
        "source_manifest": _jsonable(dict(source_manifest)),
        "retrieval_operating_points": points,
        "labels": [
            "stock PostgreSQL 16, x86_64; not the GX10 fork",
            "graph reads are commit-visible, not snapshot-isolated",
            "HotpotQA metrics use only the fully_resolved subset",
            "the candidate pool was built to contain the gold titles",
            "conformance is reported per condition; no wholesale label is claimed",
        ],
    }


def run(
    memory: TriDBGovernedMemory,
    *,
    slice_dir: Path,
    scope_id: str,
    phase: str = "all",
    reset: bool = False,
    question_limit: int | None = None,
    k: int = 10,
    term_cond: int = 32,
    model_name: str = DEFAULT_MODEL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one phase or all phases and return ``(results, manifest)``."""
    memory.init_schema()
    wiki = adapter.WikiSlice.load(slice_dir)
    coverage, questions = load_resolved_questions(slice_dir / "questions.json")
    if question_limit is not None:
        questions = questions[:question_limit]
    if reset:
        reset_evidence = reset_scope(memory, scope_id)
    else:
        reset_evidence = None

    results: dict[str, Any] = {
        "scope_id": scope_id,
        "phase": phase,
        "coverage": coverage,
        "questions_run": len(questions),
        "k": k,
        "term_cond": term_cond,
        "reset": reset_evidence,
        "acts": {},
    }

    if phase in ("all", "ingest"):
        results["acts"]["ingest"] = run_ingest(memory, wiki, scope_id)
    if phase in ("all", "retrieve"):
        results["acts"]["retrieve"] = run_retrieve(
            memory, questions, scope_id, k=k, term_cond=term_cond
        )
    if phase in ("all", "revise"):
        results["acts"]["revise"] = run_revise(memory, wiki, scope_id)

    # Conditions C1-C4 and C6 are sampled before forgetting tombstones topology.
    preforget = conformance.run(
        memory,
        scope_id,
        configuration={"scope_id": scope_id},
        only=["C1", "C2", "C3", "C4", "C6"],
    )
    conditions = _condition_payload(preforget)

    if phase in ("all", "forget"):
        results["acts"]["forget"] = run_forget(memory, scope_id)
        beta = int(
            memory.store.conn.execute(
                "SELECT COALESCE(max(active_units), 0) FROM gem_transition"
                " WHERE scope_id = %s",
                (scope_id,),
            ).fetchone()[0]
        )
        c5 = conformance.run(memory, scope_id, beta=beta, only=["C5"])
        conditions.extend(_condition_payload(c5))

    results["conformance"] = {
        "reporting": "per-condition only",
        "satisfied": [
            row["condition"] for row in conditions if row.get("holds") is True
        ],
        "violated": [
            row["condition"] for row in conditions if row.get("holds") is False
        ],
        "unchecked": [
            row["condition"] for row in conditions if row.get("holds") is None
        ],
        "results": conditions,
        "caveats": {
            "graph_read_visibility": "commit_visible, not snapshot_isolated",
            "repeatable_read_topology_claims": "not supported on this engine",
        },
    }
    results["trajectory"] = _jsonable(memory.trajectory(scope_id))
    manifest = environment_manifest(
        memory,
        source_manifest=wiki.manifest,
        scope_id=scope_id,
        model_name=model_name,
    )
    return results, manifest


def connect(
    dsn: str,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> TriDBGovernedMemory:
    """Construct the measured stock-PG operating point."""
    from bench.agent_memory.tridbBackend.backend import FastEmbedder

    embedder = FastEmbedder(model_name, batch_size=batch_size)
    return TriDBGovernedMemory.connect(
        dsn,
        dim=384,
        embedder=embedder,
        policy_engine=PolicyEngine(extra=seed_policies()),
    )
