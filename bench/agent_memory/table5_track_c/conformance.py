"""Reusable Track C adapter conformance gate and machine-readable receipt."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .dataset import EventItem, LoCoMoCorpus, QueryItem, WARMUP_SAMPLE_ID
from .protocol import (
    Adapter,
    _base_receipt,
    _retrieval_query,
    _write_json,
    add_receipt_contract,
    search_receipt_contract,
)

VISIBILITY_TARGET_TEXT = (
    "Please remember this for future conversations: my personal emergency "
    "rendezvous location is Silver Harbor, and my access code is 8427."
)
SCOPE_CONTROL_TEXT = (
    "Please remember this for future conversations: my backup meeting location "
    "is Violet Meadow, and my access code is 1936."
)


def _compact_search(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_count": result.get("result_count"),
        "hit_ids": result.get("hit_ids"),
        "contexts": result.get("contexts"),
        "context_tokens": result.get("context_tokens"),
        "empty": result.get("empty"),
        "cost": result.get("cost"),
        "probes": result.get("probes"),
    }


def run_conformance(
    *,
    adapter_factory: Callable[[], Adapter],
    corpus: LoCoMoCorpus,
    build_id: str,
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite conformance receipt: {output}")
    adapter = adapter_factory()
    receipt = _base_receipt(
        corpus=corpus, adapter=adapter, build_id=build_id, phase="conformance"
    )
    receipt["status"] = "running"
    _write_json(output, receipt)
    events = [
        event
        for event in corpus.events_by_sample[WARMUP_SAMPLE_ID]
        if event.session_id in {"S1", "S2", "S3"}
    ]
    queries = corpus.queries_by_sample[WARMUP_SAMPLE_ID][:5]
    target = EventItem(
        sample_id=WARMUP_SAMPLE_ID,
        event_id=f"{build_id}_visibility_target",
        session_id="S99",
        timestamp="2026-08-19 18:00",
        role="ConformanceTester",
        text=VISIBILITY_TARGET_TEXT,
        ordinal=99_000_001,
        metadata={"source": "track_c_conformance"},
    )
    other = EventItem(
        sample_id="conv-30",
        event_id=f"{build_id}_scope_control",
        session_id="S99",
        timestamp="2026-08-19 18:01",
        role="ConformanceTester",
        text=SCOPE_CONTROL_TEXT,
        ordinal=99_000_002,
        metadata={"source": "track_c_conformance"},
    )
    try:
        receipt["schema_gate"] = adapter.init_schema()
        started = time.perf_counter()
        receipt["build"] = adapter.ingest_history(events)
        receipt["build_finalize"] = adapter.finalize_build()
        receipt["build_outer_seconds"] = time.perf_counter() - started
        search_results = [
            adapter.search(_retrieval_query(query), top_k=35) for query in queries
        ]
        receipt["searches"] = [_compact_search(result) for result in search_results]
        receipt["target_add"] = adapter.add(target, visibility=True)
        receipt["scope_control_add"] = adapter.add(other, visibility=True)
        receipt["stats_before_restart"] = adapter.stats()
        adapter.close()

        adapter = adapter_factory()
        receipt["restart_schema_gate"] = adapter.init_schema()
        receipt["restart_visible"] = adapter.visibility_probe(target)
        cross_scope_query = QueryItem(
            sample_id=other.sample_id,
            question_id=f"scope:{target.event_id}",
            question=target.text,
            answer="",
            category="scope_probe",
            evidence_ids=(target.event_id,),
            ordinal=0,
        )
        cross_scope = adapter.search(_retrieval_query(cross_scope_query), top_k=35)
        receipt["cross_scope_search"] = _compact_search(cross_scope)
        receipt["scope_isolated"] = target.event_id not in cross_scope.get(
            "hit_ids", []
        )
        receipt["stats_after_restart"] = adapter.stats()
        required = {
            "events_exact": len(events) == 58,
            "five_searches": len(receipt["searches"]) == 5,
            "all_searches_nonempty": all(
                int(result.get("result_count") or 0) > 0
                for result in receipt["searches"]
            ),
            "search_receipt_contract": search_receipt_contract(
                [{"success": True, "receipt": result} for result in search_results]
            ),
            "add_creation_receipt_contract": add_receipt_contract(
                [
                    {"success": True, "receipt": receipt["target_add"]},
                    {"success": True, "receipt": receipt["scope_control_add"]},
                ]
            ),
            "target_visible_on_add": receipt["target_add"].get("visibility_probe")
            is True,
            "control_visible_on_add": receipt["scope_control_add"].get(
                "visibility_probe"
            )
            is True,
            "restart_persistent": receipt["restart_visible"] is True,
            "scope_isolated": receipt["scope_isolated"] is True,
        }
        receipt["checks"] = required
        receipt["status"] = "passed" if all(required.values()) else "failed"
        _write_json(output, receipt)
        if receipt["status"] != "passed":
            raise RuntimeError(f"conformance checks failed: {json.dumps(required)}")
        return receipt
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        _write_json(output, receipt)
        raise
    finally:
        adapter.close()
