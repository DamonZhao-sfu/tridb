"""Small live conformance gate for the pinned paper-era EverMemOS server."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bench.agent_memory.table5_track_c.dataset import load_locomo

from .paper_era_adapter import EverMemOSPaperAdapter, EverMemOSPaperConfig


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8195")
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "evermemos_paper_track_c_conformance_v0.1.0",
        "status": "running",
        "started_at": now(),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    adapter = EverMemOSPaperAdapter(
        EverMemOSPaperConfig(
            dataset_path=args.dataset,
            base_url=args.base_url,
            answer_base_url=args.answer_base_url,
            answer_model=args.answer_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            build_scope_workers=1,
        )
    )
    try:
        corpus = load_locomo(args.dataset)
        sample_id = corpus.sample_ids[0]
        events = corpus.events_by_sample[sample_id][:6]
        query = corpus.queries_by_sample[sample_id][0]
        receipt["schema_gate"] = adapter.init_schema()
        receipt["ingest"] = adapter.ingest_history(events)
        receipt["finalize"] = adapter.finalize_build()
        receipt["search"] = adapter.search(query, top_k=10)
        receipt["fingerprint"] = adapter.snapshot_fingerprint()
        receipt["checks"] = {
            "source_commit": receipt["schema_gate"]["source"]["commit"]
            == "806ad0555a09245ec90ad936e848bee9c64c6a49",
            "source_clean": receipt["schema_gate"]["source"]["dirty"] is False,
            "events_ingested": receipt["ingest"]["events"] == 6,
            "synchronous_boundary": receipt["finalize"]["converged"] is True,
            "search_contract": all(
                key in receipt["search"]
                for key in ("result_count", "hit_ids", "contexts", "empty")
            ),
            "snapshot_nonempty": receipt["fingerprint"]["total_count"] > 0,
        }
        receipt["passed"] = all(receipt["checks"].values())
        receipt["status"] = "complete" if receipt["passed"] else "failed"
        receipt["completed_at"] = now()
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0 if receipt["passed"] else 1
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["passed"] = False
        receipt["failed_at"] = now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
