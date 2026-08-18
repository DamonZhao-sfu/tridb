"""Re-audit the E0 query set with the OFFICIAL stark-qa API, independently of its deriver.

WHY THIS IS A SEPARATE MODULE
-----------------------------
`tools/e0/query_expand.py` emits `path_audit.all_official_answers_reachable = true` on every
row it produces. That claim is worth nothing on its own: the same CSR reimplementation that
SEARCHED for the typed-hop envelope is the one asserting the envelope is correct. A bug
shared by both halves is invisible.

This module re-derives the reachable set with `stark_qa.skb.prime.PrimeSKB` through
`stark_prime_prepare._typed_reachable` -- the auditor the 10 hand-audited pilot rows were
validated with, over the official processed graph rather than the normalized parquet. Two
independent implementations, two independent data paths, one claim.

It checks four things per query:
  * every official answer lies inside (anchor, edge_types, hop_limit)
  * the anchor id really carries the anchor name the row states
  * every official answer really has the declared target entity type
  * the anchor name actually occurs in the query text (the row is about what was asked)

Exit status is non-zero if any query fails, so it can gate a Makefile pipeline.

CLI:
    python -m tools.e0.query_audit
    python -m tools.e0.query_audit --queries path/to/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.e0.common import environment_record, write_json


def audit(queries_path: Path, raw_root: Path) -> dict[str, Any]:
    try:
        from stark_qa.skb.prime import PrimeSKB
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("stark-qa is required; install requirements-e0.txt") from exc

    from tools.e0.stark_prime_prepare import _typed_reachable

    skb = PrimeSKB(root=str(raw_root), download_processed=False)
    rows = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    failures: list[dict[str, Any]] = []
    for row in rows:
        qid = row["query_id"]
        anchor = int(row["anchor_ids"][0])
        reach = _typed_reachable(
            skb, anchor, list(row["edge_types"]), int(row["hop_limit"])
        )
        missing = [a for a in row["answer_ids"] if a not in reach]
        if missing:
            failures.append(
                {"query_id": qid, "check": "unreachable_answers", "detail": missing}
            )
        actual_name = str(skb.node_info[anchor].get("name", ""))
        if actual_name != row["anchor_names"][0]:
            failures.append(
                {
                    "query_id": qid,
                    "check": "anchor_name_mismatch",
                    "detail": [actual_name, row["anchor_names"][0]],
                }
            )
        answer_types = sorted({skb.node_info[a]["type"] for a in row["answer_ids"]})
        if answer_types != [row["target_entity_type"]]:
            failures.append(
                {
                    "query_id": qid,
                    "check": "target_type_mismatch",
                    "detail": answer_types,
                }
            )
        if row["anchor_names"][0].casefold() not in row["query_text"].casefold():
            failures.append(
                {
                    "query_id": qid,
                    "check": "anchor_not_named_in_text",
                    "detail": row["anchor_names"][0],
                }
            )

    result = {
        "schema_version": "e0-query-audit-v0.1.0",
        "environment": environment_record(),
        "queries": str(queries_path),
        "auditor": "stark_qa.skb.prime.PrimeSKB via stark_prime_prepare._typed_reachable",
        "total": len(rows),
        "failures": failures,
        "passed": not failures,
        "status_distribution": dict(
            Counter(r.get("annotation_status", "unknown") for r in rows)
        ),
        "hop_distribution": dict(Counter(int(r["hop_limit"]) for r in rows)),
        "template_distribution": dict(Counter(r["template"] for r in rows)),
        "edge_type_set_sizes": dict(Counter(len(r["edge_types"]) for r in rows)),
        "answers_per_query": {
            "min": min((len(r["answer_ids"]) for r in rows), default=0),
            "median": sorted(len(r["answer_ids"]) for r in rows)[len(rows) // 2]
            if rows
            else 0,
            "max": max((len(r["answer_ids"]) for r in rows), default=0),
        },
    }
    write_json(queries_path.with_suffix(".audit.json"), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/e0/stark_prime/normalized/queries_v0.2.jsonl"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/e0/stark_prime/raw/prime")
    )
    args = parser.parse_args(argv)

    result = audit(args.queries, args.raw_root)
    print(
        f"[query_audit] {result['total']} queries re-audited with the official stark-qa API"
    )
    print(f"[query_audit] status={result['status_distribution']}")
    print(
        f"[query_audit] hops={result['hop_distribution']} "
        f"templates={result['template_distribution']}"
    )
    print(f"[query_audit] answers/query={result['answers_per_query']}")
    if result["failures"]:
        print(f"[query_audit] FAILURES: {len(result['failures'])}")
        for failure in result["failures"][:10]:
            print(f"  {failure['query_id']}: {failure['check']} {failure['detail']}")
        return 1
    print("[query_audit] PASSED: 0 failures")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
