"""Behavioral parity gate for live and stored node-artifact embeddings.

The physical-plan experiment consumes node IDs, not cosine values.  The hard gate is
therefore exact top-m seed identity under deterministic tie-breaking.  Re-embedding
cosine remains an auditable diagnostic and can be restored as a hard condition with
``--require-cosine`` for older experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.agent_memory.gem_eg.store import DEFAULT_DSN, EgStore
from tools.evotrace.embed import (
    BATCH,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    embed_batch,
    node_text,
)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _top_m(query: np.ndarray, matrix: np.ndarray, uids: list[str], m: int) -> list[str]:
    distances = 1.0 - matrix @ query
    return [
        uids[i]
        for i in sorted(range(len(uids)), key=lambda i: (distances[i], uids[i]))[:m]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--domain", default="math")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--topm-queries", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.99999)
    parser.add_argument(
        "--require-cosine",
        action="store_true",
        help="also require every sampled cosine to meet --threshold; the approved "
        "Math/ALE plan uses top-m identity as the hard behavioral gate",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample < 2:
        parser.error("--sample must be at least 2")

    store = EgStore.connect(args.dsn)
    try:
        all_rows = store.conn.execute(
            "SELECT v.uid,v.embedding::text FROM gem_eg_vertex v"
            " WHERE v.scope_id=%s AND v.kind='node' AND v.domain=%s"
            " AND v.embedding IS NOT NULL"
            " ORDER BY v.uid",
            (args.scope, args.domain),
        ).fetchall()
        detail = store.conn.execute(
            "SELECT v.id,n.node_uid,n.language,n.status,n.changes,n.error_signature,a.payload"
            " FROM gem_eg_vertex v JOIN gem_eg_node n ON n.id=v.id"
            " LEFT JOIN gem_eg_artifact a ON a.artifact_uid=n.artifact_uid"
            " WHERE v.scope_id=%s AND v.domain=%s AND v.embedding IS NOT NULL"
            " ORDER BY n.node_uid",
            (args.scope, args.domain),
        ).fetchall()
    finally:
        store.close()
    if len(detail) < args.sample:
        raise SystemExit(
            f"only {len(detail)} embedded nodes, need sample {args.sample}"
        )
    indexes = [
        round(i * (len(detail) - 1) / (args.sample - 1)) for i in range(args.sample)
    ]
    sample = [detail[i] for i in indexes]
    try:
        texts = [node_text(row) for row in sample]
        vectors = []
        # Match the corpus writer's request size; pooling kernels can otherwise make
        # a strict 0.99999 gate measure serving batch shape rather than model parity.
        for start in range(0, len(texts), BATCH):
            vectors.extend(
                embed_batch(
                    texts[start : start + BATCH],
                    endpoint=args.endpoint,
                    model=args.model,
                )
            )
        fresh = np.asarray(vectors, dtype=np.float32)
    except Exception as exc:  # fail closed and leave an auditable receipt
        receipt = {
            "schema_version": "live_node_embedding_gate_v2",
            "scope_id": args.scope,
            "endpoint": args.endpoint,
            "model": args.model,
            "domain": args.domain,
            "sample": args.sample,
            "top_m": args.top_m,
            "hard_gate": "cosine_and_top_m_identity"
            if args.require_cosine
            else "top_m_identity",
            "passed": False,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
        return 2
    stored_by_uid = {str(uid): json.loads(vector) for uid, vector in all_rows}
    stored_sample = np.asarray(
        [stored_by_uid[row[1]] for row in sample], dtype=np.float32
    )
    fresh = _normalise(fresh)
    stored_sample = _normalise(stored_sample)
    cosines = np.sum(fresh * stored_sample, axis=1)

    uids = [str(uid) for uid, _ in all_rows]
    matrix = _normalise(
        np.asarray([json.loads(vector) for _, vector in all_rows], dtype=np.float32)
    )
    topm_checks = []
    for index, row in enumerate(sample[: args.topm_queries]):
        old = _top_m(stored_sample[index], matrix, uids, args.top_m)
        new = _top_m(fresh[index], matrix, uids, args.top_m)
        topm_checks.append(
            {"query_uid": row[1], "stored": old, "fresh": new, "equal": old == new}
        )

    cosine_diagnostic_passed = bool(np.all(cosines >= args.threshold))
    topm_identity_passed = all(row["equal"] for row in topm_checks)
    passed = topm_identity_passed and (
        cosine_diagnostic_passed if args.require_cosine else True
    )
    receipt = {
        "schema_version": "live_node_embedding_gate_v2",
        "scope_id": args.scope,
        "endpoint": args.endpoint,
        "model": args.model,
        "domain": args.domain,
        "dimension": int(fresh.shape[1]),
        "sample": len(sample),
        "threshold": args.threshold,
        "min_cosine": float(np.min(cosines)),
        "cosine_diagnostic_passed": cosine_diagnostic_passed,
        "require_cosine": args.require_cosine,
        "top_m": args.top_m,
        "topm_checks": topm_checks,
        "topm_identity_passed": topm_identity_passed,
        "hard_gate": "cosine_and_top_m_identity"
        if args.require_cosine
        else "top_m_identity",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
