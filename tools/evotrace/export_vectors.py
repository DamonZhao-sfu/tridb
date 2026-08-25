"""Pull the Experience Graph's vectors out of the database for the exact oracle.

    python3 tools/evotrace/export_vectors.py --dsn ... --scope evotrace:349117b0

The oracle must compute the right answer *independently of the engine*, but it has to
score against the same vectors the engine indexed — otherwise a parity failure could
mean either "the operator is wrong" or "the two sides embedded different text", and the
gate would tell us nothing. So the vectors come FROM the database, and everything else
about the oracle stays separate.

Only vertices that actually carry an embedding are exported. A vertex with none is not
missing data: Sessions and Prompts are hops on the path to an answer, and `tjs_open`
skips vectorless candidates outright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.agent_memory.gem_eg.store import DEFAULT_DIM, DEFAULT_DSN, EgStore

DEFAULT_OUT = Path("data/evotrace/normalized/vectors.npz")


def export(store: EgStore, scope_id: str, out: Path) -> dict[str, object]:
    rows = store.conn.execute(
        "SELECT uid, kind, embedding_source, embedding_model, embedding::text"
        " FROM gem_eg_vertex"
        " WHERE scope_id = %s AND embedding IS NOT NULL"
        " ORDER BY id",
        (scope_id,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"no embeddings in scope {scope_id!r}; run tools/evotrace/embed.py first")

    uids = [r[0] for r in rows]
    kinds = [r[1] for r in rows]
    sources = [r[2] or "" for r in rows]
    models = sorted({r[3] for r in rows if r[3]})
    # pgvector's text form is "[a,b,c]"; np.fromstring on the trimmed body is both
    # faster and less fragile than json.loads per row at 10k+ rows.
    matrix = np.stack(
        [np.fromstring(r[4].strip()[1:-1], sep=",", dtype=np.float32) for r in rows]
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        uids=np.array(uids, dtype=object).astype("U"),
        kinds=np.array(kinds, dtype=object).astype("U"),
        sources=np.array(sources, dtype=object).astype("U"),
        vectors=matrix,
    )
    by_source: dict[str, int] = {}
    for source in sources:
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "scope_id": scope_id,
        "path": str(out),
        "rows": len(uids),
        "dimension": int(matrix.shape[1]),
        "by_source": by_source,
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    store = EgStore.connect(args.dsn, dim=args.dim)
    try:
        summary = export(store, args.scope, args.out)
    finally:
        store.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
