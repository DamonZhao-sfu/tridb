"""Arm C's precondition -- does Polyglot answer the SAME question as GEM?

Arm B and arm C exist to compare two SYSTEMS, not two retrieval designs. If they
return different result sets, an agent-outcome difference says nothing about the
systems and everything about the answers. So before arm C may run, the two must agree
on a sample of the same queries, and any disagreement must be visible.

WHY THIS GATE IS NOT OPTIONAL
-----------------------------
On 2026-08-18 the E0 polyglot numbers were retracted wholesale (commit b3b4e6d):
1,010 of 1,010 cells returned an empty `result_ids` because the loader finished two
minutes AFTER the measurement started, and every empty answer was silently scored as
zero. Inside an agent run that failure is even harder to see -- it presents as
"memory did not help". The two rules below are the direct response:

  1. a load receipt must exist before anything is measured;
  2. an empty result set is an OUTAGE, never an answer.

    python3 -m tools.evotrace.gate_polyglot_parity --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import GroundTruth
from bench.agent_memory.gem_eg.metrics import parity
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import EgStore

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
DEFAULT_SCOPE = "evotrace:349117b0"
DEFAULT_NORMALIZED = Path("data/evotrace/normalized")


def require_load_receipt(path: Path) -> dict[str, Any]:
    """Rule 1. No receipt, no measurement -- this is the retraction's root cause."""
    if not path.is_file():
        raise SystemExit(
            f"no polyglot load receipt at {path}. The E0 polyglot numbers were "
            "retracted because measurement began before the loader finished; a "
            "receipt naming the loaded row counts and completion time is required "
            "before any cell is timed."
        )
    receipt = json.loads(path.read_text())
    for field in ("completed_at", "rows_loaded", "scope"):
        if field not in receipt:
            raise SystemExit(f"load receipt {path} lacks required field {field!r}")
    if not receipt["rows_loaded"]:
        raise SystemExit(f"load receipt {path} reports zero rows loaded")
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    ap.add_argument("--load-receipt", type=Path,
                    default=Path("bench/out/polyglot/load_receipt.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--queries", default="W1.a")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-exact", type=float, default=0.95,
                    help="required fraction of queries where both systems return the "
                         "identical ordered result set")
    args = ap.parse_args(argv)

    receipt = require_load_receipt(args.load_receipt)
    print(f"polyglot load receipt: {receipt['rows_loaded']:,} rows, "
          f"completed {receipt['completed_at']}")

    # Import late: arm C's backend is optional, and a missing Milvus client should
    # produce a clear gate failure rather than an ImportError at module load.
    try:
        from experiments.e0.plan_spread.live_backend import PolyglotBackend
    except ImportError as exc:
        raise SystemExit(f"polyglot backend unavailable: {exc}") from None

    from bench.agent_memory.gem_oe.retrievers import GemRetriever, PolyglotRetriever

    corpus = Corpus.load(args.normalized, vectors=args.normalized / "vectors.npz")
    points = [dp for dp in GroundTruth(corpus).all_points() if dp.query_id == args.queries]
    import random

    chosen = random.Random(args.seed).sample(points, min(args.limit, len(points)))

    store = EgStore.connect(args.dsn)
    store.bootstrap_edge_types()
    gem = GemRetriever(store=store, scope_id=args.scope, knobs=Knobs())
    polyglot = PolyglotRetriever(backend=PolyglotBackend(), gem=gem)

    rows: list[dict[str, Any]] = []
    empties = 0
    started = time.perf_counter()
    for dp in chosen:
        spec = gem._spec(dp.task_uid, _StubParent(dp.parent_fitness), args.k, dp.iteration)
        gem_ids = list(W1Engine(store, args.scope).run(spec, gem.knobs).ids)
        poly_ids, _ = polyglot.backend.reuse_query(spec)
        poly_ids = list(poly_ids)
        # Rule 2. An empty answer from a multi-system pipeline is far more often a
        # stage that did not connect than a genuinely empty eligible set.
        if not poly_ids:
            empties += 1
        exact, recall, tie_equivalent = parity(poly_ids, gem_ids, distances=None)
        rows.append({
            "dp_id": dp.dp_id,
            "task_uid": dp.task_uid,
            "gem": gem_ids,
            "polyglot": poly_ids,
            "exact": exact,
            "recall": recall,
            "tie_equivalent": tie_equivalent,
        })

    n = len(rows) or 1
    summary = {
        "queries": args.queries,
        "n": len(rows),
        "exact_rate": round(sum(r["exact"] for r in rows) / n, 4),
        "tie_equivalent_rate": round(sum(r["tie_equivalent"] for r in rows) / n, 4),
        "mean_recall": round(sum(r["recall"] for r in rows) / n, 4),
        "polyglot_empty_results": empties,
        "seconds": round(time.perf_counter() - started, 1),
        "load_receipt": receipt,
    }
    summary["passed"] = (
        empties == 0
        and summary["exact_rate"] >= args.min_exact
        and len(rows) > 0
    )
    for key, value in summary.items():
        if key != "load_receipt":
            print(f"  {key:<26} {value}")
    if empties:
        print(f"\n  {empties} queries returned NOTHING -- treated as an outage, "
              "not as an empty eligible set")
    print(f"\ngate: {'PASS' if summary['passed'] else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"receipt: {args.out}")
    return 0 if summary["passed"] else 1


class _StubParent:
    """Just enough of `Program` for `GemRetriever._predicate`."""

    def __init__(self, fitness: float | None) -> None:
        self.metrics = {"combined_score": fitness} if fitness is not None else {}
        self.id = "stub"


if __name__ == "__main__":
    raise SystemExit(main())
