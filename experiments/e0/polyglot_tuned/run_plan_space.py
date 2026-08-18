"""Enumerate the frozen E0 plan space over every query and record every execution.

Reads configs/e0/plan_space_v0.1.yaml (frozen BEFORE measurement) and runs every
(shape, k, hops, predicate_placement) combination against every query, `repetitions` times,
writing one row per (query, plan) to a parquet file. Analysis lives in analyze.py -- this
module only measures, so a re-analysis never needs a re-run.

Query embeddings are computed ONCE, before the sweep, through the same vLLM endpoint that
embedded the corpus (Qwen3-Embedding-0.6B, 1024-d) and cached to disk. No plan is charged
for embedding, and the cache makes the sweep reproducible without the model server.

Checkpointing: rows are flushed per query, and a re-run skips queries already present in the
output. A 2,880-execution sweep must survive being interrupted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.e0.polyglot_tuned.executor import (  # noqa: E402
    PlanSpec,
    PolyglotTuned,
    Query,
    load_queries,
    quality,
)

STORE_CFG = {
    "milvus_host": "localhost",
    "milvus_port": "19530",
    "milvus_collection": "e0_stark_prime",
    "neo4j_uri": "bolt://localhost:7688",
    "neo4j_user": "neo4j",
    "neo4j_password": "testpassword",
    "neo4j_label": "Node",
    "pg_host": "127.0.0.1",
    "pg_port": 5434,
    "pg_db": "tridb_wiki",
    "pg_user": "postgres",
    "pg_password": "postgres",
    "pg_table": "e0_stark_prime_node",
}


def embed_queries(
    queries: list[Query], cache_path: Path, base_url: str, model: str
) -> dict[str, list[float]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if all(q.query_id in cached for q in queries):
            return cached
    import urllib.request

    out: dict[str, list[float]] = {}
    for query in queries:
        payload = json.dumps({"model": model, "input": query.query_text}).encode()
        request = urllib.request.Request(
            f"{base_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        vector = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        out[query.query_id] = (vector / norm).tolist()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    return out


def plan_grid(config: dict[str, Any]) -> list[PlanSpec]:
    """The cartesian product, minus combinations that are not distinct plans.

    traverse_first has nothing before the traversal, so `pre` and `during` lower to the
    identical Cypher. Enumerating both would double-count one plan and quietly bias every
    per-shape aggregate towards traverse_first. It is dropped, not silently merged.
    """
    dims = config["dimensions"]
    grid: list[PlanSpec] = []
    for shape, k, hops, placement in itertools.product(
        dims["shape"], dims["k"], dims["hops"], dims["predicate_placement"]
    ):
        if shape == "traverse_first" and placement == "pre":
            continue
        grid.append(
            PlanSpec(shape=shape, k=k, hops=hops, predicate_placement=placement)
        )
    return grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/e0/plan_space_v0.1.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/e0/plan_space/raw.jsonl"))
    parser.add_argument("--tuning", choices=["tuned", "naive"], default="tuned")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    queries = load_queries(config["queries"])
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    plans = plan_grid(config)
    repetitions = int(config["repetitions"])
    timeout_s = float(config["timeout_seconds"])

    vectors = embed_queries(
        queries,
        Path("data/e0/stark_prime/normalized/query_embeddings.json"),
        args.embed_base_url,
        args.embed_model,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["query_id"])
        print(f"[run] resuming; {len(done)} queries already recorded")

    engine = PolyglotTuned(STORE_CFG, tuning=args.tuning)
    started = time.time()
    written = 0
    mode = "a" if (args.resume and args.out.exists()) else "w"
    with args.out.open(mode, encoding="utf-8") as handle:
        for index, query in enumerate(queries):
            if query.query_id in done:
                continue
            qvec = vectors[query.query_id]
            for plan in plans:
                samples: list[float] = []
                last = None
                for _ in range(repetitions):
                    result = engine.run(query, plan, qvec)
                    samples.append(result.latency_ms)
                    last = result
                    if result.latency_ms > timeout_s * 1e3:
                        result.infeasible = True
                        break
                assert last is not None
                metrics = quality(last.ranked, query.answer_ids)
                handle.write(
                    json.dumps(
                        {
                            "query_id": query.query_id,
                            "annotation_status": query.annotation_status,
                            "template": query.template,
                            "query_hop_limit": query.hop_limit,
                            "n_answers": len(query.answer_ids),
                            "shape": plan.shape,
                            "k": plan.k,
                            "hops": plan.hops,
                            "predicate_placement": plan.predicate_placement,
                            "plan_tag": plan.tag,
                            "latency_ms": statistics.median(samples),
                            "latency_min_ms": min(samples),
                            "latency_max_ms": max(samples),
                            "repetitions": len(samples),
                            "stage_ms": last.stage_ms,
                            "round_trips": last.round_trips,
                            "bytes_shipped": last.bytes_shipped,
                            "cardinality": last.cardinality,
                            "infeasible": last.infeasible,
                            "error": last.error,
                            "returned": len(last.ranked),
                            **metrics,
                        }
                    )
                    + "\n"
                )
                written += 1
            handle.flush()
            print(
                f"[run] {index + 1}/{len(queries)} {query.query_id} "
                f"({written} rows, {time.time() - started:.0f}s)",
                flush=True,
            )
    engine.close()
    print(f"[run] wrote {written} rows to {args.out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
