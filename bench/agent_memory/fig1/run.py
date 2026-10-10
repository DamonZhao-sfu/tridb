"""Figure 1 EvoTrace experiment: mock ingest/retrieve workload, tri-modal + semantic.

    python3 -m bench.agent_memory.fig1.run --dsn postgresql://$USER@127.0.0.1:55440/fig1 \\
        --llm mock --tasks ale:ahc008,math:circle_packing --out results/fig1_semantic_v0.1.0/x

Arms (each in its own scope of the same database, replaying the same timeline):

  eager        Ingest eager; at every event run all retrieval plans (trimodal, naive,
               planned_mat, planned) on the identical state, strong consistency.
  onread       Ingest deferred (trigger on_read); retrieval 'planned', strong.
  onread_stale Ingest deferred (on_read); retrieval 'planned', stale; never resolves.
  batch        Ingest deferred (batch every N); retrieval 'planned', strong.

Every retrieval plan and every arm uses its own cache namespace (model label suffix), so
LLM-call counts are not subsidised by another arm's cache. After the replay each arm is
fully materialized and Mat(S) is compared with the eager arm. ``--replay-check`` re-ingests
the eager arm in a fresh scope with the LLM disabled (offline cache replay, plan P1 exit).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg

from bench.agent_memory.fig1.corpus import Corpus, load_corpus
from bench.agent_memory.fig1.memory import Fig1Memory, IngestPlan, RetrievePlan
from bench.agent_memory.fig1.mock import make_responder
from bench.agent_memory.semantic.cache import SemCache, SemMeter
from bench.agent_memory.semantic.llm import MockLLM, OpenAIChatClient
from bench.agent_memory.semantic.ops import SemOps
from bench.agent_memory.semantic.planner import SemPlanner

ARMS = ("eager", "eager_ctl", "onread", "onread_stale", "batch")
PLANS = ("trimodal", "naive", "planned_mat", "planned")


class Namespaced:
    """Same LLM, different cache partition: the cache key includes ``model``."""

    def __init__(self, inner: Any, label: str) -> None:
        self.inner = inner
        self.model = f"{inner.model}#{label}"

    def complete(self, system: str, user: str) -> Any:
        return self.inner.complete(system, user)

    def complete_many(self, prompts: Any) -> Any:
        return self.inner.complete_many(prompts)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": pct(values, 0.5),
        "p95": pct(values, 0.95),
        "sum": sum(values),
    }


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def arm_database(dsn: str, scope: str, *, create: bool = True) -> str:
    """Each arm gets a fresh database so arms never share an HNSW index or graph."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    name = (
        "f1_" + "".join(c if c.isalnum() else "_" for c in scope.split(":", 1)[1])[:55]
    )
    if create:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            admin.execute(f'CREATE DATABASE "{name}"')
    params = conninfo_to_dict(dsn)
    params["dbname"] = name
    return make_conninfo(**params)


def copy_cache(source_dsn: str, conn: Any) -> int:
    """Seed this arm's gem_sem_cache with the eager arm's verdicts ("given the cache")."""
    from psycopg.types.json import Jsonb

    with psycopg.connect(source_dsn) as src:
        rows = src.execute(
            "SELECT op, model, prompt_version, input_hash, output, prompt_tokens,"
            " completion_tokens, llm_seconds FROM gem_sem_cache"
        ).fetchall()
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO gem_sem_cache (op, model, prompt_version, input_hash, output,"
            " prompt_tokens, completion_tokens, llm_seconds) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT DO NOTHING",
            [(*row[:4], Jsonb(row[4]), *row[5:]) for row in rows],
        )
    conn.commit()
    return len(rows)


def build_llm(args: argparse.Namespace) -> Any:
    if args.llm == "real":
        return OpenAIChatClient(
            args.base_url, concurrency=args.concurrency, max_tokens=160
        )
    return MockLLM(
        latency_s=args.mock_latency,
        concurrency=args.concurrency,
        responder=make_responder(keep_rate=args.keep_rate, fix_rate=args.fix_rate),
    )


def run_arm(
    arm: str,
    corpus: Corpus,
    args: argparse.Namespace,
    llm: Any,
    scope: str,
    *,
    offline: bool = False,
    namespace: str | None = None,
    retrieve: bool = True,
    database_of: str | None = None,
) -> dict[str, Any]:
    target = (
        arm_database(args.dsn, database_of, create=False)
        if database_of
        else arm_database(args.dsn, scope)
    )
    conn = psycopg.connect(target, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS join_order")
    conn.commit()
    ns = namespace or ("shared" if args.shared_cache else arm)
    ingest_ops = SemOps(
        Namespaced(llm, f"{args.tag}:{ns}:ingest"), SemCache(conn), SemMeter()
    )
    ingest_ops.offline = offline
    memory = Fig1Memory(conn, scope, ingest_ops, planner=SemPlanner(conn))
    memory.setup()
    if args.shared_cache and arm != "eager" and database_of is None:
        copied = copy_cache(
            arm_database(args.dsn, f"fig1:{args.tag}:eager", create=False), conn
        )
        print(
            f"[{arm}] shared cache: copied {copied} rows from the eager arm", flush=True
        )
    if arm == "eager":
        iplan = IngestPlan(mode="eager", fix_candidates=args.fix_k)
        plans = list(args.plans)
        consistency = "strong"
    elif arm == "eager_ctl":  # control: same as eager, measures HNSW run-to-run noise
        iplan = IngestPlan(mode="eager", fix_candidates=args.fix_k)
        plans, consistency = ["planned"], "strong"
    elif arm == "batch":
        iplan = IngestPlan(
            mode="deferred",
            trigger="batch",
            batch_every=args.batch_every,
            fix_candidates=args.fix_k,
        )
        plans, consistency = ["planned"], "strong"
    else:
        iplan = IngestPlan(
            mode="deferred", trigger="on_read", fix_candidates=args.fix_k
        )
        plans = ["planned"]
        consistency = "stale" if arm == "onread_stale" else "strong"
    rops = {
        p: SemOps(
            Namespaced(llm, f"{args.tag}:{ns}:retrieve:{p}"), SemCache(conn), SemMeter()
        )
        for p in plans
    }
    rplans = {
        p: RetrievePlan(
            plan=p,
            consistency=consistency,
            k=args.k,
            k_max=args.k_max,
            m=args.m,
            d=args.d,
            c_llm=args.c_llm,
        )
        for p in plans
    }

    ingest_secs: list[float] = []
    retrieve_secs: dict[str, list[float]] = defaultdict(list)
    per_plan: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    outputs: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    for t, uid in enumerate(corpus.timeline):
        node = corpus.nodes[uid]
        if retrieve and node.parent_uid is not None and node.parent_uid in memory.vid:
            parent = corpus.nodes[node.parent_uid]
            for p in plans:
                memory.ops = (
                    ingest_ops  # derivations resolved on read stay in ingest ns
                )
                _hints, rec = memory.retrieve(parent, rplans[p], ops=rops[p])
                retrieve_secs[p].append(rec.seconds)
                outputs[p].append((rec.extra["out_ids"], rec.extra["out_text_sha"]))
                agg = per_plan[p]
                for key in (
                    "rows_pulled",
                    "sem_evaluated",
                    "hints",
                    "resolved",
                    "staleness",
                    "engine_seconds",
                ):
                    agg[key] += float(rec.extra.get(key) or 0)
                agg["closed_early"] += 1.0 if rec.extra.get("closed_early") else 0.0
                agg["retrievals"] += 1
                if args.trace and t < args.trace:
                    events.append({"t": t, "plan": p, **rec.extra, "sem": rec.sem})
        rec = memory.ingest(node, corpus.vectors[uid], iplan)
        ingest_secs.append(rec.seconds)
        if args.progress and (t + 1) % args.progress == 0:
            el = time.perf_counter() - started
            print(f"[{arm}] {t + 1}/{len(corpus)} events {el:.1f}s", flush=True)
    replay_seconds = time.perf_counter() - started

    pending_left = memory.pending.count_open(scope)
    conn.commit()
    before_final = ingest_ops.meter.snapshot()
    replay_llm = {k: v for k, v in before_final.items() if k != "by_op"}
    t0 = time.perf_counter()
    final_resolved = memory.materialize(by="batch")
    final_seconds = time.perf_counter() - t0
    final_cost = ingest_ops.meter.diff(before_final)
    state = memory.materialized_state()

    result = {
        "arm": arm,
        "scope": scope,
        "events": len(corpus),
        "ingest_plan": asdict(iplan),
        "consistency": consistency,
        "replay_seconds": replay_seconds,
        "ingest_seconds": summarize(ingest_secs),
        "ingest_llm_during_replay": replay_llm,
        "ingest_llm": {
            k: v for k, v in ingest_ops.meter.snapshot().items() if k != "by_op"
        },
        "ingest_llm_by_op": ingest_ops.meter.snapshot()["by_op"],
        "pending_left_before_final": pending_left,
        "final_materialize": {
            "resolved": final_resolved,
            "seconds": final_seconds,
            "llm": final_cost,
        },
        "retrieve": {
            p: {
                "seconds": summarize(retrieve_secs[p]),
                "llm": {
                    k: v for k, v in rops[p].meter.snapshot().items() if k != "by_op"
                },
                "totals": dict(per_plan[p]),
            }
            for p in plans
        },
        "state_summary": {
            "lessons": sum(1 for v in state["lessons"].values() if v),
            "fixed_by": len(state["fixed_by"]),
            "fixed_by_native": len(state["fixed_by_native"]),
            "native_equals_mirror": state["fixed_by_native"] == state["fixed_by"],
        },
        "trace": events,
    }
    conn.close()
    return {"result": result, "outputs": dict(outputs), "state": state}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/evotrace/normalized"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-nodes-per-session", type=int, default=None)
    ap.add_argument("--max-sessions-per-task", type=int, default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--plans", default=",".join(PLANS))
    ap.add_argument("--llm", choices=("mock", "real"), default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--mock-latency", type=float, default=0.0)
    ap.add_argument("--keep-rate", type=float, default=0.4)
    ap.add_argument("--fix-rate", type=float, default=0.5)
    ap.add_argument("--c-llm", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=48)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--fix-k", type=int, default=5)
    ap.add_argument("--batch-every", type=int, default=64)
    ap.add_argument("--tag", default=time.strftime("r%Y%m%dT%H%M%S"))
    ap.add_argument("--replay-check", action="store_true")
    ap.add_argument(
        "--shared-cache",
        action="store_true",
        help="all arms share one cache namespace, seeded from the eager arm "
        "(tests Mat equivalence 'given the cache'; LLM counts are then not comparable)",
    )
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.plans = [p for p in args.plans.split(",") if p]
    tasks = [t for t in args.tasks.split(",") if t] or None

    corpus = load_corpus(
        args.data,
        tasks=tasks,
        max_nodes_per_session=args.max_nodes_per_session,
        max_sessions_per_task=args.max_sessions_per_task,
    )
    llm = build_llm(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"corpus: {len(corpus)} events, {len(corpus.tasks)} tasks, llm={llm.model}",
        flush=True,
    )

    runs: dict[str, dict[str, Any]] = {}
    for arm in [a for a in args.arms.split(",") if a]:
        runs[arm] = run_arm(arm, corpus, args, llm, f"fig1:{args.tag}:{arm}")
        (args.out / f"{arm}.json").write_text(
            json.dumps(runs[arm]["result"], indent=2, default=str), encoding="utf-8"
        )
        r = runs[arm]["result"]
        print(
            f"[{arm}] replay {r['replay_seconds']:.1f}s ingest p50 "
            f"{r['ingest_seconds']['p50'] * 1e3:.1f}ms replay_llm_calls "
            f"{r['ingest_llm_during_replay']['llm_calls']} total {r['ingest_llm']['llm_calls']}",
            flush=True,
        )

    comparison: dict[str, Any] = {
        "tag": args.tag,
        "events": len(corpus),
        "llm": llm.model,
    }
    if "eager" in runs:
        ref_state = runs["eager"]["state"]
        ref_out = runs["eager"]["outputs"]
        comparison["mat_equal_to_eager"] = {
            arm: {
                "lessons_equal": run["state"]["lessons"] == ref_state["lessons"],
                "lessons_agree": _agree(run["state"]["lessons"], ref_state["lessons"]),
                "fixed_by_equal": run["state"]["fixed_by"] == ref_state["fixed_by"],
                "fixed_by_native_equal": run["state"]["fixed_by_native"]
                == ref_state["fixed_by_native"],
                "native_equals_mirror": run["state"]["fixed_by_native"]
                == run["state"]["fixed_by"],
                "fixed_by_jaccard": _jac_pairs(
                    run["state"]["fixed_by"], ref_state["fixed_by"]
                ),
            }
            for arm, run in runs.items()
        }
        base = ref_out.get("planned_mat") or ref_out.get("planned")
        comparison["output_agreement_vs_eager_planned_mat"] = {}
        for arm, run in runs.items():
            for p, outs in run["outputs"].items():
                if base is None or len(outs) != len(base):
                    continue
                pairs = list(zip(outs, base, strict=True))
                comparison["output_agreement_vs_eager_planned_mat"][f"{arm}/{p}"] = (
                    _agreement(pairs)
                )
        if args.replay_check:
            replay = run_arm(
                "eager",
                corpus,
                args,
                _Disabled(llm),
                f"fig1:{args.tag}:eager_replay",
                offline=True,
                namespace="shared" if args.shared_cache else "eager",
                retrieve=False,
                database_of=f"fig1:{args.tag}:eager",
            )
            comparison["offline_replay"] = {
                "lessons_equal": replay["state"]["lessons"] == ref_state["lessons"],
                "fixed_by_equal": replay["state"]["fixed_by"] == ref_state["fixed_by"],
                "fixed_by_native_equal": replay["state"]["fixed_by_native"]
                == ref_state["fixed_by_native"],
                "llm_calls": replay["result"]["ingest_llm"]["llm_calls"],
            }
    (args.out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, default=str))
    return 0


class _Disabled:
    """An LLM that must never be called (offline replay)."""

    def __init__(self, inner: Any) -> None:
        self.model = inner.model

    def complete(self, system: str, user: str) -> Any:
        raise RuntimeError("LLM disabled during offline replay")

    def complete_many(self, prompts: Any) -> Any:
        if prompts:
            raise RuntimeError("LLM disabled during offline replay")
        return []


def _agreement(pairs: list[tuple[Any, Any]]) -> dict[str, float]:
    """Per-retrieval agreement: set Jaccard, exact set, exact ordered list, same text."""
    if not pairs:
        return {
            "mean_jaccard": 1.0,
            "exact_set": 1.0,
            "exact_ordered": 1.0,
            "same_text": 1.0,
        }
    jac = [jaccard(a[0], b[0]) for a, b in pairs]
    return {
        "mean_jaccard": statistics.fmean(jac),
        "exact_set": statistics.fmean([1.0 if v == 1.0 else 0.0 for v in jac]),
        "exact_ordered": statistics.fmean(
            [1.0 if a[0] == b[0] else 0.0 for a, b in pairs]
        ),
        "same_text": statistics.fmean([1.0 if a[1] == b[1] else 0.0 for a, b in pairs]),
    }


def _agree(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(1 for k in keys if a.get(k) == b.get(k)) / len(keys)


def _jac_pairs(a: list[Any], b: list[Any]) -> float:
    sa, sb = {tuple(x) for x in a}, {tuple(x) for x in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


if __name__ == "__main__":
    raise SystemExit(main())
