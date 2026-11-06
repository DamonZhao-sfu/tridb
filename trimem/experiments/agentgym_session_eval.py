"""AgentGym official evaluator + TriMEM session memory (task seeds -> sessions -> steps).

Everything AgentGym does is unchanged (process pool, Evaluator, max_round, reward, per-item files);
before each episode this runner retrieves once with ``session_retrieval.retrieve`` through
``AgentEnvTaskAdapter`` (the same retriever and opt store as SWE-QA-Pro), appends the rendered
memory to the official instruction turn, and after the episode writes the generation cost and the
simulator quality (paired with the No-Memory result of the same item) to ``opt.session_retrieval``.
Budgets come from ``llm_budget``. Use ``.venv-agentgym``.

    .venv-agentgym/bin/python -m trimem.experiments.agentgym_session_eval --plan P2 \\
        --memory_config '{"seed_count":8,"depth":1,"fanout":4,"top_k":3}' --split test \\
        --baseline_dir results/agentgym_eval_20261015/nomemory_fc/alfworld --action_format function_calling \\
        <official openai_eval_multiprocessing arguments>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg
from psycopg.rows import dict_row

from trimem.experiments import agentgym_official_eval as base
from trimem.experiments import llm_budget
from trimem.experiments import memory_policy as mp
from trimem.experiments import quality_reuse as qr
from trimem.experiments import semfilter_v2
from trimem.experiments.colbert_rank import VERSION as COLBERT
from trimem.experiments.colbert_rank import ColBERTRank, rank_base_for
from trimem.experiments.agentenv_task_adapter import (
    AgentEnvTaskAdapter,
    dsn_for,
    render_memory,
)
from trimem.experiments.retrieval_lease import RetrievalLease, readonly_connect
from trimem.experiments.semantic_rubric import RubricSemFilter, RubricSemRank, env_card
from trimem.experiments.session_profile_store import recorded_retrieve
from trimem.experiments.session_retrieval import (
    RANK_PLANS,
    SEMFILTER_PLANS,
    V2_PLANS,
    SessionConfig,
)
from trimem.experiments.session_store_ext import ScalarSessionStore
from trimem.experiments.sweqa.session_adapter import DSN as OPT_DSN

official = base.official
EMBED = "http://127.0.0.1:8021/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
S: dict = {}  # run settings, filled in main() before the pool forks


def semantic_tokens(calls):
    if os.environ.get("TRIMEM_ABLATION_RESOURCE_LEDGER"):
        from trimem.experiments.scienceworld_trace_ablation import checked_semantic_tokens

        try:
            return checked_semantic_tokens(calls)
        except ValueError as exc:
            raise SystemExit(f"semantic usage unavailable; ablation paused: {exc}") from exc
    return sum((c["usage"] or {}).get("total_tokens", 0) for c in calls)


def query_task(item: str) -> dict:
    """Task row of an item: test items by their AgentEval id, training-panel items by the task key
    recorded in the frozen panel (``--panel``; never a held-out / AgentEval task, splits as the panel lists)."""
    with psycopg.connect(dsn_for(item.rsplit("_", 1)[0]), row_factory=dict_row) as c:
        if item in S.get("panel", {}):
            # Pool-profiling samples (user 2026-10-19) also draw valid_seen / valid_unseen (ALFWorld) and
            # dev / test-fold (ScienceWorld) tasks: the panel file lists its splits (default train only).
            row = c.execute(
                "SELECT * FROM task WHERE key = %s AND split = ANY(%s) AND NOT held_out "
                "AND agenteval_item IS NULL",
                (S["panel"][item], list(S.get("panel_splits", ["train"]))),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT * FROM task WHERE agenteval_item = %s", (item,)
            ).fetchone()
    if row is None:
        raise ValueError(f"{item}: no task row")
    row.pop("emb", None)
    return row


V2_OUTCOME_POLICY = {
    "success_only": mp.SUCCESS_ONLY,
    "success_failure": mp.SUCCESS_FAILURE,
    "success_failure_capped": mp.SUCCESS_FAILURE,
}


RETRIEVAL_SLOT_DIR = Path("/tmp/trimem-retrieval-slots")


class retrieval_slot:
    """Cross-process limiter for v2 retrievals (pool-profiling round runs many arms in parallel; PG 55450 has
    max_connections = 32): ``$TRIMEM_RETRIEVAL_SLOTS`` (default 16) lock files, one held per in-flight
    retrieval (its adapter connection + store writes). Unset slots -> the default."""

    def __init__(self):
        self.n = max(1, int(os.environ.get("TRIMEM_RETRIEVAL_SLOTS", "16")))
        self.f = None

    def __enter__(self):
        import fcntl
        import random

        RETRIEVAL_SLOT_DIR.mkdir(parents=True, exist_ok=True)
        order = list(range(self.n))
        random.shuffle(order)
        while True:
            for i in order:
                f = open(RETRIEVAL_SLOT_DIR / f"{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(0.2)

    def __exit__(self, *exc):
        self.f.close()  # closing releases the flock
        return False


SEMANTIC_ROUTING_ENV = (
    "TRIMEM_SEMANTIC_ROUTING"  # "request-load" = request-load-v1 (as the τ² pool run)
)
SEMANTIC_BASES_ENV = (
    "TRIMEM_SEMANTIC_BASES"  # comma-separated vLLM bases (default 8030,8031)
)
SEMANTIC_ROUTING = "request-load-v1"
_SEM_POOL: dict = {}


def semantic_pool():
    """Process-local ``SemanticRequestPool`` (request-load-v1, user 2026-10-21: ALF/Sci pool round as τ²) when
    ``$TRIMEM_SEMANTIC_ROUTING=request-load``, else None. SemFilter judge calls then go to the endpoint with the
    fewest running + waiting requests; the agent calls stay on the episode's (half's) endpoint. Created lazily in
    the worker process (its poll thread must not cross the pool fork)."""
    if os.environ.get(SEMANTIC_ROUTING_ENV, "").strip() != "request-load":
        return None
    pid = os.getpid()
    if _SEM_POOL.get("pid") != pid:
        from trimem.experiments.tau2.semantic_pool import SemanticRequestPool

        bases = os.environ.get(
            SEMANTIC_BASES_ENV, "http://127.0.0.1:8030,http://127.0.0.1:8031"
        ).split(",")
        _SEM_POOL.update(pid=pid, pool=SemanticRequestPool(bases).start())
    return _SEM_POOL["pool"]


def retrieve_v2(
    q: dict,
    text: str,
    vector: str,
    t0: float,
    cfg: dict,
    *,
    item: str,
    protocol_extra: dict,
    store=None,
    step_index: int = 0,
    target_override: str | None = None,
) -> dict:
    """Session space v2 retrieval (pool profiling, user 2026-10-19; ``session_space_agentenv_v2``):
    SessionConfigV2 (pool cap, similarity gate, failed-card cap), the configuration's outcome policy
    (success_only -> memory policy success-only-v0: only successful sessions, unlabelled cards, rubric v1;
    success_failure[_capped] -> success+failure-v1), its SemFilter threshold, and its memory form (space -v4,
    ``memory_forms``: raw / compact / distilled = offline per-session sidecar cards, no per-episode LLM call;
    pre-v4 configurations: raw, or a query-time SemExtract block whose LLM tokens / wall time count as
    semantic tokens / retrieval latency).
    ``cfg`` = space v2 ``runner_args`` form. Returns the same record as ``retrieve_memory``."""
    from trimem.experiments.agentenv_sem_extract import AgentEnvSemExtract
    from trimem.experiments.session_retrieval import SessionConfigV2

    v2, plan = cfg["v2"], cfg["plan"]
    store = store or S["store"]
    run_policy = S["memory_policy"]
    if run_policy != mp.SUCCESS_FAILURE:
        raise SystemExit(
            "session space v2 runs under memory policy success+failure-v1 (the store admits both "
            "outcomes; each configuration's outcome_policy restricts it)"
        )
    policy = V2_OUTCOME_POLICY[v2["outcome_policy"]]
    config = SessionConfigV2(
        plan=plan,
        edges=tuple(cfg["edges"].split("+")),
        **cfg["memory_config"],
        pool_cap=v2["pool_cap"],
        expand_limit=v2["expand_limit"],
        sim_gate=v2["sim_gate"],
        max_failed=v2["max_failed"],
        graph_impl=v2.get("graph_impl", "BFS"),
        beam_width=v2.get("beam_width"),
    )
    # periodic injection (engine extension v0.1.0) re-retrieves against the agent's recent state
    target = target_override or q.get("initial_observation") or text
    rubric = mp.rubric(policy)
    workers = max(
        1, int(os.environ.get("TRIMEM_SEMANTIC_WORKERS", "4"))
    )  # judge calls in flight per episode
    # semfilter-v2 (user 2026-10-22): OPT-IN via $TRIMEM_SEMFILTER_V2=1 (unset = the v1 judge below, unchanged):
    # top-N cosine prefilter + short {id, useful, score} verdict + compact / distilled judge input.
    sf2 = semfilter_v2.settings() if plan in SEMFILTER_PLANS else None
    sf2_view = sf2_proto = None
    if sf2 is not None:
        sem, sf2_view, sf2_proto = semfilter_v2.build(
            q["env"],
            target,
            sf2,
            base=S["semantic_base"],
            version=rubric,
            workers=workers,
            threshold=v2["semfilter_threshold"],
            sem_fallback=v2.get("sem_fallback", "none"),
        )
    else:
        sem = (
            RubricSemFilter(
                q["env"],
                target,
                env_card,
                base=S["semantic_base"],
                version=rubric,
                workers=workers,
                threshold=v2["semfilter_threshold"],
            )
            if plan
            in SEMFILTER_PLANS  # P2 / P2+Rank / P1+Rank (plan-p1rank-v1: P1+Rank filters the seeds)
            else None
        )
    routing = {}
    if sem is not None and (pool := semantic_pool()) is not None:
        sem.endpoint_pool = pool
        routing = {"semantic_routing": SEMANTIC_ROUTING}
    rank_impl = v2.get("rank_impl", "ColBERT")
    ranker = None
    if plan in RANK_PLANS:
        if rank_impl == "ColBERT":
            ranker = ColBERTRank(target, mp.rank_view(env_card), base=S["rank_base"])
        elif rank_impl == "cross-encoder":
            from trimem.experiments.rank_impls import CrossEncoderRank

            ranker = CrossEncoderRank(
                target,
                mp.rank_view(env_card),
                base=os.environ.get("TRIMEM_CROSS_ENCODER_BASE", "http://127.0.0.1:8050"),
            )
        elif rank_impl == "LLM":
            from trimem.experiments.rank_impls import LLMPointwiseRank

            ranker = LLMPointwiseRank(
                target, mp.rank_view(env_card), base=S["semantic_base"], workers=workers
            )
        else:
            raise SystemExit(f"Unsupported rank_impl {rank_impl!r}")
    meta = {
        "experiment_id": S["experiment"],
        "dataset": q["env"],
        "split": S["split"],
        "query_id": item,
        "query_context": {"task_key": q["key"], "goal": q["description"]},
        "protocol": {
            "llm_budget": llm_budget.as_dict(),
            "selection": "task-session-full-pool-v2",
            "action_format": S["action_format"],
            "isolation": "L1",
            "semantic": (sf2_proto["rubric"] if sf2_proto else rubric) if sem else None,
            "rank": (COLBERT if rank_impl == "ColBERT" else ranker.version)
            if ranker
            else None,
            **mp.protocol(run_policy),
            "session_v2": v2,
            "retrieval_outcome_policy": policy,
            **routing,
            **({"semfilter": sf2_proto} if sf2_proto is not None else {}),
            **protocol_extra,
        },
    }
    kw = {}
    if sem is not None:
        kw["semfilter"] = sem
    if sf2 is not None:
        kw["prefilter_top_n"] = sf2["top_n"]
    if (
        v2.get("sem_fallback", "none") != "none"
    ):  # semfilter-v2b (WebShop, configuration knob; absent = none)
        if sf2 is None:
            raise SystemExit(
                "sem_fallback fill-topk needs semfilter-v2 (TRIMEM_SEMFILTER_V2=1)"
            )
        kw["sem_fallback"] = v2["sem_fallback"]
    if ranker is not None:
        kw["semrank"] = ranker
    # retrieval-slot-v2 (user 2026-10-21): the slot (= this retrieval's PG 55450 connection) is held only for
    # the DB phases (seeds, expansion, card hydration, store writes); it is freed and the adapter connection
    # closed while SemFilter / ColBERT calls run, and re-acquired lazily for any later DB work.
    with RetrievalLease(
        retrieval_slot, lambda: readonly_connect(dsn_for(q["env"]))
    ) as lease:
        gstore = lease.gated(store)
        adapter = lease.bind(AgentEnvTaskAdapter(vector, q, policy=policy))
        adapter.widen_seeds = True  # v2: widen until seed_count seed tasks with an admissible session are found
        try:
            receipt = recorded_retrieve(
                adapter, config, gstore, meta, pause=lease.pause, **kw
            )
            receipt = adapter.enrich_receipt(receipt)
            if sf2 is not None:
                receipt["semfilter"] = semfilter_v2.receipt_block(
                    sf2_proto, sem, sf2_view, receipt
                )
            gstore.patch(
                receipt["retrieval_id"],
                status="retrieved_pending_evaluation",
                receipt=receipt,
            )
        finally:
            adapter.close()
    distill_tokens = 0
    cap = llm_budget.per_card(config.top_k) * 3  # ~3 chars per token, conservative
    unit = v2.get("retrieval_unit", "whole")
    if unit == "whole" and v2.get("whole_card_cap") == "none":
        cap = None  # user 2026-11-03: whole cards are never cut (render_card cap_chars=None)
    unit_calls = []
    if unit != "whole":
        # engine extension v0.1.0: window / action-level units (raw cards only)
        if v2.get("memory_form", "raw") != "raw":
            raise SystemExit("retrieval_unit window/actions requires memory_form raw")
        from trimem.experiments.retrieval_units import render_unit

        step_ranker = None
        if unit == "actions":
            step_ranker = ColBERTRank(target, lambda c: c["text"], base=S["rank_base"])
        memory, unit_receipt = render_unit(
            q["env"], receipt["cards"], unit, cap, policy,
            step_index=step_index, step_ranker=step_ranker,
        )
        unit_calls = list(step_ranker.calls) if step_ranker else []
        receipt_extra = {"retrieval_unit": unit_receipt}
    elif v2.get(
        "memory_forms_version"
    ):  # space -v4: raw / compact / distilled (offline sidecar), no LLM call
        from trimem.experiments import memory_forms as mf

        form = v2["memory_form"]
        dstore = (
            mf.DistilledStore.for_dataset(q["env"]) if form == "distilled" else None
        )
        memory, form_receipt = mf.render_memory_agentenv(
            q["env"], receipt["cards"], form, cap, policy, dstore
        )
        receipt_extra = {
            "memory_form": form_receipt
            | {"memory_tokens": mf.count_tokens(memory) if memory else 0}
        }
    elif (
        v2["memory_form"] == "distilled"
    ):  # pre-v4 configurations: query-time SemExtract (legacy)
        ex = AgentEnvSemExtract(q["env"], target, env_card, base=S["semantic_base"])
        ex.open(receipt["cards"])
        memory = ex.next() or ""
        ex.close()
        distill_tokens = semantic_tokens(ex.calls)
        receipt_extra = {"sem_extract": ex.receipt, "sem_extract_calls": ex.calls}
    else:  # pre-v4 raw / success_only_raw
        memory = render_memory(q["env"], receipt["cards"], cap, policy)
        receipt_extra = {}
    sem_calls = sem.calls if sem else []
    rank_calls = ranker.calls if rank_impl == "LLM" and ranker is not None else []
    return {
        "rid": receipt["retrieval_id"],
        "memory": memory,
        "retrieval_ms": (time.monotonic() - t0) * 1000,
        "semantic_tokens": semantic_tokens(sem_calls) + semantic_tokens(rank_calls) + distill_tokens,
        "rank_impl": rank_impl,
        "unit_calls": len(unit_calls),
        "selected": receipt["selected_ids"],
        "memory_policy": policy,
        "card_outcomes": mp.card_outcomes(receipt["cards"]),
        "v2": {
            "gated": receipt.get("gated"),
            "pool_size": receipt.get("pool_size"),
            "pool_cap_hit": receipt.get("pool_cap_hit"),
            "expansion_truncated": receipt.get("expansion_truncated"),
            "best_cosine": receipt.get("best_cosine"),
            "distill_tokens": distill_tokens,
            **receipt_extra,
            **(
                {"semfilter": receipt["semfilter"]}
                if receipt.get("semfilter") is not None
                else {}
            ),
        },
    }


def retrieve_memory(data_idx: int, args: dict) -> dict:
    item = f"{args['task_name']}_{data_idx}"
    q = query_task(item)
    # external memory baseline (external_baselines.runner)
    if S.get("external") is not None:
        from trimem.experiments.external_baselines import runner as external

        return external.retrieve(q, item, S)
    text = f"{q['env']}: {q['description'] or q['key']}"  # the store's task-vector text format
    t0 = time.monotonic()
    r = httpx.post(EMBED, json={"model": EMBED_MODEL, "input": [text]}, timeout=120)
    r.raise_for_status()
    vector = json.dumps(r.json()["data"][0]["embedding"])
    if S.get("space_v2") is not None:
        return retrieve_v2(
            q,
            text,
            vector,
            t0,
            S["space_v2"],
            item=item,
            protocol_extra=S.get("protocol_extra", {}),
        )
    policy = S["memory_policy"]
    adapter = AgentEnvTaskAdapter(vector, q, policy=policy)
    target = q.get("initial_observation") or text
    sem, ranker, plan = None, None, S["config"].plan
    rubric = mp.rubric(policy)
    if plan in ("P1", "P2", "P2+Rank"):
        sem = RubricSemFilter(
            q["env"], target, env_card, base=S["semantic_base"], version=rubric
        )
    elif plan == "VGR+SemRank":
        sem = RubricSemRank(
            q["env"], target, env_card, base=S["semantic_base"], version=rubric
        )
    if plan in ("VGR+Rank", "P2+Rank"):
        ranker = ColBERTRank(target, mp.rank_view(env_card), base=S["rank_base"])
    meta = {
        "experiment_id": S["experiment"],
        "dataset": q["env"],
        "split": S["split"],
        "query_id": item,
        "query_context": {"task_key": q["key"], "goal": q["description"]},
        "protocol": {
            "llm_budget": llm_budget.as_dict(),
            "selection": "task-session-full-pool-v2",
            "action_format": S["action_format"],
            "isolation": "L1",
            "semantic": rubric if sem else None,
            "rank": COLBERT if ranker else None,
            **mp.protocol(policy),
            **S.get("protocol_extra", {}),
        },
    }
    try:
        kw = {}
        if plan == "VGR+SemRank":
            kw["semrank"] = sem
        elif sem is not None:
            kw["semfilter"] = sem
        if ranker is not None:
            kw["semrank"] = ranker
        receipt = recorded_retrieve(adapter, S["config"], S["store"], meta, **kw)
        receipt = adapter.enrich_receipt(receipt)
        S["store"].patch(
            receipt["retrieval_id"],
            status="retrieved_pending_evaluation",
            receipt=receipt,
        )
    finally:
        adapter.close()
    cap = llm_budget.per_card(S["config"].top_k) * 3  # ~3 chars per token, conservative
    memory = render_memory(q["env"], receipt["cards"], cap, policy)
    sem_calls = sem.calls if sem else []
    return {
        "rid": receipt["retrieval_id"],
        "memory": memory,
        "retrieval_ms": (time.monotonic() - t0) * 1000,
        "semantic_tokens": semantic_tokens(sem_calls),
        "selected": receipt["selected_ids"],
        "memory_policy": policy,
        "card_outcomes": mp.card_outcomes(receipt["cards"]),
    }


def quality_of(task_name: str, reward: float) -> float:
    """Simulator quality in [0, 1]: ScienceWorld score/100 (a negative score is a failed task); ALFWorld won;
    WebShop the official reward (attribute / option / price match of the purchase, 1.0 = success)."""
    return max(reward, 0) / 100 if task_name == "sciworld" else float(reward)


METRIC = {"alfworld": "won", "sciworld": "score/100", "webshop": "reward"}


PERIODIC_OBS_CHARS = 400


def periodic_query(q: dict, conversation: list, n: int) -> tuple[str, str]:
    """(query text, latest observation): task + the last ``n`` (action, observation) steps."""
    turns = [m for m in conversation if m.get("role") in ("assistant", "user")]
    pairs, last_obs = [], ""
    for i, m in enumerate(turns):
        if m["role"] == "assistant":
            obs = turns[i + 1]["content"] if i + 1 < len(turns) else ""
            pairs.append((m["content"], obs))
    if conversation and conversation[-1].get("role") == "user":
        last_obs = conversation[-1]["content"]
    recent = pairs[-n:]
    lines = [
        f"Action: {' '.join(str(a).split())[:200]}\nObservation: {' '.join(str(o).split())[:PERIODIC_OBS_CHARS]}"
        for a, o in recent
    ]
    text = f"{q['env']}: {q['description'] or q['key']}\nRecent steps:\n" + "\n".join(lines)
    return text, last_obs


def periodic_hook(conversation: list) -> None:
    """Engine extension v0.1.0: every ``interval`` agent steps, re-retrieve against the agent's
    recent state and append the memory block to the newest environment observation. Runs inside
    the episode wall clock; its retrieval latency and tokens are added to the episode record."""
    st = S.get("periodic")
    if not st or not conversation or conversation[-1].get("role") != "user":
        return
    steps = sum(1 for m in conversation if m.get("role") == "assistant")
    if steps < st["next"]:
        return
    st["next"] += st["interval"]
    text, last_obs = periodic_query(st["q"], conversation, st["interval"])
    t0 = time.monotonic()
    r = httpx.post(EMBED, json={"model": EMBED_MODEL, "input": [text]}, timeout=120)
    r.raise_for_status()
    vector = json.dumps(r.json()["data"][0]["embedding"])
    mem = retrieve_v2(
        st["q"], text, vector, t0, S["space_v2"], item=st["item"],
        protocol_extra={**S.get("protocol_extra", {}),
                        "injection": {"timing": "periodic", "step": steps, "parent": st["rid"]}},
        step_index=steps, target_override=last_obs or text,
    )
    S["store"].patch(mem["rid"], status="periodic_injection")
    if mem["memory"]:
        conversation[-1]["content"] = conversation[-1]["content"] + mem["memory"]
    st["records"].append({"step": steps, "rid": mem["rid"], "retrieval_ms": mem["retrieval_ms"],
                          "semantic_tokens": mem["semantic_tokens"], "memory_chars": len(mem["memory"]),
                          "selected": mem["selected"]})


def install_periodic_hook() -> None:
    """Wrap whatever generate is installed (the worker's strict generate included)."""
    inner = base.NoThinkingAPIAgent.generate
    if getattr(inner, "_periodic_wrapped", False):
        return

    def generate(self, conversation):
        periodic_hook(conversation)
        return inner(self, conversation)

    generate._periodic_wrapped = True
    base.NoThinkingAPIAgent.generate = generate


def process_with_memory(data_idx: int, args: dict):
    if S.get("protocol_extra", {}).get("holdout_identity"):
        # The holdout ledger admits one episode attempt, including failures.
        args = {**args, "episode_fail_fast": True}
    out = Path(args["output_dir"]) / f"{args['task_name']}_{data_idx}.json"
    if out.exists():  # official resume semantics
        return S["official_process"](data_idx, args)
    mem = retrieve_memory(data_idx, args)
    (Path(args["output_dir"]) / "memory").mkdir(exist_ok=True)
    (Path(args["output_dir"]) / "memory" / f"{data_idx}.json").write_text(
        json.dumps(mem, ensure_ascii=False)
    )
    mem_sha = hashlib.sha256(mem["memory"].encode()).hexdigest()
    mem_key = memory_key_of(mem)
    if S.get("reuse"):
        hit = reused_episode(f"{args['task_name']}_{data_idx}", mem_sha, mem_key)
        if hit is not None:
            return reuse(hit, mem, mem_sha, out, data_idx, args)
    if S.get("reuse_from"):
        hit = reused_from_episode(f"{args['task_name']}_{data_idx}", mem_sha, mem_key)
        if hit is not None:
            return reuse(hit, mem, mem_sha, out, data_idx, args)
    client = official.evaluator.tasks[0].clients[0]
    orig = client.conversation_start
    first = dict(orig[0])
    first["value"] = first["value"] + mem["memory"]
    client.conversation_start = (first, *orig[1:])
    v2opts = (S.get("space_v2") or {}).get("v2") or {}
    initial_retrieval_ms = mem["retrieval_ms"]
    if v2opts.get("injection_timing", "start") == "periodic":
        interval = int(v2opts["retrieval_interval"])
        S["periodic"] = {"interval": interval, "next": interval, "q": query_task(f"{args['task_name']}_{data_idx}"),
                         "item": f"{args['task_name']}_{data_idx}", "rid": mem["rid"], "records": []}
    base.CURRENT_ITEM = data_idx
    t0 = time.monotonic()
    try:
        score, success = S["official_process"](data_idx, args)
    finally:
        client.conversation_start = orig
        base.CURRENT_ITEM = None
        periodic = S.pop("periodic", None)
    wall_ms = (time.monotonic() - t0) * 1000
    if periodic is not None:
        # All retrievals' cost belongs to the episode. wall_ms already contains the periodic
        # retrievals (they run inside the episode); retrieval_ms / semantic_tokens report totals.
        mem["periodic"] = periodic["records"]
        mem["semantic_tokens"] += sum(r["semantic_tokens"] for r in periodic["records"])
        mem["retrieval_ms"] += sum(r["retrieval_ms"] for r in periodic["records"])
        (Path(args["output_dir"]) / "memory" / f"{data_idx}.json").write_text(
            json.dumps(mem, ensure_ascii=False)
        )
    calls = (
        [json.loads(x) for x in open(base.CALL_LOG)] if base.CALL_LOG.exists() else []
    )
    mine = [
        c for c in calls if c.get("item") == data_idx and c.get("pid") == os.getpid()
    ]
    if os.environ.get("TRIMEM_ABLATION_RESOURCE_LEDGER") and any(
        c.get("prompt_tokens") is None or c.get("completion_tokens") is None for c in mine
    ):
        raise SystemExit("agent token usage unavailable; ablation paused")
    gen_tokens = sum(
        (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in mine
    )
    from trimem.experiments.agent_step_metrics import annotate_agentgym_result

    steps = annotate_agentgym_result(out)
    reward = json.load(open(out))["reward"]
    # ScienceWorld scores can be negative (wrong focus = -100); a negative score is a failed task.
    quality = quality_of(args["task_name"], reward)
    store = S["store"]
    store.generation(
        mem["rid"],
        {
            **steps,
            "reward": reward,
            "llm_calls": len(mine),
            "generation_tokens": gen_tokens,
            "semantic_tokens": mem["semantic_tokens"],
            "episode_ms": wall_ms,
            "retrieval_ms": mem["retrieval_ms"],
            "memory_chars": len(mem["memory"]),
            "memory_sha256": mem_sha,
            "memory_key": mem_key,
            "result_file": str(out.resolve()),
            **episode_policy(mem),
        },
        total_tokens=gen_tokens + mem["semantic_tokens"],
        # periodic retrievals are inside wall_ms already: add only the initial one
        e2e_ms=wall_ms + initial_retrieval_ms,
    )
    store.evaluate_scalar(
        mem["rid"],
        quality,
        {"reward": reward, "metric": METRIC[args["task_name"]]},
        S["baseline"].get(data_idx),
    )
    return score, success


def memory_key_of(mem: dict) -> str:
    """``quality_reuse.memory_key``: the exact injected text + memory_form / render version (v2 receipt)."""
    form = (mem.get("v2") or {}).get("memory_form") or {}
    return qr.memory_key(
        mem["memory"],
        form.get("memory_form", "legacy"),
        form.get("memory_forms_version", ""),
    )


def episode_policy(mem: dict) -> dict:
    """Per-episode memory policy and successful / failed card counts (opt ``generation`` + memory file);
    session space v2 runs also record gate / pool-cap / distillation facts (``v2``)."""
    out = {
        "memory_policy": mem.get("memory_policy"),
        "card_outcomes": mem.get("card_outcomes"),
    }
    if mem.get("v2") is not None:
        out["v2"] = {
            k: v for k, v in mem["v2"].items() if k not in ("sem_extract_calls",)
        }
    return out


# The memory policy is part of every reuse key (rows that predate it are success-only-v0).
POLICY_SQL = "AND coalesce(protocol->>'memory_policy', 'success-only-v0') = %s "
REUSE_SQL = (
    "SELECT retrieval_id, generation, total_tokens, e2e_ms, quality, judge, protocol->'space_config' AS "
    "source_config FROM opt.session_retrieval "
    "WHERE experiment_id = %s AND split = %s AND query_id = %s AND status = 'evaluated' "
    "AND generation->>'memory_sha256' = %s AND generation->>'reused_from' IS NULL "
    "AND generation->>'memory_key' = %s "  # quality-reuse-v1: + memory_form / render version
    "AND protocol->>'panel_hash' = %s " + POLICY_SQL + "ORDER BY created_at LIMIT 1"
)


# latency-regime-v1 (user 2026-10-22, ALFWorld v7 c128 -> c16 from round 6): a reused episode carries its source's
# episode wall time, so sources are restricted to rows created at / after the regime start (ISO time; set by
# bo_pool_driver from state.json latency_regime). Unset = every source (unchanged behaviour).
REUSE_SINCE_ENV = "TRIMEM_REUSE_SINCE"


def reuse_sql() -> tuple[str, tuple]:
    """REUSE_SQL, restricted to sources created at / after ``$TRIMEM_REUSE_SINCE`` when it is set."""
    since = os.environ.get(REUSE_SINCE_ENV, "").strip()
    if not since:
        return REUSE_SQL, ()
    head, tail = REUSE_SQL.split("ORDER BY created_at", 1)
    return head + "AND created_at >= %s ORDER BY created_at" + tail, (since,)


# max-round-v1 (user 2026-10-23, WebShop max_round 10): a runner whose --max_round differs from the native 30 reuses
# only sources recorded with the same max_round (opt row protocol.max_round; absent = 30). At 30 the SQL and its
# parameters are unchanged (ALFWorld / ScienceWorld / earlier WebShop rounds: byte-identical).
NATIVE_MAX_ROUND = 30
MAX_ROUND_SQL = "AND coalesce(protocol->>'max_round', '30') = %s "


def with_max_round_sql(sql: str) -> tuple[str, tuple]:
    """``sql`` + the max_round clause (inserted before the memory-policy clause) and its parameter, for a runner at
    ``S['max_round']`` != 30; else ``(sql, ())``."""
    n = S.get("max_round")
    if n is None or int(n) == NATIVE_MAX_ROUND:
        return sql, ()
    head, tail = sql.split(POLICY_SQL, 1)
    return head + MAX_ROUND_SQL + POLICY_SQL + tail, (str(int(n)),)


def reused_episode(item: str, mem_sha: str, mem_key: str) -> dict | None:
    """Quality reuse (RAG-Stack ``gen_fingerprint_dedup``; user decision 2026-10-16: ScienceWorld only):
    an earlier ORIGINAL episode of the same BO experiment, panel and query whose injected memory text is
    byte-identical has the same prompt, so its outcome is reused instead of re-running the agent."""
    pe = S.get("protocol_extra", {})
    sql, extra = reuse_sql()
    sql, mr = with_max_round_sql(sql)
    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        row = c.execute(
            sql,
            (
                S["experiment"],
                S["split"],
                item,
                mem_sha,
                mem_key,
                pe.get("panel_hash"),
                *mr,
                S["memory_policy"],
                *extra,
            ),
        ).fetchone()
    if row is None or not Path(row["generation"].get("result_file", "")).exists():
        return None
    return row


# Cross-experiment reuse (user 2026-10-17, train500 profiling): ORIGINAL training episodes of the listed
# source experiments (e.g. the finished train100 BO run) for the same query, under the same llm_budget
# protocol and action format, never a noise repeat. The memory identity is checked in Python because older
# rows (ALFWorld train100) predate generation.memory_sha256 / result_file: their memory file (which carries
# the retrieval id) is found under the source run's config directories (``--reuse_from_dirs``), and the
# result file next to it is the episode.
REUSE_FROM_SQL = (
    "SELECT retrieval_id, experiment_id, generation, total_tokens, e2e_ms, quality, judge "
    "FROM opt.session_retrieval WHERE experiment_id = ANY(%s) AND split = 'train' AND query_id = %s "
    "AND status = 'evaluated' AND generation->>'reused_from' IS NULL "
    "AND protocol->'llm_budget' = %s::jsonb AND protocol->>'action_format' = %s "
    + POLICY_SQL
    + "AND experiment_id NOT LIKE '%%:noise-repeat' ORDER BY created_at"
)


def _memory_files(item: str, dirs: tuple[str, ...]) -> dict[str, Path]:
    """retrieval id -> memory file of ``item`` under the source config directories."""
    idx = item.rsplit("_", 1)[1]
    out = {}
    for root in dirs:
        for mf in Path(root).glob(f"*/memory/{idx}.json"):
            try:
                out[str(json.load(open(mf))["rid"])] = mf
            except (ValueError, KeyError):
                continue
    return out


def source_episode(
    row: dict, item: str, dirs: tuple[str, ...] = ()
) -> tuple[str | None, Path | None]:
    """(memory sha256, result file) of a source row; (None, None) when either cannot be established."""
    g = row["generation"] or {}
    rf = Path(g["result_file"]) if g.get("result_file") else None
    if g.get("memory_sha256") and rf is not None:
        return g["memory_sha256"], rf
    idx = item.rsplit("_", 1)[1]
    mf = (
        rf.parent / "memory" / f"{idx}.json"
        if rf is not None
        else _memory_files(item, dirs).get(str(row["retrieval_id"]))
    )
    if mf is None or not mf.exists():
        return None, None
    m = json.load(open(mf))
    if str(m.get("rid")) != str(row["retrieval_id"]) or "memory" not in m:
        return None, None
    return (
        g.get("memory_sha256") or hashlib.sha256(m["memory"].encode()).hexdigest(),
        rf or mf.parent.parent / f"{item}.json",
    )


def pick_reuse_source(
    rows: list[dict],
    mem_sha: str,
    item: str,
    dirs: tuple[str, ...] = (),
    mem_key: str | None = None,
) -> dict | None:
    """First (oldest) source row whose injected memory is byte-identical and whose result file exists and
    belongs to the item. Returned as a copy with ``generation.result_file`` set (older rows lack it). With
    ``mem_key`` (quality-reuse-v1 memory key: memory_form + render version + text), a source row that records a
    different memory_key is skipped (rows that predate memory_key are matched on the text alone, as before)."""
    for row in rows:
        rk = (row["generation"] or {}).get("memory_key")
        if mem_key is not None and rk is not None and rk != mem_key:
            continue
        sha, rf = source_episode(row, item, dirs)
        if sha != mem_sha or rf is None or not rf.exists():
            continue
        try:
            if json.load(open(rf)).get("item_id") != item:
                continue
        except ValueError:
            continue
        return {
            **row,
            "generation": {**(row["generation"] or {}), "result_file": str(rf)},
        }
    return None


def reused_from_episode(
    item: str, mem_sha: str, mem_key: str | None = None
) -> dict | None:
    sql, mr = with_max_round_sql(REUSE_FROM_SQL)
    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        rows = c.execute(
            sql,
            (
                list(S["reuse_from"]),
                item,
                json.dumps(llm_budget.as_dict()),
                S["action_format"],
                *mr,
                S["memory_policy"],
            ),
        ).fetchall()
    return pick_reuse_source(
        rows, mem_sha, item, tuple(S.get("reuse_from_dirs", ())), mem_key
    )


def reuse(hit: dict, mem: dict, mem_sha: str, out: Path, data_idx: int, args: dict):
    src = json.load(open(hit["generation"]["result_file"]))
    if src.get("item_id") != f"{args['task_name']}_{data_idx}":
        raise ValueError("reuse source belongs to another item")
    g = hit["generation"]
    from trimem.experiments.agent_step_metrics import agentgym_steps

    steps = agentgym_steps(src)
    store = S["store"]
    # Same episode outcome and generation cost; this configuration's own retrieval cost is recorded.
    store.generation(
        mem["rid"],
        {
            **steps,
            **{
                k: g[k]
                for k in ("reward", "llm_calls", "generation_tokens", "episode_ms")
            },
            "semantic_tokens": mem["semantic_tokens"],
            "retrieval_ms": mem["retrieval_ms"],
            "memory_chars": len(mem["memory"]),
            "memory_sha256": mem_sha,
            "memory_key": memory_key_of(mem),
            "result_file": str(out.resolve()),
            "reused_from": str(hit["retrieval_id"]),
            "reused_from_file": g.get("result_file"),
            **episode_policy(mem),
            **(
                {"reused_from_experiment": hit["experiment_id"]}
                if hit.get("experiment_id") not in (None, S["experiment"])
                else {}
            ),
        },
        total_tokens=g["generation_tokens"] + mem["semantic_tokens"],
        e2e_ms=g["episode_ms"] + mem["retrieval_ms"],
    )
    reward = src["reward"]
    quality = quality_of(args["task_name"], reward)
    store.evaluate_scalar(
        mem["rid"],
        quality,
        {
            "reward": reward,
            "metric": METRIC[args["task_name"]],
            "reused_from": str(hit["retrieval_id"]),
        },
        S["baseline"].get(data_idx),
    )
    tmp = out.with_suffix(
        ".reuse.tmp"
    )  # result file last: completion = file + evaluated row
    rec = {  # reuse record (quality-reuse-v1): the source's outcome + where it came from
        **src,
        **steps,
        "reuse": True,
        "reuse_protocol": qr.PROTOCOL,
        "memory_key": memory_key_of(mem),
        "memory_sha256": mem_sha,
        "reused_from": {
            "path": g.get("result_file"),
            "retrieval_id": str(hit["retrieval_id"]),
            "config": hit.get("source_config"),
            "experiment": hit.get("experiment_id") or S["experiment"],
        },
    }
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.replace(out)
    print(
        f"[quality-reuse] hit {args['task_name']}_{data_idx} <- {hit['retrieval_id']}",
        flush=True,
    )
    return src["reward"], src["success"]


def main() -> None:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--plan",
        required=True,
        choices=("VGR", "P1", "P2", "VGR+SemRank", "VGR+Rank", "P2+Rank", "P1+Rank"),
    )
    extra.add_argument(
        "--memory_config", default='{"seed_count":8,"depth":1,"fanout":4,"top_k":3}'
    )
    extra.add_argument("--split", default="test", choices=("train", "test", "smoke"))
    extra.add_argument("--baseline_dir", required=True)
    extra.add_argument("--experiment", default=None)
    extra.add_argument("--semantic_base", default="http://127.0.0.1:8031")
    extra.add_argument("--action_format", default="react")
    extra.add_argument(
        "--rank_base", default=None, help="ColBERT server (default: LLM port + 10)"
    )
    extra.add_argument(
        "--edges",
        default="similar",
        help="session edges joined by '+', e.g. similar+similar_up",
    )
    extra.add_argument(
        "--panel", default=None, help="frozen training panel (item_id -> task_key)"
    )
    extra.add_argument(
        "--protocol_extra",
        default="{}",
        help="JSON merged into every opt row's protocol",
    )
    extra.add_argument(
        "--reuse_identical_memory",
        action="store_true",
        help="reuse the outcome of an earlier episode with byte-identical memory (same experiment)",
    )
    extra.add_argument(
        "--reuse_from",
        default="",
        help="comma-separated source experiment ids (training only): reuse an ORIGINAL episode of "
        "the same query with byte-identical memory, same llm_budget and action format",
    )
    extra.add_argument(
        "--memory_policy",
        default=None,
        choices=mp.POLICIES,
        help=f"which stored sessions memory may inject (default ${mp.ENV_VAR} or {mp.DEFAULT})",
    )
    extra.add_argument(
        "--reuse_from_dirs",
        default="",
        help="comma-separated config directories of the source runs (memory files of rows that "
        "predate generation.result_file)",
    )
    extra.add_argument(
        "--space_v2",
        default=None,
        help="session space v2 options (JSON: sim_gate, memory_form, outcome_policy, "
        "semfilter_threshold, pool_cap, expand_limit, max_failed); see retrieve_v2",
    )
    extra.add_argument(
        "--external",
        default=None,
        help="external memory baseline (JSON: system, url, trace_label, params); replaces TriMEM "
        "retrieval, see trimem.experiments.external_baselines.runner",
    )
    known, rest = extra.parse_known_args()
    if known.external and (
        known.reuse_identical_memory or known.reuse_from or known.space_v2
    ):
        raise SystemExit("--external runs without reuse and without --space_v2")
    reuse_from = tuple(x for x in known.reuse_from.split(",") if x)
    reuse_from_dirs = tuple(x for x in known.reuse_from_dirs.split(",") if x)
    if reuse_from and (
        known.split != "train"
        or (known.experiment or "").endswith(":noise-repeat")
        or any(x.endswith(":noise-repeat") for x in reuse_from)
    ):
        raise SystemExit(
            "--reuse_from is for training profiling only (never test arms or noise repeats)"
        )
    llm_base = next(rest[i + 1] for i, a in enumerate(rest) if a == "--base_url")
    # an external system's parameters live in --external (recorded per opt row)
    if known.external:
        cfg = None
    elif (
        known.space_v2
    ):  # session space v2 bounds (seed <= 128, depth <= 4, fanout <= 64, top_k <= 10)
        from trimem.experiments.session_retrieval import SessionConfigV2

        o = json.loads(known.space_v2)
        cfg = SessionConfigV2(
            plan=known.plan,
            edges=tuple(known.edges.split("+")),
            **json.loads(known.memory_config),
            pool_cap=o["pool_cap"],
            expand_limit=o["expand_limit"],
            sim_gate=o["sim_gate"],
            max_failed=o["max_failed"],
            graph_impl=o.get("graph_impl", "BFS"),
            beam_width=o.get("beam_width"),
        )
    else:
        cfg = SessionConfig(
            plan=known.plan,
            edges=tuple(known.edges.split("+")),
            **json.loads(known.memory_config),
        )
    if known.split == "train" and not known.panel:
        raise SystemExit(
            "--split train requires --panel (training items are looked up by task key)"
        )
    panel, panel_splits = {}, ["train"]
    if known.panel:
        pnl = json.load(open(known.panel))
        panel = {x["item_id"]: x["task_key"] for x in pnl["items"]}
        panel_splits = list(pnl.get("splits", ["train"]))
    baseline = {}
    for f in Path(known.baseline_dir).glob("*_*.json"):
        if f.stem.split("_")[-1].isdigit():
            d = json.load(open(f))
            idx = int(d["item_id"].split("_")[1])
            baseline[idx] = quality_of(d["item_id"].rsplit("_", 1)[0], d["reward"])
    task = next(rest[i + 1] for i, a in enumerate(rest) if a == "--task_name")
    policy = mp.validate(known.memory_policy or mp.current())
    if "--output_dir" not in rest:
        raise SystemExit("--output_dir is required")
    mp.claim_dir(
        Path(rest[rest.index("--output_dir") + 1]), policy
    )  # result dirs never mix policies
    space_v2 = None
    if known.space_v2:
        if known.plan not in V2_PLANS:
            raise SystemExit(f"--space_v2 supports the plans {sorted(V2_PLANS)}")
        space_v2 = {
            "plan": known.plan,
            "edges": known.edges,
            "memory_config": json.loads(known.memory_config),
            "v2": json.loads(known.space_v2),
        }
    S.update(
        space_v2=space_v2,
        external=json.loads(known.external) if known.external else None,
    )
    S.update(
        # the runner's --max_round (max-round-v1 reuse key; the official default when absent)
        max_round=int(rest[rest.index("--max_round") + 1])
        if "--max_round" in rest
        else None,
        panel=panel,
        panel_splits=panel_splits,
        protocol_extra=json.loads(known.protocol_extra),
        reuse=known.reuse_identical_memory,
        reuse_from=reuse_from,
        reuse_from_dirs=reuse_from_dirs,
    )
    S.update(
        config=cfg,
        split=known.split,
        baseline=baseline,
        semantic_base=known.semantic_base,
        action_format=known.action_format,
        store=ScalarSessionStore(OPT_DSN),
        rank_base=known.rank_base or rank_base_for(llm_base),
        memory_policy=policy,
        experiment=mp.tag(
            known.experiment
            or (
                f"{task}:agenteval200:session-v2:{known.plan}:{llm_budget.PROTOCOL}"
                + (f":{mp.rubric(policy)}" if known.plan != "VGR" else "")
            ),
            policy,
        ),
        official_process=official.process,
    )
    official.process = process_with_memory
    install_periodic_hook()
    sys.argv = [sys.argv[0], "--action_format", known.action_format, *rest]
    base.main()


if __name__ == "__main__":
    main()
