"""τ²-bench official orchestrator + TriMEM session memory (task seeds -> sessions -> steps).

Uses the pinned official ``LLMAgent`` prompt, user simulator, tools, orchestrator,
evaluator and per-trial seed derivation with local model/decoding settings. This is not a
reproduction of the released GPT-4.1 runs. Memory mode registers ``trimem_memory_agent`` (official agent +
``<memory>`` block, see ``agent``); at the customer's first message the agent retrieves once with
``session_retrieval.retrieve`` through ``Tau2TaskAdapter`` and ``recorded_retrieve`` (the same retriever
and opt store as SWE-QA-Pro / AgentGym). After the episode the generation cost and the official reward
(quality in [0, 1], paired with the No-Memory result of the same task) go to ``opt.session_retrieval``
via ``ScalarSessionStore``. No-Memory mode runs the official ``llm_agent`` and writes result files only.

Agent, user simulator and NL-assertion judge all run on one OpenAI-compatible endpoint through
litellm (``openai/<model>`` + ``api_base``), temperature 0, thinking disabled, ``llm_budget`` output cap.
Per-task result files make the run resumable (file written last; completion = file [+ evaluated row]).

    .venv-tau2/bin/python -m trimem.experiments.tau2.runner --mode nomemory --split train \\
        --out_dir results/tau2_20261016/nomemory_train
    .venv-tau2/bin/python -m trimem.experiments.tau2.runner --mode memory --plan VGR --split train \\
        --baseline_dir results/tau2_20261016/nomemory_train --out_dir results/tau2_20261016/vgr_train
"""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from trimem.experiments import llm_budget, rss_guard
from trimem.experiments import memory_policy as mp
from trimem.experiments.session_retrieval import SessionConfig
from trimem.experiments.tau2 import entities as ent
from trimem.experiments.tau2 import trajectories as tr
from trimem.experiments.tau2.adapter import DSN as STORE_DSN
from trimem.experiments.tau2.adapter import (
    ISOLATION_BY_POLICY,
    Tau2TaskAdapter,
    render_memory,
    tau2_card,
)

MODEL = "openai/Qwen/Qwen3.8-27B-FP8"
API_BASE = "http://127.0.0.1:8030/v1"
SEMANTIC_BASE = "http://127.0.0.1:8031"
EMBED = "http://127.0.0.1:8021/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
OPT_DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa"  # shared opt schema (= sweqa OPT_DSN)
DATASET = "tau2"
PLANS = ("VGR", "P1", "P2", "VGR+SemRank", "VGR+Rank", "P2+Rank")
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
TAU2_KNOWLEDGE = (
    "which lookup tools identified the customer and the record (user id via name + zip or e-mail, order or "
    "reservation details); the checks the policy requires before a write (explicit confirmation, eligibility "
    "such as cabin / fare / insurance rules, delivered vs pending status); which write tool resolved the "
    "request and with which kind of arguments; a telecom troubleshooting step and the device check that "
    "confirmed the fix; a refusal or transfer the policy requires"
)


CURRENT_DOMAIN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tau2_domain", default=None
)
# Episode-scoped state set by ``run_task`` in the episode's thread (episodes run on a thread pool).
CURRENT_TRIAL: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "tau2_trial", default=None
)
CURRENT_ENDPOINT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tau2_endpoint", default=None
)
# Quality-reuse claims (``quality_reuse.Claim``) the memory provider took in this episode: released by
# ``run_task`` after the result file is written (or the episode failed), so a waiting same-key episode reuses it.
EPISODE_CLAIMS: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "tau2_reuse_claims", default=None
)
# Within a trial, start the longest domain first so the arm's tail is made of short episodes.
LONGEST_FIRST = ("telecom", "retail", "airline")


class ReuseEpisode(Exception):
    """Raised by the memory provider when an earlier episode had byte-identical memory."""

    def __init__(self, hit: dict, mem: dict):
        super().__init__("identical memory: reuse earlier episode")
        self.hit, self.mem = hit, mem


# ---------------------------------------------------------------- pure helpers (unit-tested)


def llm_args(
    api_base: str,
    *,
    max_tokens: int = llm_budget.OUTPUT_TOKENS,
    json_mode: bool = False,
) -> dict:
    args = {
        "temperature": 0.0,
        "api_base": api_base,
        "api_key": "EMPTY",
        "max_tokens": max_tokens,
        "extra_body": dict(NO_THINKING),
    }
    if json_mode:
        args["response_format"] = {"type": "json_object"}
    return args


def trial_seed(seed: int, trial: int = 0) -> int:
    """Official per-trial seed: ``random.seed(config.seed); [randint(0, 1e6) for trials]``."""
    rnd = random.Random(seed)
    return [rnd.randint(0, 1000000) for _ in range(trial + 1)][trial]


def quality_of(reward: Any) -> float:
    """Official τ² reward (product of the task's reward-basis checks) as quality in [0, 1]."""
    q = float(reward)
    if math.isnan(q):
        raise ValueError("reward is NaN")
    return min(1.0, max(0.0, q))


def nl_judge_integrity(task, reward_info: dict) -> dict:
    """Observe official judge completeness without changing its reward or rerunning it."""
    criteria = getattr(task, "evaluation_criteria", None)
    expected = list(getattr(criteria, "nl_assertions", None) or [])
    if not expected or "NL_ASSERTION" not in (
        reward_info.get("reward_breakdown") or {}
    ):
        return {"status": "not_applicable_or_not_evaluated"}
    actual = [r["nl_assertion"] for r in (reward_info.get("nl_assertions") or [])]
    return {
        "status": "complete" if Counter(expected) == Counter(actual) else "mismatch",
        "expected_count": len(expected),
        "actual_count": len(actual),
        "official_reward_unchanged": True,
    }


def result_name(domain: str, task_id: str, trial: int | None = None) -> str:
    """Per-episode file name; ``trial=None`` is the single-trial (legacy) name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id)[:80].strip("_")
    suffix = "" if trial is None else f"__t{trial}"
    return f"{domain}__{slug}__{hashlib.sha1(task_id.encode()).hexdigest()[:8]}{suffix}.json"


def result_path(
    out_dir: Path, domain: str, task_id: str, trial: int | None = None
) -> Path:
    return Path(out_dir) / result_name(domain, task_id, trial)


def episode_key(domain: str, task_id: str, trial: int | None = None) -> str:
    """``domain:task_id`` (single trial) or ``domain:task_id#t<trial>`` (multi-trial runs)."""
    key = ent.task_key(domain, task_id)
    return key if trial is None else f"{key}#t{trial}"


def result_episode_key(r: dict) -> str:
    multi = (r.get("num_trials") or 1) > 1
    return episode_key(r["domain"], str(r["task_id"]), r["trial"] if multi else None)


def trial_items(items: list, num_trials: int) -> list:
    """(domain, task) -> episodes. One trial keeps ``items``; several give trial-major
    ``(domain, task, trial)`` with the longest domain first inside each trial."""
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if num_trials == 1:
        return list(items)
    rank = {d: i for i, d in enumerate(LONGEST_FIRST)}
    ordered = sorted(items, key=lambda it: rank.get(it[0], len(rank)))
    return [(d, t, k) for k in range(num_trials) for d, t in ordered]


class EndpointPool:
    """Pins each episode to one OpenAI-compatible endpoint (agent, user simulator, NL judge and
    semantic operators of that episode), choosing the endpoint with the fewest in-flight episodes.
    One endpoint per episode keeps the agent's growing conversation in that server's prefix cache."""

    def __init__(self, bases: list[str] | tuple[str, ...]):
        bases = tuple(b.rstrip("/") for b in bases)
        if not bases or len(set(bases)) != len(bases):
            raise ValueError("endpoint pool needs distinct endpoints")
        self.bases = bases
        self.load = dict.fromkeys(bases, 0)
        self.assigned = dict.fromkeys(bases, 0)
        self._lock = threading.Lock()

    @contextmanager
    def lease(self) -> Iterator[str]:
        with self._lock:
            base = min(self.bases, key=lambda b: self.load[b])
            self.load[base] += 1
            self.assigned[base] += 1
        try:
            yield base
        finally:
            with self._lock:
                self.load[base] -= 1


class EpisodeJudgeArgs(Mapping):
    """Official NL-judge kwargs whose ``api_base`` follows the calling episode's endpoint."""

    def __init__(self, args: dict):
        self.args = dict(args)

    def _current(self) -> dict:
        base = CURRENT_ENDPOINT.get()
        return self.args if base is None else {**self.args, "api_base": base}

    def __getitem__(self, key):
        return self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())


def read_result(path: Path) -> dict | None:
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return (
        r
        if isinstance(r, dict) and r.get("status") == "done" and "reward" in r
        else None
    )


def write_result(path: Path, result: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, default=str))
    tmp.replace(path)


def load_baseline(folder: Path) -> dict[str, float]:
    """``domain:task_id`` -> No-Memory quality, from a runner output directory."""
    out = {}
    for f in sorted(Path(folder).glob("*.json")):
        r = read_result(f)
        if r is not None:
            out[result_episode_key(r)] = quality_of(r["reward"])
    return out


def usage_tokens(messages: list[dict]) -> dict:
    agent = user = calls_a = calls_u = 0
    for m in messages:
        u = m.get("usage") or {}
        n = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        if not u:
            continue
        if m.get("role") == "assistant":
            agent, calls_a = agent + n, calls_a + 1
        elif m.get("role") == "user":
            user, calls_u = user + n, calls_u + 1
    return {
        "agent_tokens": agent,
        "user_tokens": user,
        "agent_calls": calls_a,
        "user_calls": calls_u,
    }


def experiment_id(
    plan: str, edges: tuple[str, ...] = ("similar",), policy: str | None = None
) -> str:
    """Default experiment id; success-only ids are the pre-2026-10-18 ids, others carry the policy."""
    policy = mp.validate(policy or mp.current())
    eid = f"{DATASET}:train178-test100:session-v1:{plan}:{llm_budget.PROTOCOL}"
    if plan != "VGR":
        eid += f":{mp.rubric(policy)}"
    eid = eid if edges == ("similar",) else f"{eid}:edges={'+'.join(edges)}"
    return mp.tag(eid, policy)


# ---------------------------------------------------------------- settings + memory provider


@dataclass
class Settings:
    mode: str  # "memory" | "nomemory"
    out_dir: Path
    split: str  # task split: train | test
    opt_split: str = "train"  # opt row split: train | test | smoke
    model: str = MODEL
    api_base: str = API_BASE
    user_api_base: str | None = None
    judge_api_base: str | None = None
    judge_json_mode: bool = True
    seed: int = 300
    trial: int = 0
    num_trials: int = 1  # >1: episodes are (task, trial) with per-trial official seeds
    api_bases: tuple[
        str, ...
    ] = ()  # endpoint pool for agent/user/judge/semantic (overrides)
    semantic_workers: int = 4  # concurrent SemFilter/SemRank judge calls per episode
    pool: Any = None  # EndpointPool built from api_bases
    max_steps: int = 200
    max_errors: int = 10
    timeout: float | None = None
    config: SessionConfig | None = None
    experiment: str | None = None
    baseline: dict = field(default_factory=dict)
    protocol_extra: dict = field(default_factory=dict)
    semantic_base: str = SEMANTIC_BASE
    rank_base: str | None = None
    store_dsn: str = STORE_DSN
    opt_dsn: str = OPT_DSN
    strict_entities: bool = False
    reuse: bool = False
    store: Any = None  # ScalarSessionStore
    embed_fn: Callable[[str], str] | None = None  # text -> pgvector literal
    adapter_factory: Callable | None = None  # (vector, query) -> SessionAdapter
    reuse_lookup: Callable | None = None  # (query_id, memory_sha) -> hit | None
    memory_policy: str = field(
        default_factory=mp.current
    )  # which stored sessions memory may inject
    run_protocol_sha256: str | None = None
    semantic_pool: Any = None
    transport_change: Path | None = None
    accepted_protocols: tuple[str, ...] = ()
    drain_file: Path | None = None
    base_results: dict = field(default_factory=dict)
    serving_segment_sha256: str | None = None
    reuse_writer: Callable | None = (
        None  # (task, domain, s, ReuseEpisode, path) -> result; default ``reuse``
    )
    result_extra: dict = field(
        default_factory=dict
    )  # merged into every result file (e.g. the load regime)


def http_embed(url: str = EMBED, model: str = EMBED_MODEL) -> Callable[[str], str]:
    def embed(text: str) -> str:
        import httpx

        r = httpx.post(url, json={"model": model, "input": [text]}, timeout=120)
        r.raise_for_status()
        return json.dumps(r.json()["data"][0]["embedding"])

    return embed


def episode_api_base(s: Settings, default: str | None = None) -> str:
    """The episode's pinned endpoint when a pool is active, else ``default`` (or ``api_base``)."""
    base = CURRENT_ENDPOINT.get() if s.api_bases else None
    return base or default or s.api_base


def episode_semantic_base(s: Settings) -> str:
    base = CURRENT_ENDPOINT.get() if s.api_bases else None
    return base.removesuffix("/v1") if base else s.semantic_base


def semantic_ops(plan: str, target: str, s: Settings) -> tuple[Any, Any]:
    from trimem.experiments.semantic_rubric import (
        KNOWLEDGE,
        RubricSemFilter,
        RubricSemRank,
    )

    def tau2_judge(cls):
        # The rubric module only knows ALFWorld / ScienceWorld / SWE-QA; reuse its prompt with τ² knowledge
        # without touching the shared module (running experiments import it).
        j = cls(
            "sweqa",
            target,
            tau2_card,
            base=episode_semantic_base(s),
            workers=s.semantic_workers,
            version=mp.rubric(s.memory_policy),
        )
        j.dataset = DATASET
        j.system = j.system.replace(KNOWLEDGE["sweqa"], TAU2_KNOWLEDGE)
        if cls is RubricSemFilter:
            j.endpoint_pool = s.semantic_pool
        return j

    sem = ranker = None
    if plan in ("P1", "P2", "P2+Rank"):
        sem = tau2_judge(RubricSemFilter)
    elif plan == "VGR+SemRank":
        sem = tau2_judge(RubricSemRank)
    if plan in ("VGR+Rank", "P2+Rank"):
        from trimem.experiments.colbert_rank import ColBERTRank, rank_base_for

        ranker = ColBERTRank(
            target,
            mp.rank_view(tau2_card),  # Rank sees content, not the outcome label
            base=s.rank_base or rank_base_for(s.api_base.removesuffix("/v1")),
        )
    return sem, ranker


REUSE_SQL = (
    "SELECT retrieval_id, generation, total_tokens, e2e_ms, quality, judge FROM opt.session_retrieval "
    "WHERE experiment_id = %s AND split = %s AND query_id = %s AND status = 'evaluated' "
    "AND generation->>'memory_sha256' = %s AND generation->>'reused_from' IS NULL "
    "AND protocol->>'panel_hash' IS NOT DISTINCT FROM %s "
    "AND coalesce(protocol->>'memory_policy', 'success-only-v0') = %s ORDER BY created_at LIMIT 1"
)  # the memory policy is part of the reuse key (rows that predate it are success-only-v0)


def db_reuse_lookup(s: Settings) -> Callable:
    def lookup(query_id: str, mem_sha: str) -> dict | None:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
            row = c.execute(
                REUSE_SQL,
                (
                    s.experiment,
                    s.opt_split,
                    query_id,
                    mem_sha,
                    s.protocol_extra.get("panel_hash"),
                    s.memory_policy,
                ),
            ).fetchone()
        if row is None or not Path(row["generation"].get("result_file", "")).exists():
            return None
        return row

    return lookup


def episode_trial(s: Settings) -> int:
    trial = CURRENT_TRIAL.get()
    return s.trial if trial is None else trial


class Provider:
    """``provider(task, first_user_message)`` for ``TrimemMemoryAgent``: retrieve once, render, record."""

    def __init__(self, s: Settings):
        self.s = s

    def __call__(self, task, text: str) -> dict:
        from trimem.experiments.session_profile_store import recorded_retrieve

        s = self.s
        tdict = (
            task.model_dump(mode="json") if hasattr(task, "model_dump") else dict(task)
        )
        domain = (
            CURRENT_DOMAIN.get()
        )  # set by run_task in the episode's thread (task ids repeat across domains)
        if domain not in ent.DOMAINS:
            raise ValueError(
                f"cannot determine the τ² domain of task {tdict.get('id')!r}"
            )
        q = ent.identity(domain, tdict, strict=s.strict_entities)
        t0 = time.monotonic()
        vector = s.embed_fn(tr.task_text(domain, text))
        adapter = (
            s.adapter_factory
            or (
                lambda v, qq: Tau2TaskAdapter(
                    v, qq, s.store_dsn, policy=s.memory_policy
                )
            )
        )(vector, q)
        if getattr(adapter, "policy", s.memory_policy) != s.memory_policy:
            raise ValueError("retrieval adapter runs under another memory policy")
        sem, ranker = (
            semantic_ops(s.config.plan, text, s)
            if s.config.plan != "VGR"
            else (None, None)
        )
        meta = {
            "experiment_id": s.experiment,
            "dataset": DATASET,
            "split": s.opt_split,
            "query_id": q["key"],
            "query_context": {
                "domain": domain,
                "task_id": q["task_id"],
                "first_user_message": text,
                "task_split": s.split,
            },
            "protocol": {
                "llm_budget": llm_budget.as_dict(),
                "selection": "task-session-full-pool-v2",
                "isolation": "L1",
                "isolation_rule": ISOLATION_BY_POLICY[s.memory_policy],
                "strict_entities": s.strict_entities,
                "query": "first user message",
                "model": s.model,
                "seed": s.seed,
                "trial": episode_trial(s),
                "num_trials": s.num_trials,
                "endpoint": CURRENT_ENDPOINT.get(),
                "semantic_routing": "request-load-v1"
                if s.semantic_pool
                else "episode-endpoint",
                "serving_segment_sha256": s.serving_segment_sha256,
                "semantic": f"{mp.rubric(s.memory_policy)}:tau2" if sem else None,
                "rank": "colbert" if ranker else None,
                **mp.protocol(s.memory_policy),
                **s.protocol_extra,
                "run_protocol_sha256": s.run_protocol_sha256,
            },
        }
        kw: dict = {}
        if s.config.plan == "VGR+SemRank":
            kw["semrank"] = sem
        elif sem is not None:
            kw["semfilter"] = sem
        if ranker is not None:
            kw["semrank"] = ranker
        try:
            receipt = recorded_retrieve(adapter, s.config, s.store, meta, **kw)
        finally:
            adapter.close()
        cap = (
            llm_budget.per_card(s.config.top_k) * 3
        )  # ~3 chars per token, conservative
        memory = render_memory(receipt["cards"], cap, s.memory_policy)
        source_tasks = [c["task_id"] for c in receipt["cards"]]
        sem_calls = (getattr(sem, "calls", None) or []) + (
            getattr(ranker, "calls", None) or []
        )
        mem = {
            "rid": receipt["retrieval_id"],
            "semantic_endpoint_counts": dict(
                Counter(c["endpoint"] for c in sem_calls if c.get("endpoint"))
            ),
            "memory": memory,
            "retrieval_ms": (time.monotonic() - t0) * 1000,
            "semantic_tokens": sum(
                ((c.get("usage") or {}).get("total_tokens", 0)) for c in sem_calls
            ),
            "selected": receipt["selected_ids"],
            "memory_sha256": hashlib.sha256(memory.encode()).hexdigest(),
            "query_id": q["key"],
            "memory_policy": s.memory_policy,
            "card_outcomes": mp.card_outcomes(receipt["cards"]),
            "selected_source_task_ids": source_tasks,
            "selected_source_task_keys": [c["key"] for c in receipt["cards"]],
        }
        if s.reuse:
            hit = (s.reuse_lookup or db_reuse_lookup(s))(q["key"], mem["memory_sha256"])
            if hit is not None:
                raise ReuseEpisode(hit, mem)
        return mem


class SlotLimitedProvider:
    """Bounds concurrent retrievals (each holds a corpus connection to PG 55450, whose
    max_connections is 32, plus SemFilter calls) independently of episode concurrency."""

    def __init__(self, provider: Callable, slots: int):
        self.provider = provider
        self.slots = threading.BoundedSemaphore(slots)

    def __call__(self, task, text: str) -> dict:
        with self.slots:
            return self.provider(task, text)

    def __getattr__(self, name: str):
        return getattr(self.provider, name)


# ---------------------------------------------------------------- episode


def configure_http_pool(max_connections: int) -> None:
    """Raise the official client's connection cap (transport only, no request change).

    ``tau2.utils.llm_utils`` installs one process-wide ``litellm.client_session`` with
    ``httpx.Limits(max_connections=10)``, so at most 10 agent/user/judge calls are in flight per
    process however many episode threads run (diagnosed 2026-10-19: 40 workers -> 10 sockets, the
    other threads blocked on the pool). Replace it with an otherwise identical client."""
    import httpx
    import litellm

    import tau2.utils.llm_utils  # noqa: F401  (its import installs the 10-connection client)

    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=max_connections
    )
    litellm.client_session = httpx.Client(limits=limits)


def configure_judge(model: str, args: dict) -> None:
    """Point the official NL-assertion judge at the local model (module constants; prompt unchanged)."""
    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = model
    # The judge runs inside run_simulation in the episode's thread: follow its pinned endpoint.
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = EpisodeJudgeArgs(args)


def official_simulate(task, domain: str, s: Settings):
    """Build the official text orchestrator and run + evaluate it. Returns (SimulationRun, agent)."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import build_text_orchestrator, run_simulation

    from trimem.experiments.tau2.agent import AGENT_NAME

    config = TextRunConfig(
        domain=domain,
        agent=AGENT_NAME if s.mode == "memory" else "llm_agent",
        llm_agent=s.model,
        llm_args_agent=llm_args(episode_api_base(s)),
        user="user_simulator",
        llm_user=s.model,
        llm_args_user=llm_args(episode_api_base(s, s.user_api_base)),
        max_steps=s.max_steps,
        max_errors=s.max_errors,
        seed=s.seed,
        num_trials=1,
        timeout=s.timeout,
    )
    orch = build_text_orchestrator(config, task, seed=trial_seed(s.seed, s.trial))
    sim = run_simulation(orch)
    from trimem.experiments.agent_step_metrics import tau2_steps

    initial = getattr(getattr(task, "initial_state", None), "message_history", None)
    initial_count = len(initial) if initial else 1  # text orchestrator's fixed greeting
    sim.info = {
        **(sim.info or {}),
        "trimem_step_metrics": tau2_steps(
            sim.model_dump(mode="json"),
            initial_message_count=initial_count,
            orchestrator_steps=orch.step_count,
        ),
    }
    return sim, orch.agent


def _base_result(task, domain: str, s: Settings) -> dict:
    return {
        "status": "done",
        "task_id": str(task.id),
        "domain": domain,
        "split": s.split,
        "mode": s.mode,
        "model": s.model,
        "seed": s.seed,
        "trial": s.trial,
        "num_trials": s.num_trials,
        "trial_seed": trial_seed(s.seed, s.trial),
        "experiment": s.experiment,
        "config": None if s.config is None else dataclasses.asdict(s.config),
        "run_protocol_sha256": s.run_protocol_sha256,
        **s.result_extra,
    }


def run_task(
    task,
    domain: str,
    s: Settings,
    simulate: Callable = official_simulate,
    trial: int | None = None,
) -> dict:
    """One episode (task, trial); returns the result dict (also written to ``out_dir``).
    Skips finished episodes. ``trial=None`` = the single-trial run at ``s.trial``."""
    if trial is not None:
        s = dataclasses.replace(s, trial=trial)
    path = result_path(s.out_dir, domain, str(task.id), trial)
    done = read_result(path)
    if done is not None:
        if (
            s.transport_change
            and done.get("run_protocol_sha256") != s.run_protocol_sha256
        ):
            if (
                s.base_results.get(path.name)
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"unregistered base result: {path}")
        if s.run_protocol_sha256 and done.get("run_protocol_sha256") not in (
            s.run_protocol_sha256,
            *s.accepted_protocols,
        ):
            raise ValueError(f"{path}: result belongs to another run protocol")
        return {**done, "resumed": True}
    key = episode_key(domain, str(task.id), trial)
    claims: list = []
    try:
        with ExitStack() as episode:
            if s.pool is not None:
                episode.enter_context(
                    _bind(CURRENT_ENDPOINT, episode.enter_context(s.pool.lease()))
                )
            episode.enter_context(_bind(CURRENT_TRIAL, s.trial))
            episode.enter_context(_bind(CURRENT_DOMAIN, domain))
            episode.enter_context(_bind(EPISODE_CLAIMS, claims))
            t0 = time.monotonic()
            try:
                sim, agent = simulate(task, domain, s)
            except ReuseEpisode as r:
                return (s.reuse_writer or reuse)(task, domain, s, r, path)
            endpoint = CURRENT_ENDPOINT.get()
        return _finish(task, domain, s, sim, agent, key, path, t0, endpoint)
    finally:  # after the result file (or the failure): same-key episodes waiting on the claim proceed
        for c in claims:
            c.release()


@contextmanager
def _bind(var: contextvars.ContextVar, value) -> Iterator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def _finish(task, domain, s, sim, agent, key, path, t0, endpoint) -> dict:
    wall_ms = (time.monotonic() - t0) * 1000
    simd = sim.model_dump(mode="json") if hasattr(sim, "model_dump") else dict(sim)
    reward = (simd.get("reward_info") or {}).get("reward")
    if reward is None:
        raise ValueError(f"{key}: simulation has no reward")
    quality = quality_of(reward)
    usage = usage_tokens(simd.get("messages") or [])
    from trimem.experiments.agent_step_metrics import tau2_steps

    steps = tau2_steps(simd)
    result = {
        **steps,
        **_base_result(task, domain, s),
        "endpoint": endpoint,
        "serving_segment_sha256": s.serving_segment_sha256,
        "reward": float(reward),
        "quality": quality,
        "reward_info": simd.get("reward_info"),
        "nl_judge_integrity": nl_judge_integrity(task, simd.get("reward_info") or {}),
        "termination_reason": simd.get("termination_reason"),
        **usage,
        "wall_ms": wall_ms,
        "memory": None,
        "retrieval_id": None,
    }
    if s.mode == "memory":
        mem = getattr(agent, "memory_meta", None)
        if (
            mem is None
        ):  # the customer never sent a text message: no retrieval, no opt row
            result["memory_status"] = "no_first_user_message"
        else:
            episode_ms = wall_ms - mem["retrieval_ms"]
            gen = {
                **steps,
                "reward": float(reward),
                "llm_calls": usage["agent_calls"],
                "generation_tokens": usage["agent_tokens"],
                "user_sim_tokens": usage["user_tokens"],
                "semantic_tokens": mem["semantic_tokens"],
                "episode_ms": episode_ms,
                "retrieval_ms": mem["retrieval_ms"],
                "memory_chars": len(mem["memory"]),
                "memory_sha256": mem["memory_sha256"],
                "result_file": str(path.resolve()),
                "termination_reason": simd.get("termination_reason"),
                "memory_policy": mem.get("memory_policy"),
                "card_outcomes": mem.get("card_outcomes"),
            }
            # Cost = the agent's tokens + semantic operators (the user simulator is the environment).
            s.store.generation(
                mem["rid"],
                gen,
                total_tokens=usage["agent_tokens"] + mem["semantic_tokens"],
                e2e_ms=wall_ms,
            )
            s.store.evaluate_scalar(
                mem["rid"],
                quality,
                {
                    "reward": float(reward),
                    "metric": "tau2 reward",
                    "reward_breakdown": (simd.get("reward_info") or {}).get(
                        "reward_breakdown"
                    ),
                },
                s.baseline.get(key),
            )
            result.update(
                retrieval_id=mem["rid"],
                memory={k: v for k, v in mem.items() if k != "memory"},
                memory_text=mem["memory"],
                first_user_message=getattr(agent, "query_text", None),
            )
    result["simulation"] = simd
    # process-wide (all episode threads) RSS / fds / threads after this episode: leaks show in the data
    result["resources"] = rss_guard.usage()
    write_result(path, result)  # last: completion = file (+ evaluated row)
    return result


def reuse(task, domain: str, s: Settings, r: ReuseEpisode, path: Path) -> dict:
    hit, mem = r.hit, r.mem
    src = read_result(Path(hit["generation"]["result_file"]))
    if src is None or src.get("task_id") != str(task.id) or src.get("domain") != domain:
        raise ValueError("reuse source belongs to another task")
    g = hit["generation"]
    from trimem.experiments.agent_step_metrics import tau2_steps

    steps = tau2_steps(src.get("simulation") or {})
    key = ent.task_key(domain, str(task.id))
    s.store.generation(
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
            "memory_sha256": mem["memory_sha256"],
            "result_file": str(path.resolve()),
            "reused_from": str(hit["retrieval_id"]),
            "memory_policy": mem.get("memory_policy"),
            "card_outcomes": mem.get("card_outcomes"),
        },
        total_tokens=g["generation_tokens"] + mem["semantic_tokens"],
        e2e_ms=g["episode_ms"] + mem["retrieval_ms"],
    )
    s.store.evaluate_scalar(
        mem["rid"],
        quality_of(src["reward"]),
        {
            "reward": src["reward"],
            "metric": "tau2 reward",
            "reused_from": str(hit["retrieval_id"]),
        },
        s.baseline.get(key),
    )
    out = {
        **src,
        **steps,
        "experiment": s.experiment,
        "retrieval_id": mem["rid"],
        "reused_from": str(hit["retrieval_id"]),
        "memory": {k: v for k, v in mem.items() if k != "memory"},
        "memory_text": mem["memory"],
        "config": _base_result(task, domain, s)["config"],
    }
    write_result(path, out)
    return out


# ---------------------------------------------------------------- CLI


def load_split(domain: str, split: str) -> list:
    from tau2.runner.helpers import load_tasks

    return load_tasks(domain, split)


def run_all(
    items: list[tuple[str, Any]],
    s: Settings,
    workers: int,
    retries: int,
    simulate: Callable = official_simulate,
    guard: rss_guard.RssGuard | None = None,
) -> dict:
    """Run ``items`` -- ``(domain, task)`` or ``(domain, task, trial)`` -- on ``workers`` threads. Tasks are submitted as threads free up (never more than
    ``workers`` in flight); with an RSS ``guard`` no new task starts once the process is at 90 % of
    ``--max_rss_gb`` -- in-flight episodes finish, ``stats["stopped_rss"]`` is set, a rerun resumes."""
    s.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "done": 0,
        "resumed": 0,
        "failed": 0,
        "reward_sum": 0.0,
        "stopped_rss": None,
        "stopped_requested": None,
        "stopped_service": None,
    }
    guard = guard or rss_guard.RssGuard(None)

    def one(domain, task, trial):
        for attempt in range(retries + 1):
            try:
                return run_task(task, domain, s, simulate, trial)
            except Exception:
                err = traceback.format_exc()
                if attempt == retries:
                    (s.out_dir / "errors").mkdir(exist_ok=True)
                    (
                        s.out_dir
                        / "errors"
                        / result_name(domain, str(task.id), trial).replace(
                            ".json", ".txt"
                        )
                    ).write_text(err)
                    raise
                time.sleep(5 * (attempt + 1))

    workers = max(1, workers)
    todo = list(items)
    todo.reverse()  # pop() from the end keeps submission order without O(n) pops
    with ThreadPoolExecutor(workers) as ex:
        futs: dict = {}
        while todo or futs:
            while (
                todo
                and len(futs) < workers
                and not any(
                    stats[k]
                    for k in ("stopped_rss", "stopped_requested", "stopped_service")
                )
            ):
                if s.drain_file and s.drain_file.exists():
                    stats["stopped_requested"] = f"drain requested: {s.drain_file}"
                    break
                if s.semantic_pool and (reason := s.semantic_pool.unhealthy_reason()):
                    stats["stopped_service"] = reason
                    break
                rss = rss_guard.usage()["total_rss_mb"] if guard.enabled else 0.0
                if guard.soft_exceeded(rss):
                    stats["stopped_rss"] = guard.message(rss, "is at 90% of")
                    print("ERROR: " + stats["stopped_rss"], file=sys.stderr, flush=True)
                    break
                d, t, *rest = todo.pop()
                trial = rest[0] if rest else None
                label = t.id if trial is None else f"{t.id}#t{trial}"
                futs[ex.submit(one, d, t, trial)] = (d, label)
            if not futs:
                break
            finished, _ = wait(futs, return_when=FIRST_COMPLETED)
            for f in finished:
                d, tid = futs.pop(f)
                try:
                    r = f.result()
                except Exception as exc:  # recorded in errors/; the run continues
                    stats["failed"] += 1
                    print(f"FAILED {d}:{tid}: {exc!r}", flush=True)
                    continue
                stats["resumed" if r.get("resumed") else "done"] += 1
                if r.get("reuse") and not r.get("resumed"):
                    stats["reused"] = stats.get("reused", 0) + 1
                stats["reward_sum"] += r["reward"]
                res = r.get("resources") or {}
                print(
                    f"{d}:{tid} reward={r['reward']:.3f}{' (resumed)' if r.get('resumed') else ''}"
                    + (" (reused)" if r.get("reuse") and not r.get("resumed") else "")
                    + (f" rss_mb={res['total_rss_mb']}" if res else ""),
                    flush=True,
                )
    stats["not_started"] = len(todo)
    n = stats["done"] + stats["resumed"]
    stats["pass^1"] = stats["reward_sum"] / n if n else None
    return stats


def main(argv: list[str] | None = None) -> int:
    with ExitStack() as resources:
        return _main(argv, resources)


def _main(argv: list[str] | None, resources: ExitStack) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    ap.add_argument("--mode", required=True, choices=("memory", "nomemory"))
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument("--domains", default=",".join(ent.DOMAINS))
    ap.add_argument(
        "--task_ids", default=None, help="comma-separated task ids (within --domains)"
    )
    ap.add_argument(
        "--num_tasks", type=int, default=None, help="first N tasks per domain (smoke)"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="opt rows go to split 'smoke' (never BO training)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--baseline_dir",
        default=None,
        help="No-Memory output dir (required for --mode memory)",
    )
    ap.add_argument("--plan", default="VGR", choices=PLANS)
    ap.add_argument(
        "--memory_config", default='{"seed_count":8,"depth":1,"fanout":4,"top_k":3}'
    )
    ap.add_argument(
        "--edges",
        default="similar",
        help="session edges joined by '+', e.g. similar+similar_up",
    )
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--protocol_extra", default="{}")
    ap.add_argument(
        "--freeze_protocol",
        action="store_true",
        help="persist effective settings/code/task hashes and reject mismatched resume",
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--api_base", default=API_BASE)
    ap.add_argument(
        "--user_api_base",
        default=None,
        help="user simulator endpoint (default: --api_base)",
    )
    ap.add_argument(
        "--judge_api_base",
        default=None,
        help="NL-assertion judge endpoint (default: --api_base)",
    )
    ap.add_argument(
        "--no_judge_json_mode",
        action="store_true",
        help="do not request JSON output from the judge",
    )
    ap.add_argument("--semantic_base", default=SEMANTIC_BASE)
    ap.add_argument("--semantic_load_balance", action="store_true")
    ap.add_argument("--transport_change", type=Path)
    ap.add_argument("--drain_file", type=Path)
    ap.add_argument("--serving_segment", type=Path)
    ap.add_argument("--rank_base", default=None)
    ap.add_argument("--embed_url", default=EMBED)
    ap.add_argument("--store_dsn", default=STORE_DSN)
    ap.add_argument("--opt_dsn", default=OPT_DSN)
    ap.add_argument(
        "--strict_entities",
        action="store_true",
        help="isolation keeps domain-constant entities",
    )
    ap.add_argument("--reuse_identical_memory", action="store_true")
    ap.add_argument(
        "--memory_policy",
        choices=mp.POLICIES,
        default=None,
        help=f"which stored sessions memory may inject (default ${mp.ENV_VAR} or {mp.DEFAULT})",
    )
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_errors", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="trials per task (official per-trial seeds); >1 writes one file per (task, trial)",
    )
    ap.add_argument(
        "--api_bases",
        default=None,
        help="comma-separated endpoint pool; each episode is pinned to the least-loaded one for "
        "agent, user simulator, NL judge and semantic operators (overrides the per-role bases)",
    )
    ap.add_argument(
        "--semantic_workers",
        type=int,
        default=4,
        help="concurrent SemFilter/SemRank judge calls inside one episode's retrieval",
    )
    ap.add_argument(
        "--max_concurrency", type=int, default=4, help="concurrent episodes"
    )
    ap.add_argument(
        "--llm_http_connections",
        type=int,
        default=0,
        help="litellm HTTP connection cap (0 = max(16, 2 x --max_concurrency); official is 10)",
    )
    ap.add_argument(
        "--max_retrieval_concurrency",
        type=int,
        default=0,
        help="concurrent memory retrievals across episodes (0 = no extra limit)",
    )
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--log_level", default="ERROR")
    ap.add_argument(
        "--max_rss_gb",
        type=float,
        default=0.0,
        help="process (+ children) RSS budget: start no new task at 90%% of it, hard-exit at 100%% "
        "(0 = off). Finished tasks are saved; rerun resumes",
    )
    a = ap.parse_args(argv)
    if a.max_rss_gb < 0:
        ap.error("--max_rss_gb must be >= 0")
    if a.reuse_identical_memory and a.freeze_protocol:
        ap.error("frozen runs do not reuse historical episode rewards")
    if a.max_retrieval_concurrency < 0 or a.llm_http_connections < 0:
        ap.error("--max_retrieval_concurrency / --llm_http_connections must be >= 0")
    http_connections = a.llm_http_connections or max(16, 2 * a.max_concurrency)
    if a.num_trials < 1 or a.semantic_workers < 1 or a.max_concurrency < 1:
        ap.error("--num_trials, --semantic_workers and --max_concurrency must be >= 1")
    api_bases = tuple(
        b.strip().rstrip("/") for b in (a.api_bases or "").split(",") if b.strip()
    )
    if api_bases and (a.user_api_base or a.judge_api_base):
        ap.error("--api_bases replaces --user_api_base / --judge_api_base")
    if a.semantic_load_balance and (len(api_bases) < 2 or a.mode != "memory"):
        ap.error("--semantic_load_balance requires memory mode and two endpoints")
    if a.semantic_load_balance and (
        a.plan != "P2" or not a.serving_segment or not a.freeze_protocol
    ):
        ap.error(
            "--semantic_load_balance requires P2, --serving_segment and --freeze_protocol"
        )
    if a.transport_change and (not a.semantic_load_balance or not a.freeze_protocol):
        ap.error(
            "--transport_change requires frozen protocol and semantic load balancing"
        )
    if a.reuse_identical_memory and a.num_trials > 1:
        ap.error("identical-memory reuse is single-trial only")

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=a.log_level)
    s = Settings(
        mode=a.mode,
        out_dir=Path(a.out_dir),
        split=a.split,
        opt_split="smoke" if a.smoke else a.split,
        model=a.model,
        api_base=a.api_base,
        user_api_base=a.user_api_base,
        judge_api_base=a.judge_api_base,
        judge_json_mode=not a.no_judge_json_mode,
        seed=a.seed,
        max_steps=a.max_steps,
        max_errors=a.max_errors,
        timeout=a.timeout,
        protocol_extra=json.loads(a.protocol_extra),
        semantic_base=a.semantic_base,
        rank_base=a.rank_base,
        store_dsn=a.store_dsn,
        opt_dsn=a.opt_dsn,
        strict_entities=a.strict_entities,
        reuse=a.reuse_identical_memory,
        memory_policy=mp.validate(a.memory_policy or mp.current()),
        num_trials=a.num_trials,
        api_bases=api_bases,
        semantic_workers=a.semantic_workers,
        pool=EndpointPool(api_bases) if api_bases else None,
        transport_change=a.transport_change,
        drain_file=a.drain_file,
    )
    if a.semantic_load_balance:
        from trimem.experiments.tau2.semantic_pool import SemanticRequestPool

        s.semantic_pool = SemanticRequestPool(api_bases).start()
        resources.callback(s.semantic_pool.close)
    if a.freeze_protocol:
        from trimem.experiments.tau2.run_protocol import exclusive_arm

        resources.enter_context(exclusive_arm(s.out_dir))
    if a.serving_segment:
        from trimem.experiments.tau2.run_protocol import claim_serving_segment

        s.serving_segment_sha256 = claim_serving_segment(s, a.serving_segment)
    configure_http_pool(http_connections)
    configure_judge(
        a.model,
        llm_args(
            api_bases[0] if api_bases else a.judge_api_base or a.api_base,
            json_mode=s.judge_json_mode,
        ),
    )
    if a.mode == "memory":
        if not a.baseline_dir:
            raise SystemExit(
                "--mode memory requires --baseline_dir (paired No-Memory results)"
            )
        from trimem.experiments.session_store_ext import ScalarSessionStore
        from trimem.experiments.tau2.agent import register, set_memory_provider

        cfg = json.loads(a.memory_config)
        s.config = SessionConfig(plan=a.plan, edges=tuple(a.edges.split("+")), **cfg)
        mp.claim_dir(s.out_dir, s.memory_policy)
        s.experiment = mp.tag(
            a.experiment or experiment_id(a.plan, s.config.edges, s.memory_policy),
            s.memory_policy,
        )
        s.baseline = load_baseline(Path(a.baseline_dir))
        s.store = ScalarSessionStore(a.opt_dsn)
        s.embed_fn = http_embed(a.embed_url)
        register()
        provider = Provider(s)
        if a.max_retrieval_concurrency > 0:
            provider = SlotLimitedProvider(provider, a.max_retrieval_concurrency)
        set_memory_provider(provider)
    wanted = set(a.task_ids.split(",")) if a.task_ids else None
    items = []
    for d in a.domains.split(","):
        tasks = [
            t for t in load_split(d, a.split) if wanted is None or str(t.id) in wanted
        ]
        items += [(d, t) for t in tasks[: a.num_tasks or None]]
    if not items:
        raise SystemExit("no tasks selected")
    if a.freeze_protocol:
        from trimem.experiments.tau2.run_protocol import claim_protocol

        s.run_protocol_sha256 = claim_protocol(s, items, a.embed_url)
    if a.mode == "memory":
        trials = range(s.num_trials) if s.num_trials > 1 else (None,)
        missing = [
            key
            for d, t in items
            for key in (episode_key(d, str(t.id), k) for k in trials)
            if key not in s.baseline
        ]
        if missing:
            print(
                f"WARNING: {len(missing)} tasks have no No-Memory baseline (paired_gain NULL): {missing[:3]}"
            )
    print(
        json.dumps(
            {
                "tasks": len(items),
                "num_trials": s.num_trials,
                "episodes": len(items) * s.num_trials,
                "api_bases": list(api_bases),
                "max_concurrency": a.max_concurrency,
                "max_retrieval_concurrency": a.max_retrieval_concurrency,
                "llm_http_connections": http_connections,
                "mode": a.mode,
                "split": a.split,
                "experiment": s.experiment,
                "out_dir": str(s.out_dir),
            }
        ),
        flush=True,
    )
    guard = rss_guard.RssGuard(a.max_rss_gb)
    guard.start_watchdog()
    try:
        stats = run_all(
            trial_items(items, s.num_trials),
            s,
            a.max_concurrency,
            a.max_retries,
            guard=guard,
        )
        if s.pool is not None:
            stats["episodes_per_endpoint"] = dict(s.pool.assigned)
    finally:
        guard.stop()
    print(json.dumps(stats), flush=True)
    if stats["stopped_rss"]:
        return rss_guard.EXIT_RSS
    if stats.get("stopped_service") or stats["failed"]:
        return 1
    if stats.get("stopped_requested"):
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
