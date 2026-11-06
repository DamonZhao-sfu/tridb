"""τ² episode runner of the pool-profiling round (session space ``session-space-tau2-v2``; driver
``trimem.experiments.tau2.pool_profile``; plan docs/trimem_pool_profiling_plan_v0.1.0.md, τ² addendum).

A NEW entry point next to ``tau2.runner`` (which stays unchanged): it reuses the runner's official episode
machinery (``run_all`` / ``run_task`` / ``official_simulate`` / ``_finish``: official orchestrator, user
simulator, evaluator, per-trial seeds, result files, opt rows) and adds only

* the v2 retrieval (``PoolProvider``): ``SessionConfigV2`` (candidate pool cap 256 by query cosine, expansion
  guard 4,096, ``sim_gate``, ``max_failed``), the configuration's outcome policy (success_only ->
  success-only-v0 = successful sessions, unlabelled cards, rubric v1; success_failure[_capped] ->
  success+failure-v1), its SemFilter threshold, ColBERT on the episode's own GPU (8030 -> 8040, 8031 -> 8041)
  and the v2 seed widening (``WidenedTau2Adapter``: widen the seed search until ``seed_count`` seed tasks WITH
  an admissible session are found -- matters for success_only, telecom has 93/296 successful sessions);
* exact task selection ``--task_keys domain:task,...`` (the stock ``--task_ids`` matches ids in every listed
  domain, and airline / retail reuse numeric ids);
* a host-wide retrieval cap (``--retrieval_slots``, flock slots under ``/tmp/trimem-tau2-retrieval-slots``:
  several arms run in parallel and every retrieval holds a PG 55450 connection, max_connections 32);
* host-wide episode -> endpoint affinity (``endpoint_affinity.HostEndpointPool``: all LLM calls of an episode
  on one endpoint, the endpoint with the fewest in-flight episodes over ALL arm processes);
* ``arm.json``: the arm's identity (configuration key, experiment, split, trials, seed); a resume under a
  different identity is refused.

Run memory policy: success+failure-v1 (the store admits both outcomes; each configuration's outcome_policy
restricts it). memory_form (-v5: fixed ``compact`` for τ²; the runner still renders any form): raw / compact / distilled cards (``memory_forms``; distilled = the offline
τ² sidecar of ``memory_distill``, loaded once per process, never an LLM call per episode).

    .venv-tau2/bin/python -m trimem.experiments.tau2.runner_pool --mode memory --split train \\
        --space_config '<json>' --experiment <id> --baseline_dir <NoMemory dir> --out_dir <dir> \\
        --num_trials 4 --api_bases http://127.0.0.1:8030/v1,http://127.0.0.1:8031/v1 --max_concurrency 8
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from trimem.experiments import llm_budget, retrieval_lease, rss_guard
from trimem.experiments import memory_forms as mf
from trimem.experiments import memory_policy as mp
from trimem.experiments import quality_reuse as qr
from trimem.experiments.session_retrieval import SessionConfigV2
from trimem.experiments.tau2 import design as tau2_design
from trimem.experiments.tau2 import entities as ent
from trimem.experiments.tau2 import runner as R
from trimem.experiments.tau2 import trajectories as tr
from trimem.experiments.tau2.adapter import (
    SEED_SQL,
    Tau2TaskAdapter,
    tau2_card,
)
from trimem.experiments.tau2.endpoint_affinity import (
    LEASE_DIR,
    ROUTING,
    HostEndpointPool,
)

# session space of the selected pool design (TAU2_POOL_DESIGN, inherited from pool_profile): session-space-tau2-v5
# (``session_space_tau2_v2``) unless explore-v6 (pool-v6, user 2026-10-23: + plan P1, SemFilter judge built for it)
space = tau2_design.space_module()

RUN_POLICY = mp.SUCCESS_FAILURE
OUTCOME_POLICY = {
    "success_only": mp.SUCCESS_ONLY,
    "success_failure": mp.SUCCESS_FAILURE,
    "success_failure_capped": mp.SUCCESS_FAILURE,
}
SLOT_DIR = Path("/tmp/trimem-tau2-retrieval-slots")
ARM_FILE = "arm.json"
# Load regime of the episodes that predate the ``load`` block (pool -v5 rounds 0-5: <= 64 concurrent episodes).
LEGACY_REGIME = "c64"


# ---------------------------------------------------------------- pure helpers (unit-tested)


def session_config(cfg: dict) -> SessionConfigV2:
    """``SessionConfigV2`` of a τ²-v2 configuration (runner form of ``session_space_tau2_v2``)."""
    ra = space.runner_args(cfg)
    o = ra["v2"]
    return SessionConfigV2(
        plan=ra["plan"],
        edges=tuple(ra["edges"].split("+")),
        **ra["memory_config"],
        pool_cap=o["pool_cap"],
        expand_limit=o["expand_limit"],
        sim_gate=o["sim_gate"],
        max_failed=o["max_failed"],
    )


def outcome_policy(cfg: dict) -> str:
    return OUTCOME_POLICY[space.validate(cfg)["outcome_policy"]]


def parse_task_keys(text: str | None) -> dict[str, set[str]] | None:
    """``domain:task,...`` -> domain -> task ids (None = the whole split). Task ids may contain ':'."""
    if not text:
        return None
    out: dict[str, set[str]] = {}
    for k in (x for x in text.split(",") if x):
        d, _, t = k.partition(":")
        if d not in ent.DOMAINS or not t:
            raise ValueError(f"bad τ² task key {k!r}")
        out.setdefault(d, set()).add(t)
    return out


def select_items(
    tasks_by_domain: dict[str, list], wanted: dict[str, set[str]] | None
) -> list[tuple[str, object]]:
    """(domain, task) of exactly the wanted keys (domain order, split order); raises on unknown keys."""
    items = []
    for d, tasks in tasks_by_domain.items():
        if wanted is not None and d not in wanted:
            continue
        items += [(d, t) for t in tasks if wanted is None or str(t.id) in wanted[d]]
    if wanted is not None:
        got = {(d, str(t.id)) for d, t in items}
        need = {(d, t) for d, ts in wanted.items() for t in ts}
        if got != need:
            raise ValueError(f"unknown τ² task keys: {sorted(need - got)[:5]}")
    return items


def arm_identity(
    *,
    mode: str,
    split: str,
    cfg: dict | None,
    experiment: str | None,
    num_trials: int,
    seed: int,
) -> dict:
    return {
        "mode": mode,
        "split": split,
        "space_version": space.VERSION if cfg else None,
        "config_key": space.key(cfg) if cfg else None,
        "experiment": experiment,
        "num_trials": num_trials,
        "seed": seed,
        "run_memory_policy": RUN_POLICY,
    }


def claim_arm(out: Path, ident: dict) -> None:
    p = Path(out) / ARM_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = json.loads(p.read_text())
        if old != ident:
            raise SystemExit(f"{p} belongs to another arm ({old}); refusing to mix")
        return
    p.write_text(json.dumps(ident, indent=1))


class HostSlots:
    """At most ``n`` retrievals in flight HOST-wide (flock slot files; released when the holder exits). ``start``
    > 0 restricts this holder to slots ``start..n-1``: semantic (SemFilter / ColBERT) retrievals hold their slot,
    i.e. one PG 55450 connection, for minutes, so slots ``0..start-1`` stay reserved for fast (VGR) retrievals
    (``fast_reserve``; live τ² round 6, 2026-10-20: all 18 slots held by P2+Rank retrievals, VGR episodes queued)."""

    def __init__(self, n: int, folder: Path = SLOT_DIR, start: int = 0):
        if n > 0 and not 0 <= start < n:
            raise ValueError(f"slot start {start} outside 0..{n - 1}")
        self.n, self.folder, self.start = n, Path(folder), start
        self.folder.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        if self.n <= 0:
            self.f = None
            return self
        while True:
            for i in range(self.start, self.n):
                f = open(self.folder / f"{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(0.2)

    def __exit__(self, *exc):
        if self.f is not None:
            self.f.close()
        return False

    def slot(self) -> HostSlots:  # one context object per acquisition (thread-safe)
        return HostSlots(self.n, self.folder, self.start)


def slow_plan(plan: str) -> bool:
    """Retrievals that make LLM judge (SemFilter) or ColBERT calls while holding their slot."""
    return plan in space.SEMFILTER_PLANS or plan.endswith("+Rank")


def fast_reserve(n: int, reserve: int) -> int:
    """Slots reserved for fast retrievals: ``reserve`` (-1 = auto: n // 3 when n >= 6, else 0)."""
    if reserve < 0:
        reserve = n // 3 if n >= 6 else 0
    return min(reserve, max(n - 1, 0))


class WidenedTau2Adapter(Tau2TaskAdapter):
    """v2 seed widening (as ``AgentEnvTaskAdapter.widen_seeds``): grow the task search until ``count`` seed
    tasks WITH at least one admissible session are found (or the domain is exhausted). The stock adapter stops
    once ``count`` isolation-admissible tasks are listed, so under success_only a seed task whose sessions all
    failed is lost. Same SQL, same isolation, same receipts."""

    def seeds(self, count: int) -> list[dict]:
        limit = max(32, count * 2)
        while True:
            self.seed_tasks, self.task_membership_paths = [], []
            self.isolation_rejections = {}
            rows = self._read(SEED_SQL, (self.query["env"], self.vector, limit))
            sessions: dict[int, dict] = {}
            for t in rows:
                if len(self.seed_tasks) >= count:
                    break
                if not self._task_ok(t):
                    continue
                kept = self._admissible_sessions(t)
                if kept:
                    self.seed_tasks.append(
                        {k: t[k] for k in ("id", "vid", "key", "env", "family")}
                    )
                    for s in kept:
                        sessions[s["id"]] = s
            if len(self.seed_tasks) >= count or len(rows) < limit or limit >= 4096:
                return list(sessions.values())
            limit *= 2

    def _admissible_sessions(self, t: dict) -> list[dict]:
        kept = []
        for n in self._read(
            "SELECT dst FROM trimem.expand(%s, 'has_session', 64)", (t["vid"],)
        ):
            s = self._session(n["dst"])
            if s is None or s["task_id"] != t["id"]:
                raise ValueError("task/session ownership mismatch")
            self.task_membership_paths.append(
                {
                    "src": t["vid"],
                    "dst": s["vid"],
                    "edge": "has_session",
                    "task_id": t["id"],
                    "session_id": s["id"],
                }
            )
            if self.allowed(s):
                kept.append(s)
        return kept


def tau2_semfilter_v2(
    target: str, s: R.Settings, threshold: float, sf2: dict, policy_rubric: str
):
    """semfilter-v2 (user 2026-10-22; ``semfilter_v2``, opt-in ``TRIMEM_SEMFILTER_V2=1``) for τ²: the batched
    score-only judge (top-N query-cosine prefilter applied by the retrieval, K candidates per guided json_schema call,
    integer tenths) with the τ² rubric entries of ``semantic_rubric`` (KNOWLEDGE / CONFLICTS ``tau2``; rubric =
    ``sf2['rubric']``, default semantic-rubric-v3 incl. the CONFLICT clause), judge input = the τ² judge card
    (``tau2_card``, = v1's input; raw only). Returns (judge, protocol block)."""
    from trimem.experiments import semfilter_v2 as sf

    if sf2["judge_input"] != "raw":
        raise ValueError("τ² semfilter-v2: judge input raw (tau2_card) only")
    sem = sf.BatchScoreSemFilter(
        R.DATASET,
        target,
        tau2_card,
        base=R.episode_semantic_base(s),
        version=sf2["rubric"],
        workers=s.semantic_workers,
        threshold=threshold,
        batch=sf2["batch"],
        max_tokens=sf2["max_tokens"],
        guided=sf2.get("guided", True),
    )
    sem.endpoint_pool = s.semantic_pool
    proto = {
        "protocol": sf.PROTOCOL,
        "rubric": sf2["rubric"],
        "rubric_dataset": R.DATASET,
        "policy_rubric": policy_rubric,
        "top_n": sf2["top_n"],
        "prefilter_order": sf.PREFILTER_ORDER,
        "prefilter_rest": "not passed (dropped; not a semantic decision)",
        "batch": sf2["batch"],
        "max_tokens": sf2["max_tokens"],
        "output": "score-only",
        "guided": "json_schema strict (ids -> 11-level enum)"
        if sf2.get("guided", True)
        else "json_object",
        "useful_rule": "score >= semfilter_threshold",
        "fallback": "per-candidate score-only call for ids missing / duplicated / non-numeric in a batch answer",
        "judge_input": "raw",
        "judge_card": "tau2.adapter.tau2_card",
    }
    return sem, proto


class _RawView:
    """``semfilter_v2.receipt_block`` view stand-in (raw τ² cards: no sidecar misses)."""

    distilled_missing = 0


def semfilter_receipt(proto: dict, sem, receipt: dict) -> dict:
    """``receipt.semfilter`` of a τ² semfilter-v2 retrieval (calls, tokens, batch errors, cutoff sizes)."""
    from trimem.experiments import semfilter_v2 as sf

    return sf.receipt_block(proto, sem, _RawView(), receipt)


def semantic_ops_v2(
    plan: str,
    target: str,
    s: R.Settings,
    policy: str,
    threshold: float,
    sf2: dict | None = None,
):
    """(SemFilter, ColBERT ranker) of a v2 configuration: τ² rubric judge with the configuration's threshold and
    the outcome policy's rubric version (as ``agentgym_session_eval.retrieve_v2``), on the episode's endpoint;
    ColBERT on the episode's GPU. ``sf2`` (``semfilter_v2.settings()``, opt-in): the semfilter-v2 judge
    (``tau2_semfilter_v2``) instead of the v1 judge, its protocol block as ``sem.sf2_proto``."""
    from trimem.experiments.colbert_rank import ColBERTRank, rank_base_for
    from trimem.experiments.semantic_rubric import KNOWLEDGE, RubricSemFilter

    sem = ranker = None
    if plan in space.SEMFILTER_PLANS and sf2 is not None:
        sem, proto = tau2_semfilter_v2(target, s, threshold, sf2, mp.rubric(policy))
        sem.sf2_proto = proto
    elif plan in space.SEMFILTER_PLANS:
        sem = RubricSemFilter(
            "sweqa",
            target,
            tau2_card,
            base=R.episode_semantic_base(s),
            workers=s.semantic_workers,
            version=mp.rubric(policy),
            threshold=threshold,
        )
        sem.dataset = R.DATASET
        sem.system = sem.system.replace(KNOWLEDGE["sweqa"], R.TAU2_KNOWLEDGE)
        # request-load-v1 (2026-10-20): SemFilter judge calls go to the endpoint with the fewest running + waiting
        # requests (vLLM metrics, host-wide) instead of the episode's endpoint; agent / user / NL-judge calls stay
        # pinned to the episode endpoint (prefix cache)
        sem.endpoint_pool = s.semantic_pool
    if plan.endswith("+Rank"):
        ranker = ColBERTRank(
            target,
            mp.rank_view(tau2_card),
            base=s.rank_base
            or rank_base_for(R.episode_api_base(s).removesuffix("/v1")),
        )
    return sem, ranker


def source_regime(r: dict) -> str:
    return ((r.get("load") or {}).get("regime")) or LEGACY_REGIME


def result_memory_key(r: dict) -> str | None:
    """Memory key (``quality_reuse.memory_key``) of a memory-mode result file (None = no injected-memory record)."""
    form = (r.get("memory") or {}).get("memory_form") or {}
    if r.get("memory_text") is None or not form.get("memory_form"):
        return None
    return qr.memory_key(
        r["memory_text"], form["memory_form"], form.get("memory_forms_version", "")
    )


def result_episode_key(
    r: dict, experiment: str, run_policy: str = RUN_POLICY
) -> str | None:
    mk = result_memory_key(r)
    if mk is None:
        return None
    return qr.episode_key(
        experiment=experiment,
        split=r["split"],
        task=f"{r['domain']}:{r['task_id']}",
        trial=r.get("trial") or 0,
        trial_seed=r["trial_seed"],
        run_policy=run_policy,
        model=r["model"],
        mem_key=mk,
    )


def valid_source(
    r: dict | None,
    *,
    experiment: str,
    split: str,
    task: str,
    trial: int,
    model: str,
    mem_key: str,
) -> bool:
    """An ORIGINAL (not reused) evaluated memory episode of the same experiment / split / task / trial / model
    whose injected memory has ``mem_key``."""
    return (
        r is not None
        and not r.get("reuse")
        and r.get("mode") == "memory"
        and r.get("experiment") == experiment
        and r.get("split") == split
        and f"{r.get('domain')}:{r.get('task_id')}" == task
        and (r.get("trial") or 0) == trial
        and r.get("model") == model
        and result_memory_key(r) == mem_key
    )


class PoolReuse:
    """Quality reuse of one arm process (``quality_reuse``): key the episode's injected memory, reuse an evaluated
    same-key episode (waiting for an in-flight one), else claim the key for this episode."""

    def __init__(self, s: R.Settings, index: qr.ReuseIndex, regime: str):
        self.s, self.index, self.regime = s, index, regime
        self._src: dict[tuple, dict | None] = {}

    def _read(self, path: str) -> dict | None:
        try:
            st = Path(path).stat()
        except FileNotFoundError:
            return None
        ck = (path, st.st_mtime_ns, st.st_size)
        if ck not in self._src:
            self._src[ck] = R.read_result(Path(path))
        return self._src[ck]

    def validator(self, *, task: str, trial: int, mem_key: str):
        s = self.s

        def valid(entry: dict) -> dict | None:
            r = self._read(entry["source"])
            ok = valid_source(
                r,
                experiment=s.experiment,
                split=s.split,
                task=task,
                trial=trial,
                model=s.model,
                mem_key=mem_key,
            )
            return r if ok else None

        return valid

    def __call__(self, mem: dict, domain: str, task_id: str, form: dict) -> None:
        s = self.s
        trial = R.episode_trial(s)
        task = f"{domain}:{task_id}"
        mk = qr.memory_key(
            mem["memory"], form["memory_form"], form["memory_forms_version"]
        )
        ek = qr.episode_key(
            experiment=s.experiment,
            split=s.split,
            task=task,
            trial=trial,
            trial_seed=R.trial_seed(s.seed, trial),
            run_policy=RUN_POLICY,
            model=s.model,
            mem_key=mk,
        )
        mem["memory_key"], mem["reuse_key"] = mk, ek
        path = R.result_path(
            s.out_dir, domain, task_id, trial if s.num_trials > 1 else None
        )
        entry = {
            "source": str(Path(path).resolve()),
            "config_key": mem.get("space_config_key"),
            "memory_key": mk,
            "task": task,
            "trial": trial,
            "split": s.split,
            "regime": self.regime,
            "pid": os.getpid(),
        }
        kind, x = self.index.acquire(
            ek, self.validator(task=task, trial=trial, mem_key=mk), entry
        )
        if kind == "hit":
            e, src = x
            raise R.ReuseEpisode(
                {"entry": e, "source": src, "key": ek, "memory_key": mk}, mem
            )
        claims = R.EPISODE_CLAIMS.get()
        if claims is None:
            x.release()
            raise RuntimeError(
                "quality reuse needs runner.run_task (EPISODE_CLAIMS unbound)"
            )
        claims.append(x)


def write_reuse(
    task, domain: str, s: R.Settings, r: R.ReuseEpisode, path: Path
) -> dict:
    """Reuse record: the source episode's outcome, agent tokens and episode latency; THIS configuration's own
    retrieval (receipt, semantic tokens, retrieval ms). Latency = source wall - source retrieval + own retrieval."""
    hit, mem = r.hit, r.mem
    src = hit["source"]
    if src.get("task_id") != str(task.id) or src.get("domain") != domain:
        raise ValueError("reuse source belongs to another task")
    smem = src.get("memory") or {}
    src_ret = float(smem.get("retrieval_ms") or 0.0)
    wall = float(src["wall_ms"]) - src_ret + float(mem["retrieval_ms"])
    reused_from = {
        "path": hit["entry"]["source"],
        "config_key": smem.get("space_config_key"),
        "retrieval_id": src.get("retrieval_id"),
        "regime": source_regime(src),
    }
    key = ent.task_key(domain, str(task.id))
    if s.store is not None:
        s.store.generation(
            mem["rid"],
            {
                "reward": float(src["reward"]),
                "llm_calls": src.get("agent_calls"),
                "generation_tokens": src.get("agent_tokens"),
                "user_sim_tokens": src.get("user_tokens"),
                "semantic_tokens": mem["semantic_tokens"],
                "episode_ms": float(src["wall_ms"]) - src_ret,
                "retrieval_ms": mem["retrieval_ms"],
                "memory_chars": len(mem["memory"]),
                "memory_sha256": mem["memory_sha256"],
                "memory_key": hit["memory_key"],
                "result_file": str(Path(path).resolve()),
                "reused_from": str(src.get("retrieval_id")),
                "reused_from_file": reused_from["path"],
                "memory_policy": mem.get("memory_policy"),
                "card_outcomes": mem.get("card_outcomes"),
            },
            total_tokens=(src.get("agent_tokens") or 0) + mem["semantic_tokens"],
            e2e_ms=wall,
        )
        s.store.evaluate_scalar(
            mem["rid"],
            R.quality_of(src["reward"]),
            {
                "reward": src["reward"],
                "metric": "tau2 reward",
                "reused_from": str(src.get("retrieval_id")),
            },
            s.baseline.get(key),
        )
    base = R._base_result(task, domain, s)
    out = {
        **src,
        "experiment": s.experiment,
        "config": base["config"],
        "trial": base["trial"],
        "num_trials": base["num_trials"],
        "trial_seed": base["trial_seed"],
        **s.result_extra,
        "retrieval_id": mem["rid"],
        "memory": {k: v for k, v in mem.items() if k != "memory"},
        "memory_text": mem["memory"],
        "first_user_message": mem.get("query_text"),
        "wall_ms": wall,
        "reuse": True,
        "reuse_protocol": qr.PROTOCOL,
        "reuse_key": hit["key"],
        "memory_key": hit["memory_key"],
        "reused_from": reused_from,
        "source_regime": reused_from["regime"],
        "source_wall_ms": float(src["wall_ms"]),
        "source_retrieval_ms": src_ret,
        "source_semantic_tokens": smem.get("semantic_tokens", 0),
        "resources": rss_guard.usage(),
    }
    R.write_result(path, out)
    return out


class PoolProvider:
    """``provider(task, first_user_message)`` of ``TrimemMemoryAgent`` for one τ²-v2 configuration."""

    def __init__(
        self, s: R.Settings, cfg: dict, slots: HostSlots, reuse: PoolReuse | None = None
    ):
        self.s, self.cfg, self.slots = s, space.validate(cfg), slots
        self.reuse = reuse
        self.config = session_config(self.cfg)
        self.policy = outcome_policy(self.cfg)
        self.v2 = space.runner_args(self.cfg)["v2"]
        self.form = self.v2["memory_form"]
        self.distilled = (
            mf.DistilledStore.for_dataset(R.DATASET)
            if self.form == "distilled"
            else None
        )
        self.form_protocol = {
            "memory_form": self.form,
            "memory_forms_version": mf.VERSION,
            "token_counter": mf.counter(),
            **(self.distilled.protocol() if self.distilled else {}),
        }

    def __call__(self, task, text: str) -> dict:
        from trimem.experiments.session_profile_store import recorded_retrieve

        s, cfg, policy = self.s, self.cfg, self.policy
        tdict = (
            task.model_dump(mode="json") if hasattr(task, "model_dump") else dict(task)
        )
        domain = R.CURRENT_DOMAIN.get()
        if domain not in ent.DOMAINS:
            raise ValueError(
                f"cannot determine the τ² domain of task {tdict.get('id')!r}"
            )
        q = ent.identity(domain, tdict, strict=s.strict_entities)
        t0 = time.monotonic()
        vector = s.embed_fn(tr.task_text(domain, text))
        # semfilter-v2 (user 2026-10-22): OPT-IN via $TRIMEM_SEMFILTER_V2=1 (unset = the v1 judge, unchanged); the
        # pool driver exports the knobs recorded in state.json ``semfilter`` (tau2.sfv2_rerun)
        from trimem.experiments import semfilter_v2

        sf2 = semfilter_v2.settings() if cfg["plan"] in space.SEMFILTER_PLANS else None
        sem, ranker = semantic_ops_v2(
            cfg["plan"], text, s, policy, cfg["semfilter_threshold"], sf2
        )
        meta = {
            "experiment_id": s.experiment,
            "dataset": R.DATASET,
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
                "isolation_rule": R.ISOLATION_BY_POLICY[policy],
                "strict_entities": s.strict_entities,
                "query": "first user message",
                "model": s.model,
                "seed": s.seed,
                "trial": R.episode_trial(s),
                "num_trials": s.num_trials,
                "endpoint": R.CURRENT_ENDPOINT.get(),
                "semantic_routing": "request-load-v1"
                if s.semantic_pool
                else "episode-endpoint",
                "endpoint_routing": ROUTING,
                "semantic": (
                    f"{sf2['rubric']}:tau2"
                    if sf2 is not None
                    else f"{mp.rubric(policy)}:tau2"
                )
                if sem
                else None,
                **({"semfilter": sem.sf2_proto} if sf2 is not None else {}),
                "rank": "colbert" if ranker else None,
                **mp.protocol(s.memory_policy),
                "session_v2": self.v2,
                "retrieval_outcome_policy": policy,
                "seed_widening": True,
                "space_version": space.VERSION,
                "space_config": cfg,
                "memory_form": self.form_protocol,
                **s.protocol_extra,
            },
        }
        kw = {}
        if sem is not None:
            kw["semfilter"] = sem
        if sf2 is not None:
            kw["prefilter_top_n"] = sf2["top_n"]
        if ranker is not None:
            kw["semrank"] = ranker
        factory = s.adapter_factory or (
            lambda v, qq: WidenedTau2Adapter(v, qq, s.store_dsn, policy=policy)
        )
        # retrieval-slot-v2 (user 2026-10-21): the slot (= this retrieval's PG 55450 connection) is held only for
        # the DB phases; freed, with the adapter connection closed, while SemFilter / ColBERT calls run. A custom
        # ``adapter_factory`` (tests) keeps its own connection: only the slot is freed.
        reconnect = (
            None
            if s.adapter_factory
            else (lambda: retrieval_lease.readonly_connect(s.store_dsn))
        )
        with retrieval_lease.RetrievalLease(self.slots.slot, reconnect) as lease:
            adapter = lease.bind(factory(vector, q))
            if getattr(adapter, "policy", policy) != policy:
                adapter.close()
                raise ValueError("retrieval adapter runs under another outcome policy")
            try:
                receipt = recorded_retrieve(
                    adapter,
                    self.config,
                    lease.gated(s.store),
                    meta,
                    pause=lease.pause,
                    **kw,
                )
            finally:
                adapter.close()
        memory, form_receipt = mf.render_memory_tau2(
            receipt["cards"],
            self.form,
            llm_budget.per_card(self.config.top_k) * 3,
            policy,
            self.distilled,
        )
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
            "memory_policy": policy,
            "card_outcomes": mp.card_outcomes(receipt["cards"]),
            "selected_source_task_ids": [c["task_id"] for c in receipt["cards"]],
            "selected_source_task_keys": [c["key"] for c in receipt["cards"]],
            "space_config_key": space.key(cfg),
            "memory_form": form_receipt
            | {"memory_tokens": mf.count_tokens(memory) if memory else 0},
            # semfilter-v2: the labelled judge receipt in every result file (None = v1 judge / no SemFilter)
            "semfilter": semfilter_receipt(sem.sf2_proto, sem, receipt)
            if sf2 is not None
            else None,
            "v2": {
                k: receipt.get(k)
                for k in (
                    "gated",
                    "pool_size",
                    "pool_cap_hit",
                    "expansion_truncated",
                    "best_cosine",
                )
            }
            | {"n_seed_tasks": len(receipt.get("seed_ids") or [])},
            "query_text": text,
        }
        if (
            self.reuse is not None
        ):  # may raise ReuseEpisode (identical memory: reuse the earlier episode)
            self.reuse(mem, domain, q["task_id"], form_receipt)
        return mem


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    ap.add_argument("--mode", required=True, choices=("memory", "nomemory"))
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument(
        "--task_keys",
        default=None,
        help="comma-separated domain:task keys (default: whole split)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--space_config", default=None, help="τ²-v2 configuration (JSON; memory mode)"
    )
    ap.add_argument("--experiment", default=None)
    ap.add_argument(
        "--baseline_dir",
        default=None,
        help="paired No-Memory dir (opt paired_gain; may be partial)",
    )
    ap.add_argument("--protocol_extra", default="{}")
    ap.add_argument("--model", default=R.MODEL)
    ap.add_argument(
        "--api_bases", default="http://127.0.0.1:8030/v1,http://127.0.0.1:8031/v1"
    )
    ap.add_argument(
        "--rank_base",
        default=None,
        help="fixed ColBERT base (default: the episode GPU's, LLM port + 10)",
    )
    ap.add_argument("--embed_url", default=R.EMBED)
    ap.add_argument("--store_dsn", default=R.STORE_DSN)
    ap.add_argument("--opt_dsn", default=R.OPT_DSN)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--num_trials", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_errors", type=int, default=10)
    ap.add_argument("--semantic_workers", type=int, default=4)
    ap.add_argument("--max_concurrency", type=int, default=4)
    ap.add_argument(
        "--llm_http_connections",
        type=int,
        default=0,
        help="litellm HTTP pool (0 = max(16, 2 x --max_concurrency); official default 10)",
    )
    ap.add_argument(
        "--retrieval_slots",
        type=int,
        default=12,
        help="host-wide concurrent retrievals (0 = no cap)",
    )
    ap.add_argument(
        "--semantic_routing",
        default="request-load",
        choices=("request-load", "episode"),
        help="SemFilter judge calls: least running+waiting endpoint per request (vLLM metrics; default) or the "
        "episode's endpoint",
    )
    ap.add_argument(
        "--retrieval_fast_reserve",
        type=int,
        default=-1,
        help="slots reserved for non-semantic (VGR) retrievals; semantic plans use the rest (-1 = n // 3)",
    )
    ap.add_argument(
        "--endpoint_lease_dir",
        type=Path,
        default=LEASE_DIR,
        help="host-wide episode leases (endpoint_affinity): every arm process balances on the same counts",
    )
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--max_rss_gb", type=float, default=4.0)
    ap.add_argument(
        "--drain_file",
        type=Path,
        default=None,
        help="when this file exists no new episode starts; in-flight ones finish; exit 75 (rebalancing)",
    )
    ap.add_argument(
        "--reuse_index",
        type=Path,
        default=None,
        help="quality reuse (quality-reuse-v1): host-wide index file of this experiment + split (memory mode)",
    )
    ap.add_argument(
        "--load_regime",
        default=None,
        help="load regime label recorded in every result (e.g. c100); latency comparability",
    )
    ap.add_argument("--total_concurrency", type=int, default=0)
    ap.add_argument("--parallel_arms", type=int, default=0)
    ap.add_argument("--log_level", default="ERROR")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if (
        a.num_trials < 1
        or a.max_concurrency < 1
        or a.semantic_workers < 1
        or a.max_rss_gb < 0
    ):
        ap.error(
            "--num_trials / --max_concurrency / --semantic_workers >= 1, --max_rss_gb >= 0"
        )
    if (a.mode == "memory") != (a.space_config is not None):
        ap.error("--space_config is required in memory mode and only there")
    if a.mode == "memory" and not a.experiment:
        ap.error("--experiment is required in memory mode")
    cfg = space.validate(json.loads(a.space_config)) if a.space_config else None
    if cfg is not None and cfg["plan"] == "NoMemory":
        ap.error("NoMemory runs with --mode nomemory")
    api_bases = tuple(
        b.strip().rstrip("/") for b in a.api_bases.split(",") if b.strip()
    )
    if not api_bases:
        ap.error("--api_bases is empty")
    out = Path(a.out_dir)
    if a.mode == "memory":
        # claim the memory-policy marker BEFORE arm.json exists: claim_dir treats any *.json as old results.
        # Heal dirs where an earlier launch wrote only arm.json and then died on that check.
        if not (out / mp.MARKER).exists() and [p.name for p in out.glob("*.json")] == [
            ARM_FILE
        ]:
            (out / ARM_FILE).unlink()
        mp.claim_dir(out, RUN_POLICY)
    claim_arm(
        out,
        arm_identity(
            mode=a.mode,
            split=a.split,
            cfg=cfg,
            experiment=a.experiment,
            num_trials=a.num_trials,
            seed=a.seed,
        ),
    )
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=a.log_level)
    s = R.Settings(
        mode=a.mode,
        out_dir=out,
        split=a.split,
        opt_split=a.split,
        model=a.model,
        api_base=api_bases[0],
        seed=a.seed,
        max_steps=a.max_steps,
        max_errors=a.max_errors,
        protocol_extra=json.loads(a.protocol_extra),
        rank_base=a.rank_base,
        store_dsn=a.store_dsn,
        opt_dsn=a.opt_dsn,
        memory_policy=RUN_POLICY,
        num_trials=a.num_trials,
        api_bases=api_bases,
        semantic_workers=a.semantic_workers,
        pool=HostEndpointPool(api_bases, a.endpoint_lease_dir),
        drain_file=a.drain_file,
    )
    if a.load_regime:
        s.result_extra = {
            "load": {
                "regime": a.load_regime,
                "arm_workers": a.max_concurrency,
                "total_concurrency": a.total_concurrency or None,
                "parallel_arms": a.parallel_arms or None,
            }
        }
    http = a.llm_http_connections or max(16, 2 * a.max_concurrency)
    R.configure_http_pool(http)
    R.configure_judge(a.model, R.llm_args(api_bases[0], json_mode=True))
    if a.mode == "memory":
        from trimem.experiments.session_store_ext import ScalarSessionStore
        from trimem.experiments.tau2.agent import register, set_memory_provider

        s.config = session_config(cfg)
        mp.claim_dir(out, RUN_POLICY)
        s.experiment = a.experiment
        s.baseline = (
            R.load_baseline(Path(a.baseline_dir))
            if a.baseline_dir and Path(a.baseline_dir).exists()
            else {}
        )
        s.store = ScalarSessionStore(a.opt_dsn)
        s.embed_fn = R.http_embed(a.embed_url)
        register()
        if (
            a.semantic_routing == "request-load"
            and len(api_bases) >= 2
            and cfg["plan"] in space.SEMFILTER_PLANS
        ):
            from trimem.experiments.tau2.semantic_pool import SemanticRequestPool

            s.semantic_pool = SemanticRequestPool(api_bases).start()
        reuse = None
        if a.reuse_index is not None:
            reuse = PoolReuse(
                s, qr.ReuseIndex(a.reuse_index), a.load_regime or "unknown"
            )
            s.reuse_writer = write_reuse
        start = (
            fast_reserve(a.retrieval_slots, a.retrieval_fast_reserve)
            if slow_plan(cfg["plan"])
            else 0
        )
        set_memory_provider(
            PoolProvider(s, cfg, HostSlots(a.retrieval_slots, start=start), reuse)
        )
    wanted = parse_task_keys(a.task_keys)
    items = select_items({d: R.load_split(d, a.split) for d in ent.DOMAINS}, wanted)
    if not items:
        raise SystemExit("no tasks selected")
    print(
        json.dumps(
            {
                "tasks": len(items),
                "num_trials": s.num_trials,
                "episodes": len(items) * s.num_trials,
                "api_bases": list(api_bases),
                "endpoint_routing": ROUTING,
                "max_concurrency": a.max_concurrency,
                "llm_http_connections": http,
                "retrieval_slots": a.retrieval_slots,
                "reuse_index": str(a.reuse_index) if a.reuse_index else None,
                "load": s.result_extra.get("load"),
                "mode": a.mode,
                "split": a.split,
                "experiment": s.experiment,
                "space_config": cfg,
                "out_dir": str(out),
            }
        ),
        flush=True,
    )
    guard = rss_guard.RssGuard(a.max_rss_gb)
    guard.start_watchdog()
    try:
        stats = R.run_all(
            R.trial_items(items, s.num_trials),
            s,
            a.max_concurrency,
            a.max_retries,
            guard=guard,
        )
        stats["episodes_per_endpoint"] = dict(s.pool.assigned)
    finally:
        guard.stop()
    print(json.dumps(stats), flush=True)
    if stats["stopped_rss"]:
        return rss_guard.EXIT_RSS
    if stats["failed"] or stats.get("stopped_service"):
        return 1
    return 75 if stats.get("stopped_requested") else 0


if __name__ == "__main__":
    raise SystemExit(main())
