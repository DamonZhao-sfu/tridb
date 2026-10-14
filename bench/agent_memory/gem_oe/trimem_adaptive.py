"""TriMEM with a downstream-aware plan choice per evolution step (plan v0.2.0, D1/D3).

Every step: one tri-modal pool of M improving descendants for the parent, one SemScore and one
SemFilter pass over it (`trimem.objective.pool`), 13 candidates (none + {trimodal, hard, rank} ×
k ∈ {1,3,5,8}), a policy choice (`trimem.objective.policy`: random / O1 / O2 × offline / online),
and injection of the chosen candidate's programs. The decision and, once the child is evaluated,
its outcome (Δ = max(0, child − parent), validity) go to `trimem_outcome` with the propensity, so
the random arm is an unbiased exploration log and the adaptive arms are auditable receipts.

`observe(child, iteration)` is installed on `GemMemoryDatabase.add` by run_arm. OpenEvolve runs
several iterations concurrently, so a child is matched to its decision by OpenEvolve iteration
(decision counter + offset) and checked against the parent id; a mismatch falls back to the
oldest open decision of that parent and is recorded.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from bench.agent_memory.gem_oe.memory_database import RetrievedProgram
from trimem.objective.models import BayesLinear, HurdleModel
from trimem.objective.policy import Policy, PolicyV2
from trimem.objective.pool import build_pool, build_pools, featurize, featurize_v2
from trimem.objective.store import OUTCOME_DDL, count_tokens
from trimem.replay.memory import Embedder

INSERT_SQL = """
INSERT INTO trimem_outcome (session, task, iteration, policy, candidate, propensity, features,
                            candidates, parent_fitness, tok_injected, ret_tokens, ret_ms,
                            best_fitness, space)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (session, iteration) DO UPDATE SET
  policy = EXCLUDED.policy, candidate = EXCLUDED.candidate, propensity = EXCLUDED.propensity,
  features = EXCLUDED.features, candidates = EXCLUDED.candidates,
  parent_fitness = EXCLUDED.parent_fitness, tok_injected = EXCLUDED.tok_injected,
  ret_tokens = EXCLUDED.ret_tokens, ret_ms = EXCLUDED.ret_ms, decided_at = now(),
  best_fitness = EXCLUDED.best_fitness, space = EXCLUDED.space, delta_best = NULL,
  child_fitness = NULL, delta = NULL, valid = NULL, tok_child = NULL, observed_at = NULL
"""
OBSERVE_SQL = """
UPDATE trimem_outcome SET child_fitness = %s, delta = %s, valid = %s, tok_child = %s,
       delta_best = %s, observed_at = now()
WHERE session = %s AND iteration = %s
"""


def load_prior(path: str | None) -> dict[str, Any]:
    """{'hurdle': HurdleModel, 'X': ndarray, 'delta': ndarray} written by trimem.objective.train."""
    if not path:
        return {}
    return pickle.loads(Path(path).read_bytes())


def build_policy(
    kind: str,
    mode: str,
    *,
    prior_path: str | None,
    seed: int,
    epsilon: float,
    lam_t: float,
    lam_a: float,
) -> Policy:
    prior = load_prior(prior_path)
    if kind != "random" and not prior:
        raise SystemExit(
            f"--policy {kind} needs --policy-prior (train it with trimem.objective.train)"
        )
    offline: HurdleModel | None = prior.get("hurdle")
    online = None
    if mode == "online-cold":
        online = BayesLinear.cold(
            prior["X"]
        )  # only the feature scaling comes from the log
    elif mode == "online-warm":
        online = BayesLinear.warm(prior["X"], prior["delta"])
    return Policy(
        kind=kind,
        mode=mode,
        epsilon=epsilon,
        lam_t=lam_t,
        lam_a=lam_a,
        seed=seed,
        offline=offline,
        online=online,
    )


def build_policy_v2(kind: str, mode: str, *, prior_path: str | None, seed: int, epsilon: float,
                    lam_t: float, tol: float, space: str = "v2", lam_l: float = 0.1,
                    out_budget: float | None = None) -> PolicyV2:
    """Prior pickle (trimem.objective.train --space v2): {'mem': HurdleModel, 'none': HurdleModel,
    'X_mem', 'y_mem', 'X_none', 'y_none'}; labels are Δ_best."""
    prior = load_prior(prior_path)
    if kind not in ("random", "fixed") and not prior:
        raise SystemExit(f"--policy {kind} needs --policy-prior (train it with trimem.objective.train --space v2)")
    pol = PolicyV2(kind=kind, mode=mode, epsilon=epsilon, lam_t=lam_t, seed=seed, tol=tol, space=space,
                   lam_l=lam_l, offline_mem=prior.get("mem"), offline_none=prior.get("none"))
    if kind in ("o4", "o5"):
        from trimem.objective.policy import LatencyModel

        a, b, d = prior["latency"]
        pol.latency = LatencyModel(a, b, d, prior["out_model"], prior["valid_model"])
    if kind == "o5":
        pol.out_budget = float(out_budget if out_budget is not None else prior.get("out_budget", 0.0))
        if mode.startswith("online"):  # P(new best) online models, scaled to Var(1[Δ_best > 0])
            vb = prior.get("var_bin", 0.1)
            pk = {"tau2": vb, "sigma2": vb}
            if mode == "online-warm":
                pol.online_pmem = BayesLinear.warm(prior["X_mem"], (prior["y_mem"] > 0).astype(float), **pk)
                pol.online_pnone = BayesLinear.warm(prior["X_none"], (prior["y_none"] > 0).astype(float), **pk)
            else:
                pol.online_pmem = BayesLinear.cold(prior["X_mem"], **pk)
                pol.online_pnone = BayesLinear.cold(prior["X_none"], **pk)
    # v3: the online prior and noise are scaled to the observed spread of Δ_best (τ² = σ² = Var Δ_best);
    # v2 used τ² = 1, σ² = 0.01, which made Thompson samples dwarf the signal.
    kw = {"tau2": prior["var_y"], "sigma2": prior["var_y"]} if space == "v3" and "var_y" in prior else {}
    if mode == "online-cold":  # only the feature scaling comes from the log
        pol.online_mem, pol.online_none = BayesLinear.cold(prior["X_mem"], **kw), BayesLinear.cold(prior["X_none"], **kw)
    elif mode == "online-warm":
        pol.online_mem = BayesLinear.warm(prior["X_mem"], prior["y_mem"], **kw)
        pol.online_none = BayesLinear.warm(prior["X_none"], prior["y_none"], **kw)
    return pol


@dataclass
class _Open:
    counter: int
    parent_id: str
    parent_fitness: float
    features: dict
    best_fitness: float = 0.0


@dataclass
class TrimemAdaptiveRetriever:
    dsn: str
    policy: Policy | PolicyV2
    session: str
    horizon: int
    prompt_version: str = "oe"
    telemetry_path: Path | None = None
    embedding_url: str = "http://127.0.0.1:8011/v1"
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    db: Any = None  # GemMemoryDatabase, for the own inspirations a candidate replaces
    require_changes: bool = False  # change-summary injection (ALE): pools keep only rows with a summary

    def __post_init__(self) -> None:
        self.name = f"trimem-adaptive[{self.policy.name}]"
        self.embedder = Embedder(self.embedding_url)
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.conn.execute(OUTCOME_DDL)
        for stmt in (
            "SET jit = off",
            "SET trimem.timeout_ms = 900000",
            "SET hnsw.iterative_scan = relaxed_order",
            "SET hnsw.ef_search = 100",
            f"SET trimem.prompt_version = '{self.prompt_version}'",
        ):
            self.conn.execute(stmt)
        self.offset: int | None = None
        self.open: dict[int, _Open] = {}
        self._own_tokens: dict[str, int] = {}

    # -- decision ---------------------------------------------------------

    def _own_token_list(self) -> list[int]:
        own: Sequence[Any] = getattr(self.db, "last_own", None) or []
        out = []
        for p in own:
            if p.id not in self._own_tokens:
                self._own_tokens[p.id] = count_tokens(p.code)
            out.append(self._own_tokens[p.id])
        return out

    def _refresh_token_model(self) -> None:
        if not (self.telemetry_path and self.telemetry_path.exists()):
            return
        recs = [
            json.loads(line)
            for line in self.telemetry_path.read_text().splitlines()
            if line.strip()
        ]
        recs = [
            r for r in recs if r.get("status") == "success" and r.get("prompt_tokens")
        ]
        if recs:
            self.policy.tokens.refresh(
                float(np.mean([r["prompt_tokens"] for r in recs])),
                float(np.mean([r["completion_tokens"] for r in recs])),
            )

    def retrieve(
        self, *, task_uid: str, parent: Any, k: int, iteration: int
    ) -> list[RetrievedProgram]:
        started = time.perf_counter()
        self._task = task_uid
        if self.offset is None and self.db is not None:
            # OpenEvolve iteration = counter + offset; a resumed run's counter starts at last_iteration
            self.offset = int(getattr(self.db, "last_iteration", 0) or 0) - iteration + 1
        fitness = float((parent.metrics or {}).get("combined_score") or 0.0)
        q, emb_tokens, emb_ms = self.embedder.embed(parent.code)
        if isinstance(self.policy, PolicyV2):
            return self._retrieve_v2(task_uid, parent, k, iteration, fitness, q, emb_tokens, emb_ms, started)
        pool = build_pool(
            self.conn,
            task=task_uid,
            q=q,
            parent_fitness=fitness,
            prompt_version=self.prompt_version,
        )
        cands = [
            c for c in pool.candidates if c.k <= k
        ]  # never more than the slot budget
        parent_tokens = self._own_tokens.setdefault(
            parent.id, count_tokens(parent.code)
        )
        for c in cands:
            c.features = featurize(
                c,
                parent_fitness=fitness,
                iteration=iteration,
                horizon=self.horizon,
                parent_tokens=parent_tokens,
                parent_session=None,
            )
        self._refresh_token_model()
        own_tokens = self._own_token_list()
        decision = self.policy.choose(
            cands, own_tokens=own_tokens, sem_tokens=pool.sem_tokens
        )
        chosen = cands[decision.index]
        receipt = [
            dict(r, features=c.features)
            for r, c in zip(decision.receipt, cands, strict=True)
        ]
        oe_iter = (self.offset or 0) + iteration
        self.conn.execute(
            INSERT_SQL,
            (
                self.session,
                task_uid,
                oe_iter,
                self.policy.name,
                chosen.name,
                decision.propensity,
                json.dumps(chosen.features),
                json.dumps(receipt),
                fitness,
                int(chosen.features["tokens_injected"]),
                pool.sem_tokens,
                pool.ret_ms + pool.sem_ms,
                None,
                "v1",
            ),
        )
        self.open[oe_iter] = _Open(iteration, parent.id, fitness, chosen.features)
        self.telemetry.append(
            {
                "iteration": iteration,
                "oe_iteration": oe_iter,
                "k_budget": k,
                "pool": len(pool.rows),
                "chosen": chosen.name,
                "propensity": decision.propensity,
                "returned": len(chosen.rows),
                "ret_ms": pool.ret_ms,
                "sem_ms": pool.sem_ms,
                "sem_tokens": pool.sem_tokens,
                "embed_ms": emb_ms,
                "embed_tokens": emb_tokens,
                "own_slots": len(own_tokens),
                "total_ms": (time.perf_counter() - started) * 1000,
                "receipt": decision.receipt,
                "ids": [r.id for r in chosen.rows],
                "code_chars": [len(r.code) for r in chosen.rows],
            }
        )
        return [
            RetrievedProgram(
                uid=self._uid(r.id),
                code=r.code,
                metrics={"combined_score": float(r.fitness)},
                changes_description=r.changes or "",
                provenance={
                    "memory": "trimem-adaptive",
                    "candidate": chosen.name,
                    "attempt_id": r.id,
                    "parent_fitness": float(r.parent_fitness),
                    "score": r.score,
                    "verdict": r.verdict,
                },
            )
            for r in chosen.rows
        ]

    def _scale(self, task_uid: str) -> float:
        """|seed score| for ALE (scale-free features), 1 for math."""
        if not task_uid.startswith("ale:"):
            return 1.0
        if not hasattr(self, "_scale_cache"):
            r = self.conn.execute(
                "SELECT fitness FROM attempt WHERE task = %s AND parent_id IS NULL AND cell IS NULL AND is_valid "
                "GROUP BY fitness ORDER BY count(*) DESC LIMIT 1", (task_uid,)).fetchone()
            self._scale_cache = max(abs(float(r[0])), 1e-9) if r else 1.0
        return self._scale_cache

    def _run_best(self, parent_fitness: float) -> float:
        own = self.db.own_programs().values() if self.db is not None else []
        return max([parent_fitness, *(float((p.metrics or {}).get("combined_score") or 0.0) for p in own)])

    def _retrieve_v2(self, task_uid: str, parent: Any, k: int, iteration: int, fitness: float,
                     q: list[float], emb_tokens: int, emb_ms: float, started: float) -> list[RetrievedProgram]:
        from trimem.objective.space import GRAPH_SPACE, GRAPH_SPACE_V3

        res = build_pools(self.conn, task=task_uid, q=q, parent_fitness=fitness, prompt_version=self.prompt_version,
                          configs=GRAPH_SPACE_V3 if getattr(self.policy, "space", "v2") == "v3" else GRAPH_SPACE,
                          require_changes=self.require_changes)
        best = self._run_best(fitness)
        cands = [c for c in res.candidates if c.k <= k]
        parent_tokens = self._own_tokens.setdefault(parent.id, count_tokens(parent.code))
        for c in cands:
            c.features = featurize_v2(c, parent_fitness=fitness, best_fitness=best, iteration=iteration,
                                      horizon=self.horizon, parent_tokens=parent_tokens,
                                      pool_n=len(res.pools[c.graph.name]) if c.graph else 0,
                                      union_n=res.union_n, scale=self._scale(task_uid))
        self._refresh_token_model()
        own_tokens = self._own_token_list()
        decision = self.policy.choose(cands, own_tokens=own_tokens, sem_tokens=res.sem_tokens)
        chosen = cands[decision.index]
        oe_iter = (self.offset or 0) + iteration
        self.conn.execute(INSERT_SQL, (
            self.session, task_uid, oe_iter, self.policy.name, chosen.name, decision.propensity,
            json.dumps(chosen.features), json.dumps(decision.receipt), fitness,
            int(chosen.features["tokens_injected"]), res.sem_tokens, res.ret_ms + res.sem_ms, best,
            getattr(self.policy, "space", "v2")))
        self.open[oe_iter] = _Open(iteration, parent.id, fitness, chosen.features, best)
        self.telemetry.append({
            "iteration": iteration, "oe_iteration": oe_iter, "k_budget": k, "pool": res.union_n,
            "chosen": chosen.name, "propensity": decision.propensity, "returned": len(chosen.rows),
            "ret_ms": res.ret_ms, "sem_ms": res.sem_ms, "sem_tokens": res.sem_tokens, "best": best,
            "embed_ms": emb_ms, "embed_tokens": emb_tokens, "own_slots": len(own_tokens),
            "total_ms": (time.perf_counter() - started) * 1000, "receipt": decision.receipt,
            "ids": [r.id for r in chosen.rows], "code_chars": [len(r.code) for r in chosen.rows],
        })
        return [
            RetrievedProgram(
                uid=self._uid(r.id), code=r.code, metrics={"combined_score": float(r.fitness)},
                changes_description=r.changes or "",
                provenance={"memory": "trimem-adaptive-v2", "candidate": chosen.name, "attempt_id": r.id,
                            "parent_fitness": float(r.parent_fitness), "score": r.score, "verdict": r.verdict},
            )
            for r in chosen.rows
        ]

    def _uid(self, attempt_id: int) -> str:
        return self.conn.execute(
            "SELECT uid FROM attempt WHERE id = %s", (attempt_id,)
        ).fetchone()[0]

    # -- outcome ----------------------------------------------------------

    def observe(self, child: Any, iteration: int | None) -> None:
        if not child.parent_id or iteration is None:
            return
        match = "iteration"
        dec = self.open.get(iteration)
        if dec is None or dec.parent_id != child.parent_id:
            same = sorted(
                (i for i, d in self.open.items() if d.parent_id == child.parent_id)
            )
            if not same:
                return  # a child of an iteration that made no decision (e.g. retrieval skipped)
            match, iteration = "parent_fifo", same[0]
            dec = self.open[iteration]
        del self.open[iteration]
        score = float((child.metrics or {}).get("combined_score") or 0.0)
        delta = max(0.0, score - dec.parent_fitness)
        delta_best = max(0.0, score - dec.best_fitness) / self._scale(getattr(self, "_task", ""))
        self.conn.execute(
            OBSERVE_SQL,
            (
                score,
                delta,
                score > 0,
                count_tokens(child.code),
                delta_best,
                self.session,
                iteration,
            ),
        )
        self.policy.observe(dec.features, delta_best if isinstance(self.policy, PolicyV2) else delta)
        self.telemetry.append(
            {
                "observed": iteration,
                "match": match,
                "child_fitness": score,
                "delta": delta,
            }
        )

    def close(self) -> None:
        """Iterations whose generation failed produce no child: Δ = 0, not valid."""
        for it, dec in list(self.open.items()):
            self.conn.execute(OBSERVE_SQL, (None, 0.0, False, None, 0.0, self.session, it))
            self.policy.observe(dec.features, 0.0)
        self.open.clear()
        if self.telemetry_path is not None:  # policy state (online posterior, token model, RNG) for a resume
            (self.telemetry_path.parent / "policy.pkl").write_bytes(pickle.dumps(self.policy))


def install_observer(memory_db: Any, retriever: TrimemAdaptiveRetriever) -> None:
    """Route every program OpenEvolve adds (children, not the external injections) to observe()."""
    retriever.db = memory_db
    base_add = memory_db.add

    def add(
        program: Any, iteration: int | None = None, target_island: int | None = None
    ) -> str:
        pid = base_add(program, iteration=iteration, target_island=target_island)
        retriever.observe(program, iteration)
        return pid

    memory_db.add = add
