"""D3: objectives O1/O2 and the per-iteration policies (random, adaptive offline/online).

O1  argmin_c  Tok(c) / max(ĝ(c), ε)                         (tokens per unit of expected improvement)
O2  argmax_c  ĝ(c)/max_c ĝ − λ_T·norm(Tok(c)) − λ_A·Anchor(c)  (AgentQE / CA-RAG utility form)
Tok(c) = T0 + T_inj(c) − T_repl(c) + Ŝ_out + R_sem(c): base prompt, injected minus replaced own
program tokens (exact), mean output tokens, and the semantic-operator tokens of the families that
read the semantic passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trimem.objective.models import BayesLinear, HurdleModel
from trimem.objective.pool import Candidate, vector

EPS = 1e-3


@dataclass
class TokenModel:
    """Prompt + output tokens of the generation call each candidate would cause.

    Injection replaces inspiration slots (policy ``fixed``): injecting k programs drops the first
    min(k, n_own) of the run's own inspirations, so the prompt changes by
    Σ tokens(injected) − Σ tokens(replaced own) — both exact at plan time. The base prompt and the
    output length are running means from the cell's LLM telemetry.
    """

    base_prompt: float = 8000.0
    out_tokens: float = 1500.0

    def refresh(self, prompt_mean: float | None, out_mean: float | None) -> None:
        if prompt_mean:
            self.base_prompt = prompt_mean
        if out_mean:
            self.out_tokens = out_mean

    def tokens(self, c: Candidate, own_tokens: list[int], sem_tokens: int) -> float:
        replaced = sum(own_tokens[: min(len(c.rows), len(own_tokens))])
        sem = (
            sem_tokens / 2 if c.family in ("hard", "rank") else 0.0
        )  # two families share the passes
        return (
            self.base_prompt
            + c.features["tokens_injected"]
            - replaced
            + self.out_tokens
            + sem
        )


def objective_scores(
    kind: str,
    g: np.ndarray,
    tok: np.ndarray,
    anchor: np.ndarray,
    *,
    lam_t: float = 0.2,
    lam_a: float = 0.0,
) -> np.ndarray:
    """Higher is better for both objectives (O1 is negated)."""
    if kind == "o1":
        return -(tok / np.maximum(g, EPS)) - lam_a * anchor * tok.mean() / max(
            g.mean(), EPS
        )
    gmax = max(float(g.max()), EPS)
    span = float(tok.max() - tok.min()) or 1.0
    return g / gmax - lam_t * (tok - tok.min()) / span - lam_a * anchor


@dataclass
class Decision:
    index: int
    propensity: float
    receipt: list[dict]


@dataclass
class Policy:
    kind: str  # random | o1 | o2
    mode: str = "offline"  # offline | online-cold | online-warm
    epsilon: float = 0.1
    lam_t: float = 0.2
    lam_a: float = 0.0
    seed: int = 0
    offline: HurdleModel | None = None
    online: BayesLinear | None = None
    tokens: TokenModel = field(default_factory=TokenModel)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "random" if self.kind == "random" else f"{self.kind}-{self.mode}"

    def predict_gain(self, X: np.ndarray) -> np.ndarray:
        if self.mode.startswith("online") and self.online is not None:
            return self.online.sample(X, self.rng)  # Thompson sample
        if self.offline is not None:
            return self.offline.predict(X)
        return np.zeros(len(X))

    def choose(
        self, cands: list[Candidate], *, own_tokens: list[int], sem_tokens: int
    ) -> Decision:
        n = len(cands)
        X = np.vstack([vector(c.features) for c in cands])
        tok = np.array([self.tokens.tokens(c, own_tokens, sem_tokens) for c in cands])
        anchor = np.array([c.features["sim_mean"] for c in cands])
        if self.kind == "random":
            i = int(self.rng.integers(n))
            return Decision(
                i,
                1.0 / n,
                [
                    {"candidate": c.name, "tok": float(t)}
                    for c, t in zip(cands, tok, strict=True)
                ],
            )
        g = self.predict_gain(X)
        score = objective_scores(
            self.kind, g, tok, anchor, lam_t=self.lam_t, lam_a=self.lam_a
        )
        best = int(np.argmax(score))
        explore = self.rng.random() < self.epsilon
        i = int(self.rng.integers(n)) if explore else best
        prop = self.epsilon / n + (1 - self.epsilon) * (i == best)
        receipt = [
            {
                "candidate": c.name,
                "g_hat": float(gi),
                "tok": float(t),
                "score": float(s),
            }
            for c, gi, t, s in zip(cands, g, tok, score, strict=True)
        ]
        return Decision(i, float(prop), receipt)

    def observe(self, features: dict, delta: float) -> None:
        if self.mode.startswith("online") and self.online is not None:
            self.online.update(vector(features), delta)


# ---------------------------------------------------------------- v2 (plan v0.3.0)
# Target: improvement over the run's best-so-far, Δ_best = max(0, child − best) — the reported
# quality is best-so-far, so beating a weak sampled parent is not progress. "Inject nothing" gets
# its own estimator (its few features carry no pool information, and one shared linear model let
# the k > 0 rows' gains leak into it). O3 is a constraint objective: the cheapest candidate whose
# predicted gain is at least that of rank:5 on the v1 graph configuration.

REF_NAME = "rank:5@lineage/down/1/8"


def tokens_v2(tm: TokenModel, c: Candidate, own_tokens: list[int], sem_tokens: int) -> float:
    replaced = sum(own_tokens[: min(len(c.rows), len(own_tokens))])
    sem = sem_tokens * c.features.get("sem_share", 0.0) / 2  # hard and rank share the passes
    return tm.base_prompt + c.features["tokens_injected"] - replaced + tm.out_tokens + sem


@dataclass
class LatencyModel:
    """O4 cost: expected LLM inference seconds per VALID child.

    seconds = a + b·prompt_tokens + d·output_tokens (fitted on the joined telemetry; measured
    R² 0.85, d ≈ 400·b, so the input barely matters), divided by P̂(valid | c) because a failed
    generation wastes the whole call. Output tokens and validity are predicted per candidate."""

    a: float
    b: float
    d: float
    out_model: object  # regressor on vector_v3 -> completion tokens
    valid_model: object  # classifier on vector_v3 -> P(child valid)
    p_floor: float = 0.2

    def seconds(self, X: np.ndarray, prompt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = np.maximum(self.out_model.predict(X), 0.0)
        pv = np.clip(self.valid_model.predict_proba(X)[:, 1], self.p_floor, 1.0)
        return (self.a + self.b * prompt + self.d * out) / pv, out, pv


@dataclass
class PolicyV2:
    kind: str  # random | o3 | o4 | o1 | o2
    mode: str = "offline"  # offline | online-cold | online-warm
    epsilon: float = 0.1
    lam_t: float = 0.2
    seed: int = 0
    tol: float = 0.0  # O3: feasible if ĝ(c) >= (1 - tol) · ĝ(ref)
    offline_mem: HurdleModel | None = None
    offline_none: HurdleModel | None = None
    online_mem: BayesLinear | None = None
    online_none: BayesLinear | None = None
    tokens: TokenModel = field(default_factory=TokenModel)
    space: str = "v2"  # v2 | v3 (v3 = + mixed path, vector_v3 features)
    # O5 (BRANE form): argmax p̂(c) − λ·E[output tokens | c]/1000, λ by online dual ascent on a budget
    out_budget: float = 0.0  # B: target mean output tokens per call (rank's mean); 0 = no budget
    eta: float = 0.05
    lam: float = 0.0
    fixed_name: str = ""  # kind "fixed" (offline profiling): always inject this candidate
    online_pmem: BayesLinear | None = None  # online models of P(new best) (linear probability, Thompson)
    online_pnone: BayesLinear | None = None
    lam_l: float = 0.1  # O4 weight of the normalised latency term
    latency: LatencyModel | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return f"{self.space}-random" if self.kind == "random" else f"{self.space}-{self.kind}-{self.mode}"

    def _vec(self, f: dict) -> np.ndarray:
        from trimem.objective.pool import vector_v2, vector_v3

        return vector_v3(f) if self.space == "v3" else vector_v2(f)

    def _predict(self, cands: list[Candidate]) -> np.ndarray:
        from trimem.objective.pool import vector_none

        none = np.array([c.family == "none" for c in cands])
        g = np.zeros(len(cands))
        online = self.mode.startswith("online")
        for mask, vec, on, off in ((none, vector_none, self.online_none, self.offline_none),
                                   (~none, self._vec, self.online_mem, self.offline_mem)):
            if not mask.any():
                continue
            X = np.vstack([vec(c.features) for c, m in zip(cands, mask, strict=True) if m])
            if online and on is not None:
                g[mask] = on.sample(X, self.rng)
            elif off is not None:
                g[mask] = off.predict(X)
        return g

    @staticmethod
    def _groups(cands: list[Candidate]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cands):
            groups.setdefault(f"{c.family}:{c.k}", []).append(i)
        return groups

    def _p_uniform(self, cands: list[Candidate], i: int) -> float:
        groups = self._groups(cands)
        return 1.0 / (len(groups) * len(groups[f"{cands[i].family}:{cands[i].k}"]))

    def _p_success(self, cands: list[Candidate]) -> np.ndarray:
        """P̂(new best | c): offline = hurdle classifier part; online = Thompson sample of a linear
        probability model updated on 1[Δ_best > 0]."""
        from trimem.objective.pool import vector_none

        none = np.array([c.family == "none" for c in cands])
        p = np.zeros(len(cands))
        online = self.mode.startswith("online")
        for mask, vec, on, off in ((none, vector_none, self.online_pnone, self.offline_none),
                                   (~none, self._vec, self.online_pmem, self.offline_mem)):
            if not mask.any():
                continue
            X = np.vstack([vec(c.features) for c, m in zip(cands, mask, strict=True) if m])
            if online and on is not None:
                mean, cov = on.posterior()
                w = self.rng.multivariate_normal(mean, cov)
                p[mask] = np.clip(on._phi(X) @ w, 0.0, 1.0)
            elif off is not None:
                p[mask] = off.p_improve(X)
        return p

    def _random_index(self, cands: list[Candidate]) -> tuple[int, float]:
        """Uniform over the 13 (family, k) choices, then uniform over the graph configurations."""
        groups = self._groups(cands)
        keys = sorted(groups)
        key = keys[int(self.rng.integers(len(keys)))]
        idx = groups[key][int(self.rng.integers(len(groups[key])))]
        return idx, self._p_uniform(cands, idx)

    def choose(self, cands: list[Candidate], *, own_tokens: list[int], sem_tokens: int) -> Decision:
        tok = np.array([tokens_v2(self.tokens, c, own_tokens, sem_tokens) for c in cands])
        if self.kind == "fixed":
            i = next((j for j, c in enumerate(cands) if c.name == self.fixed_name), None)
            if i is None:
                raise ValueError(f"profiling configuration {self.fixed_name!r} is not a candidate")
            return Decision(i, 1.0, [{"candidate": cands[i].name, "tok": float(tok[i])}])
        if self.kind == "random":
            i, prop = self._random_index(cands)
            return Decision(i, prop, [{"candidate": cands[i].name, "tok": float(tok[i])}])
        g = self._predict(cands)
        extra: dict[int, dict] = {}
        if self.kind == "o4":
            # O4 = ĝ(c)/ĝ_ref − λ_L · L̂(c)/L_ref, ref = rank:5 on lineage/down/1/8; input tokens enter
            # only through the latency model (tiny coefficient), never as a direct penalty.
            ref = next((j for j, c in enumerate(cands) if c.name == REF_NAME), int(np.argmax(g)))
            X = np.vstack([self._vec(c.features) for c in cands])
            prompt = tok - self.tokens.out_tokens - np.array(
                [sem_tokens * c.features.get("sem_share", 0.0) / 2 for c in cands])
            lat, out, pv = self.latency.seconds(X, prompt)
            score = g / max(float(g[ref]), 1e-4) - self.lam_l * lat / max(float(lat[ref]), 1e-6)
            extra = {j: {"lat_s": float(lat[j]), "out_tok": float(out[j]), "p_valid": float(pv[j])}
                     for j in range(len(cands))}
        elif self.kind == "o5":
            # λ is the shadow price of an output token (BRANE / ParetoBandit dual ascent): it rises while
            # this run's mean output exceeds the budget B and decays to 0 when it is under it.
            if self.out_budget > 0 and self.tokens.out_tokens:
                self.lam = max(0.0, self.lam + self.eta * (self.tokens.out_tokens / self.out_budget - 1.0))
            p = self._p_success(cands)
            X = np.vstack([self._vec(c.features) for c in cands])
            out = np.maximum(self.latency.out_model.predict(X), 0.0)
            score = p - self.lam * out / 1000.0
            g = p
            extra = {j: {"p_new_best": float(p[j]), "out_tok": float(out[j]), "lambda": self.lam}
                     for j in range(len(cands))}
            ref = next((j for j, c in enumerate(cands) if c.name == REF_NAME), 0)
        elif self.kind == "o3":
            ref = next((j for j, c in enumerate(cands) if c.name == REF_NAME), int(np.argmax(g)))
            feasible = g >= (1 - self.tol) * g[ref] - 1e-12
            score = np.where(feasible, -tok + 1e-6 * g, -np.inf)  # cheapest feasible; ties -> larger ĝ
        else:
            score = objective_scores(self.kind, g, tok, np.zeros(len(cands)), lam_t=self.lam_t)
        best = int(np.argmax(score))
        i = self._random_index(cands)[0] if self.rng.random() < self.epsilon else best
        prop = (1 - self.epsilon) * (i == best) + self.epsilon * self._p_uniform(cands, i)
        order = np.argsort(-score)[:15]  # receipt: the 15 best-scored candidates + the reference
        keep = sorted(set(order.tolist()) | {i} | ({ref} if self.kind in ("o3", "o4", "o5") else set()))
        receipt = [{"candidate": cands[j].name, "g_hat": float(g[j]), "tok": float(tok[j]),
                    "score": float(score[j]) if np.isfinite(score[j]) else None, **extra.get(j, {})}
                   for j in keep]
        return Decision(i, float(prop), receipt)

    def observe(self, features: dict, delta_best: float) -> None:
        from trimem.objective.pool import vector_none

        if not self.mode.startswith("online"):
            return
        if features.get("fam_none", 0.0) == 1.0:
            if self.online_none is not None:
                self.online_none.update(vector_none(features), delta_best)
            if self.online_pnone is not None:
                self.online_pnone.update(vector_none(features), float(delta_best > 0))
        else:
            if self.online_mem is not None:
                self.online_mem.update(self._vec(features), delta_best)
            if self.online_pmem is not None:
                self.online_pmem.update(self._vec(features), float(delta_best > 0))
