"""Downstream-aware plan choice (plan v0.2.0): candidates, features, estimators, objectives."""

from __future__ import annotations

import numpy as np
import pytest

from trimem.objective.models import BayesLinear, HurdleModel
from trimem.objective.policy import Policy, TokenModel, objective_scores
from trimem.objective.pool import FEATURE_ORDER, PoolRow, candidates, featurize, vector


def _rows(n: int = 10) -> list[PoolRow]:
    return [
        PoolRow(
            id=i,
            fitness=0.5 + 0.05 * (n - i),
            parent_fitness=0.5,
            changes=f"c{i}",
            code="x" * 10,
            tokens=100 + i,
            sim=0.9 - 0.01 * i,
            session=f"s{i % 3}",
            score=1.0 if i % 2 else 0.0,
            verdict=bool(i % 2),
        )
        for i in range(n)
    ]


def _cands(parent_fitness: float = 0.5):
    cs = candidates(_rows())
    for c in cs:
        c.features = featurize(
            c,
            parent_fitness=parent_fitness,
            iteration=3,
            horizon=20,
            parent_tokens=120,
            parent_session="s0",
        )
    return cs


def test_thirteen_candidates_and_family_rules():
    cs = _cands()
    assert [c.name for c in cs][:2] == ["none:0", "trimodal:1"] and len(cs) == 13
    by = {c.name: c for c in cs}
    assert by["none:0"].rows == []
    assert all(r.verdict for r in by["hard:8"].rows) and len(by["hard:8"].rows) == 5
    assert [r.id for r in by["trimodal:3"].rows] == [0, 1, 2]
    # rank: gain × (0.5 + score) promotes high-score rows
    assert by["rank:1"].rows[0].id == 1


def test_features_exact_and_ordered():
    by = {c.name: c for c in _cands()}
    f = by["trimodal:3"].features
    assert (
        f["tokens_injected"] == 100 + 101 + 102
        and f["k"] == 3
        and f["fam_trimodal"] == 1.0
    )
    assert f["other_session_frac"] == pytest.approx(2 / 3)
    assert vector(f).shape == (len(FEATURE_ORDER),)
    assert by["hard:8"].features["k"] == 5 and by["hard:8"].features["k_requested"] == 8


def test_token_model_counts_replaced_own_slots():
    by = {c.name: c for c in _cands()}
    tm = TokenModel(base_prompt=1000, out_tokens=500)
    own = [200, 200, 200]
    assert tm.tokens(by["none:0"], own, 0) == 1500
    assert tm.tokens(by["trimodal:1"], own, 0) == 1500 + 100 - 200
    assert tm.tokens(by["trimodal:5"], own, 0) == 1500 + sum(range(100, 105)) - 600
    assert (
        tm.tokens(by["rank:1"], own, 40) == tm.tokens(by["trimodal:1"], own, 0) + 1 + 20
    )


def test_objectives_o1_prefers_cheap_gain_o2_trades_off():
    g = np.array([0.0, 0.10, 0.11])
    tok = np.array([1000.0, 2000.0, 8000.0])
    anchor = np.zeros(3)
    assert int(np.argmax(objective_scores("o1", g, tok, anchor))) == 1
    assert int(np.argmax(objective_scores("o2", g, tok, anchor, lam_t=0.2))) == 1
    assert int(np.argmax(objective_scores("o2", g, tok, anchor, lam_t=0.0))) == 2


def test_random_policy_uniform_propensity_and_epsilon_propensity():
    pytest.importorskip("sklearn")
    cs = _cands()
    d = Policy(kind="random", seed=1).choose(cs, own_tokens=[100] * 10, sem_tokens=0)
    assert d.propensity == pytest.approx(1 / 13)
    X = np.vstack([vector(c.features) for c in cs])
    delta = X[:, FEATURE_ORDER.index("gain_vs_parent_max")] * 0.3
    hm = HurdleModel().fit(
        np.vstack([X] * 8), np.concatenate([delta * (i % 2) for i in range(8)])
    )
    pol = Policy(kind="o2", epsilon=0.0, offline=hm, seed=0)
    d = pol.choose(cs, own_tokens=[100] * 10, sem_tokens=0)
    assert d.propensity == pytest.approx(1.0) and len(d.receipt) == 13
    assert cs[d.index].name != "none:0"


def test_hurdle_learns_signal():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    delta = np.where(X[:, 0] > 0.3, 0.1 + 0.05 * X[:, 1], 0.0).clip(0)
    hm = HurdleModel().fit(X, delta)
    hi, lo = hm.predict(np.array([[2.0, 0, 0]])), hm.predict(np.array([[-2.0, 0, 0]]))
    assert hi[0] > 0.05 > lo[0]


def test_bayes_linear_online_update_moves_toward_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 0.2 * X[:, 0]
    bl = BayesLinear.cold(X)
    before = bl.predict(np.array([[2.0, 0.0]]))[0]
    for x, d in zip(X[:100], y[:100], strict=True):
        bl.update(x, d)
    after = bl.predict(np.array([[2.0, 0.0]]))[0]
    assert before == 0.0 and after == pytest.approx(0.4, abs=0.05)
    warm = BayesLinear.warm(X, y)
    assert warm.predict(np.array([[2.0, 0.0]]))[0] == pytest.approx(0.4, abs=0.05)
    assert warm.sample(np.array([[2.0, 0.0]]), rng).shape == (1,)


# ---------------------------------------------------------------- v2 space (plan v0.3.0)
from trimem.objective.policy import REF_NAME, PolicyV2, tokens_v2  # noqa: E402
from trimem.objective.pool import featurize_v2, vector_none, vector_v2  # noqa: E402
from trimem.objective.space import GRAPH_SPACE, GraphConfig, pool_sql  # noqa: E402


def _cands_v2():
    from trimem.objective.pool import FAMILIES, KS, Candidate, _order

    rows = _rows()
    cands = [Candidate("none", 0, [])]
    for g in GRAPH_SPACE:
        pool = rows[: 4 + (g.fanout // 4)]
        for fam in FAMILIES:
            for k in KS:
                cands.append(Candidate(fam, k, _order(pool, fam)[:k], graph=g))
    for c in cands:
        c.features = featurize_v2(c, parent_fitness=0.5, best_fitness=0.7, iteration=3, horizon=20,
                                  parent_tokens=120, pool_n=len(c.rows), union_n=10)
    return cands


def test_v2_space_and_sql():
    assert len(GRAPH_SPACE) == 24 and GraphConfig() in GRAPH_SPACE
    assert GraphConfig("similar", "up", 2, 16).edge_type == "similar_up"
    sql = pool_sql(GraphConfig("lineage", "up", 2, 4))
    assert sql.count("trimem.expand") == 2 and "'lineage_up', 4" in sql
    assert pool_sql(GraphConfig()).count("trimem.expand") == 1
    cands = _cands_v2()
    assert len(cands) == 1 + 24 * 12 and any(c.name == REF_NAME for c in cands)
    f = next(c for c in cands if c.name == "hard:3@similar/up/2/16").features
    assert f["edge_similar"] == f["dir_up"] == f["hops2"] == 1.0 and f["best_gap"] == pytest.approx(0.2)
    assert vector_v2(f).shape[0] > vector_none(f).shape[0]


class _Const:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return np.array([self.fn(x) for x in X])


def test_o3_picks_cheapest_feasible_and_routes_none():
    cands = _cands_v2()
    ref = next(c for c in cands if c.name == REF_NAME)
    ref_g = 0.05
    # memory model: gain grows with k; none model: tiny gain -> infeasible
    k_idx = __import__("trimem.objective.pool", fromlist=["FEATURE_ORDER_V2"]).FEATURE_ORDER_V2.index("k")
    pol = PolicyV2(kind="o3", epsilon=0.0, offline_mem=_Const(lambda x: 0.01 * x[k_idx]),
                   offline_none=_Const(lambda x: 0.0))
    d = pol.choose(cands, own_tokens=[100] * 10, sem_tokens=0)
    chosen = cands[d.index]
    g_ref = 0.01 * ref.features["k"]
    assert chosen.family != "none" and 0.01 * chosen.features["k"] >= g_ref - 1e-12
    tok = lambda c: tokens_v2(pol.tokens, c, [100] * 10, 0)  # noqa: E731
    assert all(tok(chosen) <= tok(c) + 1e-9 for c in cands if 0.01 * c.features.get("k", 0) >= g_ref and c.family != "none")
    assert d.propensity == pytest.approx(1.0) and ref_g > 0


def test_v2_random_propensity_and_online_split():
    cands = _cands_v2()
    pol = PolicyV2(kind="random", seed=3)
    d = pol.choose(cands, own_tokens=[], sem_tokens=0)
    c = cands[d.index]
    assert d.propensity == pytest.approx(1 / 13 if c.family == "none" else 1 / (13 * 24))
    X = np.vstack([vector_v2(c.features) for c in cands])
    on = PolicyV2(kind="o3", mode="online-cold", online_mem=BayesLinear.cold(X),
                  online_none=BayesLinear.cold(np.vstack([vector_none(c.features) for c in cands])))
    on.observe(cands[0].features, 0.3)
    on.observe(cands[5].features, 0.1)
    assert on.online_none.n == 1 and on.online_mem.n == 1


# ---------------------------------------------------------------- v3: mixed path + O4 (plan v0.4.0)
def test_mixed_path_sql_and_space():
    from trimem.objective.space import GRAPH_SPACE_V3

    g = GraphConfig("mixed", "down", 1, 8)  # similar, then 1 lineage step
    sql = pool_sql(g)
    assert g in GRAPH_SPACE_V3 and len(GRAPH_SPACE_V3) == 27
    assert "'similar', 8" in sql and "'lineage', 8" in sql and "x AS (SELECT dst FROM h2)" in sql


class _Out:
    def predict(self, X):
        return np.full(len(X), 1500.0)


class _Valid:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - self.p(X), self.p(X)])


def test_o4_prefers_quality_and_charges_invalid_latency():
    from trimem.objective.policy import LatencyModel
    from trimem.objective.pool import FEATURE_ORDER_V3

    cands = _cands_v2()
    for c in cands:
        c.features["edge_mixed"] = 0.0
    ki = FEATURE_ORDER_V3.index("k")
    ni = FEATURE_ORDER_V3.index("fam_none")
    lat = LatencyModel(2.0, 0.0002, 0.03, _Out(), _Valid(lambda X: np.where(X[:, ni] == 1, 0.3, 0.9)))
    pol = PolicyV2(kind="o4", epsilon=0.0, space="v3", lam_l=0.1, latency=lat,
                   offline_mem=_Const(lambda x: 0.01 * x[ki]), offline_none=_Const(lambda x: 0.005))
    d = pol.choose(cands, own_tokens=[100] * 10, sem_tokens=0)
    chosen = cands[d.index]
    # input tokens are nearly free under O4, so the largest predicted gain (k = 8) wins
    assert chosen.family != "none" and chosen.features["k"] == max(c.features["k"] for c in cands)
    rec = {r["candidate"]: r for r in d.receipt}
    assert "lat_s" in rec[chosen.name] and rec[chosen.name]["p_valid"] == pytest.approx(0.9)


def test_o5_dual_lambda_trades_success_for_output_tokens():
    from trimem.objective.policy import LatencyModel

    cands = _cands_v2()
    for c in cands:
        c.features["edge_mixed"] = 0.0

    class _P:  # P(new best) grows with k
        def p_improve(self, X):
            return np.clip(0.05 * X[:, 4], 0, 1)

        def predict(self, X):
            return self.p_improve(X)

    class _OutK:  # more context -> longer output
        def predict(self, X):
            return 1000.0 + 300.0 * X[:, 4]

    lat = LatencyModel(2.0, 0.0002, 0.03, _OutK(), _Valid(lambda X: np.full(len(X), 0.9)))
    class _PNone:
        def p_improve(self, X):
            return np.full(len(X), 0.01)

        def predict(self, X):
            return self.p_improve(X)

    pol = PolicyV2(kind="o5", epsilon=0.0, space="v3", latency=lat, offline_mem=_P(), offline_none=_PNone(),
                   out_budget=1500.0)
    pol.tokens.out_tokens = 1000.0  # under budget -> λ stays 0 -> pure success probability
    big = cands[pol.choose(cands, own_tokens=[], sem_tokens=0).index]
    assert pol.lam == 0.0 and big.features["k"] == 8
    pol.lam, pol.tokens.out_tokens = 0.5, 3000.0  # far over budget -> λ grows -> shorter outputs win
    small = cands[pol.choose(cands, own_tokens=[], sem_tokens=0).index]
    assert pol.lam > 0.5 and small.features["k"] < 8
