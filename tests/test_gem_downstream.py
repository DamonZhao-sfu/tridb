"""Unit tests for the downstream E[#attempts | O] estimator (plan P5). No network, no DB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.agent_memory.gem.downstream import (
    AttemptEstimator,
    Candidate,
    Hint,
    LogisticModel,
    choose_k,
    expected_attempts,
    features_of,
    fit_estimator,
    fit_logistic,
    load_exg_rows,
    split_rows,
)


def _synthetic(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Rows where success rises with the golden fraction and the fix coverage."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for _ in range(n):
        k = int(rng.integers(1, 6))
        g = int(rng.integers(0, k + 1))
        fixed = int(rng.integers(0, k - g + 1))
        hints = [Hint(f"g{i}", "golden", tokens=100.0) for i in range(g)]
        warns = [
            Hint(f"w{i}", "warning", repair_case_uid=f"g{i}" if i < fixed else None)
            for i in range(k - g)
        ]
        x = features_of(hints + warns)
        logit = -1.0 + 2.5 * x[1] + 1.5 * x[3]
        rows.append(x)
        ys.append(float(rng.random() < 1 / (1 + np.exp(-logit))))
    return np.stack(rows), np.array(ys)


def test_features_of_counts_fix_coverage_and_empty() -> None:
    hints = [
        Hint("a", "golden", similarity=0.8, tokens=10),
        Hint("b", "warning", repair_case_uid="a", similarity=0.4, tokens=30),
        Hint("c", "warning", repair_case_uid="zzz"),
    ]
    x = features_of(hints)
    assert x[0] == 3
    assert x[1] == pytest.approx(1 / 3)
    assert x[2] == pytest.approx(2 / 3)
    assert x[3] == pytest.approx(0.5)  # b's partner is in O, c's is not
    assert x[4] == pytest.approx(0.6)
    assert x[5] == pytest.approx(np.log1p(40))
    assert np.all(features_of([]) == 0)


def test_fit_is_deterministic() -> None:
    x, y = _synthetic(7)
    a = fit_logistic(x, y, l2=1.0)
    b = fit_logistic(x.copy(), y.copy(), l2=1.0)
    assert a == b
    x2, y2 = _synthetic(7)
    assert (
        fit_logistic(x2, y2) == a
    )  # same seed for the data gives the same coefficients


def test_split_is_seeded_and_70_30() -> None:
    from bench.agent_memory.gem.downstream import TaskRow

    rows = [TaskRow(f"t{i:03d}", i, (), i % 2, 1, 0) for i in range(100)]
    tr1, te1 = split_rows(rows, seed=42)
    tr2, te2 = split_rows(list(reversed(rows)), seed=42)
    assert [r.task_id for r in te1] == [r.task_id for r in te2]
    assert (len(tr1), len(te1)) == (70, 30)
    assert {r.task_id for r in tr1}.isdisjoint({r.task_id for r in te1})


def test_monotone_more_golden_and_fix_coverage_not_more_attempts() -> None:
    x, y = _synthetic(3)
    est = fit_estimator(x, y)
    by_golden = [
        est.expected_attempts(
            [Hint(f"g{i}", "golden") for i in range(g)]
            + [Hint(f"w{i}", "warning") for i in range(4 - g)]
        )
        for g in range(5)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(by_golden, by_golden[1:], strict=False))
    by_fix = [
        est.expected_attempts(
            [Hint("g0", "golden")]
            + [
                Hint(f"w{i}", "warning", repair_case_uid="g0" if i < f else None)
                for i in range(3)
            ]
        )
        for f in range(4)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(by_fix, by_fix[1:], strict=False))


def test_expected_attempts_cap_handling() -> None:
    assert float(expected_attempts(0.3, cap=1)) == pytest.approx(1.0)
    assert float(expected_attempts(0.3, cap=2)) == pytest.approx(
        1.7
    )  # EXG 1 + (1 - p1)
    assert float(expected_attempts(0.5, cap=3)) == pytest.approx(1.75)
    assert float(expected_attempts(1.0, cap=5)) == pytest.approx(1.0)
    assert float(expected_attempts(0.0, cap=4)) == pytest.approx(4.0)
    big = float(expected_attempts(0.25, cap=200))
    assert big == pytest.approx(1 / 0.25)  # geometric limit 1 / p1
    with pytest.raises(ValueError):
        expected_attempts(0.5, cap=0)


def _fixed_estimator(cap: int = 2) -> AttemptEstimator:
    """p1 = sigmoid(2 * frac_golden - 1); hand-set so expected costs are exact."""
    coef = (0.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    model = LogisticModel((0.0,) * 6, (1.0,) * 6, coef, -1.0, 0.0)
    return AttemptEstimator(model, cap=cap)


def test_choose_k_picks_argmin_with_breakdown() -> None:
    est = _fixed_estimator()
    w = Hint("w", "warning", tokens=100)
    g1, g2 = Hint("g1", "golden", tokens=100), Hint("g2", "golden", tokens=5000)
    cands = {
        1: Candidate((w,)),  # p1 = s(-1), cheap prompt
        2: Candidate((w, g1), n_sem=1),  # p1 = s(0)
        3: Candidate((w, g1, g2), n_sem=1),  # p1 = s(1/3), huge prompt
    }
    res = choose_k(
        cands,
        c_db={1: 1.0, 2: 2.0, 3: 3.0},
        c_llm=10.0,
        c_agent_per_token=0.1,
        estimator=est,
        c_agent_fixed=100.0,
    )

    def sig(z: float) -> float:
        return 1 / (1 + np.exp(-z))

    expect = {
        1: 1.0 + (2 - sig(-1)) * (100 + 10),
        2: 2.0 + 10 + (2 - sig(0)) * (100 + 20),
        3: 3.0 + 10 + (2 - sig(1 / 3)) * (100 + 520),
    }
    got = {r.k: r.total for r in res.table}
    for k, v in expect.items():
        assert got[k] == pytest.approx(v)
    assert res.k == min(expect, key=expect.get)
    assert res.best.total == pytest.approx(min(expect.values()))
    row2 = next(r for r in res.table if r.k == 2)
    assert row2.retrieval_llm == pytest.approx(10.0)
    assert row2.o_tokens == pytest.approx(200.0)
    assert row2.downstream == pytest.approx(row2.expected_attempts * 120.0)


def test_choose_k_cap_changes_choice_and_accepts_plain_hint_lists() -> None:
    w, g = Hint("w", "warning", tokens=0), Hint("g", "golden", tokens=0)
    # k=0 is cheaper to retrieve but has p1 = s(-1); k=1 has p1 = s(1).
    cands = {0: [w], 1: [g]}
    kw = dict(
        c_db={0: 0.0, 1: 30.0}, c_llm=0.0, c_agent_per_token=0.0, c_agent_fixed=100.0
    )
    low = choose_k(cands, estimator=_fixed_estimator(cap=1), **kw)
    high = choose_k(cands, estimator=_fixed_estimator(cap=2), **kw)
    assert low.k == 0  # one attempt regardless: retrieval cost decides
    assert high.k == 1  # retries make the better O pay for itself
    with pytest.raises(ValueError):
        choose_k({}, 0.0, 0.0, 0.0, estimator=_fixed_estimator())


def test_loader_uses_logged_hints_and_drops_self_hints(tmp_path: Path) -> None:
    def task(seq: int, tid: str, statuses: list[str], hints: list[dict]) -> None:
        d = tmp_path / "tasks" / "000" / f"{seq:06d}"
        (d / "attempt_1").mkdir(parents=True)
        atts = [{"status": s, "case_id": i} for i, s in enumerate(statuses)]
        (d / "completed.json").write_text(
            json.dumps({"sequence": seq, "task_id": tid, "attempts": atts})
        )
        mem = {"query_seeds": [{"id": 1, "cosine": 0.9}], "hints": hints}
        prep = {"memory": mem, "messages": [{"content": "x" * 400}]}
        (d / "attempt_1" / "prepared.json").write_text(json.dumps(prep))
        traj = {"call": {"guard": {"count": 100}}}
        (d / "attempt_1" / "trajectory.json").write_text(json.dumps(traj))

    h = {
        "case_uid": "u0",
        "source_task": "T0",
        "status": "warning",
        "repair_case_uid": None,
        "provenance": {"id": 1},
    }
    task(0, "T0", ["golden"], [])
    task(1, "T1", ["warning", "golden"], [h])
    task(2, "T2", ["golden"], [dict(h, source_task="T2")])  # leak: own task
    task(3, "T3", ["infrastructure"], [])
    rows, stats = load_exg_rows(tmp_path)
    assert [r.task_id for r in rows] == ["T0", "T1"]
    assert stats["self_hint_leak"] == 1 and stats["infrastructure"] == 1
    r1 = rows[1]
    assert (r1.first_success, r1.attempts_used, r1.retried) == (0, 2, 1)
    assert r1.hints[0].similarity == pytest.approx(0.9)
    assert r1.hints[0].tokens == pytest.approx(len(json.dumps(h)) * 0.25)
