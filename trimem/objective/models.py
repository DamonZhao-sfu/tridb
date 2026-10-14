"""D2/D3 estimators of the expected score improvement ĝ(c) = E[Δ | features of candidate c].

* ``HurdleModel`` (offline prior): P(Δ>0) by logistic regression with isotonic calibration, times
  E[Δ | Δ>0] by gradient boosting. Fitted on logged decisions, frozen during a run.
* ``BayesLinear`` (online): Bayesian linear regression of Δ on standardised features with a
  Gaussian prior; posterior updated after every observed iteration; Thompson samples for
  exploration. Cold start: prior mean 0. Warm start: prior mean = ridge fit on offline data.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        sd = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(sd > 1e-9, sd, 1.0))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class HurdleModel:
    def __init__(self) -> None:
        self.clf = None
        self.reg = None
        self.pos_mean = 0.0
        self.base_rate = 0.0

    def fit(self, X: np.ndarray, delta: np.ndarray) -> "HurdleModel":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = (delta > 0).astype(int)
        self.base_rate = float(y.mean()) if len(y) else 0.0
        if 0 < y.sum() < len(y) and min(y.sum(), len(y) - y.sum()) >= 3:
            base = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)
            )
            self.clf = CalibratedClassifierCV(
                base, method="isotonic" if len(y) >= 60 else "sigmoid", cv=3
            ).fit(X, y)
        pos = delta > 0
        self.pos_mean = float(delta[pos].mean()) if pos.any() else 0.0
        if pos.sum() >= 8:
            self.reg = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                random_state=0,
            ).fit(X[pos], delta[pos])
        return self

    def p_improve(self, X: np.ndarray) -> np.ndarray:
        return (
            self.clf.predict_proba(X)[:, 1]
            if self.clf is not None
            else np.full(len(X), self.base_rate)
        )

    def gain_if_improve(self, X: np.ndarray) -> np.ndarray:
        return (
            np.maximum(self.reg.predict(X), 0.0)
            if self.reg is not None
            else np.full(len(X), self.pos_mean)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.p_improve(X) * self.gain_if_improve(X)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "HurdleModel":
        return pickle.loads(Path(path).read_bytes())


@dataclass
class BayesLinear:
    """Δ ≈ wᵀ[1, z(x)] + ε, ε ~ N(0, σ²), prior w ~ N(μ0, τ² I)."""

    scaler: Standardizer
    mu0: np.ndarray
    tau2: float = 1.0
    sigma2: float = 0.01
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        d = len(self.mu0)
        self.A = np.eye(d) / self.tau2
        self.b = self.mu0 / self.tau2

    def _phi(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler(np.atleast_2d(X))
        return np.hstack([np.ones((len(Z), 1)), Z])

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        cov = np.linalg.inv(self.A)
        return cov @ self.b, cov

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.posterior()
        return np.maximum(self._phi(X) @ mean, 0.0)

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean, cov = self.posterior()
        w = rng.multivariate_normal(mean, cov)
        return np.maximum(self._phi(X) @ w, 0.0)

    def update(self, x: np.ndarray, delta: float) -> None:
        phi = self._phi(x)[0]
        self.A += np.outer(phi, phi) / self.sigma2
        self.b += phi * delta / self.sigma2
        self.n += 1

    @classmethod
    def cold(cls, X_ref: np.ndarray, **kw) -> "BayesLinear":
        scaler = (
            Standardizer.fit(X_ref)
            if len(X_ref) > 1
            else Standardizer(np.zeros(X_ref.shape[1]), np.ones(X_ref.shape[1]))
        )
        return cls(scaler, np.zeros(X_ref.shape[1] + 1), **kw)

    @classmethod
    def warm(
        cls, X: np.ndarray, delta: np.ndarray, ridge: float = 1.0, **kw
    ) -> "BayesLinear":
        scaler = Standardizer.fit(X)
        Z = np.hstack([np.ones((len(X), 1)), scaler(X)])
        mu0 = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ delta)
        return cls(scaler, mu0, **kw)
