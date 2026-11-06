"""Frozen affine map and LatentCodec for TRACE latent-MOBO (spec section 6).

The codec defines the coordinate system u in [0,1]^d used by the latent GP, the trust region
and the proposer. It duck-types as ``codec`` for ``plan_surrogate.AcquisitionSnapshot``
(``.space``, ``.encode``, ``.key``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trimem.optimizer.latent_seq_decode import decode_configs
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import load_checkpoint, to_matrix
from trimem.optimizer.plan_schema import digest

STD_FLOOR = 1e-3
SIGMA = 4.0
MARGIN = 0.05
BATCH = 4096


@dataclass(frozen=True)
class Affine:
    """Per-dimension box [lo, hi]; maps z to u = (z-lo)/(hi-lo). No clipping."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]

    def _bounds(self, ref: torch.Tensor):
        lo = torch.as_tensor(self.lo, dtype=torch.double, device=ref.device)
        hi = torch.as_tensor(self.hi, dtype=torch.double, device=ref.device)
        return lo, hi - lo

    def to_u(self, z: torch.Tensor) -> torch.Tensor:
        lo, w = self._bounds(z)
        return (z.to(torch.double) - lo) / w

    def to_z(self, u: torch.Tensor) -> torch.Tensor:
        lo, w = self._bounds(u)
        return u.to(torch.double) * w + lo


def _encode_z(model, grammar: SeqGrammar, configs) -> torch.Tensor:
    """Posterior means (double, [N,d]) in batches of BATCH."""
    configs = list(configs)
    model.eval()
    dim = model.d
    if not configs:
        return torch.zeros((0, dim), dtype=torch.double)
    x = torch.as_tensor(to_matrix(grammar, configs))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            out.append(model.encode_mean(x[i : i + BATCH, 1:]).to(torch.double))
    z = torch.cat(out, 0)
    if not torch.isfinite(z).all():
        raise ValueError("encoder produced non-finite latent values")
    return z


def fit_affine(model, grammar: SeqGrammar, train_configs, init_configs=()) -> Affine:
    """mean +- 4 std (std floor 1e-3) over train; widen to cover train and init; add 5 % of
    the original (mean +- 4 std) width on each side."""
    zt = _encode_z(model, grammar, train_configs).numpy()
    if len(zt) == 0:
        raise ValueError("fit_affine needs at least one train config")
    zi = _encode_z(model, grammar, init_configs).numpy()
    return _affine_from_z(zt, zi)


def _affine_from_z(zt: np.ndarray, zi: np.ndarray) -> Affine:
    mean = zt.mean(0)
    std = np.maximum(zt.std(0), STD_FLOOR)
    lo0, hi0 = mean - SIGMA * std, mean + SIGMA * std
    width0 = hi0 - lo0
    allz = zt if len(zi) == 0 else np.concatenate([zt, zi], 0)
    lo = np.minimum(lo0, allz.min(0)) - MARGIN * width0
    hi = np.maximum(hi0, allz.max(0)) + MARGIN * width0
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (hi > lo).all()):
        raise ValueError("affine bounds are non-finite or empty")
    return Affine(tuple(float(v) for v in lo), tuple(float(v) for v in hi))


def save_affine(path, affine: Affine, meta: dict) -> None:
    payload = {"lo": list(affine.lo), "hi": list(affine.hi), "meta": dict(meta)}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_affine(path) -> tuple[Affine, dict]:
    payload = json.loads(Path(path).read_text())
    return Affine(tuple(payload["lo"]), tuple(payload["hi"])), dict(payload.get("meta", {}))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class LatentCodec:
    """u(c) = A(mu(c)) with decode(U) = masked-greedy(A^-1(U))."""

    cat_dims: tuple = ()

    def __init__(self, space, model, affine: Affine, *, checkpoint_sha256: str):
        if len(affine.lo) != model.d or len(affine.hi) != model.d:
            raise ValueError("affine dimension does not match the model latent size")
        model.eval()  # dropout off: u(c) must be deterministic
        self.space, self.model, self.affine = space, model, affine
        self.checkpoint_sha256 = checkpoint_sha256
        self.grammar = SeqGrammar(space)
        self.d = model.d
        self.identity = digest(
            [space.identity, checkpoint_sha256, [list(affine.lo), list(affine.hi)]]
        )
        self._cache: dict[str, list[float]] = {}

    def key(self, c) -> str:
        return self.space.key(c)

    def encode(self, c) -> list[float]:
        k = self.key(c)
        hit = self._cache.get(k)
        if hit is None:
            hit = self.encode_many([c])[0].tolist()
        return list(hit)

    def encode_many(self, configs) -> torch.Tensor:
        configs = list(configs)
        if not configs:
            return torch.zeros((0, self.d), dtype=torch.double)
        U = self.affine.to_u(_encode_z(self.model, self.grammar, configs))
        # first writer wins: u(c) never changes once cached
        for c, row in zip(configs, U):
            self._cache.setdefault(self.key(c), row.tolist())
        return torch.tensor([self._cache[self.key(c)] for c in configs], dtype=torch.double)

    def decode(self, U: torch.Tensor) -> list[dict]:
        z = self.affine.to_z(torch.as_tensor(U))
        dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            return decode_configs(self.model, self.grammar, z.to(dtype))

    def roundtrip(self, c) -> bool:
        u = torch.tensor([self.encode(c)], dtype=torch.double)
        return self.decode(u)[0] == self.space.canonicalize(c)


def load_codec(checkpoint, affine, space) -> LatentCodec:
    aff, meta = load_affine(affine)
    sha = file_sha256(checkpoint)  # verify before unpickling
    if sha != meta.get("checkpoint_sha256"):
        raise ValueError(
            f"checkpoint sha256 {sha} does not match affine meta {meta.get('checkpoint_sha256')}"
        )
    model, _ = load_checkpoint(checkpoint, space)
    return LatentCodec(space, model, aff, checkpoint_sha256=sha)
