"""VAE over TRACE configurations (latent-BO plan, Task 2; 2026-11-04).

Encoder: per-slot embedding (16) -> MLP 256-256 -> (mu, logvar) in R^d.
Decoder: z -> MLP 256-256 -> per-slot logits over the slot's domain values (no PAD class).
Loss: per-slot cross-entropy on ACTIVE slots only (inactive slots are ignored by
``canonicalize``) + beta * KL, beta = 0.1 (BayesQO / LOL-BO ``kl_factor``).
Decoding is deterministic: argmax per slot, then the grammar's canonical form
(``latent_grammar.canonical_token_matrix``). Training data: canonical configurations only;
no episode outcomes.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .latent_grammar import PAD, canonical_token_matrix, sample_canonical_tokens
from .plan_schema import FIELD_NAMES, digest


class TraceVAE(nn.Module):
    def __init__(self, domain_sizes: list[int], d: int, emb: int = 16, hidden: int = 256):
        super().__init__()
        self.domain_sizes, self.d = list(domain_sizes), d
        self.embed = nn.ModuleList(nn.Embedding(s + 1, emb) for s in domain_sizes)  # +PAD
        S = len(domain_sizes)
        self.enc = nn.Sequential(nn.Linear(S * emb, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu, self.logvar = nn.Linear(hidden, d), nn.Linear(hidden, d)
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, sum(domain_sizes)))
        self.offsets = np.cumsum([0, *domain_sizes])

    def _h(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.enc(torch.cat([e(tokens[:, i]) for i, e in enumerate(self.embed)], dim=1))

    def encode(self, tokens):
        h = self._h(tokens)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def encode_mean(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encode(tokens)[0]

    def logits(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = self.dec(z)
        return [out[:, self.offsets[i]:self.offsets[i + 1]] for i in range(len(self.domain_sizes))]

    def decode_index(self, z: torch.Tensor) -> np.ndarray:
        """argmax value index per slot (N, S)."""
        with torch.no_grad():
            return torch.stack([lg.argmax(1) for lg in self.logits(z)], dim=1).cpu().numpy()


def _loss(model, tokens, beta):
    mu, logvar = model.encode(tokens)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    ce = 0.0
    for i, lg in enumerate(model.logits(z)):
        t = tokens[:, i]
        active = t != PAD
        if active.any():
            ce = ce + nn.functional.cross_entropy(lg[active], t[active] - 1, reduction="sum")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    n = len(tokens)
    return (ce + beta * kl) / n, ce.item() / n, kl.item() / n


def reconstruct(model, space, tokens: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Deterministic encode-mean -> argmax decode -> canonical tokens."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            t = torch.as_tensor(tokens[i:i + batch])
            out.append(canonical_token_matrix(space, model.decode_index(model.encode_mean(t))))
    return np.concatenate(out)


def evaluate(model, space, tokens: np.ndarray) -> dict:
    rec = reconstruct(model, space, tokens)
    exact = (rec == tokens).all(axis=1)
    per_slot = {}
    for i, n in enumerate(FIELD_NAMES):
        active = tokens[:, i] != PAD
        if active.any() and len(space.domains[n]) > 1:
            per_slot[n] = float((rec[active, i] == tokens[active, i]).mean())
    return {"exact": float(exact.mean()), "n": int(len(tokens)), "per_slot": per_slot}


def latent_bounds(model, tokens: np.ndarray, widen: float = 0.05) -> list[list[float]]:
    with torch.no_grad():
        z = torch.cat([model.encode_mean(torch.as_tensor(tokens[i:i + 65536]))
                       for i in range(0, len(tokens), 65536)])
    lo, hi = z.min(0).values, z.max(0).values
    pad = (hi - lo) * widen
    return [(lo - pad).tolist(), (hi + pad).tolist()]


def train_vae(space, d: int, *, n_samples: int, epochs: int, seed: int, holdout: int = 200_000,
              beta: float = 0.1, batch: int = 4096, lr: float = 1e-3, extra_tokens=None, log=print):
    torch.manual_seed(seed)
    tokens = sample_canonical_tokens(space, n_samples + holdout, seed=seed)
    train, held = tokens[:n_samples], tokens[n_samples:]
    if extra_tokens is not None and len(extra_tokens):
        train = np.concatenate([train, np.asarray(extra_tokens, dtype=np.int64)])
    model = TraceVAE([len(space.domains[n]) for n in FIELD_NAMES], d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tt = torch.as_tensor(train)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tt))
        tot = [0.0, 0.0, 0]
        for i in range(0, len(tt), batch):
            b = tt[perm[i:i + batch]]
            loss, ce, kl = _loss(model, b, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot[0] += ce * len(b); tot[1] += kl * len(b); tot[2] += len(b)
        sched.step()
        log(f"d={d} epoch {ep + 1}/{epochs} ce={tot[0] / tot[2]:.4f} kl={tot[1] / tot[2]:.3f} {time.time() - t0:.0f}s")
    metrics = {"d": d, "beta": beta, "n_train": int(len(train)), "epochs": epochs, "seed": seed,
               "heldout": evaluate(model, space, held), "train_sample": evaluate(model, space, train[:200_000]),
               "bounds": latent_bounds(model, train), "seconds": time.time() - t0}
    return model, metrics


def save_checkpoint(path, model, space, metrics):
    torch.save({"state": model.state_dict(), "domain_sizes": model.domain_sizes, "d": model.d,
                "space": digest(space.manifest()), "domains": digest(space.domains),
                "metrics": metrics}, path)


def load_checkpoint(path, space):
    ck = torch.load(path, weights_only=False)
    # The grammar depends on the field domains only (not on the run's execution fingerprint).
    if ck.get("domains") != digest(space.domains):
        raise ValueError("VAE checkpoint was trained for different field domains")
    model = TraceVAE(ck["domain_sizes"], ck["d"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["metrics"]
