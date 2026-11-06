"""Masked deterministic decoder for the TRACE sequence VAE (spec section 4).

Replaces the grammar's repair rule on the live path. At each step the logits outside the
grammar's allowed set are set to -inf, so every emitted token is valid and no repair
happens. The allowed set comes from the grammar's own state machine (``SeqGrammar.decode``)
through a subclass that intercepts ``_pick``; no second parser exists.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from trimem.optimizer.latent_seq_grammar import BOS, EOS, PAD, SeqGrammar
from trimem.optimizer.latent_seq_vae import MAX_LEN, SeqVAE


class _Need(Exception):
    """Raised by the probe when the parser needs one more token; carries the valid symbols."""

    def __init__(self, valid: list[str]):
        super().__init__("need more tokens")
        self.valid = list(valid)


class _Probe(SeqGrammar):
    """SeqGrammar whose ``_pick`` is strict: it never repairs and reports what is needed."""

    def __init__(self, space):
        super().__init__(space)
        self.consumed = 0

    def _pick(self, token, valid):
        if token is None:
            raise _Need(valid)
        sym = self.vocab[token]
        if sym not in valid:
            raise ValueError(f"token {sym!r} is not valid here; valid: {valid}")
        self.consumed += 1
        return sym


def allowed_next(grammar: SeqGrammar, prefix: Sequence[int]) -> tuple[int, ...]:
    """Token indices allowed after ``prefix`` (BOS/PAD ignored; ``prefix`` must be valid).

    Fully parsed prefix -> (EOS,). Prefix ending in EOS -> (). Invalid prefix -> ValueError.
    """
    toks = [int(t) for t in prefix if grammar.vocab[int(t)] not in (PAD, BOS)]
    eos = grammar.index[EOS]
    ended = bool(toks) and toks[-1] == eos
    if ended:
        toks = toks[:-1]
    if eos in toks:
        raise ValueError("EOS inside prefix")
    probe = _Probe(grammar.space)
    try:
        probe.decode(toks)
    except _Need as need:
        if ended:
            raise ValueError("EOS before the plan is complete") from None
        return tuple(grammar.index[s] for s in need.valid)
    if probe.consumed != len(toks):
        raise ValueError("tokens after the last group")
    return () if ended else (eos,)


@torch.no_grad()
def decode_masked(model: SeqVAE, grammar: SeqGrammar, z: torch.Tensor) -> list[list[int]]:
    """Batched greedy decoding under the grammar mask. Returns tokens (no BOS) ending at EOS."""
    z = torch.as_tensor(z)
    if z.ndim != 2:
        raise ValueError("z must be 2-D (n, d)")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("non-finite latent")
    was_training = model.training
    model.eval()
    try:
        n, eos, bos = len(z), grammar.index[EOS], grammar.index[BOS]
        out: list[list[int]] = [[] for _ in range(n)]
        done = [False] * n
        seq = torch.full((n, 1), bos, dtype=torch.long, device=z.device)
        for _ in range(MAX_LEN - 1):
            if all(done):
                break
            logits = model._decode_logits(z, seq)[:, -1].float()
            mask = torch.full_like(logits, float("-inf"))
            for i in range(n):
                if not done[i]:
                    mask[i, list(allowed_next(grammar, out[i]))] = 0.0
            nxt = (logits + mask).argmax(-1)  # first max index on ties
            col = torch.zeros(n, dtype=torch.long, device=z.device)
            for i in range(n):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t not in allowed_next(grammar, out[i]):
                    raise RuntimeError("decoder picked a token outside the allowed set")
                out[i].append(t)
                col[i] = t
                done[i] = t == eos
            seq = torch.cat([seq, col[:, None]], 1)
        if not all(done):
            raise RuntimeError("decoding did not reach EOS within MAX_LEN-1 steps")
        return out
    finally:
        model.train(was_training)


def decode_configs(model: SeqVAE, grammar: SeqGrammar, z: torch.Tensor) -> list[dict]:
    """Canonical configs for latent rows ``z`` (ValueError on non-finite z)."""
    return [grammar.space.canonicalize(grammar.decode(t)) for t in decode_masked(model, grammar, z)]
