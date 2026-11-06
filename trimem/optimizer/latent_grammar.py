"""Fixed-slot grammar for TRACE plan+configuration (latent-BO plan, Task 1; 2026-11-04).

One slot per ``plan_schema.FIELDS`` field, in schema order. Slot vocabulary: ``PAD`` (0)
plus the field's domain values (value i -> token i + 1). Encoding writes ``PAD`` for fields
that are inactive under the configuration's plan (``JointSpace.active_mask``).

Decoding maps ANY integer vector to a valid canonical configuration (BayesQO §4.1 "decoding
validity", by modulo instead of BayesQO's index-into-valid-list): PAD -> the schema default,
token t > 0 -> ``domain[(t - 1) % len(domain)]``, then ``JointSpace.canonicalize``. Canonical
configurations round-trip exactly (the grammar is injective on canonical configurations).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .plan_schema import FIELD_NAMES

PAD = 0


def slot_sizes(space) -> list[int]:
    """Vocabulary size (PAD included) of every slot, in FIELD_NAMES order."""
    return [len(space.domains[n]) + 1 for n in FIELD_NAMES]


def encode_tokens(space, config: Mapping) -> list[int]:
    c = space.canonicalize(config)
    return [PAD if c[n] is None else space.domains[n].index(c[n]) + 1 for n in FIELD_NAMES]


def decode_tokens(space, tokens: Sequence[int]) -> dict:
    if len(tokens) != len(FIELD_NAMES):
        raise ValueError(f"Expected {len(FIELD_NAMES)} tokens, got {len(tokens)}")
    raw = {}
    for n, t in zip(FIELD_NAMES, tokens):
        t = int(t)
        dom = space.domains[n]
        raw[n] = None if t <= PAD else dom[(t - 1) % len(dom)]
    return space.canonicalize(raw)


def canonical_token_matrix(space, idx: np.ndarray) -> np.ndarray:
    """Vectorized: per-slot value indices (N, S) -> canonical token matrix (N, S).

    Mirrors ``JointSpace.active_mask`` (checked against ``encode_tokens`` in the tests):
    inactive fields become PAD, active fields token = index + 1.
    """
    col = {n: i for i, n in enumerate(FIELD_NAMES)}
    tok = idx.astype(np.int64) + 1
    dom = space.domains

    def is_(field, values):
        return np.isin(idx[:, col[field]], [dom[field].index(v) for v in values if v in dom[field]])

    pre = is_("filter_order", ("before", "both"))
    post = is_("filter_order", ("after", "both"))
    inactive = {
        "filter_pre_impl": ~pre,
        "filter_post_impl": ~post,
        "beam_width": ~is_("graph_impl", ("beam",)),
        "sem_threshold_pre": ~(pre & is_("filter_pre_impl", ("LLM",))),
        "sem_threshold_post": ~(post & is_("filter_post_impl", ("LLM",))),
        "retrieval_interval": ~is_("injection_timing", ("periodic",)),
    }
    for field, mask in inactive.items():
        tok[mask, col[field]] = PAD
    return tok


def sample_canonical_tokens(space, n: int, seed: int) -> np.ndarray:
    """n distinct canonical token vectors (uniform per-slot draws, deduplicated)."""
    rng = np.random.default_rng(seed)
    sizes = [len(space.domains[f]) for f in FIELD_NAMES]
    out, seen = [], set()
    for _ in range(50):  # bounded: n must be well below the canonical space size (2,857,680 live)
        if len(out) >= n:
            break
        idx = np.stack([rng.integers(0, s, size=2 * n) for s in sizes], axis=1)
        for row in map(tuple, canonical_token_matrix(space, idx)):
            if row not in seen:
                seen.add(row)
                out.append(row)
                if len(out) == n:
                    break
    if len(out) < n:
        raise ValueError(f"Only {len(out)} distinct canonical configurations drawn")
    return np.array(out, dtype=np.int64)
