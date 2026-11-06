import numpy as np

from trimem.optimizer.latent_grammar import (
    PAD, canonical_token_matrix, decode_tokens, encode_tokens, sample_canonical_tokens, slot_sizes,
)
from trimem.optimizer.plan_schema import FIELD_NAMES, JointSpace

SPACE = JointSpace.live()


def test_roundtrip_canonical():
    for c in SPACE.sample(2000, seed=1):
        assert decode_tokens(SPACE, encode_tokens(SPACE, c)) == SPACE.canonicalize(c)


def test_any_tokens_decode_valid():
    rng = np.random.default_rng(0)
    for row in rng.integers(0, 10**6, size=(5000, len(FIELD_NAMES))):
        c = decode_tokens(SPACE, row)
        assert c == SPACE.canonicalize(c)
        SPACE.key(c)


def test_inactive_slots_are_pad():
    c = SPACE.default(filter_order="none")
    t = dict(zip(FIELD_NAMES, encode_tokens(SPACE, c)))
    for f in ("sem_threshold_pre", "sem_threshold_post", "filter_pre_impl", "filter_post_impl"):
        assert t[f] == PAD


def test_vectorized_matches_encoder():
    toks = sample_canonical_tokens(SPACE, 5000, seed=3)
    for row in toks:
        assert list(row) == encode_tokens(SPACE, decode_tokens(SPACE, row))
    assert len({tuple(r) for r in toks}) == 5000
    assert toks.shape[1] == len(slot_sizes(SPACE))
