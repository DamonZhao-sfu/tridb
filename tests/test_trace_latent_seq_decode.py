import pytest
import torch

from tests.latent_fixtures import enumerate_space, tiny_model, tiny_space
from trimem.optimizer.latent_seq_decode import allowed_next, decode_configs, decode_masked
from trimem.optimizer.latent_seq_grammar import EOS, SeqGrammar

torch.set_num_threads(4)
S = tiny_space()
G = SeqGrammar(S)
M = tiny_model()


def _z(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 4, generator=g) * 5.0
    z[:8] = torch.randn(len(z[:8]), 4, generator=g).sign() * 1e3 / 2.0  # |z| = 1e3
    return z


def test_encoded_sequences_are_accepted_step_by_step():
    for c in enumerate_space(S):
        toks = G.encode(c)
        for i, t in enumerate(toks):
            assert t in allowed_next(G, toks[:i])
        assert allowed_next(G, toks) == ()


def test_masked_decode_needs_no_repair():
    for toks in decode_masked(M, G, _z()):
        assert all(t in allowed_next(G, toks[:i]) for i, t in enumerate(toks))
        assert G.decode(toks) == S.canonicalize(G.decode(toks)) and toks[-1] == G.index[EOS]
        assert G.encode(G.decode(toks)) == toks


def test_decode_is_deterministic():
    z = _z(32, 1)
    assert decode_configs(M, G, z) == decode_configs(M, G, z.clone())


def test_nonfinite_latent_rejected():
    for bad in (float("nan"), float("inf")):
        z = _z(4)
        z[1, 2] = bad
        with pytest.raises(ValueError):
            decode_configs(M, G, z)


def test_beam_width_only_after_beam():
    for toks in decode_masked(M, G, _z()):
        syms = [G.vocab[t] for t in toks]
        for i, s in enumerate(syms):
            if s.startswith("beam="):
                assert syms[i - 1] == "impl:beam"
        if "impl:BFS" in syms:
            assert not any(s.startswith("beam=") for s in syms)
    for c in decode_configs(M, G, _z()):
        if c["graph_impl"] == "BFS":
            assert c["beam_width"] == S.canonicalize(c)["beam_width"]
