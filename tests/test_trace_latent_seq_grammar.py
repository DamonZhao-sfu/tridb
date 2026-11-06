import numpy as np

from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.plan_schema import JointSpace

SPACE = JointSpace.live()
G = SeqGrammar(SPACE)


def test_roundtrip_canonical():
    for c in SPACE.sample(3000, seed=2):
        assert G.decode(G.encode(c)) == SPACE.canonicalize(c)


def test_any_sequence_decodes_valid():
    rng = np.random.default_rng(0)
    for n in (0, 3, 10, 25, 60):
        for row in rng.integers(0, 10**6, size=(500, n)):
            c = G.decode(row)
            assert c == SPACE.canonicalize(c)


def test_structure_is_explicit_and_variable_length():
    vgr = G.symbols(SPACE.default(filter_order="none", graph_impl="BFS", injection_timing="start"))
    p2 = G.symbols(SPACE.default(filter_order="after", graph_impl="beam", injection_timing="periodic"))
    assert "SEMFILTER" not in vgr and p2.index("SEMFILTER") > p2.index("G")
    assert any(s.startswith("beam=") for s in p2) and len(p2) > len(vgr)
