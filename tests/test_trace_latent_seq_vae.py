import pytest
import torch

from tests.latent_fixtures import enumerate_space, synthetic_receipt, tiny_model, tiny_space
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import load_checkpoint, save_checkpoint, to_matrix
from trimem.optimizer.plan_schema import JointSpace

torch.set_num_threads(4)


def test_checkpoint_roundtrip_cpu(tmp_path):
    m = tiny_model()
    p = tmp_path / "t.pt"
    save_checkpoint(p, m, tiny_space(), {"d": 4})
    m2, _ = load_checkpoint(p, tiny_space())
    x = torch.as_tensor(to_matrix(SeqGrammar(tiny_space()), enumerate_space(tiny_space())[:8]))
    assert not m2.training and torch.equal(m.encode_mean(x[:, 1:]), m2.encode_mean(x[:, 1:]))
    assert all(p.device.type == "cpu" for p in m2.parameters())


def test_checkpoint_domain_mismatch_raises(tmp_path):
    p = tmp_path / "t.pt"
    save_checkpoint(p, tiny_model(), tiny_space(), {"d": 4})
    with pytest.raises(ValueError):
        load_checkpoint(p, JointSpace.live())


def test_enumerate_tiny_space():
    space = tiny_space()
    cs = enumerate_space(space)
    assert len(cs) == 576
    assert len({space.key(c) for c in cs}) == 576


def test_synthetic_receipt_shape_and_monotonicity():
    space = tiny_space()
    m = {"protocol": "p"}
    lo = space.default(seed_count=8, rank_impl="ColBERT", depth=2, graph_impl="BFS")
    hi = space.default(seed_count=32, rank_impl="LLM", depth=3, graph_impl="beam", beam_width=4)
    a, b = synthetic_receipt(lo, "t1", m), synthetic_receipt(hi, "t1", m)
    assert a["synthetic"] is True and a["split"] == "TRAIN" and isinstance(a["agent_steps"], int)
    assert b["raw_score"] > a["raw_score"] and b["total_tokens"] > a["total_tokens"]
    assert b["latency_seconds"] > a["latency_seconds"]
