"""T10: VAE vocabulary / round trip / tiny CPU training on the v0.7.0 space; the v0.6.0 checkpoint is refused."""

from pathlib import Path

import pytest
import torch

from tests import v070_fixtures as fx
from tools import latent_readiness as lr
from tools import latent_readiness_v070 as lr7
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import load_checkpoint
from trimem.optimizer.latent_seq_vae_v2 import save_checkpoint_v2, train_seq_vae_v2
from trimem.optimizer.plan_schema import JointSpace
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/trace_latent_seqvae_v2_20261104/seqvae_v2_d16.pt"
torch.set_num_threads(4)


def test_vocabulary_has_rank_none_not_llm():
    vocab = SeqGrammar(JointSpaceV070.live()).vocab
    assert "rank:none" in vocab and "rank:LLM" not in vocab
    assert "unit:whole" in vocab and "unit:actions" not in vocab
    assert [v for v in vocab if v.startswith("dir=")] == ["dir=Both"]


def test_grammar_round_trip_live_space():
    space = JointSpaceV070.live()
    g = SeqGrammar(space)
    configs = space.sample(200, seed=3)
    assert any(c["rank_impl"] == "none" for c in configs)
    for c in configs:
        assert g.decode(g.encode(c)) == c
        if c["rank_impl"] == "none":
            assert "rank:none" in g.symbols(c)


def test_tiny_cpu_training_and_checkpoint(tmp_path):
    space = fx.tiny_space_v070()
    model, metrics = train_seq_vae_v2(space, 4, n_samples=160, holdout=16, epochs=1, seed=0, layers=1, batch=64,
                                      log=lambda *_: None)
    ck = tmp_path / "seqvae_v2_d4.pt"
    save_checkpoint_v2(ck, model, space, metrics)
    loaded, _ = load_checkpoint(ck, space)
    assert loaded.d == 4
    stock = JointSpace({k: tuple(v for v in vals if v != "none") or vals for k, vals in fx.TINY_V070.items()
                        if k != "rank_impl"} | {"rank_impl": ("ColBERT",)}, execution_fingerprint="x")
    with pytest.raises(ValueError, match="different field domains"):
        load_checkpoint(ck, stock)


@pytest.mark.skipif(not OLD.exists(), reason="v0.6.0 checkpoint not present")
def test_v060_checkpoint_refused():
    with pytest.raises(ValueError, match="different field domains"):
        load_checkpoint(OLD, JointSpaceV070.live())


def test_readiness_family_includes_rank():
    configs = [{"filter_order": "none", "graph_impl": "BFS", "rank_impl": r} for r in ("none", "ColBERT", "none")]
    worst, per = lr7.family_rates_v070(configs, [True, True, False])
    assert per == {"none|BFS|none": 0.5, "none|BFS|ColBERT": 1.0} and worst == 0.5
    with lr7.patches(None):
        assert lr.family_rates is lr7.family_rates_v070
    assert lr.family_rates is not lr7.family_rates_v070
