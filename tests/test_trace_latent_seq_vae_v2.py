"""SeqVAEv2: parameter-name symbol + continuous value embedding, order-aware output scoring and a
linear order head on mu (user 2026-11-04). Tiny space, CPU only."""

import functools
import itertools
import json
import math
import sys

import numpy as np
import pytest
import torch

from tests.latent_fixtures import enumerate_space, tiny_model, tiny_space
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_seq_decode import allowed_next, decode_masked
from trimem.optimizer.latent_seq_grammar import BOS, EOS, SeqGrammar
from trimem.optimizer.latent_seq_vae import (
    MAX_LEN,
    SeqVAE,
    load_checkpoint,
    save_checkpoint,
    to_matrix,
)
from trimem.optimizer.latent_seq_vae_v2 import (
    KIND,
    SeqVAEv2,
    aux_loss,
    aux_targets,
    save_checkpoint_v2,
    spearman,
    train_seq_vae_v2,
    value_spec,
)
from trimem.optimizer.plan_schema import JointSpace

torch.set_num_threads(4)
S = tiny_space()
G = SeqGrammar(S)


D = 12  # d=4 cannot carry the 576 tiny configs (~6.4 nats) at beta=0.1 in a short CPU run; d=12
# and 200 epochs reach held-out exact 1.0 on seeds 0, 1 and 2 (120 epochs: 0.615 on seed 2)


@functools.cache
def trained():
    """Short tiny-space training run shared by the tests (layers=1, CPU, ~70 s). 480 train + 96
    held-out configs: space.sample draws distinct configs, so held-out configs are unseen."""
    model, metrics = train_seq_vae_v2(
        S,
        D,
        n_samples=480,
        holdout=96,
        epochs=200,
        seed=0,
        layers=1,
        batch=32,
        lr=1e-3,
        log=lambda *_: None,
    )
    model.eval()
    return model, metrics


def _x(configs):
    return torch.as_tensor(to_matrix(G, configs))


def test_v2_interface_matches_v1():
    m, _ = trained()
    v1 = tiny_model(4)
    for attr in ("vocab", "d", "cfg", "mem_tokens", "d_model"):
        assert hasattr(m, attr)
    assert m.vocab == v1.vocab == len(G.vocab) and m.d == D
    x = _x(enumerate_space(S)[:16])
    with torch.no_grad():
        z = m.encode_mean(x[:, 1:])
        mu, logvar = m.encode(x[:, 1:])
        assert z.shape == mu.shape == logvar.shape == (16, D)
        z1 = v1.encode_mean(x[:, 1:])
        assert (
            m._decode_logits(z, x[:, :-1]).shape
            == v1._decode_logits(z1, x[:, :-1]).shape
            == (16, MAX_LEN - 1, m.vocab)
        )
        assert m.decode_greedy(z, G.index[BOS]).shape == (16, MAX_LEN - 1)
    # the masked decoder and the codec work unchanged with a v2 model
    g = torch.Generator().manual_seed(0)
    for toks in decode_masked(m, G, torch.randn(32, D, generator=g) * 3):
        assert (
            all(t in allowed_next(G, toks[:i]) for i, t in enumerate(toks))
            and toks[-1] == G.index[EOS]
        )
    configs = enumerate_space(S)
    codec = LatentCodec(S, m, fit_affine(m, G, configs), checkpoint_sha256="x")
    U = codec.encode_many(configs[:20])
    assert U.shape == (20, D) and len(codec.decode(U)) == 20
    assert isinstance(codec.roundtrip(configs[0]), bool)


def test_phi_monotone_and_log_scale():
    live = JointSpace.live()
    for space in (live, S):
        spec = value_spec(SeqGrammar(space))
        prefixes = {f["prefix"] for f in spec}
        assert (
            "dir" not in prefixes and "cat" not in prefixes
        )  # non-numeric stay ordinary symbols
        for f in spec:
            phi, vals = f["phi"], f["values"]
            assert len(phi) == len(vals) == len(f["tokens"])
            if len(vals) > 1:
                assert (
                    phi[0] == 0.0
                    and phi[-1] == 1.0
                    and all(a < b for a, b in itertools.pairwise(phi))
                )
            else:
                assert phi == [0.5]
    spec = {f["field"]: f for f in value_spec(SeqGrammar(live))}
    assert set(spec) == {
        "seed_count",
        "beam_width",
        "depth",
        "fanout",
        "returned_k",
        "retrieval_interval",
        "sem_threshold_pre",
        "sem_threshold_post",
    }
    for field in (
        "seed_count",
        "beam_width",
        "fanout",
        "returned_k",
        "retrieval_interval",
    ):
        assert spec[field]["scale"] == "log2"
    for field in ("depth", "sem_threshold_pre", "sem_threshold_post"):
        assert spec[field]["scale"] == "linear"
    seed = spec["seed_count"]
    assert seed["phi"][seed["values"].index(16)] == pytest.approx(
        (math.log2(16) - 2) / (7 - 2)
    )
    k = spec["returned_k"]
    assert k["phi"][k["values"].index(5)] == pytest.approx(
        (math.log2(5) - math.log2(3)) / (math.log2(10) - math.log2(3))
    )
    assert spec["depth"]["phi"] == [0.0, 0.5, 1.0]
    assert spec["sem_threshold_pre"]["phi"] == pytest.approx([0.0, 0.5, 1.0])
    # the model's vectorised lookup tables agree with the spec
    g = SeqGrammar(live)
    m = SeqVAEv2.from_grammar(g, 4, layers=1)
    t = g.index["depth=3"]
    assert bool(m.is_num[t]) and float(m.phi[t]) == pytest.approx(0.5)
    assert not bool(m.is_num[g.index["dir=Both"]]) and not bool(m.is_num[g.index["V"]])
    assert m.cfg["lambda_ord"] == 1.0 and m.cfg["numeric"] == value_spec(g)


def test_checkpoint_roundtrip_and_dispatch(tmp_path):
    m, metrics = trained()
    p = tmp_path / "v2.pt"
    save_checkpoint_v2(p, m, S, metrics)
    ck = torch.load(p, weights_only=False)
    assert ck["kind"] == KIND == "seq-v2"
    assert ck["cfg"]["lambda_ord"] == m.cfg["lambda_ord"] and ck["cfg"][
        "numeric"
    ] == value_spec(G)
    m2, met2 = load_checkpoint(p, S)
    assert type(m2) is SeqVAEv2 and not m2.training and met2 == metrics
    assert all(q.device.type == "cpu" for q in m2.parameters())
    x = _x(enumerate_space(S)[:12])
    with torch.no_grad():
        z = m.encode_mean(x[:, 1:])
        assert torch.equal(z, m2.encode_mean(x[:, 1:]))
        assert torch.equal(
            m._decode_logits(z, x[:, :-1]), m2._decode_logits(z, x[:, :-1])
        )
    with pytest.raises(ValueError):
        load_checkpoint(p, JointSpace.live())
    # v1 checkpoints still load as plain SeqVAE
    p1 = tmp_path / "v1.pt"
    save_checkpoint(p1, tiny_model(4), S, {"d": 4})
    m1, _ = load_checkpoint(p1, S)
    assert type(m1) is SeqVAE
    # unknown kinds are refused
    ck["kind"] = "seq-v9"
    torch.save(ck, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "bad.pt", S)


def test_aux_loss_masks_inactive_fields():
    m = SeqVAEv2.from_grammar(G, 4, layers=1)
    cols = {f: i for i, f in enumerate(m.aux_fields)}
    assert set(cols) == {
        "seed_count",
        "depth",
        "beam_width",
        "returned_k",
    }  # >= 2 values in the tiny space
    bfs = S.default(graph_impl="BFS", seed_count=32, depth=3, returned_k=5)
    beam = S.default(
        graph_impl="beam", beam_width=4, seed_count=8, depth=2, returned_k=3
    )
    x = _x([bfs, beam])[:, 1:]
    target, mask = aux_targets(m, x)
    assert mask.shape == target.shape == (2, 4)
    assert mask[0, cols["beam_width"]] == 0 and mask[1, cols["beam_width"]] == 1
    assert mask[0].sum() == 3 and mask[1].sum() == 4
    assert target[0, cols["seed_count"]] == 1.0 and target[0, cols["depth"]] == 1.0
    assert target[1, cols["beam_width"]] == 1.0 and target[1, cols["returned_k"]] == 0.0
    # the loss ignores the prediction of an inactive field
    mu = torch.randn(1, 4)
    with torch.no_grad():
        base = float(aux_loss(m, mu, x[:1]))
        both = float(aux_loss(m, mu.repeat(2, 1), x))
        m.aux.weight[cols["beam_width"]] += 10.0
        m.aux.bias[cols["beam_width"]] += 10.0
        assert float(aux_loss(m, mu, x[:1])) == pytest.approx(
            base
        )  # beam inactive in row 0
        assert (
            float(aux_loss(m, mu.repeat(2, 1), x)) > both + 1.0
        )  # beam active in row 1


def test_spearman_ties():
    assert spearman(
        np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])
    ) == pytest.approx(1.0)
    assert spearman(
        np.array([1, 1, 2, 2.0]), np.array([4, 3, 2, 1.0])
    ) == pytest.approx(-0.894427, abs=1e-5)
    assert math.isnan(spearman(np.ones(4), np.arange(4.0)))


def test_tiny_training_reconstructs_and_orders():
    _, metrics = trained()
    h = metrics["heldout"]
    assert (
        h["exact_masked"] >= 0.9
        and h["exact_repaired"] >= 0.9
        and h["exact"] == h["exact_repaired"]
    )
    assert set(h["per_field"]) >= {"seed_count", "depth", "graph_impl"}
    order = h["order"]
    assert order["depth"]["aux_spearman"] > 0.8
    assert set(order["depth"]) >= {"aux_spearman", "best_dim_spearman", "best_dim", "n"}
    assert 0.0 <= h["collapsed_fraction"] <= 1.0
    assert metrics["arch"]["lambda_ord"] == 1.0 and metrics["kind"] == "seq-v2"
    json.dumps(metrics)  # the CLI writes it as JSON


def test_cli_arch_v2_writes_v2_checkpoint(tmp_path, monkeypatch):
    from tools import train_trace_seq_vae as cli

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"space": S.manifest()}))
    out = tmp_path / "out"
    argv = [
        "x",
        "--run-dir",
        str(run),
        "--out",
        str(out),
        "--dim",
        "4",
        "--n",
        "64",
        "--holdout",
        "16",
        "--epochs",
        "1",
        "--layers",
        "1",
        "--threads",
        "4",
        "--arch",
        "v2",
        "--lambda-ord",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    m, met = load_checkpoint(out / "seqvae_v2_d4.pt", S)
    assert type(m) is SeqVAEv2 and m.cfg["lambda_ord"] == 0.5
    assert (
        json.loads((out / "seqvae_v2_d4.json").read_text())["heldout"]["n"]
        == 16
        == met["heldout"]["n"]
    )
