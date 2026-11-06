import numpy as np
import pytest
import torch

from tests.latent_fixtures import enumerate_space, tiny_model, tiny_space
from trimem.optimizer.latent_plan_codec import (
    Affine,
    LatentCodec,
    _affine_from_z,
    file_sha256,
    fit_affine,
    load_affine,
    load_codec,
    save_affine,
)
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint

torch.set_num_threads(4)


def test_affine_rule():
    rng = np.random.default_rng(0)
    zt = rng.normal(size=(50, 3))
    zt[:, 2] = 1.0  # zero std -> floor
    zi = np.array([[10.0, 0.0, 1.0], [0.0, -10.0, 1.0]])
    a = _affine_from_z(zt, zi)
    lo, hi = np.array(a.lo), np.array(a.hi)
    allz = np.concatenate([zt, zi])
    assert (lo <= allz.min(0)).all() and (hi >= allz.max(0)).all()
    m, s = zt.mean(0), np.maximum(zt.std(0), 1e-3)
    w0 = 8 * s
    # untouched dim 2: exactly mean+-4std then 5 % margin
    assert lo[2] == pytest.approx(m[2] - 4 * s[2] - 0.05 * w0[2])
    assert hi[2] == pytest.approx(m[2] + 4 * s[2] + 0.05 * w0[2])
    # widened dim 0: hi = max(init) + 5 % of original width
    assert hi[0] == pytest.approx(10.0 + 0.05 * w0[0])


def test_affine_does_not_clip():
    a = Affine((0.0,), (1.0,))
    u = a.to_u(torch.tensor([[2.0]], dtype=torch.double))
    assert u.item() == 2.0 and a.to_z(u).item() == 2.0


@pytest.fixture(scope="module")
def codec():
    space, model = tiny_space(), tiny_model(4)
    grammar = SeqGrammar(space)
    configs = enumerate_space(space)
    aff = fit_affine(model, grammar, configs)
    return LatentCodec(space, model, aff, checkpoint_sha256="x"), configs


def test_train_encodings_inside_unit_box(codec):
    c, configs = codec
    assert len(configs) == 576
    U = c.encode_many(configs)
    assert U.dtype == torch.double and U.shape == (576, 4)
    assert (U >= 0).all() and (U <= 1).all()


def test_encode_cached_and_deterministic(codec):
    c, configs = codec
    c2 = LatentCodec(c.space, c.model, c.affine, checkpoint_sha256="x")
    assert not c2._cache
    a = c2.encode(configs[0])
    assert c2.encode(configs[0]) == a and isinstance(a, list) and len(a) == 4
    c2.encode_many(configs[:10])
    assert all(c2.key(x) in c2._cache for x in configs[:10])
    assert c2.encode(configs[3]) == c2.encode_many([configs[3]])[0].tolist()  # first writer wins
    assert c2.identity == c.identity


def test_decode_casts_dtype_and_roundtrip(codec):
    c, configs = codec
    U = c.encode_many(configs[:8])
    out = c.decode(U)
    assert len(out) == 8 and all(isinstance(x, dict) for x in out)
    assert isinstance(c.roundtrip(configs[0]), bool)


def test_codec_rejects_changed_checkpoint(tmp_path):
    space, model = tiny_space(), tiny_model(4)
    ck = tmp_path / "m.pt"
    save_checkpoint(ck, model, space, {})
    aff = fit_affine(model, SeqGrammar(space), enumerate_space(space)[:50])
    save_affine(tmp_path / "a.json", aff, {"checkpoint_sha256": file_sha256(ck)})
    a2, meta = load_affine(tmp_path / "a.json")
    assert a2 == aff
    codec = load_codec(ck, tmp_path / "a.json", space)
    assert codec.d == 4
    raw = bytearray(ck.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ck.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        load_codec(ck, tmp_path / "a.json", space)


def test_codec_sets_eval_mode():
    space, model = tiny_space(), tiny_model(4)
    model.train()
    try:
        LatentCodec(space, model, Affine((0.0,) * 4, (1.0,) * 4), checkpoint_sha256="x")
        assert not model.training
    finally:
        model.eval()
