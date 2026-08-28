"""Tests for the wiki embedding densify step — no model, no network, no corpus.

Builds synthetic tools/wiki_embed_hybrid checkpoints on disk and drives
tools.wiki_densify_emb over them. The load-bearing contract under test is the one
every wiki harness assumes and none of them can verify: row i of
`dense_id_aligned.npy` is article id i. The permuted-checkpoint case is the whole
reason the step is a scatter rather than a copy — that is the case a reshape would
silently get wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.wiki_densify_emb import (  # noqa: E402
    CheckpointError,
    default_out,
    densify,
    main,
)

DIM = 4
SHARD = 2


def write_checkpoint(
    d: Path,
    ids: list[int],
    vectors: np.ndarray,
    *,
    shards_done: list[int] | None = None,
    normalized: bool = True,
) -> Path:
    """Materialize a wiki_embed_hybrid-shaped checkpoint directory."""
    emb = d / "emb"
    emb.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    vectors.astype(np.float32).tofile(emb / "vectors.f32")
    np.save(emb / "ids.i64.npy", np.asarray(ids, dtype=np.int64))
    n_shards = -(-n // SHARD)
    done = np.asarray(
        shards_done if shards_done is not None else [1] * n_shards, dtype=np.uint8
    )
    done.tofile(emb / "shards.done")
    (emb / "meta.json").write_text(
        json.dumps(
            {
                "model": "BAAI/bge-small-en-v1.5",
                "dim": DIM,
                "N": n,
                "shard_size": SHARD,
                "normalized": normalized,
                "status": "complete",
            }
        )
    )
    return emb


def rows(n: int) -> np.ndarray:
    """Distinguishable rows: row r is all-r, so a mis-scatter is visible."""
    return np.tile(np.arange(n, dtype=np.float32).reshape(n, 1), (1, DIM))


def test_identity_checkpoint_round_trips(tmp_path):
    emb = write_checkpoint(tmp_path, [0, 1, 2, 3], rows(4))
    stats = densify(emb, default_out(emb))

    out = np.load(default_out(emb))
    assert out.shape == (4, DIM)
    np.testing.assert_array_equal(out, rows(4))
    assert stats["was_identity"] is True
    assert stats["zero_rows"] == 0
    assert stats["embedded_rows"] == 4


def test_permuted_checkpoint_is_scattered_to_id_order(tmp_path):
    # The embedder's work-queue slot order is NOT article-id order here: slot 0
    # holds article 3, slot 1 holds article 0, and so on. A reshape would keep the
    # slot order and mislabel every vector.
    emb = write_checkpoint(tmp_path, [3, 0, 2, 1], rows(4))
    stats = densify(emb, default_out(emb))

    out = np.load(default_out(emb))
    # article 3 got slot 0's vector (all-0), article 0 got slot 1's (all-1), ...
    np.testing.assert_array_equal(out[3], np.full(DIM, 0.0, dtype=np.float32))
    np.testing.assert_array_equal(out[0], np.full(DIM, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(out[2], np.full(DIM, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.full(DIM, 3.0, dtype=np.float32))
    assert stats["was_identity"] is False


def test_incomplete_run_is_refused_but_allow_partial_zero_fills(tmp_path):
    emb = write_checkpoint(tmp_path, [0, 1, -1, -1], rows(4), shards_done=[1, 0])

    with pytest.raises(CheckpointError, match="INCOMPLETE"):
        densify(emb, default_out(emb))

    stats = densify(emb, default_out(emb), strict=False, allow_partial=True)
    out = np.load(default_out(emb))
    assert out.shape == (2, DIM)  # max embedded id is 1
    assert stats["embedded_rows"] == 2
    assert stats["zero_rows"] == 0


def test_gap_is_a_zero_row_and_strict_refuses_it(tmp_path):
    # article 1 was never embedded; ids 0 and 2 were. Dropping row 1 would shift
    # article 2 into row 1 and mislabel every downstream recall number.
    emb = write_checkpoint(tmp_path, [0, 2], rows(2), shards_done=[1])

    with pytest.raises(CheckpointError, match="no vector"):
        densify(emb, default_out(emb))

    stats = densify(emb, default_out(emb), strict=False)
    out = np.load(default_out(emb))
    assert out.shape == (3, DIM)
    np.testing.assert_array_equal(out[1], np.zeros(DIM, dtype=np.float32))
    np.testing.assert_array_equal(out[2], np.full(DIM, 1.0, dtype=np.float32))
    assert stats["zero_rows"] == 1


def test_duplicate_id_is_always_refused(tmp_path):
    emb = write_checkpoint(tmp_path, [0, 0, 1, 2], rows(4))
    # Not recoverable by --no-strict/--allow-partial: one of the two vectors would
    # silently win the row.
    with pytest.raises(CheckpointError, match="DUPLICATE"):
        densify(emb, default_out(emb), strict=False, allow_partial=True)


def test_meta_sidecar_carries_provenance(tmp_path):
    emb = write_checkpoint(tmp_path, [0, 1, 2, 3], rows(4), normalized=False)
    densify(emb, default_out(emb))

    side = json.loads(Path(str(default_out(emb)) + ".meta.json").read_text())
    assert side["model"] == "BAAI/bge-small-en-v1.5"
    assert side["dim"] == DIM
    assert side["rows"] == 4
    assert side["normalized"] is False
    assert side["shards_done"] == side["shards_total"] == 2


def test_cli_refuses_with_nonzero_exit(tmp_path, capsys):
    emb = write_checkpoint(tmp_path, [0, 1, -1, -1], rows(4), shards_done=[1, 0])
    assert main(["--emb", str(emb)]) == 1
    assert "REFUSED" in capsys.readouterr().out

    assert main(["--emb", str(emb), "--allow-partial", "--no-strict"]) == 0
    assert default_out(emb).is_file()


def test_cli_warns_on_unnormalized(tmp_path, capsys):
    emb = write_checkpoint(tmp_path, [0, 1], rows(2), shards_done=[1], normalized=False)
    assert main(["--emb", str(emb)]) == 0
    assert "normalized=false" in capsys.readouterr().out
