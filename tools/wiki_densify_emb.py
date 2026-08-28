"""Collapse the hybrid-embedder checkpoint into the id-aligned matrix every harness reads.

WHY THIS EXISTS
---------------
`tools/wiki_embed_hybrid.py` writes a RESUMABLE checkpoint, not a corpus matrix:

    vectors.f32   float32 memmap (N, dim)   -- row = the embedder's work-queue slot
    ids.i64.npy   int64 (N,)                -- row i -> article id (-1 = unfilled)
    shards.done   uint8 (n_shards,)         -- 1 = shard embedded AND flushed
    meta.json     {model, dim, N, shard_size, normalized, status, ...}

Every consumer in this repo instead reads ONE file, `dense_id_aligned.npy`, whose
contract is row i == article id i (bench/wiki_h2h.Cfg.emb_path, and through it
bench/wiki_fusion, bench/wiki_h2h_queryset, bench/wiki_ppr_gate). Nothing in the
repository produced it for the wiki corpus -- the wiki matrix was assembled on the
Spark and only the wikidata twin (tools/wikidata_embed) emits the dense form
directly. This module is that missing step, and it is deliberately the ONLY place
that knows how to turn the checkpoint into the contract.

WHY A SCATTER AND NOT A RESHAPE. The extractor guarantees dense 0..N-1 article ids
(docs/wiki_scale_load_design_v0.1.0.md §0), so on a healthy full run `ids[i] == i`
and the scatter is the identity. It is still written as a scatter because that
assumption is exactly what a partial/resumed/`--limit`ed embedding run breaks, and
a silently mis-permuted embedding matrix produces a benchmark that looks fine and
measures nothing. `--strict` (default) refuses anything that would make row i mean
something other than article i.

GAPS ARE ZERO ROWS, NOT DROPPED ROWS -- the same rule as tools/wikidata_embed: a
dropped row would shift every later id by one, while a zero vector can never win a
cosine ranking, so it is dropped-from-the-leg while keeping id alignment. Gaps are
counted and reported; `--strict` refuses to emit any.

OUTPUT (--out, default <emb>/dense_id_aligned.npy) via np.lib.format.open_memmap so
peak RAM is one row-block at any corpus size, plus a `<out>.meta.json` sidecar
carrying the embedder's provenance forward (model, dim, rows, normalized, gaps).

CLI:
    python -m tools.wiki_densify_emb --emb data/wiki/enwiki/emb
    python -m tools.wiki_densify_emb --emb data/wiki/enwiki/emb --allow-partial
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DENSIFY_VERSION = "0.1.0"
BLOCK = 65_536  # rows copied per scatter block; bounds peak RAM independent of N


class CheckpointError(RuntimeError):
    """The checkpoint cannot produce a trustworthy id-aligned matrix."""


def default_out(emb_dir: Path) -> Path:
    """The Cfg.emb_path default location: <emb>/dense_id_aligned.npy."""
    return emb_dir / "dense_id_aligned.npy"


def read_meta(emb_dir: Path) -> dict:
    meta_path = emb_dir / "meta.json"
    if not meta_path.is_file():
        raise CheckpointError(
            f"no meta.json in {emb_dir} -- run tools/wiki_embed_hybrid.py first"
        )
    return json.loads(meta_path.read_text())


def completed_shards(emb_dir: Path, n: int, shard_size: int) -> tuple[int, int]:
    """(done_shards, total_shards) from the shards.done marker file."""
    total = math.ceil(n / shard_size) if shard_size else 0
    done_path = emb_dir / "shards.done"
    if not done_path.is_file():
        return 0, total
    done = np.memmap(done_path, dtype=np.uint8, mode="r")
    return int(np.count_nonzero(done)), total


def densify(
    emb_dir: Path,
    out_path: Path,
    *,
    strict: bool = True,
    allow_partial: bool = False,
) -> dict:
    """Scatter the checkpoint's rows to article-id order; return a stats dict.

    Raises CheckpointError rather than emitting a matrix whose row i does not mean
    article i (incomplete run, duplicate id, id below zero) unless the caller has
    explicitly opted out via allow_partial / strict=False.
    """
    meta = read_meta(emb_dir)
    n = int(meta["N"])
    dim = int(meta["dim"])
    shard_size = int(meta.get("shard_size", 0))
    status = meta.get("status", "unknown")

    done, total = completed_shards(emb_dir, n, shard_size)
    if done != total and not allow_partial:
        raise CheckpointError(
            f"embedding run is INCOMPLETE: {done}/{total} shards flushed "
            f"(meta status={status!r}). Finish tools/wiki_embed_hybrid.py, or pass "
            f"--allow-partial to densify what exists (missing ids become zero rows)."
        )

    vectors = np.memmap(emb_dir / "vectors.f32", dtype=np.float32, mode="r").reshape(
        n, dim
    )
    ids = np.load(emb_dir / "ids.i64.npy")
    if ids.shape != (n,):
        raise CheckpointError(f"ids.i64.npy is {ids.shape}, expected {(n,)}")

    filled = ids >= 0
    n_filled = int(np.count_nonzero(filled))
    if n_filled == 0:
        raise CheckpointError(f"no embedded rows in {emb_dir} (every id is -1)")

    present = ids[filled]
    max_id = int(present.max())
    rows = max_id + 1

    # A duplicate id means two different vectors claim the same row -- last write
    # would silently win. That is never acceptable, partial run or not.
    uniq = np.unique(present)
    if uniq.size != present.size:
        raise CheckpointError(
            f"{present.size - uniq.size} DUPLICATE article ids in ids.i64.npy -- "
            f"the checkpoint is corrupt; re-run the embedder with --no-resume"
        )

    gaps = rows - uniq.size
    if gaps and strict:
        raise CheckpointError(
            f"{gaps} of {rows} article ids have no vector -- emitting them as zero "
            f"rows would silently remove them from the vector leg. Pass --no-strict "
            f"to accept zero rows, or finish the embedding run."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dense = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(rows, dim)
    )
    dense[:] = 0.0  # gaps stay zero: never wins a cosine ranking, keeps alignment

    src_rows = np.nonzero(filled)[0]
    for start in range(0, src_rows.size, BLOCK):
        block = src_rows[start : start + BLOCK]
        dense[ids[block]] = vectors[block]
    dense.flush()

    identity = bool(uniq.size == rows and np.array_equal(present, np.arange(n_filled)))
    stats = {
        "densify_version": DENSIFY_VERSION,
        "source": str(emb_dir),
        "out": str(out_path),
        "model": meta.get("model"),
        "dim": dim,
        "rows": rows,
        "embedded_rows": n_filled,
        "zero_rows": gaps,
        "normalized": meta.get("normalized"),
        "checkpoint_status": status,
        "shards_done": done,
        "shards_total": total,
        # True => the checkpoint was already in article-id order and the scatter
        # was the identity. False is NOT an error, it is why this step exists.
        "was_identity": identity,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--emb",
        type=Path,
        default=Path("data/wiki/enwiki/emb"),
        help="the tools/wiki_embed_hybrid checkpoint directory",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .npy (default <emb>/dense_id_aligned.npy)",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="densify a checkpoint whose shards are not all flushed",
    )
    ap.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="emit zero rows for article ids that have no vector",
    )
    a = ap.parse_args(argv)

    out = a.out or default_out(a.emb)
    try:
        stats = densify(a.emb, out, strict=a.strict, allow_partial=a.allow_partial)
    except CheckpointError as exc:
        print(f"[densify] REFUSED: {exc}")
        return 1

    print(
        f"[densify] {stats['rows']} rows x {stats['dim']} -> {out} "
        f"({stats['embedded_rows']} embedded, {stats['zero_rows']} zero, "
        f"identity={stats['was_identity']}, normalized={stats['normalized']})"
    )
    if not stats["normalized"]:
        print(
            "[densify] WARNING: meta.json says normalized=false. The harnesses rank "
            "with L2 `<->` as a cosine proxy -- unnormalized vectors make that ranking "
            "disagree with the cosine oracle."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
