"""Train the sequence VAE v2 on the TRACE v0.7.0 live space (user 2026-11-05).

Same trainer as ``tools.train_trace_seq_vae --arch v2`` (``latent_seq_vae_v2``), but the space is
``plan_schema_v070.JointSpaceV070.live()`` (rank optional incl. "none", no LLM rank, graph_direction Both,
retrieval_unit whole) instead of a run manifest. The name keeps ``train_trace_seq_vae`` so
``trace_latent_live.check_no_vae_training`` still refuses live evaluation while this runs.

Usage: python -m tools.train_trace_seq_vae_v070 --out DIR --dim D [--n 1000000] [--epochs 20] [--layers 6]
           [--holdout 20000] [--seed 20261104] [--threads 8] [--device cuda] [--lambda-ord 1.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trimem.optimizer.plan_schema_v070 import LIVE_DOMAINS, SCHEMA_VERSION, JointSpaceV070


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dim", required=True, type=int)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--holdout", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lambda-ord", type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from trimem.optimizer.latent_seq_vae_v2 import evaluate_v2, save_checkpoint_v2, train_seq_vae_v2

    space = JointSpaceV070.live(execution_fingerprint="vae-training")
    a.out.mkdir(parents=True, exist_ok=True)
    model, m = train_seq_vae_v2(space, a.dim, n_samples=a.n, holdout=a.holdout, epochs=a.epochs, seed=a.seed,
                                layers=a.layers, lambda_ord=a.lambda_ord, device=a.device)
    m["space"] = {"schema": SCHEMA_VERSION, "domains": {k: list(v) for k, v in LIVE_DOMAINS.items()}}
    stem = f"seqvae_v2_d{a.dim}"
    save_checkpoint_v2(a.out / f"{stem}.pt", model, space, m)
    (a.out / f"{stem}.json").write_text(json.dumps(m, indent=1, default=str))
    print(json.dumps({"d": a.dim, "heldout_exact": m["heldout"]["exact"],
                      "heldout_exact_masked": m["heldout"].get("exact_masked")}), flush=True)


if __name__ == "__main__":
    main()
