"""Train TRACE latent VAEs for several latent sizes and report decoder reconstruction.

Usage: python -m tools.train_trace_latent_vae --run-dir <run with manifest> --out <dir>
           [--dims 4,6,8,12,16] [--n 1000000] [--epochs 20] [--measured <run> ...]
Writes vae_d{d}.pt and vae_metrics.json. "exact" = the decoded canonical configuration equals
the input canonical configuration (all slots). Measured configurations of finished runs are
evaluated separately (they are not in the training set unless --train-on-measured).
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from trimem.experiments.trace_joint_plan_search import get_space
from trimem.optimizer.latent_grammar import encode_tokens
from trimem.optimizer.latent_vae import evaluate, save_checkpoint, train_vae


def measured_tokens(space, run_dirs):
    out = []
    for r in run_dirs:
        db = sqlite3.connect(f"file:{Path(r) / 'ledger.sqlite'}?mode=ro", uri=True)
        out += [encode_tokens(space, json.loads(b)) for (b,) in db.execute("SELECT body FROM configs")]
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dims", default="4,6,8,12,16")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--measured", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(8)
    space = get_space(json.loads((a.run_dir / "manifest.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    meas = measured_tokens(space, a.measured) if a.measured else None
    report = {}
    for d in map(int, a.dims.split(",")):
        model, m = train_vae(space, d, n_samples=a.n, epochs=a.epochs, seed=a.seed, beta=a.beta)
        if meas is not None and len(meas):
            m["measured"] = evaluate(model, space, meas)
        save_checkpoint(a.out / f"vae_d{d}.pt", model, space, m)
        report[d] = {k: v for k, v in m.items() if k != "bounds"}
        print(json.dumps({"d": d, "heldout_exact": m["heldout"]["exact"],
                          "measured_exact": m.get("measured", {}).get("exact")}), flush=True)
        (a.out / "vae_metrics.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
