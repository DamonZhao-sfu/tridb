"""Train the BayesQO-style sequence VAE for TRACE and report decoder reconstruction.

Usage: python -m tools.train_trace_seq_vae --run-dir <run> --out <dir> --dim D
           [--n 300000] [--epochs 10] [--layers 6] [--measured <run> ...]
           [--arch v1|v2] [--lambda-ord 1.0] [--holdout 20000]
"exact" = the greedy-decoded, grammar-repaired configuration equals the input canonical config.
--arch v2 (latent_seq_vae_v2): continuous value embedding + linear order head on mu; writes
seqvae_v2_d{D}.pt/.json (v1 writes seqvae_d{D}.pt/.json) and also reports the masked decoder.
"""
import argparse, json, sqlite3
from pathlib import Path
import torch
from trimem.optimizer.plan_schema import JointSpace
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import evaluate, save_checkpoint, train_seq_vae


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dim", required=True, type=int)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--holdout", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--measured", nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arch", choices=("v1", "v2"), default="v1")
    ap.add_argument("--lambda-ord", type=float, default=1.0, help="v2 only: weight of the order (aux) loss")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sm = json.loads((a.run_dir / "manifest.json").read_text())["space"]
    space = JointSpace(sm["domains"], execution_fingerprint=sm["execution_fingerprint"])  # no botorch import (GPU env)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.arch == "v2":
        from trimem.optimizer.latent_seq_vae_v2 import evaluate_v2, save_checkpoint_v2, train_seq_vae_v2

        model, m = train_seq_vae_v2(space, a.dim, n_samples=a.n, holdout=a.holdout, epochs=a.epochs, seed=a.seed,
                                    layers=a.layers, lambda_ord=a.lambda_ord, device=a.device)
        stem, save, ev = f"seqvae_v2_d{a.dim}", save_checkpoint_v2, lambda md, g, cs: evaluate_v2(md, g, cs, full=False)
    else:
        model, m = train_seq_vae(space, a.dim, n_samples=a.n, holdout=a.holdout, epochs=a.epochs, seed=a.seed,
                                 layers=a.layers, device=a.device)
        stem, save, ev = f"seqvae_d{a.dim}", save_checkpoint, evaluate
    meas = []
    for r in a.measured:
        db = sqlite3.connect(f"file:{Path(r) / 'ledger.sqlite'}?mode=ro", uri=True)
        meas += [json.loads(b) for (b,) in db.execute("SELECT body FROM configs")]
    if meas:
        m["measured"] = ev(model, SeqGrammar(space), meas)
    save(a.out / f"{stem}.pt", model, space, m)
    (a.out / f"{stem}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({"arch": a.arch, "d": a.dim, "heldout_exact": m["heldout"]["exact"],
                      "heldout_exact_masked": m["heldout"].get("exact_masked"),
                      "measured_exact": m.get("measured", {}).get("exact")}), flush=True)


if __name__ == "__main__":
    main()
