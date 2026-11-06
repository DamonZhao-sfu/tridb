"""Readiness gates R1-R6 for the TRACE sequence VAE and the d / surrogate-variant choice.

Spec section 5 acceptance, applied to the existing checkpoints (global constraints, resolution 1):
  R1 masked-decoder exact match on the checkpoint's own held-out set (>= 0.95)
  R2 min over families (filter_order, graph_impl) of the R1 rate (>= 0.90)
  R3 valid decodes of 10,000 Sobol u in [0,1]^d under a provisional affine fit on train u (== 1)
  R4 near-duplicate u pairs (r = 1e-6) over train u and held-out u, distinct keys only (== 0)
  R5 exact match on fresh test configs that are in neither train nor held-out (reported)
  R6 (freeze-affine) init configs whose masked decode of their own u equals the config
The repaired decoder (SeqGrammar.decode) is reported next to R1, R2 and R5. The training corpus
is uniform Sobol, so no held-out combination test exists ("not established").
CPU only (4 threads).

Usage:
  python -m tools.latent_readiness run --space-run R --checkpoints P... --out readiness.json
      [--probe 10000] [--test 20000] [--affine-cache DIR]
  python -m tools.latent_readiness choose --readiness readiness.json --gates gates.json --out chosen.json
  python -m tools.latent_readiness freeze-affine --checkpoint P --shared-init R0 --out A
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from trimem.optimizer.latent_plan_codec import (
    Affine,
    LatentCodec,
    _affine_from_z,
    _encode_z,
    file_sha256,
    fit_affine,
    load_affine,
    save_affine,
)
from trimem.optimizer.latent_seq_grammar import BOS, SeqGrammar
from trimem.optimizer.latent_seq_vae import load_checkpoint
from trimem.optimizer.plan_schema import FIELD_NAMES

OBJ = ("quality", "tokens", "latency")
VARIANTS = {
    "A": {"kernel": "saas", "structure": "independent"},
    "B": {"kernel": "saas", "structure": "coupled-cost"},
    "C": {"kernel": "matern", "structure": "independent"},
    # user 2026-11-04 ("先用1+2"): faster SAAS variants
    "D": {"kernel": "map-saas", "structure": "independent"},
    "E": {"kernel": "saas", "structure": "independent", "nuts": [256, 128, 16]},
    # user 2026-11-04: the same SAAS model with NumPyro/JAX NUTS (trimem/optimizer/latent_numpyro_saas.py)
    "F": {"kernel": "saas-numpyro", "structure": "independent"},
    # user 2026-11-04: Matern quality GP + joint (tokens, latency) multi-task GP
    "H": {"kernel": "matern", "structure": "coupled-cost"},
    "G": {"kernel": "saas-numpyro", "structure": "independent", "nuts": [256, 128, 16]},
}
PROVISIONAL_KIND = "provisional-train-v1"
FROZEN_KIND = "frozen-train-init-v1"
COMBINATION = "not established (uniform corpus)"
DECODE_BATCH = 2048
COLLISION_R = 1e-6


# ------------------------------------------------------------------ corpus and affine
def corpus(space, metrics: dict) -> tuple[list[dict], list[dict]]:
    """(train, held-out) exactly as train_seq_vae drew them."""
    n_train = int(metrics["n_train"])
    holdout = int(metrics.get("holdout", metrics["heldout"]["n"]))
    total = n_train + holdout
    configs = space.sample(total, seed=int(metrics["seed"]), max_draws=4 * total)
    if len(configs) != total:
        raise ValueError(f"re-derived corpus has {len(configs)} configs, expected {total}")
    return configs[:n_train], configs[n_train:]


def affine_cache_path(cache_dir, checkpoint) -> Path:
    return Path(cache_dir) / f"{Path(checkpoint).stem}.provisional_affine.json"


def provisional_affine(checkpoint, space, model, metrics: dict, *, cache_dir=None, zt=None,
                       sha: str | None = None) -> Affine:
    """fit_affine(train u) for the checkpoint; cached in cache_dir keyed by checkpoint sha256."""
    sha = sha or file_sha256(checkpoint)
    path = affine_cache_path(cache_dir, checkpoint) if cache_dir is not None else None
    if path is not None and path.exists():
        aff, meta = load_affine(path)
        if meta.get("checkpoint_sha256") == sha and meta.get("kind") == PROVISIONAL_KIND:
            return aff
    if zt is None:
        train, _ = corpus(space, metrics)
        zt = _encode_z(model, SeqGrammar(space), train)
    aff = _affine_from_z(np.asarray(zt), np.zeros((0, model.d)))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"kind": PROVISIONAL_KIND, "checkpoint_sha256": sha, "n_train": int(metrics["n_train"]),
                "seed": int(metrics["seed"])}
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        save_affine(tmp, aff, meta)
        os.replace(tmp, path)  # atomic: readiness and gates may share the cache
    return aff


# ------------------------------------------------------------------ decoding helpers
def _batched(fn, rows) -> list:
    """fn over DECODE_BATCH chunks; a chunk that raises is retried row by row (None = failure)."""
    out: list = []
    for i in range(0, len(rows), DECODE_BATCH):
        chunk = rows[i : i + DECODE_BATCH]
        try:
            out += fn(chunk)
        except Exception:  # noqa: BLE001 - a decode that raises is a counted failure
            for j in range(len(chunk)):
                try:
                    out += fn(chunk[j : j + 1])
                except Exception:  # noqa: BLE001
                    out.append(None)
    return out


def masked_decode(codec: LatentCodec, U: torch.Tensor) -> list[dict | None]:
    return _batched(codec.decode, U)


def repaired_decode(model, grammar: SeqGrammar, z: torch.Tensor) -> list[dict | None]:
    bos = grammar.index[BOS]
    dtype = next(model.parameters()).dtype

    def fn(zz):
        with torch.no_grad():
            seq = model.decode_greedy(zz.to(dtype), bos)
        return [grammar.decode(row.tolist()) for row in seq]

    return _batched(fn, z)


def _hits(space, decoded, truth) -> list[bool]:
    return [d is not None and d == space.canonicalize(t) for d, t in zip(decoded, truth)]


def _rate(hits) -> float:
    return float(sum(hits) / len(hits)) if len(hits) else float("nan")


def _valid(space, c) -> bool:
    if c is None:
        return False
    try:
        return space.canonicalize(c) == c
    except (ValueError, KeyError, TypeError):
        return False


def family_rates(configs, hits) -> tuple[float, dict[str, float]]:
    """Exact-match rate per family (filter_order, graph_impl) and its minimum."""
    tot: dict[str, list[int]] = {}
    for c, h in zip(configs, hits):
        t = tot.setdefault(f"{c['filter_order']}|{c['graph_impl']}", [0, 0])
        t[0] += bool(h)
        t[1] += 1
    per = {k: v[0] / v[1] for k, v in tot.items()}
    return (min(per.values()) if per else float("nan")), per


def collisions(U, keys) -> int:
    """Pairs of distinct keys whose u lie within COLLISION_R (first row per key)."""
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    pts = np.asarray(U, dtype=float)[sorted(first.values())]
    if len(pts) < 2:
        return 0
    return len(cKDTree(pts).query_pairs(r=COLLISION_R, output_type="ndarray"))


# ------------------------------------------------------------------ readiness
def readiness(checkpoint: Path, space, *, probe: int = 10_000, test: int = 20_000,
              affine_cache=None) -> dict:
    t0, phases, clock = time.time(), {}, [time.time()]

    def mark(name):
        now = time.time()
        phases[name] = round(now - clock[0], 2)
        clock[0] = now

    sha = file_sha256(checkpoint)
    model, metrics = load_checkpoint(checkpoint, space)
    grammar = SeqGrammar(space)
    train, val = corpus(space, metrics)
    mark("corpus")
    zt, zv = _encode_z(model, grammar, train), _encode_z(model, grammar, val)
    mark("encode_train_val")
    cache = Path(checkpoint).parent if affine_cache is None else affine_cache
    affine = provisional_affine(checkpoint, space, model, metrics, cache_dir=cache, zt=zt, sha=sha)
    codec = LatentCodec(space, model, affine, checkpoint_sha256=sha)
    uv = affine.to_u(zv)
    val_m = masked_decode(codec, uv)
    val_r = repaired_decode(model, grammar, affine.to_z(uv))
    hits, hits_r = _hits(space, val_m, val), _hits(space, val_r, val)
    r2, fam = family_rates(val, hits)
    r2_r, _ = family_rates(val, hits_r)
    per_field = {f: _rate([d is not None and d[f] == t[f] for d, t in zip(val_m, val)]) for f in FIELD_NAMES}
    mark("decode_val")
    train_keys, val_keys = [space.key(c) for c in train], [space.key(c) for c in val]
    r4 = collisions(affine.to_u(torch.cat([zt, zv])).numpy(), train_keys + val_keys)
    mark("collisions")
    sobol = torch.quasirandom.SobolEngine(model.d, scramble=True, seed=int(metrics["seed"]))
    probe_dec = masked_decode(codec, sobol.draw(probe).double()) if probe else []
    ok = [_valid(space, c) for c in probe_dec]
    unique = len({space.key(c) for c, v in zip(probe_dec, ok) if v})
    mark("probe")
    blocked = set(train_keys) | set(val_keys)
    tests = space.sample(test, seed=int(metrics["seed"]) + 1, avoid=blocked, max_draws=8 * test + 4096)
    us = codec.encode_many(tests) if tests else torch.zeros((0, model.d), dtype=torch.double)
    test_m, test_r = masked_decode(codec, us), repaired_decode(model, grammar, affine.to_z(us))
    mark("test")
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha, "d": int(model.d),
        "n_train": len(train), "n_val": len(val), "n_probe": int(probe), "n_test": len(tests),
        "R1_val_exact": _rate(hits), "R1_val_exact_repaired": _rate(hits_r),
        "R2_family_min": r2, "R2_family_min_repaired": r2_r, "family_exact": fam,
        "R3_probe_valid": _rate(ok), "probe_unique_rate": unique / probe if probe else float("nan"),
        "R4_collisions": r4,
        "R5_test_exact": _rate(_hits(space, test_m, tests)),
        "R5_test_exact_repaired": _rate(_hits(space, test_r, tests)),
        "per_field_exact": per_field, "combination_holdout": COMBINATION,
        "affine": {"kind": PROVISIONAL_KIND, "cache": str(affine_cache_path(cache, checkpoint))},
        "phase_seconds": phases, "seconds": time.time() - t0,
    }


def passes_readiness(r: dict) -> bool:
    return bool(r["R1_val_exact"] >= 0.95 and r["R2_family_min"] >= 0.90
                and r["R3_probe_valid"] == 1.0 and r["R4_collisions"] == 0)


# ------------------------------------------------------------------ gate rule and choice
def gate_pass(latent: dict, native: dict, *, spearman_margin: float = 0.10, lpd_margin: float = 0.5) -> bool:
    """latent: {run: {"G2_spearman", "G3": {obj: {spearman, lpd}}}}; native: {run: {obj: {...}}}.
    NaN anywhere fails (comparisons with NaN are False)."""
    if not native:
        return False
    for run, nat in native.items():
        lat = latent.get(run)
        if lat is None or not lat["G2_spearman"] > 0:
            return False
        for o in OBJ:
            ls, ll = lat["G3"][o]["spearman"], lat["G3"][o]["lpd"]
            if not (ls >= nat[o]["spearman"] - spearman_margin and ll >= nat[o]["lpd"] - lpd_margin):
                return False
    return True


def _complete(g) -> bool:
    return isinstance(g, dict) and "pass" in g and "runs" in g


def mean_lpd(g: dict, variant: str) -> float:
    vals = [r["variants"][variant]["G3"][o]["lpd"] for r in g["runs"].values() for o in OBJ]
    return float(np.mean(vals)) if vals else float("nan")


def choose_d(readiness: dict, gates: dict) -> dict:
    """Smallest d with a checkpoint, passing readiness and >= 1 passing variant; that d's
    passing variant with the highest mean CV LPD (3 objectives x all runs)."""
    ds = sorted(set(readiness) | set(gates))
    missing = []
    for d in ds:
        what = (["readiness"] if readiness.get(d) is None else []) + ([] if _complete(gates.get(d)) else ["gates"])
        if what:
            missing.append({"d": d, "missing": what})
    notes = []
    for d in ds:
        r, g = readiness.get(d), gates.get(d)
        if r is None or not _complete(g):
            notes.append(f"d{d}: missing")
            continue
        if not passes_readiness(r):
            vals = ", ".join(f"{k}={r.get(k)}" for k in ("R1_val_exact", "R2_family_min", "R3_probe_valid", "R4_collisions"))
            notes.append(f"d{d}: readiness failed ({vals})")
            continue
        passing = sorted(v for v, ok in g["pass"].items() if ok)
        if not passing:
            notes.append(f"d{d}: no surrogate variant passes the gates")
            continue
        lpds = {v: mean_lpd(g, v) for v in passing}
        best = max(passing, key=lambda v: lpds[v])
        notes.append(f"d{d}: smallest passing d; variant {best} has the highest mean CV LPD of {passing}")
        return {"d": d, "variant": best, "mean_lpd": lpds, "reason": "; ".join(notes), "missing": missing}
    return {"d": None, "variant": None, "mean_lpd": {}, "reason": "; ".join(notes) or "no checkpoints",
            "missing": missing}


# ------------------------------------------------------------------ freeze-affine (R6)
def freeze_affine(checkpoint: Path, shared_init: Path, out: Path) -> dict:
    """fit_affine(train, shared-init configs) -> out (save_affine); report -> <out>.json."""
    from trimem.experiments.trace_joint_plan_search import get_space, observations
    from trimem.optimizer.joint_search_ledger import SearchLedger

    t0 = time.time()
    shared_init, out = Path(shared_init), Path(out)
    manifest = json.loads((shared_init / "manifest.json").read_text())
    space = get_space(manifest)
    with SearchLedger(shared_init / "ledger.sqlite", manifest) as led:
        init = [o.config for o in observations(led, space, manifest)]
    if not init:
        raise ValueError(f"shared init {shared_init} has no complete configs")
    sha = file_sha256(checkpoint)
    model, metrics = load_checkpoint(checkpoint, space)
    train, _ = corpus(space, metrics)
    affine = fit_affine(model, SeqGrammar(space), train, init)
    save_affine(out, affine, {"kind": FROZEN_KIND, "checkpoint_sha256": sha, "n_train": len(train),
                              "n_init": len(init), "shared_init": str(shared_init)})
    codec = LatentCodec(space, model, affine, checkpoint_sha256=sha)

    def roundtrip(c) -> bool:
        try:
            return codec.roundtrip(c)
        except Exception:  # noqa: BLE001 - a decode that raises is a failed round trip
            return False

    flags = [roundtrip(c) for c in init]
    rep = {"R6_init_roundtrip": int(sum(flags)), "n_init": len(init),
           "failed_keys": [space.key(c) for c, f in zip(init, flags) if not f],
           "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
           "affine": str(out), "affine_sha256": file_sha256(out), "shared_init": str(shared_init),
           "seconds": time.time() - t0}
    Path(f"{out}.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------ CLI
def _d_of(path) -> int:
    m = re.search(r"d(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"cannot read d from checkpoint name {path}")
    return int(m.group(1))


def _by_d(block: dict) -> dict[int, dict | None]:
    return {int(str(k).lstrip("d")): v for k, v in block.items()}


def _write(path: Path, obj) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--space-run", required=True, type=Path)
    r.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--probe", type=int, default=10_000)
    r.add_argument("--test", type=int, default=20_000)
    r.add_argument("--affine-cache", type=Path, default=None, help="default: the checkpoint's directory")
    c = sub.add_parser("choose")
    for name in ("--readiness", "--gates", "--out"):
        c.add_argument(name, required=True, type=Path)
    f = sub.add_parser("freeze-affine")
    for name in ("--checkpoint", "--shared-init", "--out"):
        f.add_argument(name, required=True, type=Path)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    if a.cmd == "run":
        from trimem.experiments.trace_joint_plan_search import get_space

        space = get_space(json.loads((a.space_run / "manifest.json").read_text()))
        report = {"space_run": str(a.space_run), "probe": a.probe, "test": a.test,
                  "loadavg_start": list(os.getloadavg()), "checkpoints": {}}
        for ck in a.checkpoints:
            if not ck.exists():
                report["checkpoints"][f"d{_d_of(ck)}"] = None
            else:
                res = readiness(ck, space, probe=a.probe, test=a.test, affine_cache=a.affine_cache)
                report["checkpoints"][f"d{res['d']}"] = {**res, "passes": passes_readiness(res)}
            _write(a.out, report)
            print(ck.name, json.dumps(report["checkpoints"][f"d{_d_of(ck)}"])[:600], flush=True)
    elif a.cmd == "choose":
        rd = _by_d(json.loads(a.readiness.read_text())["checkpoints"])
        gd = _by_d(json.loads(a.gates.read_text()).get("d", {}))
        out = choose_d(rd, gd)
        out["surrogate"] = VARIANTS[out["variant"]] if out["variant"] else None
        out["readiness"] = {f"d{d}": (None if v is None else {**{k: v.get(k) for k in (
            "R1_val_exact", "R2_family_min", "R3_probe_valid", "R4_collisions", "R5_test_exact")},
            "passes": passes_readiness(v)}) for d, v in rd.items()}
        out["gates"] = {f"d{d}": (g.get("pass") if isinstance(g, dict) else None) for d, g in gd.items()}
        _write(a.out, out)
        print(json.dumps({k: out[k] for k in ("d", "variant", "reason")}), flush=True)
    else:
        print(json.dumps(freeze_affine(a.checkpoint, a.shared_init, a.out)), flush=True)


if __name__ == "__main__":
    main()
