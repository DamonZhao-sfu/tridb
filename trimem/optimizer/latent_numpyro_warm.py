"""Warm-start state for latent_numpyro_saas: persistence, provenance, aging (fix round 1).

A warm state is the final NUTS state of the previous round's vmapped SAAS fit, per objective:
constrained last sample, adapted step size, dense inverse mass matrix, plus bookkeeping
(objective name, n, generation, cold_fit_n, d, site order). It is stored as JSON next to the
round's model artifacts with provenance meta. ``load_states`` returns it only when the provenance
matches the current fit exactly (checkpoint sha256, affine sha256, objective.json sha256, run id,
panel identity, objective names) and the stored n is <= the current n.

Aging rule (mass-matrix aging, fix round 1): a warm fit keeps the previous inverse mass matrix
fixed. To stop it from going stale, ``latent_numpyro_saas.warm_state_valid`` refuses a state (and
the fit falls back to the cold 512-warmup fit, which re-adapts the mass matrix) when
  * generation >= MAX_WARM_GENERATIONS (3): at most 3 consecutive warm fits after a cold fit, or
  * n > MAX_N_GROWTH (1.5) x cold_fit_n: the data grew by more than 50 % since the last cold fit.
generation is 0 for a state produced by a cold fit and previous + 1 for a warm fit; cold_fit_n is
the n of the last cold fit in the chain.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

OBJECTIVE_NAMES = ("quality", "tokens", "latency")
WARM_STATE_VERSION = "trimem-numpyro-saas-warm-v2"
PROVENANCE_KEYS = ("checkpoint_sha256", "affine_sha256", "objective_sha256", "run_id", "panel_identity")
MAX_WARM_GENERATIONS = 3
MAX_N_GROWTH = 1.5


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def objective_sha256(objective) -> str:
    """sha256 of the frozen objective normalization (objective.json lo / span)."""
    return _sha({"lo": [float(v) for v in objective.lo], "span": [float(v) for v in objective.span]})


def affine_sha256(affine) -> str:
    """sha256 of the codec's affine map (lo / hi), the part of the codec identity besides the checkpoint."""
    return _sha({"lo": np.ravel(np.asarray(affine.lo, dtype=float)).tolist(),
                 "hi": np.ravel(np.asarray(affine.hi, dtype=float)).tolist()})


def warm_meta(codec, objective, observations, *, run_id: str) -> dict:
    """Provenance meta for save_warm_state / the `expect` of load_warm_state (add "n" for load)."""
    panels = {o.panel_identity for o in observations}
    if len(panels) != 1:
        raise ValueError("warm state observations must share one TRAIN panel")
    return {"checkpoint_sha256": codec.checkpoint_sha256, "affine_sha256": affine_sha256(codec.affine),
            "objective_sha256": objective_sha256(objective), "run_id": str(run_id),
            "panel_identity": next(iter(panels))}


def _check_meta(meta: dict, extra=()) -> None:
    if not isinstance(meta, dict):
        raise TypeError("warm-state meta must be a dict")
    for key in (*PROVENANCE_KEYS, *extra):
        if key not in meta or meta[key] in (None, ""):
            raise ValueError(f"warm-state meta is missing {key!r}")


def save_states(path: str | os.PathLike, states: dict, *, meta: dict) -> Path:
    """Write {objective: state} with provenance meta (JSON, temp file + atomic replace)."""
    _check_meta(meta)
    if set(states) != set(OBJECTIVE_NAMES):
        raise ValueError(f"warm state needs objectives {OBJECTIVE_NAMES}, got {sorted(states)}")
    path = Path(path)
    doc = {"version": WARM_STATE_VERSION, "meta": {k: meta[k] for k in PROVENANCE_KEYS}, "objectives": states}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, path)
    return path


def load_states(path: str | os.PathLike, *, expect: dict) -> dict | None:
    """{objective: state}, or None (-> cold fit) unless the file exists, parses, has this version,
    every provenance key equals expect, the objectives are exactly OBJECTIVE_NAMES with matching
    names inside, and every stored n <= expect["n"]."""
    _check_meta(expect, extra=("n",))
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("version") != WARM_STATE_VERSION:
        return None
    meta, states = doc.get("meta") or {}, doc.get("objectives")
    if any(meta.get(k) != expect[k] for k in PROVENANCE_KEYS):
        return None
    if not isinstance(states, dict) or set(states) != set(OBJECTIVE_NAMES):
        return None
    try:
        for name, st in states.items():
            if st.get("objective") != name or int(st["n"]) > int(expect["n"]):
                return None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return states
