"""Auxiliary GP observations from earlier profiling runs (aux-obs-v1, user-approved 2026-10-20).

The earlier train-panel BO runs (``results/bo_agentenv_20261016/<run>``; experiments ``<ds>:train100:bo3obj-v1``,
``alfworld:train500:bo3obj-v2``, ``tau2:train178:bo3obj-v1``) measured configurations under other protocols: a
success-only memory store, raw cards (τ² now runs compact), panel-paired gains against that run's own NoMemory and,
for τ², an httpx-capped latency. Their configurations with edges ``similar+similar_up`` (the current fixed edges),
sim_gate none and raw memory are turned into GP-only observations: quality = the run's paired gain (± its SE),
tokens / latency = panel means with the ln-scale SD floored at ``AUX_COST_SD_MIN`` (0.5) because raw cards and the
httpx cap make the costs incomparable. They are mapped into the current space as outcome_policy = success_only
(semfilter_threshold 0.6 = the old fixed SemFilter threshold for P2 plans); off-grid levels (depth 1, fanout 2,
top_k 1) are kept (``features_unchecked``). A binary source dimension (``session_pool.use_aux``) lets the GP learn a
systematic offset. Only split = train rows are read (``render_profiling_points.train_panel``: ``ROW_SQL`` filters
split = 'train' and refuses any other row). Read-only on PG 55450.
"""

from __future__ import annotations

import math

from trimem.experiments import session_bo_agentenv as bo

# dataset -> (dataset, panel, experiment, results run dir) of ``render_profiling_points.train_panel``
SOURCES = {
    "tau2": [("tau2", "train178", "tau2:train178:bo3obj-v1", "tau2")],
    "alfworld": [
        ("alfworld", "train100", "alfworld:train100:bo3obj-v1", "alfworld"),
        ("alfworld", "train500", "alfworld:train500:bo3obj-v2", "alfworld_train500"),
    ],
    "sciworld": [("sciworld", "train100", "sciworld:train100:bo3obj-v1", "sciworld")],
    # WebShop (user 2026-10-23): the train200 BO run used the SAME store (trimem_webshop, product families
    # asin:<ASIN>) and the same per-query product isolation; success-only-v0 era like the ALF / Sci sources
    "webshop": [
        ("webshop", "train200", "webshop:train200:bo3obj-v2", "webshop_train200")
    ],
}
EDGES = "similar+similar_up"
OLD_SEMFILTER_THRESHOLD = (
    0.6  # the v1 FILTER_THRESHOLD (= plan_defaults semfilter_threshold)
)
NOISE_MODEL = (
    "quality: the point's own paired-gain SE^2; tokens / latency (ln scale): max(relative SE^2, "
    f"{bo.AUX_COST_SD_MIN}^2) -- SD >= {bo.AUX_COST_SD_MIN} because raw cards / the httpx latency cap make the "
    "costs incomparable"
)


def map_config(old: dict, plans: tuple[str, ...]) -> dict | None:
    """An old (session-space v1) configuration in the current space form, or None when it is not admissible
    (edges other than similar+similar_up, a gate, an unknown plan)."""
    if old.get("edges") != EDGES or old.get("sim_gate", "none") != "none":
        return None
    if old.get("memory_form", "raw") != "raw" or old["plan"] not in plans:
        return None
    cfg = {
        "plan": old["plan"],
        **{k: old[k] for k in ("seed_count", "depth", "fanout", "top_k")},
        "outcome_policy": "success_only",
        "sim_gate": "none",
    }
    if old["plan"] in ("P2", "P2+Rank", "P1+Rank"):
        cfg["semfilter_threshold"] = OLD_SEMFILTER_THRESHOLD
    return cfg


def point_of(p: dict, experiment: str, cfg: dict) -> dict:
    sd2 = bo.AUX_COST_SD_MIN**2
    return {
        "experiment": experiment,
        "label": p["label"],
        "config": cfg,
        "old_config": p["cfg"],
        "gain": p["gain"],
        "gain_se": p["gain_se"],
        "tokens": p["tok"],
        "latency_ms": p["lat"] * 1000.0,
        "noise": [
            max(p["gain_se"] ** 2, 1e-8),
            max(p["tok_se"] ** 2, sd2),
            max(p["lat_se"] ** 2, sd2),
        ],
    }


def from_panels(ds: str, panels: list[dict], space) -> dict:
    """Block body from ``train_panel`` outputs (pure; tests feed synthetic panels)."""
    pts, seen = [], []
    for panel in panels:
        if panel.get("split") != "train":
            raise ValueError(
                f"{panel.get('experiment')}: auxiliary source is not a train panel"
            )
        kept = 0
        for p in panel["points"]:
            cfg = map_config(p["cfg"], tuple(space.PLANS))
            if cfg is None:
                continue
            space.features_unchecked(cfg)  # encodable (raises otherwise)
            if not all(math.isfinite(x) for x in (p["gain"], p["tok"], p["lat"])):
                raise ValueError(
                    f"{panel['experiment']}: non-finite metric for {p['label']}"
                )
            pts.append(point_of(p, panel["experiment"], cfg))
            kept += 1
        seen.append(
            {
                "experiment": panel["experiment"],
                "panel": panel["panel"],
                "n_queries": panel["n_queries"],
                "configs_complete": panel["n_configs"],
                "configs_used": kept,
                "nomemory_quality": panel["nomem"]["q"],
            }
        )
    return {
        "dataset": ds,
        "sources": seen,
        "filter": f"split = train only; edges = {EDGES}; sim_gate none; memory_form raw",
        "mapping": "outcome_policy = success_only (success-only store); sim_gate none; P2 plans semfilter_threshold "
        f"{OLD_SEMFILTER_THRESHOLD}; off-grid depth / fanout / top_k kept (GP encoding only, never proposed)",
        "quality": "paired gain vs the source run's own NoMemory on its train panel",
        "noise_model": NOISE_MODEL,
        "gp": "MixedSingleTaskGP input + 1 categorical source dimension (0 current, 1 auxiliary); candidates at 0",
        "excluded_from": "Pareto / relaxed fronts, stall and forced front, Opt-OneConfig selection, router, qfirst "
        "champion and budget, crossover anchors, DLS incumbents, NEHVI baseline and reference point",
        "points": pts,
    }


def build(ds: str, space):
    """Callable for ``session_bo_agentenv.aux_block``: reads the source panels (PG 55450, read-only)."""

    def _build() -> dict:
        from trimem.experiments import render_profiling_points as rpp

        panels = []
        for src in SOURCES[ds]:
            p = rpp.train_panel(*src)
            if p is None:
                raise SystemExit(f"auxiliary source {src[2]} unavailable")
            panels.append(p)
        return from_panels(ds, panels, space)

    return _build
