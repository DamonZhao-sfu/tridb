"""Readiness gates R1-R6 for a TRACE v0.7.0 sequence VAE (plan 2026-11-05 section 8).

``tools.latent_readiness.main`` (read-only import) with two runtime changes:

* the space is the v0.7.0 run's space (``v070_patches`` of the ``--space-run`` / ``--shared-init`` manifest):
  ``JointSpaceV070``, and for ``freeze-affine`` the init observations are each configuration's own-sample means;
* R2 family = ``filter_order|graph_impl|rank_impl`` (rank is optional in v0.7.0, so ``rank:none`` is a family).

Gates: R1 >= 0.95, R2 >= 0.90, R3 = 1, R4 = 0 (``latent_readiness.passes_readiness``); R6 on the 10 init
configurations at freeze-affine (NoMemory is not a configuration and is never encoded).

  python -m tools.latent_readiness_v070 run --space-run R0 --checkpoints CK... --out readiness.json
  python -m tools.latent_readiness_v070 freeze-affine --checkpoint CK --shared-init R0 --out AFFINE
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from tools import latent_readiness as lr

FAMILY = ("filter_order", "graph_impl", "rank_impl")


def family_rates_v070(configs, hits):
    tot: dict[str, list[int]] = {}
    for c, h in zip(configs, hits):
        t = tot.setdefault("|".join(str(c[f]) for f in FAMILY), [0, 0])
        t[0] += bool(h)
        t[1] += 1
    per = {k: v[0] / v[1] for k, v in tot.items()}
    return (min(per.values()) if per else float("nan")), per


def _run_manifest(argv) -> dict | None:
    for flag in ("--space-run", "--shared-init"):
        if flag in argv:
            return json.loads((Path(argv[argv.index(flag) + 1]) / "manifest.json").read_text())
    return None


@contextlib.contextmanager
def patches(manifest):
    from trimem.experiments.trace_v070_runtime import (
        is_v070,
        space_patches,
        v070_patches,
    )

    saved = lr.family_rates
    lr.family_rates = family_rates_v070
    try:
        if manifest is not None and not is_v070(manifest):
            raise SystemExit("latent_readiness_v070 needs a v0.7.0 run (use tools.latent_readiness otherwise)")
        with (v070_patches(manifest) if manifest is not None else space_patches()):
            yield
    finally:
        lr.family_rates = saved


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    with patches(_run_manifest(argv)):
        lr.main(argv)


if __name__ == "__main__":
    main()
