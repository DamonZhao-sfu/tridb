"""``tools.trace_v071_live`` with the stall override (``trace_v071_stall_override``). Same flags and behaviour.

    python -m tools.trace_v071_stall_live --run-dir R [--preflight-only | --baseline-only]
"""

from __future__ import annotations

from tools import trace_v071_live
from trimem.experiments import trace_v071_search as ts
from trimem.experiments import trace_v071_stall_override as so


def main(argv=None) -> None:
    so.install()
    ts.latent_search_worker_v071 = so.latent_search_worker  # v071_patches imports it by name at call time
    trace_v071_live.main(argv)


if __name__ == "__main__":
    main()
