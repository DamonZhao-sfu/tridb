"""Run Graphiti conformance through the formal per-group FIFO adapter."""

from __future__ import annotations

from experiments.graphiti_track_c import conformance as base

from .adapter import FormalGraphitiTrackCAdapter


def main() -> int:
    # The frozen conformance implementation resolves this module global in its
    # adapter factory.  Substitute only the adapter class; dataset, model,
    # persistence, isolation, and receipt gates remain byte-identical.
    base.GraphitiTrackCAdapter = FormalGraphitiTrackCAdapter
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
