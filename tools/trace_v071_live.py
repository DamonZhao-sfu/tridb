"""Launcher of a TRACE v0.7.1 run: ``tools.trace_v070_live`` (unchanged) inside ``v071_patches``.

Same flags and behaviour as the v0.7.0 launcher (``--preflight-only``, ``--baseline-only``; NoMemory stage, rounds,
TEST). On a v0.7.1 manifest the TRACE episodes (TRAIN and TEST) run through ``tools.trace_v071_episode_worker``; a v0.7.0
manifest passes through unpatched. Prints which SemFilter protocol the run uses.

    python -m tools.trace_v071_live --run-dir R [--preflight-only | --baseline-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools import trace_v070_live as tl
from trimem.experiments.trace_v071_runtime import is_v071, semfilter_mode, v071_patches


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--run-dir", required=True, type=Path)
    known, _ = ap.parse_known_args(argv)
    m = tl.load(known.run_dir)
    print(json.dumps({"trace_version": "0.7.1" if is_v071(m) else "0.7.0", "semfilter": semfilter_mode(m)}), flush=True)
    with v071_patches(m):
        tl.main(argv)


if __name__ == "__main__":
    main()
