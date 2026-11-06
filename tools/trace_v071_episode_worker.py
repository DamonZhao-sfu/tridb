"""TRACE v0.7.1 episode worker: SemFilter v2c patches, then the stock TRACE episode worker (unchanged on disk).

Run with ``.venv-agentgym`` exactly like ``tools.trace_joint_episode_worker`` (same arguments; the v0.7.1 launcher
rewrites only the module name of the worker command). Before the stock ``main`` runs, it

* requires ``TRIMEM_SEMFILTER_V2C=1`` (the v0.7.1 manifest environment) and refuses otherwise;
* requires an inference file with exactly ONE item (the episode = this process, the periodic-reuse scope);
* installs ``trimem.experiments.semfilter_v071`` (TOPN 16 via the stock knob, compact judge card, episode reuse).
The multiprocessing pool of the official evaluator forks, so the patches reach the episode process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_single_item(argv) -> str:
    if "--inference_file" not in argv:
        raise SystemExit("trace_v071_episode_worker: --inference_file is required")
    items = json.loads(Path(argv[argv.index("--inference_file") + 1]).read_text())
    if len(items) != 1:
        raise SystemExit(f"trace_v071_episode_worker: one item per episode process required, got {len(items)}")
    return items[0]["item_id"]


def main() -> None:
    from trimem.experiments import semfilter_v071 as v2c

    if not v2c.enabled():
        raise SystemExit(f"trace_v071_episode_worker needs {v2c.ENV_GATE}=1 (TRACE v0.7.1 manifest environment)")
    check_single_item(sys.argv)
    v2c.install()
    if "--output_dir" in sys.argv:  # attempt/ exists (written by the launcher before this process starts)
        v2c.LOG["path"] = str(Path(sys.argv[sys.argv.index("--output_dir") + 1]) / "attempt" / "semfilter_v2c.jsonl")
    import tools.trace_joint_episode_worker as stock

    stock.main()


if __name__ == "__main__":
    main()
