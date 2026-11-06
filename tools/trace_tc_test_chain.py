"""After trace-tc discovery: direct TEST of every pruning survivor, then resume the CA-RAG -> external-baselines chain.

User 2026-10-27: trace-tc TEST runs before the CA-RAG rerun; TEST every survivor of all 48 proposals (200 ALFWorld TEST
tasks each, same protocol and runtime as the E2 MO-TPE / GP / LHS direct-TEST points). Steps:
 1. wait for results/trace_tc_alfworld_20261027_tcprune/status.json == complete (stop on stopped_for_diagnosis);
 2. freeze a TEST workspace (copy of trimem/tests) and a source shim (the E2 ALFWorld prepare manifest with this run);
 3. alfworld_holdout_eval prepare --direct --limit 48 --label TRACE-TC --no-trace, then direct-test (takes the lease);
 4. relaunch scripts/carag_rerun_then_extbase.sh (EXTBASE_EMBED_BASE=8023) whatever the TEST outcome.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/trace_tc_alfworld_20261027_tcprune"
RUN = CAMPAIGN / "campaign/runs/alfworld_pool-strat200-v7-tc27tq__mem-sf1"
E2 = ROOT / "results/exp2_alfworld_prepare_20261024_gplhs2"
TEST = CAMPAIGN / "direct_test"
SHIM = CAMPAIGN / "test_source"
WS = CAMPAIGN / "test_workspace"
PY = ROOT / ".venv-search-baselines/bin/python"
LIMIT = 48
STATUS = CAMPAIGN / "test_chain_status.json"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def status(**kw):
    write(STATUS, {**kw, "at": time.time(), "pid": os.getpid()})


def wait_discovery():
    while True:
        s = json.loads((CAMPAIGN / "status.json").read_text())
        if s["status"] == "complete":
            return
        if s["status"] == "stopped_for_diagnosis":
            raise SystemExit(f"discovery stopped: {s.get('error')}")
        status(phase="waiting_for_discovery", discovery=s["status"])
        time.sleep(120)


def workspace():
    if not WS.exists():
        WS.mkdir(parents=True)
        for name in ("trimem", "tests"):
            shutil.copytree(
                ROOT / name, WS / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        for name in (
            "data",
            "pyproject.toml",
            "results",
            "vendor",
            "scripts",
            "paper",
            ".venv",
            ".venv-agentenv",
            ".venv-agentgym",
            ".venv-search-baselines",
        ):
            (WS / name).symlink_to(ROOT / name)


def shim():
    """The E2 ALFWorld prepare manifest (same TEST ids / env / SemFilter regime) carrying this discovery run."""
    if (SHIM / "manifest.json").exists():
        return
    e2 = json.loads((E2 / "manifest.json").read_text())
    ours = json.loads((CAMPAIGN / "prepared/manifest.json").read_text())
    run = dict(ours["runs"][0])
    run["experiment"] = json.loads((RUN / "state.json").read_text())["experiment"]
    m = {
        **e2,
        "runs": [run],
        "campaign_dir": ours["campaign_dir"],
        "reference_baseline": ours["reference_baseline"],
        "staging_mode": ours["staging_mode"],
        "shim_of": {
            "e2_manifest": str(E2 / "manifest.json"),
            "trace_tc_manifest": str(CAMPAIGN / "prepared/manifest.json"),
        },
    }
    write(SHIM / "manifest.json", m)
    import hashlib

    conf = json.loads((E2 / "confirmation.json").read_text())
    conf.update(
        manifest_sha256=hashlib.sha256(
            (SHIM / "manifest.json").read_bytes()
        ).hexdigest(),
        confirmed_by="user 2026-10-27 (trace-tc TEST before CA-RAG; all 48 proposals' survivors)",
        confirmed_by_user=True,
    )
    write(SHIM / "confirmation.json", conf)


def run(*args):
    env = dict(os.environ, PYTHONPATH=str(WS))
    subprocess.run(
        [str(PY), "-u", "-m", "trimem.experiments.alfworld_holdout_eval", *args],
        cwd=WS,
        env=env,
        check=True,
    )


def resume_chain(outcome):
    env = dict(os.environ, EXTBASE_EMBED_BASE="http://127.0.0.1:8023/v1")
    subprocess.run(
        [
            str(ROOT / "scripts/in_eval_slice.sh"),
            "carag-then-extbase",
            str(ROOT / "results/external_baselines_20261025/queue/carag_rerun.log"),
            str(ROOT / "scripts/carag_rerun_then_extbase.sh"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    status(phase="chain_resumed", test=outcome)


def main():
    outcome = "not started"
    try:
        wait_discovery()
        write(
            RUN / "supervisor.json",
            {"status": "complete", "source": str(CAMPAIGN / "status.json")},
        )
        workspace()
        shim()
        if not (TEST / "manifest.json").exists():
            status(phase="prepare")
            run(
                "prepare",
                "--out",
                str(TEST),
                "--source",
                str(SHIM),
                "--limit",
                str(LIMIT),
                "--run-id",
                "trace-tc",
                "--label",
                "TRACE-TC",
                "--no-trace",
                "--direct",
                "--authorization",
                "User 2026-10-27: after trace-tc discovery, TEST every pruning survivor of all 48 proposals "
                "(200 ALFWorld TEST tasks each) before the CA-RAG rerun; compare with TRACE on Choose curve points.",
            )
        status(phase="direct_test")
        run("direct-test", "--out", str(TEST))
        outcome = "complete"
    except BaseException as exc:  # noqa: BLE001 - record and still release the queue to the waiting chain
        outcome = f"failed: {exc!r}"
        status(phase="failed", error=outcome)
        if isinstance(exc, SystemExit) and "discovery stopped" in str(exc):
            return  # leave everything paused for diagnosis
    resume_chain(outcome)


if __name__ == "__main__":
    sys.exit(main())
