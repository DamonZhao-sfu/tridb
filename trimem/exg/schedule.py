"""Bounded process scheduler; one semantic worker per isolated PostgreSQL instance."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

from trimem.exg.run import cell_path
from trimem.exg.runtime import ARMS, read, restrict_resources, save


def jobs(root):
    protocol = read(root / "protocol.json")
    result = []
    for rep in range(protocol["replicates"]):
        for mode in ("collection", "online", "offline"):
            for arm in ["T"] if mode == "collection" else ARMS:
                for dataset in protocol["datasets"]:
                    result.append(dict(dataset=dataset, mode=mode, arm=arm, rep=rep))
    return result


def ready(root, job):
    if job["mode"] != "offline" or job["arm"] == "N":
        return True
    source = job | dict(mode="collection", arm="T")
    return (cell_path(root, source) / "complete.json").exists()


def main():
    restrict_resources()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()
    root = args.root.resolve()
    if (root / "PAUSED.json").exists():
        raise RuntimeError("Experiment paused by user; explicit resume is required")
    lock = (root / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(root / "protocol.json")
    ports = read(root / "instances.json")["ports"][: args.slots]
    queue = jobs(root)
    running = {}
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    save(
        root / "scheduler_launch.json",
        dict(pid=os.getpid(), ports=ports, epoch=time.time()),
    )
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    while True:
        for port, state in list(running.items()):
            code = state["process"].poll()
            if code is not None:
                state["log"].close()
                folder = cell_path(root, state["job"])
                save(
                    folder / "process_exit.json",
                    dict(exit_code=code, epoch=time.time()),
                )
                if code != 0 and not (folder / "failed.json").exists():
                    save(
                        folder / "failed.json",
                        dict(
                            state["job"],
                            error=f"Process exited {code}",
                            epoch=time.time(),
                        ),
                    )
                del running[port]
        cpu = psutil.cpu_percent(interval=0.25)
        pending = [
            j
            for j in queue
            if not (cell_path(root, j) / "complete.json").exists()
            and not (cell_path(root, j) / "failed.json").exists()
            and j not in [v["job"] for v in running.values()]
        ]
        # Existing old pilot occupies at most one model request; leave one slot for it.
        pilot_alive = any(
            "trimem.experiments.exg_pilot" in " ".join(p.info["cmdline"] or [])
            for p in psutil.process_iter(["cmdline"])
            if p.pid != os.getpid()
        )
        limit = min(len(ports), 3 if pilot_alive else 4)
        if cpu < 65 and psutil.virtual_memory().available > 12 * 1024**3:
            for port in ports:
                if port in running or len(running) >= limit:
                    continue

                def assigned_here(job):
                    assignment = cell_path(root, job) / "assignment.json"
                    return not assignment.exists() or read(assignment)["port"] == port

                job = next(
                    (j for j in pending if ready(root, j) and assigned_here(j)), None
                )
                if job is None:
                    continue
                pending.remove(job)
                save(cell_path(root, job) / "assignment.json", dict(port=port))
                name = f"{job['dataset']}_{job['mode']}_r{job['rep']}_{job['arm']}"
                log = (logs / f"{name}.log").open("a")
                command = [
                    str(Path.cwd() / ".venv/bin/python"),
                    "-u",
                    "-m",
                    "trimem.exg.run",
                    "--root",
                    str(root),
                    "--job",
                    json.dumps(job),
                    "--dsn",
                    f"host=127.0.0.1 port={port} user=hza214 dbname=postgres",
                ]
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                running[port] = dict(process=process, log=log, job=job)
                print(
                    json.dumps(dict(started=job, port=port, pid=process.pid)),
                    flush=True,
                )
        complete = [j for j in queue if (cell_path(root, j) / "complete.json").exists()]
        failed = [
            j
            for j in queue
            if (cell_path(root, j) / "failed.json").exists() and j not in complete
        ]
        status = dict(
            epoch=time.time(),
            completed_cells=len(complete),
            total_cells=len(queue),
            failed=failed,
            running=[
                dict(port=p, pid=v["process"].pid, job=v["job"])
                for p, v in running.items()
            ],
            cpu_percent=cpu,
            load=os.getloadavg(),
            available_gb=psutil.virtual_memory().available / 1024**3,
        )
        save(root / "scheduler_status.json", status)
        with (root / "resource_samples.jsonl").open("a") as f:
            f.write(json.dumps(status) + "\n")
        if len(complete) == len(queue):
            save(
                root / "all_cells_complete.json",
                status | dict(expected_episodes=protocol["expected_task_episodes"]),
            )
            break
        if not running and (failed or not any(ready(root, j) for j in pending)):
            save(root / "scheduler_stopped.json", status)
            raise RuntimeError(
                "Incomplete cells require inspection; no automatic failed-request retries"
            )
        time.sleep(10)


if __name__ == "__main__":
    main()
