"""Wait for background GPU jobs, then validate and run the authorized QA matrix."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from trimem.exg.prepare import REPO, bootstrap, prepare
from trimem.exg.runtime import EMBED, MODEL, restrict_resources, save

BLOCKING_MODULES = {"bench.agent_memory.gem_oe.run_arm", "trimem.experiments.oe_matrix"}


def snapshot():
    gpu_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    gpus = [
        dict(
            zip(
                ("index", "utilization", "memory_used_mb", "memory_total_mb"),
                (int(v.strip()) for v in line.split(",")),
                strict=True,
            )
        )
        for line in gpu_text.strip().splitlines()
    ]
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    gpu_pids = {int(v.strip()) for v in compute.splitlines() if v.strip().isdigit()}
    blockers, allowed = [], set()
    for p in psutil.process_iter(["pid", "cmdline", "status"]):
        try:
            cmd = p.info["cmdline"] or []
            if p.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            module = cmd[cmd.index("-m") + 1] if "-m" in cmd else None
            if module in BLOCKING_MODULES:
                blockers.append(dict(pid=p.pid, module=module))
            if "serve" in cmd and any(m in cmd for m in (MODEL, EMBED)):
                allowed.add(p.pid)
                allowed.update(q.pid for q in p.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    services = {}
    for port, model in ((8000, MODEL), (8011, EMBED)):
        try:
            identity = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            identity.raise_for_status()
            entries = identity.json()["data"]
            assert any(e["id"] == model for e in entries), "Unexpected model identity"
            response = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
            response.raise_for_status()
            values = dict(running=0.0, waiting=0.0, completed=0.0)
            found = set()
            for line in response.text.splitlines():
                for metric, key in (
                    ("vllm:num_requests_running{", "running"),
                    ("vllm:num_requests_waiting{", "waiting"),
                    ("vllm:request_success_total{", "completed"),
                ):
                    if line.startswith(metric):
                        values[key] += float(line.rsplit(" ", 1)[1])
                        found.add(key)
            assert {"running", "waiting"} <= found, "Missing queue metrics"
            services[str(port)] = dict(status="ready", **values)
        except httpx.ConnectError:
            services[str(port)] = dict(status="absent")
        except Exception as exc:
            services[str(port)] = dict(status="unknown", error=repr(exc))
    return dict(
        epoch=time.time(),
        gpus=gpus,
        blockers=blockers,
        unknown_gpu_pids=sorted(gpu_pids - allowed),
        services=services,
        cpu_percent=psutil.cpu_percent(interval=0.25),
        available_gb=psutil.virtual_memory().available / 1024**3,
    )


def idle(sample):
    return (
        bool(sample["gpus"])
        and not sample["blockers"]
        and not sample["unknown_gpu_pids"]
        and all(g["utilization"] <= 10 for g in sample["gpus"])
        and sample["cpu_percent"] < 65
        and sample["available_gb"] > 12
        and all(
            s["status"] == "absent"
            or (s["status"] == "ready" and s["running"] == s["waiting"] == 0)
            for s in sample["services"].values()
        )
    )


def wait_idle(root, stable_seconds=120, poll_seconds=30):
    save(
        root / "pipeline_status.json", dict(state="waiting_for_idle", epoch=time.time())
    )
    since = None
    previous_counts = None
    while True:
        if (root / "PAUSED.json").exists():
            raise RuntimeError("User paused the queued run")
        try:
            sample = snapshot()
            counts = {p: s.get("completed") for p, s in sample["services"].items()}
            steady = idle(sample) and (
                previous_counts is None or counts == previous_counts
            )
            previous_counts = counts
            since = (
                (since if since is not None else time.monotonic()) if steady else None
            )
            elapsed = time.monotonic() - since if since is not None else 0
            save(
                root / "waiting_status.json",
                dict(
                    state="waiting_for_idle",
                    idle_seconds=elapsed,
                    required_idle_seconds=stable_seconds,
                    **sample,
                ),
            )
            if since is not None and elapsed >= stable_seconds:
                return sample
        except (subprocess.SubprocessError, OSError) as exc:
            since = None
            save(
                root / "waiting_status.json",
                dict(state="observation_error", error=repr(exc), epoch=time.time()),
            )
        time.sleep(poll_seconds)


def ensure_services(root, sample):
    binary = Path("/localhome/hza214/miniconda3/envs/vllm17/bin/vllm")
    started = []
    for port, model in ((8000, MODEL), (8011, EMBED)):
        if sample["services"][str(port)]["status"] != "absent":
            continue
        args = [
            str(binary),
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            model,
        ]
        if port == 8000:
            args += [
                "--data-parallel-size",
                "2",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.84",
                "--max-model-len",
                "32768",
                "--max-num-seqs",
                "32",
                "--enable-prefix-caching",
                "--reasoning-parser",
                "qwen3",
            ]
        else:
            args += [
                "--runner",
                "pooling",
                "--gpu-memory-utilization",
                "0.08",
                "--max-model-len",
                "8192",
            ]
        with (root / f"model_{port}.log").open("a") as log:
            p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        started.append(dict(pid=p.pid, command=args))
        save(root / "owned_model_processes.json", started)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if p.poll() is not None:
                raise RuntimeError(f"Model service {port} exited {p.returncode}")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                if r.status_code == 200 and any(
                    e["id"] == model for e in r.json()["data"]
                ):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(10)
        else:
            raise RuntimeError(
                f"Model service {port} readiness timeout; process retained for inspection"
            )


def step(root, label, arguments):
    save(
        root / "pipeline_status.json",
        dict(state="running", stage=label, epoch=time.time()),
    )
    with (root / f"{label}.log").open("a") as log:
        p = subprocess.Popen(
            [str(REPO / ".venv/bin/python"), "-u", "-m", *arguments],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        save(
            root / "pipeline_status.json",
            dict(state="running", stage=label, pid=p.pid, epoch=time.time()),
        )
        code = p.wait()
    if code:
        raise RuntimeError(
            f"{label} failed with exit code {code}; inspect {root / (label + '.log')}"
        )


def main():
    restrict_resources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=int, default=120)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "waiter.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(
        root / "waiter_launch.json",
        dict(
            pid=os.getpid(),
            command=sys.argv,
            epoch=time.time(),
            automatic_start_authorized=True,
        ),
    )
    try:
        sample = wait_idle(root, args.idle_seconds)
        save(root / "idle_gate_passed.json", sample)
        ensure_services(root, sample)
        ports = [55443, 55444, 55445, 55446]
        for port in ports:
            bootstrap(port)
        step(
            root,
            "native_selfcheck",
            ["trimem.exg.selfcheck", "--out", str(root / "queued_native_check")],
        )
        source = REPO / "results/trimem_exg_qwen38_v0.1.0/pilot_20261011_01"
        for name, validation in (("validation", True), ("main", False)):
            target = root / name
            prepare(target, source, validation)
            save(target / "instances.json", dict(ports=ports))
            # Recheck between validation and the long formal matrix for new outside jobs.
            if not validation:
                wait_idle(root, args.idle_seconds)
            step(
                root,
                f"{name}_scheduler",
                ["trimem.exg.schedule", "--root", str(target), "--slots", "4"],
            )
            if not (target / "all_cells_complete.json").exists():
                raise RuntimeError(f"Missing completion evidence for {target}")
            step(
                root,
                f"{name}_budget_audit",
                ["trimem.exg.budget", "--root", str(target)],
            )
        save(
            root / "pipeline_status.json",
            dict(state="experiments_complete", report_pending=True, epoch=time.time()),
        )
    except Exception as exc:
        save(
            root / "pipeline_status.json",
            dict(state="stopped", error=repr(exc), epoch=time.time()),
        )
        raise


if __name__ == "__main__":
    main()
