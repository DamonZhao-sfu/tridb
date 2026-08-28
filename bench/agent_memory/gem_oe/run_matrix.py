"""Run the (task x arm) matrix, one cell per process, across the available replicas.

    python3 -m bench.agent_memory.gem_oe.run_matrix \
        --arms none gem --endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
        --out bench/out/oe/phase2

Cells are independent, so they are pinned round-robin onto the serving replicas: two
GPUs means two cells in flight, not one cell split across two GPUs (the host has no
NVLink, so tensor-parallel would pay a PCIe all-reduce per layer for no benefit here).

Every cell writes its own receipt; this driver only schedules and collects. It makes NO
latency claim: cells share the replicas with each other, so per-request timings here
are progress instrumentation, not evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: The math tasks that ship a run-local evaluator. `math:signal_processing` is absent
#: on purpose: no session of it published one, so it cannot be scored here at all.
MATH_TASKS = [
    "math:circle_packing",
    "math:first_autocorr_ineq",
    "math:heilbronn_convex_13",
    "math:heilbronn_triangle",
    "math:second_autocorr_ineq",
    "math:third_autocorr_ineq",
    "math:uncertainty_ineq",
]

RUNNER = "bench.agent_memory.gem_oe.run_arm"


def cell_dir(out: Path, task: str, arm: str, rate: float | None = None,
             freq: float | None = None, seed: int | None = None) -> Path:
    suffix = "" if rate is None else f"__r{int(round(rate * 100)):03d}"
    if freq is not None:
        suffix += f"__p{int(round(freq * 100)):03d}"
    # The seed is part of the cell identity, not a detail: three repeats of one
    # configuration must not collide on one directory, and the 0.148 run-to-run
    # spread measured on three identical `nocontext` cells is exactly why repeats
    # exist. Omitted for a single-seed matrix so old paths keep resolving.
    if seed is not None:
        suffix += f"__s{seed}"
    return out / f"{task.replace(':', '_')}__{arm}{suffix}"


def run_cell(
    python: str,
    task: str,
    arm: str,
    endpoint: str,
    out: Path,
    extra: list[str],
    timeout: float | None = None,
    rate: float | None = None,
    freq: float | None = None,
    seed: int | None = None,
    skip_complete: bool = False,
) -> dict[str, Any]:
    target = cell_dir(out, task, arm, rate, freq, seed)
    if skip_complete:
        # Resume, so that raising --workers mid-experiment does not throw away the
        # cells already paid for. A cell counts as done only on its own receipt
        # saying `complete`; a partial cell is re-run from scratch, because
        # OpenEvolve's checkpoint is not something this harness restores.
        receipt = target / "run_receipt.json"
        if receipt.exists():
            try:
                if json.loads(receipt.read_text()).get("status") == "complete":
                    return {"task": task, "arm": arm, "dir": str(target),
                            "returncode": 0, "skipped": True, "seconds": 0.0}
            except (OSError, json.JSONDecodeError):
                pass
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "-m", RUNNER,
        "--task", task, "--arm", arm,
        "--out", str(target),
        "--llm-base", endpoint,
        *(["--injection-rate", str(rate)] if rate is not None else []),
        *(["--injection-frequency", str(freq)] if freq is not None else []),
        *(["--seed", str(seed)] if seed is not None else []),
        *extra,
    ]
    log = target / "cell.log"
    started = time.perf_counter()
    timed_out = False
    # A wall-clock cap on the cell, not politeness. OpenEvolve's own evaluator timeout
    # fires (`Evaluation timed out after 900s`) but the run does NOT recover: the
    # timed-out evaluation still holds its process-pool slot, and the cell then sits at
    # 0% CPU forever. Four cells wedged that way for 75 minutes while reporting
    # `status: running`, which is indistinguishable from slow progress unless something
    # above them is watching the clock.
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        # `start_new_session` puts the cell in its own process GROUP, which is the
        # only handle that reaches OpenEvolve's forked pool workers. Without it,
        # `subprocess.run(timeout=...)` SIGKILLs the direct child only and the
        # workers survive as orphans -- reparented to init, still holding an LLM
        # connection. Measured: one cell reported "exceeded 3600.0s and was killed"
        # and then logged a successful POST to the model 93 seconds later. A forked
        # worker inherits the parent's /proc/pid/cmdline verbatim, so the orphan is
        # indistinguishable from the cell it came from except by its PPID of 1.
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=None,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            # Reap, so the entry does not linger as a zombie holding the log fd.
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            handle.write(f"\n\n[run_matrix] cell exceeded {timeout}s; process group killed\n")
    return {
        "task": task,
        "arm": arm,
        "rate": rate,
        "freq": freq,
        "endpoint": endpoint,
        "returncode": returncode,
        "timed_out": timed_out,
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(target),
        "log": str(log),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=MATH_TASKS)
    ap.add_argument("--arms", nargs="+", default=["none", "gem"])
    ap.add_argument(
        "--skip-complete", action="store_true",
        help="Skip cells whose receipt already says complete. Use to resume a matrix "
             "at a different --workers without re-running finished cells.",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Repeat every cell once per seed. Omit for a single run at the frozen "
             "seed. Repeats are the only way to tell an arm difference from the "
             "0.148 run-to-run spread already measured on this workload.",
    )
    ap.add_argument(
        "--injection-frequencies", nargs="*", type=float, default=None,
        help="arXiv:2606.29823's p, swept per memory arm: the per-step PROBABILITY "
             "that memory fires at all. Orthogonal to --injection-rates, which sets "
             "how many programs one firing carries. The paper's Figure 4 uses "
             "{0.1, 0.5}; this experiment previously ran only at p=1.0.",
    )
    ap.add_argument(
        "--injection-rates", nargs="*", type=float, default=None,
        help="fractions of the inspiration slots to fill from memory, e.g. "
             "0.1 0.5 1.0. Each memory arm is expanded once per rate; arm `none` is "
             "not, because it IS the 0%% point. Omit to run each memory arm once at "
             "'fill every slot'.",
    )
    ap.add_argument(
        "--endpoints", nargs="+", default=["http://127.0.0.1:8001/v1"],
        help="one per serving replica; cells are pinned round-robin",
    )
    ap.add_argument(
        "--python", default=".venv-e0/bin/python",
        help="interpreter that has openevolve installed",
    )
    ap.add_argument(
        "--workers", type=int, default=None,
        help="concurrent cells; defaults to one per endpoint. A single OpenEvolve run "
             "is sequential -- it waits for one generation before starting the next -- "
             "so one cell per replica leaves the GPU idle between requests. The "
             "replica serves `--max-num-seqs` concurrent sequences, so several cells "
             "can share it; raising this trades per-cell latency for utilisation.",
    )
    ap.add_argument(
        "--cell-timeout", type=float, default=5400,
        help="wall-clock cap per cell, in seconds. A completed cell took 26-36 min; "
             "5400 leaves headroom for a slow evaluator while still killing a wedge. "
             "The recorded corpus evaluations peak at ~600s, so a cell stuck far "
             "beyond that is hung, not working.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    # arm `none` has no memory to meter, so it is the 0% point by construction and is
    # never expanded over the rate axis -- doing so would run identical cells under
    # different names.
    freqs = args.injection_frequencies or [None]
    seeds: list[int | None] = list(args.seeds) if args.seeds else [None]
    cells: list[tuple[str, str, float | None, float | None, int | None]] = []
    for task in args.tasks:
        for arm in args.arms:
            # `nocontext` and `none` inject nothing, so neither axis applies -- they
            # are the p=0 point and expanding them would run one configuration under
            # several names.
            if arm in ("none", "nocontext"):
                # Repeated too: the baseline needs its own variance estimate, or
                # there is nothing to compare an arm's spread against.
                cells.extend((task, arm, None, None, sd) for sd in seeds)
                continue
            rates = args.injection_rates or [None]
            cells.extend(
                (task, arm, r, f, sd) for r in rates for f in freqs for sd in seeds
            )
    plan = [
        (task, arm, rate, freq, sd, args.endpoints[i % len(args.endpoints)])
        for i, (task, arm, rate, freq, sd) in enumerate(cells)
    ]
    print(f"{len(plan)} cells over {len(args.endpoints)} replica(s)")
    for task, arm, rate, freq, sd, endpoint in plan:
        label = arm
        if rate is not None:
            label += f"@{rate:.0%}"
        if freq is not None:
            label += f"/p{freq:g}"
        if sd is not None:
            label += f"#s{sd}"
        print(f"  {task:<28} {label:<20} -> {endpoint}")
    if args.dry_run:
        extra = [*extra, "--dry-run"]

    started = time.perf_counter()
    workers = args.workers or len(args.endpoints)
    print(f"{workers} concurrent cell(s)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda item: run_cell(
                args.python, item[0], item[1], item[5], args.out, extra,
                timeout=None if args.dry_run else args.cell_timeout,
                rate=item[2], freq=item[3], seed=item[4],
                skip_complete=args.skip_complete,
            ),
            plan,
        ))

    ok = [r for r in results if r["returncode"] == 0]
    summary = {
        "cells": results,
        "cells_ok": len(ok),
        "cells_failed": len(results) - len(ok),
        "cells_wedged": sum(1 for r in results if r.get("timed_out")),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "arms": args.arms,
        "injection_rates": args.injection_rates,
        "injection_frequencies": args.injection_frequencies,
        "seeds": args.seeds,
        "tasks": args.tasks,
        "endpoints": args.endpoints,
        "workers": workers,
        "latency_claim": "NONE. Cells share the replicas; timings are progress only.",
    }
    (args.out / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    for row in results:
        if row["returncode"] == 0:
            flag = "ok  "
        elif row.get("timed_out"):
            flag = "WEDGED"
        else:
            flag = f"FAIL({row['returncode']})"
        label = row["arm"]
        if row.get("rate") is not None:
            label += f"@{row['rate']:.0%}"
        if row.get("freq") is not None:
            label += f"/p{row['freq']:g}"
        print(f"  {flag} {row['task']:<28} {label:<20} {row['seconds']:>8.1f}s")
    print(f"\n{len(ok)}/{len(results)} cells ok -> {args.out / 'matrix_summary.json'}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
