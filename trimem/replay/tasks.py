"""Replay tasks: evaluator, seed program (majority root of the corpus runs) and specification."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import psycopg

RAW = Path("data/evotrace/raw/shinkaevolve")
EVAL_PYTHON = str(Path(".venv-e0/bin/python").absolute())
# Evaluations are CPU-bound numerical programs: run at most EVAL_SLOTS at once, each with a fixed
# BLAS/OpenMP thread budget, so concurrent cells do not starve each other into timeouts.
EVAL_THREADS = int(os.environ.get("TRIMEM_EVAL_THREADS", "4"))
CPU_TARGET = float(os.environ.get("TRIMEM_CPU_TARGET", "0.8"))
MIN_RUNNING = int(os.environ.get("TRIMEM_EVAL_MIN_RUNNING", "2"))


class CpuGovernor:
    """Admission control for evaluations: start a new one only while whole-machine CPU
    utilisation (from /proc/stat, sampled every 2 s) is below CPU_TARGET, except that MIN_RUNNING
    evaluations may always run so the experiment keeps moving on a busy host."""

    def __init__(self) -> None:
        self.util = 0.0
        self.running = 0
        self.cv = threading.Condition()
        threading.Thread(target=self._sample, daemon=True).start()

    @staticmethod
    def _read() -> tuple[int, int]:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + v[4]
        return sum(v), idle

    def _sample(self) -> None:
        tot0, idle0 = self._read()
        while True:
            time.sleep(2.0)
            tot1, idle1 = self._read()
            if tot1 > tot0:
                self.util = 1.0 - (idle1 - idle0) / (tot1 - tot0)
            tot0, idle0 = tot1, idle1
            with self.cv:
                self.cv.notify_all()

    def __enter__(self) -> None:
        with self.cv:
            while self.running >= MIN_RUNNING and self.util >= CPU_TARGET:
                self.cv.wait(timeout=5)
            self.running += 1

    def __exit__(self, *exc) -> None:
        with self.cv:
            self.running -= 1
            self.cv.notify_all()


GOVERNOR = CpuGovernor()
EVAL_SLOTS = threading.BoundedSemaphore(
    int(os.environ.get("TRIMEM_EVAL_SLOTS", "0"))
    or max(1, (os.cpu_count() or 8) // EVAL_THREADS)
)


@dataclass(frozen=True)
class Task:
    uid: str
    key: str
    evaluator: Path
    spec: str
    seed_code: str
    seed_fitness: float


def resolve(conn: psycopg.Connection, uid: str) -> Task:
    key = uid.split(":", 1)[1]
    if uid.startswith("ale:"):  # self-contained ALE-Bench evaluator (tools/evotrace/ale_evaluator.py):
        # the first 50 public cases, as the corpus runs and arXiv 2605.20086 (not the local default of 10)
        evaluator = (Path("bench/out/ale_evaluators") / f"{key}_n50.py").resolve()
    else:
        evaluator = next(RAW.glob(f"{key}_dpsk*/evaluate.py")).resolve()
    (spec,) = conn.execute(
        "SELECT specification FROM task WHERE task = %s", (uid,)
    ).fetchone()
    roots = conn.execute(
        "SELECT code, fitness FROM attempt WHERE task = %s AND parent_id IS NULL AND cell IS NULL "
        "AND code IS NOT NULL AND is_valid",
        (uid,),
    ).fetchall()
    code, _ = Counter((c, f) for c, f in roots).most_common(1)[0][0]
    fit = Counter((c, f) for c, f in roots).most_common(1)[0][0][1]
    return Task(uid, key, evaluator, spec or "", code, float(fit or 0.0))


def evaluate(task: Task, code: str, timeout: float = 600.0) -> dict:
    """Score a program with the corpus's own evaluator in a separate process."""
    with tempfile.TemporaryDirectory(prefix="trimem-eval-") as d:
        prog = Path(d) / "program.py"
        prog.write_text(code)
        driver = Path(d) / "run.py"
        driver.write_text(
            "import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('ev', {str(task.evaluator)!r})\n"
            "ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)\n"
            f"r = ev.evaluate({str(prog)!r})\n"
            "print('__RESULT__' + json.dumps(r, default=str))\n"
        )
        env = os.environ | {
            v: str(EVAL_THREADS)
            for v in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        with EVAL_SLOTS, GOVERNOR:
            started = time.perf_counter()
            try:
                out = subprocess.run(
                    [EVAL_PYTHON, str(driver)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=d,
                    env=env,
                )
                line = next(
                    (
                        ln
                        for ln in out.stdout.splitlines()
                        if ln.startswith("__RESULT__")
                    ),
                    None,
                )
                res = (
                    json.loads(line[len("__RESULT__") :])
                    if line
                    else {"combined_score": 0.0, "error": out.stderr[-800:]}
                )
            except subprocess.TimeoutExpired:
                res = {"combined_score": 0.0, "error": f"timeout after {timeout}s"}
        res["eval_seconds"] = time.perf_counter() - started
        try:
            res["combined_score"] = float(res.get("combined_score") or 0.0)
        except (TypeError, ValueError):
            res["combined_score"] = 0.0
        return res
