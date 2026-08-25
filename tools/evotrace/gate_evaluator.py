"""Phase 0 gate 2 -- does each task's own evaluator still reproduce its recorded score?

An agent-outcome experiment compares fitness numbers our runs produce against fitness
numbers the corpus recorded. That comparison is meaningless unless the SAME evaluator,
on the SAME program, still returns the SAME score on this machine. This gate takes each
task's best recorded program, re-runs the run-local `evaluate.py`, and reports the
delta. A task whose score does not reproduce is out of the experiment: no outcome
measured against a contract that does not hold is worth anything.

    python3 -m tools.evotrace.gate_evaluator --domain math

The evaluators ship with a `sys.path.insert('/home/user/anon/skydiscover/...')` left
over from anonymisation. For the math tasks that line is vestigial -- they import only
numpy/sympy/stdlib -- so they run unmodified here. The ALE evaluators genuinely need
`ale_bench` plus per-problem test data and are expected to fail this gate until that
dependency is stood up; that is a real result, not a bug, so they are reported rather
than skipped silently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import psycopg

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
DEFAULT_RAW = Path("data/evotrace/raw")

# The evaluator is a module exposing `evaluate(program_path) -> dict[str, float]`.
# Run it in a subprocess: these are third-party scripts that install signal handlers,
# spawn children and call sys.exit, none of which belong in the harness process.
_DRIVER = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("task_evaluator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("@@RESULT@@" + json.dumps(module.evaluate(sys.argv[2])))
"""


def _evaluator_for(raw: Path, run_rel: str) -> Path | None:
    # Absolute: the evaluator subprocess runs with cwd set to a scratch directory so
    # that whatever files it writes land there, which makes any relative path here
    # resolve against the wrong root.
    path = (raw / run_rel / "evaluate.py").resolve()
    return path if path.is_file() else None


def _best_program(conn: psycopg.Connection, task_uid: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT n.node_uid, n.fitness, n.metrics, a.payload, s.run_rel"
        "  FROM gem_eg_node n"
        "  JOIN gem_eg_artifact a ON a.artifact_uid = n.artifact_uid"
        "  JOIN gem_eg_session s ON s.session_uid = n.session_uid"
        " WHERE n.task_uid = %s AND n.is_valid AND n.fitness IS NOT NULL"
        "   AND a.payload IS NOT NULL"
        " ORDER BY n.fitness DESC, n.node_uid"
        " LIMIT 1",
        (task_uid,),
    ).fetchone()
    if row is None:
        return None
    return {
        "node_uid": row[0],
        "fitness": row[1],
        "metrics": row[2],
        "payload": row[3],
        "run_rel": row[4],
    }


def _evaluator_run_rel(conn: psycopg.Connection, task_uid: str, raw: Path) -> str | None:
    """Any session of this task that shipped a run-local evaluator."""
    for (run_rel,) in conn.execute(
        "SELECT run_rel FROM gem_eg_session WHERE task_uid = %s ORDER BY session_uid",
        (task_uid,),
    ).fetchall():
        if _evaluator_for(raw, run_rel) is not None:
            return run_rel
    return None


def check(
    conn: psycopg.Connection, task_uid: str, raw: Path, timeout: int,
    tolerance: float = 1e-2,
) -> dict[str, Any]:
    result: dict[str, Any] = {"task_uid": task_uid}
    best = _best_program(conn, task_uid)
    if best is None:
        return {**result, "status": "no_program_with_payload"}
    result["node_uid"] = best["node_uid"]
    result["recorded_fitness"] = best["fitness"]

    run_rel = _evaluator_run_rel(conn, task_uid, raw)
    if run_rel is None:
        return {**result, "status": "no_run_local_evaluator"}
    evaluator = _evaluator_for(raw, run_rel)
    result["evaluator"] = str(evaluator)

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "program.py"
        program.write_text(best["payload"], encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DRIVER, str(evaluator), str(program)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {**result, "status": "timeout", "timeout_s": timeout}

    marker = "@@RESULT@@"
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None
    )
    if line is None:
        return {
            **result,
            "status": "evaluator_error",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        }

    metrics = json.loads(line[len(marker) :])
    replayed = metrics.get("combined_score")
    result["replayed_metrics"] = metrics
    result["replayed_fitness"] = replayed
    if replayed is None:
        return {**result, "status": "no_combined_score"}
    delta = abs(replayed - best["fitness"])
    result["abs_delta"] = delta
    rel = delta / abs(best["fitness"]) if best["fitness"] else None
    result["rel_delta"] = rel

    # Three classes, not pass/fail, because the evaluators are not all the same kind of
    # program. `circle_packing` and `heilbronn_triangle` are DETERMINISTIC geometric
    # constructors and reproduce bit-for-bit (rel_delta 0.0). The autocorrelation and
    # uncertainty evaluators run jax/optax optimisers, whose result depends on library
    # version and floating-point summation order; measured drift there is 1e-5 to 5e-3.
    # Calling that "the evaluation contract does not hold" would throw out four usable
    # tasks over numerical noise, and calling it "reproduced" would hide a real bound
    # on what our fitness numbers can be compared against. So it is named and bounded.
    if rel is None:
        result["status"] = "no_reference_fitness"
    elif rel == 0.0:
        result["status"] = "reproduced"
    elif rel <= tolerance:
        result["status"] = "reproduced_within_tolerance"
    else:
        result["status"] = "drifted"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--domain", default=None, help="math | ale; default both")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--tolerance", type=float, default=1e-2,
        help="relative delta still counted as a reproduction. Default 1e-2: the "
             "optimiser-based evaluators drift up to 4.6e-3 between library versions, "
             "and a task whose scores move by more than 1%% cannot be compared to its "
             "recorded numbers at all.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    conn = psycopg.connect(args.dsn)
    sql = "SELECT task_uid FROM gem_eg_task"
    params: tuple[Any, ...] = ()
    if args.domain:
        sql += " WHERE domain = %s"
        params = (args.domain,)
    tasks = [r[0] for r in conn.execute(sql + " ORDER BY task_uid", params).fetchall()]

    rows = []
    for task_uid in tasks:
        row = check(conn, task_uid, args.raw, args.timeout, args.tolerance)
        rows.append(row)
        delta = row.get("rel_delta")
        print(
            f"{task_uid:<28} {row['status']:<24}"
            f" recorded={row.get('recorded_fitness')}"
            f" replayed={row.get('replayed_fitness')}"
            f" rel_delta={f'{delta:.2e}' if delta is not None else '--'}",
            flush=True,
        )

    exact = [r for r in rows if r["status"] == "reproduced"]
    within = [r for r in rows if r["status"] == "reproduced_within_tolerance"]
    passed = exact + within
    print(f"\ngate: {len(passed)}/{len(rows)} usable "
          f"({len(exact)} exact, {len(within)} within {args.tolerance:g})")
    for row in within:
        print(f"  NOTE {row['task_uid']}: scores drift by {row['rel_delta']:.1e}; "
              "our fitness cannot be compared to the recorded value below that "
              "resolution")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tasks": rows}, indent=2), encoding="utf-8")
        print(f"receipt: {args.out}")
    return 0 if len(passed) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
