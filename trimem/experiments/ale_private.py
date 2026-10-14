"""Private (held-out) ALE-Bench Lite scores for the TriMEM ALE cells (arXiv 2605.20086 §B.7 protocol).

For every finished cell: the seed program and the cell's public-best program (latest checkpoint) are
scored with `Session.private_eval` — rank and AtCoder performance on the private case set. Private
outcomes are never fed back to search. Results are cached by (problem, code sha256).

    PYTHONPATH=. .venv-e0/bin/python -m trimem.experiments.ale_private --phase it20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

R = Path("results/trimem_v0.1.0/ale")
CACHE = R / "private_cache.json"


def _best_code(cell: Path) -> tuple[str | None, float | None]:
    cks = sorted((cell / "openevolve" / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
    if not cks:
        return None, None
    info = json.loads((cks[-1] / "best_program_info.json").read_text())
    code = next(iter(cks[-1].glob("best_program.*"))).read_text()
    return code, (info.get("metrics") or {}).get("combined_score")


def score(problem: str, code: str, cache: dict, workers: int = 8) -> dict:
    import ale_bench
    from ale_bench.code_language import CodeLanguage

    code = "\n".join(line for line in code.splitlines() if "EVOLVE-BLOCK" not in line).strip() + "\n"
    key = f"{problem}:{hashlib.sha256(code.encode()).hexdigest()[:16]}"
    if key in cache:
        return cache[key]
    session = ale_bench.start(problem_id=problem, lite_version=True, num_workers=workers)
    try:
        t = time.perf_counter()
        result, rank, perf = session.private_eval(code, CodeLanguage.CPP20)
        row = {"judge": result.overall_judge_result.value, "private_abs": result.overall_absolute_score,
               "rank": int(rank), "performance": int(perf), "cases": len(result.case_results),
               "eval_seconds": round(time.perf_counter() - t, 1)}
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001, S110
            pass
    cache[key] = row
    CACHE.write_text(json.dumps(cache, indent=1))
    return row


def main(argv: list[str] | None = None) -> int:
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="it20", choices=["it20", "it50"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=str(R), help="results root holding it20/ and it50/ (e.g. .../ale_sem)")
    ap.add_argument("--problems", default=None, help="comma-separated problems to score (default: all)")
    ap.add_argument("--cache", default=str(CACHE), help="cache file (use a separate one for a parallel run)")
    args = ap.parse_args(argv)
    os.environ.setdefault("ALE_BENCH_DATA", "/localhome/hza214/ale-bench-data")
    os.environ.setdefault("ALE_BENCH_NO_CONTAINER_USER", "1")  # rootless podman cannot setresuid
    os.environ.setdefault("DOCKER_HOST", f"unix:///run/user/{os.getuid()}/podman/podman.sock")
    CACHE = Path(args.cache)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    root = Path(args.root)
    for cell in sorted((root / args.phase).iterdir()):
        receipt = cell / "run_receipt.json"
        if not receipt.exists() or json.loads(receipt.read_text()).get("status") != "complete":
            continue
        key, plan, _ = cell.name.split("__")
        if args.problems and key not in args.problems.split(","):
            continue
        best, public_best = _best_code(cell)
        seed_code = (cell / "initial_program.py").read_text()
        if args.phase == "it50":  # resumed cell: the seed lives in the iteration-20 cell
            seed_code = (root / "it20" / cell.name / "initial_program.py").read_text()
        s = score(key, seed_code, cache)
        b = score(key, best, cache) if best else None
        r = json.loads(receipt.read_text())
        rows.append({"cell": cell.name, "problem": key, "plan": plan, "phase": args.phase,
                     "public_seed": r.get("live_seed_fitness"), "public_best": public_best,
                     "private_seed_perf": s["performance"], "private_seed_rank": s["rank"],
                     "private_best_perf": b and b["performance"], "private_best_rank": b and b["rank"],
                     "delta_performance": b and b["performance"] - s["performance"], "judge_best": b and b["judge"]})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out or root / f"private_{args.phase}.json")
    out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
