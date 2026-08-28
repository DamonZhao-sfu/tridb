"""Paper-aligned held-out ALE re-scoring for selected memory arms.

Following arXiv:2605.20086 Appendix B.7, this tool evaluates the exact seed and
final public-best programs with ALE-Bench Lite's predefined private split.  It
never reads private seeds/inputs and never feeds private outcomes back to search.

Modes:
  inventory   validate receipts, source hashes, best-info and checkpoint lineage
  seed-smoke  private-score the ten shared seed programs
  full        private-score all unique seed/final programs and emit per-cell rows
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "tridb_evotrace_ale_private_v0.1.0"
PROTOCOL = "ale_bench_lite_official_private_seed_to_final_public_best"
EXPECTED_TASKS = (
    "ahc008",
    "ahc011",
    "ahc015",
    "ahc016",
    "ahc024",
    "ahc025",
    "ahc026",
    "ahc027",
    "ahc039",
    "ahc046",
)
VALID_ARMS = ("nocontext", "gem", "polyglot", "cognee")
VALID_PHYSICAL_PLANS = ("vfwd", "rrev", "aivg")


def canonical_code(code: str) -> str:
    """Remove harness markers that are invalid C++ preprocessor directives."""
    lines = (line for line in code.splitlines() if "EVOLVE-BLOCK" not in line)
    return "\n".join(lines).strip() + "\n"


def code_sha256(code: str) -> str:
    return hashlib.sha256(canonical_code(code).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*args: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _same_number(left: Any, right: Any) -> bool:
    try:
        left_f, right_f = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return abs(left_f - right_f) <= max(1e-9, 1e-12 * max(abs(left_f), abs(right_f)))


def _frequency(receipt: Mapping[str, Any], name: str) -> float | None:
    value = receipt.get("injection_frequency")
    if value is not None:
        return float(value)
    match = re.search(r"__p(\d{3})$", name)
    return None if match is None else int(match.group(1)) / 100.0


@dataclass(frozen=True)
class CellArtifact:
    cell: str
    problem: str
    arm: str
    physical_plan: str | None
    injection_frequency: float | None
    seed_path: str
    final_path: str
    seed_sha256: str
    final_sha256: str
    final_program_id: str
    final_checkpoint: str
    final_in_trace: bool
    public_seed_fitness: float
    public_final_fitness: float
    public_delta: float


def _trace_hashes(path: Path) -> set[str]:
    hashes = set()
    if not path.is_file():
        return hashes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = row.get("child_code")
        if isinstance(code, str) and code:
            hashes.add(code_sha256(code))
    return hashes


def load_cell(cell: Path) -> CellArtifact:
    receipt_path = cell / "run_receipt.json"
    seed_path = cell / "initial_program.py"
    final_path = cell / "openevolve/best/best_program.py"
    info_path = cell / "openevolve/best/best_program_info.json"
    trace_path = cell / "evolution_trace.jsonl"
    paths = (receipt_path, seed_path, final_path, info_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{cell.name}: missing artifacts: {missing}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError(f"{cell.name}: receipt is not complete")
    problem = str(receipt.get("task_uid", "")).split(":")[-1]
    arm = str(receipt.get("arm", ""))
    if problem not in EXPECTED_TASKS or arm not in VALID_ARMS:
        raise ValueError(f"{cell.name}: unexpected problem/arm {problem}/{arm}")

    seed_code = seed_path.read_text(encoding="utf-8", errors="replace")
    final_code = final_path.read_text(encoding="utf-8", errors="replace")
    seed_hash, final_hash = code_sha256(seed_code), code_sha256(final_code)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    program_id = str(info.get("id", ""))
    public_final = (info.get("metrics") or {}).get("combined_score")
    public_seed = receipt.get("live_seed_fitness", receipt.get("seed_fitness"))
    if not program_id or public_final is None or public_seed is None:
        raise ValueError(f"{cell.name}: incomplete best-info/seed fitness")

    matches = []
    pattern = f"openevolve/checkpoints/checkpoint_*/programs/{program_id}.json"
    for candidate in sorted(cell.glob(pattern)):
        row = json.loads(candidate.read_text(encoding="utf-8"))
        if code_sha256(str(row.get("code", ""))) != final_hash:
            continue
        if not _same_number(
            (row.get("metrics") or {}).get("combined_score"), public_final
        ):
            continue
        matches.append(candidate)
    if not matches:
        raise ValueError(
            f"{cell.name}: final id/hash/public score has no checkpoint proof"
        )

    trace_hashes = _trace_hashes(trace_path)
    return CellArtifact(
        cell=cell.name,
        problem=problem,
        arm=arm,
        physical_plan=receipt.get("physical_plan"),
        injection_frequency=_frequency(receipt, cell.name),
        seed_path=str(seed_path.resolve()),
        final_path=str(final_path.resolve()),
        seed_sha256=seed_hash,
        final_sha256=final_hash,
        final_program_id=program_id,
        final_checkpoint=str(matches[-1].resolve()),
        final_in_trace=final_hash == seed_hash or final_hash in trace_hashes,
        public_seed_fitness=float(public_seed),
        public_final_fitness=float(public_final),
        public_delta=float(public_final) - float(public_seed),
    )


def inventory(matrix: Path, arms: set[str], expected_cells: int) -> list[CellArtifact]:
    selected = []
    roots = [matrix]
    if (matrix / "cells").is_dir():
        roots.append(matrix / "cells")
    candidates = {cell for root in roots for cell in root.glob("ale_*")}
    for cell in sorted(candidates):
        if not cell.is_dir() or ".outage_" in cell.name:
            continue
        receipt_path = cell / "run_receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("arm") in arms:
            selected.append(cell)
    if len(selected) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} cells for {sorted(arms)}, found {len(selected)}"
        )
    artifacts = [load_cell(cell) for cell in selected]
    physical = {row.physical_plan for row in artifacts if row.physical_plan is not None}
    if physical and not physical <= set(VALID_PHYSICAL_PLANS):
        raise ValueError(f"invalid physical plans: {sorted(physical)}")
    per_task = {task: 0 for task in EXPECTED_TASKS}
    for artifact in artifacts:
        per_task[artifact.problem] += 1
    expected_per_task = expected_cells // len(EXPECTED_TASKS)
    if expected_cells % len(EXPECTED_TASKS) or any(
        count != expected_per_task for count in per_task.values()
    ):
        raise ValueError(f"unbalanced task coverage: {per_task}")
    return artifacts


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_inventory(out_dir: Path, artifacts: Sequence[CellArtifact]) -> None:
    rows = [asdict(artifact) for artifact in artifacts]
    _write_json(out_dir / "artifact_inventory.json", rows)
    with (out_dir / "artifact_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ale_code_state() -> dict[str, str]:
    import ale_bench

    root = Path(inspect.getfile(ale_bench)).resolve().parents[2]
    commit = _run("git", "rev-parse", "HEAD", cwd=root) or "unknown"
    diff = _run("git", "diff", "--binary", cwd=root) or ""
    return {
        "root": str(root),
        "commit": commit,
        "working_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def data_state() -> tuple[str, Path, dict[str, str]]:
    import ale_bench

    repo = Path(ale_bench.get_cache_dir()) / "datasets--SakanaAI--ALE-Bench"
    ref = repo / "refs/main"
    if not ref.is_file():
        raise ValueError("ALE-Bench cache has no pinned refs/main")
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = repo / "snapshots" / revision
    hashes = {}
    for task in EXPECTED_TASKS:
        archive = snapshot / f"{task}.zip"
        if not archive.is_file():
            raise ValueError(f"fixed ALE-Bench snapshot lacks {archive.name}")
        hashes[task] = file_sha256(archive)
    return revision, snapshot, hashes


def judge_runtime_state() -> dict[str, Any]:
    """Record the actual Docker-compatible engine and immutable judge image id."""
    import docker

    client = docker.from_env()
    try:
        version = client.version()
        image = client.images.get("ale-bench:cpp20-202301")
        return {
            "docker_host": os.environ.get("DOCKER_HOST"),
            "engine_name": version.get("Platform", {}).get("Name"),
            "engine_version": version.get("Version"),
            "judge_image": "ale-bench:cpp20-202301",
            "judge_image_id": image.id,
        }
    finally:
        client.close()


class Rescorer:
    def __init__(
        self,
        cache_path: Path,
        workers: int,
        judge_version: str,
        code_state: Mapping[str, str],
        data_revision: str,
    ) -> None:
        self.cache_path = cache_path
        self.workers = workers
        self.judge_version = judge_version
        self.code_state = dict(code_state)
        self.data_revision = data_revision
        self.entries: dict[str, dict[str, Any]] = {}
        if cache_path.is_file():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache.get("schema_version") == SCHEMA_VERSION:
                self.entries = dict(cache.get("entries") or {})

    def _key(self, problem: str, code_hash: str) -> str:
        contract = {
            "problem": problem,
            "code_sha256": code_hash,
            "protocol": PROTOCOL,
            "judge_version": self.judge_version,
            "ale_bench_commit": self.code_state["commit"],
            "ale_bench_diff": self.code_state["working_tree_diff_sha256"],
            "ale_data_revision": self.data_revision,
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def _persist(self) -> None:
        _write_json(
            self.cache_path,
            {"schema_version": SCHEMA_VERSION, "entries": self.entries},
        )

    def score(self, problem: str, code: str) -> dict[str, Any]:
        from ale_bench.code_language import CodeLanguage

        import ale_bench

        clean = canonical_code(code)
        code_hash = code_sha256(clean)
        key = self._key(problem, code_hash)
        if key in self.entries:
            return self.entries[key]
        session = ale_bench.start(
            problem_id=problem, lite_version=True, num_workers=self.workers
        )
        try:
            started = time.perf_counter()
            result, rank, performance = session.private_eval(
                clean, CodeLanguage.CPP20, judge_version=self.judge_version
            )
            if len(result.case_results) != session.num_private_cases:
                raise RuntimeError(f"{problem}: private case count mismatch")
            score_type = session.problem.metadata.score_type
            sign = 1 if score_type == "maximize" else -1
            row = {
                "problem": problem,
                "code_sha256": code_hash,
                "judge_result": result.overall_judge_result.value,
                "score_type": score_type,
                "private_case_count": len(result.case_results),
                "private_raw_score": int(result.overall_absolute_score),
                "private_search_oriented_score": int(result.overall_absolute_score)
                * sign,
                "private_relative_score": result.overall_relative_score,
                "private_rank": int(rank),
                "private_performance": int(performance),
                "eval_seconds": round(time.perf_counter() - started, 3),
                "evaluated_at": datetime.now(UTC).isoformat(),
            }
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001, S110
                pass
        self.entries[key] = row
        self._persist()
        return row


def generalization_label(public_delta: float, private_delta: int) -> str:
    if public_delta > 0 and private_delta > 0:
        return "aligned"
    if public_delta > 0 and -200 <= private_delta < 0:
        return "overfit(mild)"
    if public_delta > 0 and private_delta < -200:
        return "overfit(severe)"
    return "no movement"


def unique_programs(
    artifacts: Sequence[CellArtifact], seeds_only: bool
) -> Iterable[tuple[str, str, str]]:
    programs: dict[tuple[str, str], str] = {}
    for artifact in artifacts:
        seed = Path(artifact.seed_path).read_text(encoding="utf-8", errors="replace")
        programs[(artifact.problem, artifact.seed_sha256)] = seed
        if not seeds_only:
            final = Path(artifact.final_path).read_text(
                encoding="utf-8", errors="replace"
            )
            programs[(artifact.problem, artifact.final_sha256)] = final
    for (problem, code_hash), code in sorted(programs.items()):
        yield problem, code_hash, code


def materialize_rows(
    artifacts: Sequence[CellArtifact],
    scores: Mapping[tuple[str, str], Mapping[str, Any]],
    external_controls: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    controls = dict(external_controls or {})
    for artifact in artifacts:
        if artifact.arm == "nocontext":
            controls[artifact.problem] = int(
                scores[(artifact.problem, artifact.final_sha256)]["private_performance"]
            )
    if set(controls) != set(EXPECTED_TASKS):
        raise ValueError("all ten No Memory controls are required")

    rows = []
    for artifact in artifacts:
        seed = scores[(artifact.problem, artifact.seed_sha256)]
        final = scores[(artifact.problem, artifact.final_sha256)]
        delta = int(final["private_performance"]) - int(seed["private_performance"])
        rows.append(
            {
                **asdict(artifact),
                "protocol": PROTOCOL,
                "private_case_count": final["private_case_count"],
                "seed_private_raw_score": seed["private_raw_score"],
                "seed_private_rank": seed["private_rank"],
                "seed_private_performance": seed["private_performance"],
                "final_private_raw_score": final["private_raw_score"],
                "final_private_rank": final["private_rank"],
                "final_private_performance": final["private_performance"],
                "private_performance_delta_from_seed": delta,
                "private_performance_delta_vs_nocontext": (
                    None
                    if artifact.arm == "nocontext"
                    else int(final["private_performance"]) - controls[artifact.problem]
                ),
                "generalization_label": generalization_label(
                    artifact.public_delta, delta
                ),
                "seed_private_judge_result": seed["judge_result"],
                "final_private_judge_result": final["judge_result"],
            }
        )
    return rows


def write_rows(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_json(out_dir / "paper_aligned_outcomes.json", list(rows))
    with (out_dir / "paper_aligned_outcomes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--arms", default="nocontext,gem")
    parser.add_argument(
        "--mode", choices=("inventory", "seed-smoke", "full"), default="full"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-version", default="202301")
    parser.add_argument("--expected-cells", type=int, default=40)
    parser.add_argument(
        "--control-private-json",
        type=Path,
        help="previous compatible paper_aligned_outcomes.json supplying one held-out "
        "No Memory control per ALE task when the target matrix contains only plans",
    )
    return parser


def load_external_controls(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in rows if row.get("arm") == "nocontext"]
    controls = {
        str(row["problem"]): int(row["final_private_performance"]) for row in selected
    }
    if len(selected) != len(controls) or set(controls) != set(EXPECTED_TASKS):
        raise ValueError("control private JSON must have one No Memory row per task")
    if any(row.get("protocol") != PROTOCOL for row in selected):
        raise ValueError("control private JSON uses a different protocol")
    return controls


def load_control_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = path.parent / "private_eval_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"control private manifest is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = {arm.strip() for arm in args.arms.split(",") if arm.strip()}
    unknown = arms - set(VALID_ARMS)
    if not arms or unknown:
        raise ValueError(f"invalid arms: {sorted(unknown)}")
    matrix = args.matrix.resolve()
    suffix = "_".join(sorted(arms))
    out_dir = (args.out_dir or matrix / f"private_eval_v0.1.0_{suffix}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = inventory(matrix, arms, args.expected_cells)
    external_controls = load_external_controls(args.control_private_json)
    control_manifest = load_control_manifest(args.control_private_json)
    write_inventory(out_dir, artifacts)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "mode": args.mode,
        "status": "inventory_complete" if args.mode == "inventory" else "running",
        "generated_at": datetime.now(UTC).isoformat(),
        "matrix": str(matrix),
        "arms": sorted(arms),
        "cells_expected": args.expected_cells,
        "cells_inventory_valid": len(artifacts),
        "cells_missing_final_from_trace_but_checkpoint_valid": sum(
            not artifact.final_in_trace for artifact in artifacts
        ),
        "external_no_memory_control": (
            None
            if args.control_private_json is None
            else {
                "path": str(args.control_private_json.resolve()),
                "sha256": file_sha256(args.control_private_json),
                "tasks": len(external_controls),
                "source_manifest_sha256": file_sha256(
                    args.control_private_json.parent / "private_eval_manifest.json"
                ),
            }
        ),
        "unique_seed_programs": len(
            {(row.problem, row.seed_sha256) for row in artifacts}
        ),
        "unique_final_programs": len(
            {(row.problem, row.final_sha256) for row in artifacts}
        ),
        "unique_programs_to_evaluate": len(
            {
                (row.problem, code_hash)
                for row in artifacts
                for code_hash in (row.seed_sha256, row.final_sha256)
            }
        ),
        "lite_version": True,
        "judge_version": args.judge_version,
        "workers": args.workers,
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "private_data_access_rule": (
            "Only Session.private_eval is called; private seeds/inputs and "
            "standings internals are never read."
        ),
    }
    _write_json(out_dir / "private_eval_manifest.json", manifest)
    if args.mode == "inventory":
        print(json.dumps(manifest, indent=2))
        return 0

    os.environ.setdefault("ALE_BENCH_NO_CONTAINER_USER", "1")
    if "DOCKER_HOST" not in os.environ:
        socket = Path(f"/run/user/{os.getuid()}/podman/podman.sock")
        if socket.exists():
            os.environ["DOCKER_HOST"] = f"unix://{socket}"
    code_state = ale_code_state()
    revision, snapshot, archive_hashes = data_state()
    if control_manifest is not None:
        expected_control_state = {
            "judge_version": args.judge_version,
            "ale_data_revision": revision,
            "ale_bench_commit": code_state["commit"],
            "ale_bench_diff": code_state["working_tree_diff_sha256"],
        }
        observed_control_state = {
            "judge_version": control_manifest.get("judge_version"),
            "ale_data_revision": control_manifest.get("ale_data_revision"),
            "ale_bench_commit": (control_manifest.get("ale_bench_code") or {}).get(
                "commit"
            ),
            "ale_bench_diff": (control_manifest.get("ale_bench_code") or {}).get(
                "working_tree_diff_sha256"
            ),
        }
        if observed_control_state != expected_control_state:
            raise ValueError(
                "No Memory control private state differs from current rescore: "
                f"observed={observed_control_state}, expected={expected_control_state}"
            )
        manifest["external_no_memory_control"]["state_match"] = True
    runtime_state = judge_runtime_state()
    manifest.update(
        {
            "ale_bench_code": code_state,
            "ale_data_revision": revision,
            "ale_data_snapshot": str(snapshot),
            "ale_data_zip_sha256": archive_hashes,
            "judge_runtime": runtime_state,
        }
    )
    _write_json(out_dir / "private_eval_manifest.json", manifest)
    rescorer = Rescorer(
        out_dir / "private_score_cache.json",
        args.workers,
        args.judge_version,
        code_state,
        revision,
    )
    programs = list(unique_programs(artifacts, args.mode == "seed-smoke"))
    scores = {}
    for index, (problem, code_hash, code) in enumerate(programs, 1):
        print(f"[{index}/{len(programs)}] {problem} {code_hash[:12]}", flush=True)
        scores[(problem, code_hash)] = rescorer.score(problem, code)

    if args.mode == "full":
        rows = materialize_rows(artifacts, scores, external_controls)
        if len(rows) != args.expected_cells:
            raise RuntimeError(f"materialized {len(rows)} rows")
        write_rows(out_dir, rows)
        manifest.update({"status": "complete", "cells_scored": len(rows)})
    else:
        manifest.update(
            {"status": "seed_smoke_complete", "seed_programs_scored": len(scores)}
        )
    manifest["unique_private_scores_cached"] = len(rescorer.entries)
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(out_dir / "private_eval_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
