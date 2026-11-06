"""Hash guard for TRACE v0.7.0 work: no file hashed by a live / paused / queued run may change. Fails closed.

Protected runs (fix round 1, finding 5) = every ``results/*/manifest.json`` with a non-empty ``live.artifacts`` whose
run is not v0.7.0 (discovery), plus the explicit floor ``GUARDED`` (a missing GUARDED manifest is a violation).

* ``GUARDED`` runs: every ``live.artifacts`` and ``sources`` entry matches its recorded sha256 exactly.
* ``OLDER`` (v0.6.0c, 2026-11-03c): their files must equal ``SNAPSHOT`` (taken before any v0.7.0 file was
  written); a missing snapshot is a violation.
* Every discovered run: each hashed file matches its manifest, OR equals the frozen pre-existing value in
  ``BASELINE`` (``--freeze-baseline`` writes it; its evidence block records that no hashed file had an mtime after
  the v0.7.0 work began). A file outside the baseline (a new run) must match its manifest. A missing baseline is a
  violation.
* No hashed ``.py`` file may import a ``*_v070`` / ``trace_v070*`` module.

    .venv-optimizer/bin/python -m tools.trace_v070_hash_guard            # exit 1 on any violation
    .venv-optimizer/bin/python -m tools.trace_v070_hash_guard --freeze-baseline  # once; refuses to overwrite
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDED = (
    "results/trace_latent_sciworld_20261104",
    "results/trace_latent_alfworld_20261104",
    "results/trace_pr_webshop_v060_strat_20261104",
    "results/trace_pr_sciworld_v060_strat_20261104",
    "results/trace_pr_alfworld_v060_strat_20261104",
)
OLDER = (
    "results/trace_pr_alfworld_v060_20261103c",
    "results/trace_pr_sciworld_v060_20261103c",
    "results/trace_pr_webshop_v060_20261103c",
)
E2E = ROOT / "results/trace_v070_e2e_20261105"
SNAPSHOT = E2E / "hash_guard_before.json"
BASELINE = E2E / "hash_guard_discovered_baseline.json"
WORK_START = "2026-11-05T11:20:00"
V070_IMPORT = re.compile(r"(_v070\b|trace_v070|plan_schema_v070)")


def sha(path) -> str | None:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _manifest(run: str) -> dict | None:
    try:
        m = json.loads((ROOT / run / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def hashed(run: str) -> dict[str, str]:
    m = _manifest(run)
    if m is None:
        raise FileNotFoundError(f"{run}/manifest.json missing or unreadable")
    return {**m["live"]["artifacts"], **{str(ROOT / k): v for k, v in (m.get("sources") or {}).items()}}


def discover() -> list[str]:
    runs = []
    for p in sorted((ROOT / "results").glob("*/manifest.json")):
        run = str(p.parent.relative_to(ROOT))
        m = _manifest(run)
        lv = m.get("live") if m else None
        arts = lv.get("artifacts") if isinstance(lv, dict) else None
        v070 = isinstance(m.get("v070"), dict) and m["v070"].get("version") == "0.7.0" if m else False
        if isinstance(arts, dict) and arts and not v070:
            runs.append(run)
    return runs


def protected() -> list[str]:
    return sorted(set(discover()) | set(GUARDED) | set(OLDER))


def present(runs) -> list[str]:
    return [r for r in runs if _manifest(r) is not None]


def mismatches(run: str) -> list[str]:
    return sorted(k for k, v in hashed(run).items() if sha(k) != v)


def new_drift(run: str) -> list[str]:
    before = json.loads(SNAPSHOT.read_text())["files"]
    return sorted(p for p in hashed(run) if sha(p) != before.get(str(Path(p).resolve())))


def baseline_drift(run: str, base: dict) -> list[str]:
    return sorted(p for p, v in hashed(run).items()
                  if sha(p) != v and (str(Path(p).resolve()) not in base
                                      or sha(p) != base[str(Path(p).resolve())]))


def v070_importers(runs) -> list[str]:
    files = set()
    for run in runs:
        files |= {k for k in hashed(run) if k.endswith(".py")}
    return [f for f in sorted(files) if Path(f).exists() and V070_IMPORT.search(Path(f).read_text())]


def report() -> dict:
    violations: list[str] = []
    runs = protected()
    missing = [r for r in runs if _manifest(r) is None]
    violations += [f"missing protected manifest: {r}" for r in missing]
    runs = [r for r in runs if r not in missing]
    for f, what in ((SNAPSHOT, "pre-v0.7.0 snapshot"), (BASELINE, "discovered-run baseline")):
        if not f.exists():
            violations.append(f"missing {what}: {f}")
    guarded = {r: mismatches(r) for r in GUARDED if r in runs}
    older = {r: new_drift(r) for r in OLDER if r in runs} if SNAPSHOT.exists() else {}
    base = json.loads(BASELINE.read_text())["files"] if BASELINE.exists() else None
    discovered = {r: baseline_drift(r, base) for r in runs} if base is not None else {}
    importers = v070_importers(runs)
    # fix round 2: each check is aggregated on its own (a run in several checks keeps every violation)
    for check, results in (("guarded", guarded), ("older-snapshot", older), ("discovered", discovered)):
        violations += [f"{check} {r}: {p}" for r, ps in results.items() for p in ps]
    violations += [f"hashed file imports v0.7.0: {f}" for f in importers]
    return {"protected_runs": len(runs), "guarded": guarded, "older_new_drift": older,
            "discovered_new_drift": {r: v for r, v in discovered.items() if v}, "v070_importers": importers,
            "snapshot": str(SNAPSHOT), "baseline": str(BASELINE), "violations": violations, "ok": not violations}


def freeze_baseline() -> dict:
    if BASELINE.exists():
        raise SystemExit(f"{BASELINE} exists; the baseline is frozen once")
    start = time.mktime(time.strptime(WORK_START, "%Y-%m-%dT%H:%M:%S"))
    files, late = {}, []
    for run in present(protected()):
        for p in hashed(run):
            key = str(Path(p).resolve())
            files[key] = sha(p)
            if Path(p).exists() and os.path.getmtime(p) > start:
                late.append(key)
    if late:
        raise SystemExit(f"hashed files modified after {WORK_START}: {sorted(set(late))[:10]}")
    body = {"taken_at": time.time(), "rule": "pre-existing manifest mismatches of discovered runs, frozen",
            "evidence": {"work_start": WORK_START, "files": len(files), "mtime_after_work_start": 0},
            "files": files}
    BASELINE.write_text(json.dumps(body, indent=1) + "\n")
    return body["evidence"]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--freeze-baseline" in argv:
        print(json.dumps(freeze_baseline()))
        return 0
    r = report()
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
