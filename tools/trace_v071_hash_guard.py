"""Hash guard for the TRACE v0.7.1 work: the v0.7.0 guard (unchanged; the live queue runs it) plus two checks.

* Every v0.7.0-family run whose evaluation started (``episodes/`` or ``status.json`` exists, e.g. the live ALFWorld
  init / arm) is protected strictly: every ``live.artifacts`` and ``sources`` entry matches its manifest.
* No file hashed by any protected run mentions a ``*_v071`` / ``trace_v071`` / ``semfilter_v071`` module.
Directories set aside as ``*.superseded-*`` / ``*.incomplete-*`` are not started runs. A missing v0.7.0 report input
stays a violation (the v0.7.0 guard fails closed).

    .venv-optimizer/bin/python -m tools.trace_v071_hash_guard        # exit 1 on any violation
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from tools import trace_v070_hash_guard as hg

V071_IMPORT = re.compile(r"(_v071\b|trace_v071|semfilter_v071)")


def started_v070_runs() -> list[str]:
    runs = []
    for p in sorted((hg.ROOT / "results").glob("*/manifest.json")):
        run = str(p.parent.relative_to(hg.ROOT))
        if ".superseded-" in run or ".incomplete-" in run:
            continue  # set aside by an operator: not a run any queue can resume
        m = hg._manifest(run)
        if not m or not (isinstance(m.get("v070"), dict) and m["v070"].get("version") == "0.7.0"):
            continue
        if (p.parent / "episodes").is_dir() or (p.parent / "status.json").exists():
            runs.append(run)
    return runs


def v071_importers(runs) -> list[str]:
    files = set()
    for run in runs:
        files |= {k for k in hg.hashed(run) if k.endswith(".py")}
    return [f for f in sorted(files) if Path(f).exists() and V071_IMPORT.search(Path(f).read_text())]


def report() -> dict:
    base = hg.report()
    started = started_v070_runs()
    strict = {r: hg.mismatches(r) for r in started}
    # A v0.7.1 run hashes the v0.7.1 modules by design (review 2026-11-05): only runs WITHOUT a v071 block must stay
    # free of v0.7.1 imports. Their byte-identity is still checked through ``strict`` above.
    importers = v071_importers(hg.present(hg.protected()) + [r for r in started if "v071" not in (hg._manifest(r) or {})])
    violations = list(base["violations"])
    violations += [f"started v0.7.0 run {r}: {p}" for r, ps in strict.items() for p in ps]
    violations += [f"hashed file imports v0.7.1: {f}" for f in importers]
    return {"v070_report_ok": base["ok"], "protected_runs": base["protected_runs"], "started_v070_runs": strict,
            "v071_importers": importers, "violations": violations, "ok": not violations}


def main() -> int:
    r = report()
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
