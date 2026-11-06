"""Validate frozen TEST identities; optionally launch a script after validation.

Usage: python -m tools.fixed_test_panel_guard [--run-script SCRIPT [ARGS ...]]
The wrapper leaves experiment code unchanged, including frozen source manifests.
"""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "triMEM_optimizer/test_panels/fixed_test_v1.json"


def ids_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def validate_ids(dataset, ids, *, partial=False):
    panel = json.loads(MANIFEST.read_text())["datasets"][dataset]
    expected = panel["query_ids"]
    if len(expected) != 200 or len(set(expected)) != 200:
        raise ValueError(f"Invalid frozen panel: {dataset}")
    if ids_digest(expected) != panel["ids_sha256"]:
        raise ValueError(f"Frozen panel checksum mismatch: {dataset}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate TEST IDs: {dataset}")
    foreign, missing = set(ids) - set(expected), set(expected) - set(ids)
    if foreign or (missing and not partial):
        raise ValueError(f"TEST panel mismatch {dataset}: foreign={sorted(foreign)}, missing={len(missing)}")
    return {"dataset": dataset, "covered": len(ids), "expected": len(expected),
            "complete": not missing, "ids_sha256": panel["ids_sha256"]}


def preflight():
    manifest = json.loads(MANIFEST.read_text())
    reports = []
    for ds, panel in manifest["datasets"].items():
        current = []
        for source in panel["shards"]:
            path = ROOT / source["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"TEST shard changed: {path}")
            current.extend(r["item_id"] for r in json.loads(path.read_text()))
        reports.append(validate_ids(ds, current))
        if current != panel["query_ids"]:
            raise ValueError(f"TEST ordering changed: {ds}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-script", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(json.dumps({"fixed_test_panel": preflight()}, indent=2), flush=True)
    if args.run_script is not None:
        if not args.run_script:
            parser.error("--run-script requires a script path")
        # Prevent reduced panels from masquerading as a formal baseline run.
        import os

        if any(a == "--limit" or a.startswith("--limit=") for a in args.run_script[1:]):
            if os.environ.get("EXTBASE_SMOKE") != "1":
                raise ValueError("--limit is only allowed with EXTBASE_SMOKE=1")
        sys.argv = args.run_script
        runpy.run_path(str(Path(sys.argv[0]).resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
