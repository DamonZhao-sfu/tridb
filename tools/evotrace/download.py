"""P0 — pin and download the EvoTrace corpus, then prove what landed.

The dataset is a *file-tree* dataset, not a parquet one: 121 run directories, each
carrying five canonical files plus optional config/evaluator/analysis material. We pin
one revision and record a per-file SHA-256 manifest, because every downstream count in
this track (10,809 base entities, 10,479 lineage edges, ALE 56 + math 65) is only
meaningful against a named revision.

    python3 tools/evotrace/download.py            # download at the pinned revision
    python3 tools/evotrace/download.py --verify   # re-hash what is on disk

License: CC-BY-4.0. The attribution string lives in ATTRIBUTION below and is copied
into the manifest so it travels with the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ID = "ZIB-IOL/EvoTrace"
REPO_TYPE = "dataset"

#: The audited revision. EvoTraceDoc.md's every count is stated against this SHA;
#: changing it invalidates the integrity gates in normalize.py, so it is a constant
#: rather than a flag.
REVISION = "349117b04832b681ccf69ea2518c31a579853b3e"

ATTRIBUTION = (
    "EvoTrace (ZIB-IOL/EvoTrace), revision "
    f"{REVISION}, licensed CC-BY-4.0. Paper: arXiv:2605.20086."
)

#: The five files the dataset card declares for every run. A run missing any of them
#: is not `core_graph_ready` and is reported, never silently dropped.
CANONICAL_FILES = (
    "meta.json",
    "programs.jsonl",
    "iterations.jsonl",
    "iter_scalars.jsonl",
    "logs/llm_calls.jsonl",
)

#: Present on only some runs (67/121, 42/121, 18/121 respectively per the doc's tree
#: audit). Their absence is a replay-readiness fact, not a download failure.
OPTIONAL_FILES = ("run_info.json", "run_config.yaml", "evaluate.py")

DEFAULT_ROOT = Path("data/evotrace")

_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class RunDir:
    """One EvoTrace run.

    The tree is NOT uniformly two levels deep: 101 runs sit at ``<backend>/<run>`` and
    20 at ``<backend>/<group>/<run>``, where the group is an experimental condition
    (``ablation``, ``empty_seed``, ``nodiff``, ``strong_seed``). The group is a session
    property, so it is captured rather than flattened away — two runs with the same
    name under different groups are different sessions.
    """

    backend: str
    group: str | None
    name: str

    @property
    def rel(self) -> str:
        return "/".join(p for p in (self.backend, self.group, self.name) if p)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    """Every regular file under ``root``, excluding the hub's own cache metadata."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".cache" in path.parts or ".huggingface" in path.parts:
            continue
        yield path


def discover_runs(root: Path) -> list[RunDir]:
    """A run is any directory holding a ``programs.jsonl``.

    Structural discovery rather than a hardcoded list: the backend directory names
    (``evox``, ``openevolve``, ``gepa``, ``shinka``) are data, and asserting them here
    would turn a source change into a wrong answer instead of a failed gate.
    """
    runs: list[RunDir] = []
    for programs in sorted(root.rglob("programs.jsonl")):
        rel = programs.parent.relative_to(root)
        if len(rel.parts) == 2:
            runs.append(RunDir(backend=rel.parts[0], group=None, name=rel.parts[1]))
        elif len(rel.parts) == 3:
            runs.append(RunDir(backend=rel.parts[0], group=rel.parts[1], name=rel.parts[2]))
        else:
            # Unexpected nesting depth — surface it rather than guess a backend.
            print(f"  ! unexpected run depth, skipped: {rel}", file=sys.stderr)
    return runs


def classify_domain(run: RunDir) -> str:
    """ALE (C++ heuristic contest) vs math discovery, from the run directory name.

    The dataset card's own domain counts are self-inconsistent (ALE 62 + math 65 = 127
    against a stated total of 121). We derive the split from the tree and report both
    numbers; see `reconcile_domains`.
    """
    name = run.name.lower()
    if name.startswith("ahc") or "ale" in name:
        return "ale"
    return "math"


def reconcile_domains(runs: list[RunDir]) -> dict[str, object]:
    """Compare the observed domain split against the card and the paper.

    Recording the disagreement is the point. The paper's Table 7 and the doc's tree
    audit both say ALE 56 + math 65; the card says ALE 62 + math 65. We keep the
    observed numbers and flag the card as a source issue.
    """
    observed: dict[str, int] = {}
    for run in runs:
        domain = classify_domain(run)
        observed[domain] = observed.get(domain, 0) + 1
    card = {"ale": 62, "math": 65}
    paper_table7 = {"ale": 56, "math": 65}
    return {
        "observed": observed,
        "observed_total": sum(observed.values()),
        "dataset_card_claim": card,
        "dataset_card_total": sum(card.values()),
        "paper_table7_claim": paper_table7,
        "agrees_with_paper": observed == paper_table7,
        "source_issue": (
            "Dataset card states ALE 62 + math 65 = 127, incompatible with its own "
            "stated total of 121 runs. Paper Table 7 and the pinned file tree both "
            "give ALE 56 + math 65 = 121. Manifest uses the observed tree."
        ),
    }


def run_completeness(root: Path, runs: list[RunDir]) -> list[dict[str, object]]:
    """Per-run presence of the canonical and optional files."""
    rows: list[dict[str, object]] = []
    for run in runs:
        base = root / run.rel
        canonical = {name: (base / name).is_file() for name in CANONICAL_FILES}
        optional = {name: (base / name).is_file() for name in OPTIONAL_FILES}
        rows.append(
            {
                "run": run.rel,
                "backend": run.backend,
                "group": run.group,
                "domain": classify_domain(run),
                "canonical": canonical,
                "core_graph_ready": all(canonical.values()),
                "optional": optional,
                "has_blobs": (base / "blobs").is_dir(),
                "has_analysis": (base / "analysis").is_dir(),
            }
        )
    return rows


def build_manifest(root: Path, *, hash_files: bool = True) -> dict[str, object]:
    """Hash every downloaded file and summarise the tree."""
    files: list[dict[str, object]] = []
    total_bytes = 0
    for path in iter_files(root):
        size = path.stat().st_size
        total_bytes += size
        entry: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "bytes": size,
        }
        if hash_files:
            entry["sha256"] = sha256_file(path)
        files.append(entry)

    runs = discover_runs(root)
    completeness = run_completeness(root, runs)
    return {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": REVISION,
        "attribution": ATTRIBUTION,
        "license": "CC-BY-4.0",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "run_count": len(runs),
        "core_graph_ready_runs": sum(1 for r in completeness if r["core_graph_ready"]),
        "domain_reconciliation": reconcile_domains(runs),
        "runs": completeness,
        "files": files,
    }


def download(root: Path, *, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"downloading {REPO_ID}@{REVISION[:8]} -> {raw}")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        local_dir=str(raw),
        allow_patterns=allow_patterns,
        max_workers=int(os.environ.get("EVOTRACE_HF_WORKERS", "8")),
    )
    return Path(local)


def verify(root: Path, manifest_path: Path) -> int:
    """Re-hash the tree against a stored manifest. Returns a process exit code."""
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; run without --verify first")
        return 2
    stored = json.loads(manifest_path.read_text())
    expected = {entry["path"]: entry.get("sha256") for entry in stored["files"]}

    raw = root / "raw"
    seen: set[str] = set()
    mismatched: list[str] = []
    for path in iter_files(raw):
        rel = str(path.relative_to(raw))
        seen.add(rel)
        want = expected.get(rel)
        if want is None:
            mismatched.append(f"untracked: {rel}")
        elif sha256_file(path) != want:
            mismatched.append(f"sha mismatch: {rel}")
    for rel in expected:
        if rel not in seen:
            mismatched.append(f"missing: {rel}")

    runs = discover_runs(raw)
    incomplete = [
        row["run"] for row in run_completeness(raw, runs) if not row["core_graph_ready"]
    ]

    print(f"files checked : {len(seen)} (manifest {len(expected)})")
    print(f"runs found    : {len(runs)}")
    print(f"core-ready    : {len(runs) - len(incomplete)}/{len(runs)}")
    if incomplete:
        print(f"incomplete    : {incomplete[:10]}{' ...' if len(incomplete) > 10 else ''}")
    if mismatched:
        print(f"PROBLEMS ({len(mismatched)}):")
        for line in mismatched[:20]:
            print(f"  {line}")
        return 1
    print("manifest verified")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verify", action="store_true", help="re-hash against the manifest")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="skip the download, (re)build the manifest from what is on disk",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=None,
        help="hub allow_patterns glob; repeatable. Omit for the full 1.7 GB tree.",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    manifest_path = root / "raw_manifest.json"

    if args.verify:
        return verify(root, manifest_path)

    if not args.manifest_only:
        download(root, allow_patterns=args.allow)

    print("building manifest (hashing every file)...")
    manifest = build_manifest(root / "raw")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))

    dom = manifest["domain_reconciliation"]
    print(f"files        : {manifest['file_count']}")
    print(f"bytes        : {manifest['total_bytes']:,}")
    print(f"runs         : {manifest['run_count']}")
    print(f"core-ready   : {manifest['core_graph_ready_runs']}/{manifest['run_count']}")
    print(f"domains      : {dom['observed']} (total {dom['observed_total']})")
    print(f"paper agrees : {dom['agrees_with_paper']}")
    print(f"manifest     : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
