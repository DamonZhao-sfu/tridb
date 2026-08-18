"""Capture and verify the local model endpoint used by an E0 data run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.e0.common import (
    endpoint_models,
    environment_record,
    hf_snapshot_revisions,
    write_json,
)


def capture(
    *,
    base_url: str,
    model: str,
    artifact: str,
    revision: str,
    output: Path,
) -> dict:
    advertised = endpoint_models(base_url)
    ids = {str(row.get("id")) for row in advertised}
    roots = {str(row.get("root")) for row in advertised}
    if ids != {model}:
        raise RuntimeError(f"served model mismatch: expected {model!r}, got {ids}")
    if roots != {artifact}:
        raise RuntimeError(
            f"model artifact mismatch: expected {artifact!r}, got {roots}"
        )
    local_revisions = hf_snapshot_revisions(artifact)
    if revision not in local_revisions:
        raise RuntimeError(
            f"expected model revision {revision} is not visible locally: {local_revisions}"
        )
    receipt = {
        "base_url": base_url,
        "served_model": model,
        "artifact": artifact,
        "revision": revision,
        "advertised_model": advertised,
        "locally_visible_revisions": local_revisions,
        "environment": environment_record(),
    }
    write_json(output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = capture(
        base_url=args.base_url,
        model=args.model,
        artifact=args.artifact,
        revision=args.revision,
        output=args.output,
    )
    print(
        f"verified {receipt['served_model']} artifact={receipt['artifact']} "
        f"revision={receipt['revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
