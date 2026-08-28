"""Evaluator-only EvoMemBench assets.  Do not import this from online backends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PINNED_TOOL_ANSWERS_SHA256 = (
    "e205c7118c20c19aa0d8b1850a59f85b7d6f7a91d05e32322670e4a429f43473"
)
TOOL_ANSWERS_RELATIVE_PATH = (
    "Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/"
    "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_ours.json"
)


def load_tool_ground_truth(
    path: str | Path, *, expected_sha256: str = PINNED_TOOL_ANSWERS_SHA256
) -> dict[str, Any]:
    source = Path(path)
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"CrossEp-Tool answer checksum mismatch: {observed}")
    out: dict[str, Any] = {}
    for line in payload.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row["ground_truth"]
    return out
