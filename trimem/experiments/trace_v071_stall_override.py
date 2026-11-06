"""Stall-rule override for TRACE v0.7.1 runs (user 2026-11-05 ~19:00), additive: no hashed file or manifest changes.

The ALFWorld arm's ledger freezes digest(manifest) and hashes ``trace_v071_search.py``, so the new rule is applied at
runtime through ``tools.trace_v071_stall_live``, in the launcher and again inside the spawned search worker:
* epsilon = 0.001 for every run: a round whose HV gain is at most 0.001 counts as stalled (was 1e-6). Patience stays 3.
* ``<run>/stall_override.json`` {"force_rounds": ["round_042"], ...}: those rounds are forced exploration rounds, whatever
  the counter says. The decision stays a pure function of the ledger, the round files and this file (crash replay).
The rule a round really used is recorded in its round file (``stall_rule``) and in ``hv_rounds.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

from trimem.experiments import trace_v071_search as ts

EPSILON = 1e-3
PROTOCOL = "rag-stack-stall-v1+epsilon-1e-3"
FILE = "stall_override.json"
_STOCK = {"rule": ts.stall_rule, "state": ts.stall_state, "worker": ts.latent_search_worker_v071}


def force_rounds(directory) -> set[str]:
    path = Path(directory) / FILE
    return set(json.loads(path.read_text()).get("force_rounds", [])) if path.exists() else set()


def stall_rule(manifest) -> dict | None:
    rule = _STOCK["rule"](manifest)
    return None if rule is None else {**rule, "epsilon": EPSILON, "override": PROTOCOL}


def stall_state(directory, manifest, entries, objective, *, before: str | None = None) -> dict:
    state = _STOCK["state"](directory, manifest, entries, objective, before=before)
    if before in force_rounds(directory):
        state = {**state, "force_exploration": True, "forced_by": "user"}
    return state


def install() -> None:
    ts.stall_rule, ts.stall_state = stall_rule, stall_state


def latent_search_worker(directory, manifest, entries, agent, seed, round_file):
    """Spawn target, picklable by name: a spawned child imports the stock module, so install again here."""
    install()
    return _STOCK["worker"](directory, manifest, entries, agent, seed, round_file)
