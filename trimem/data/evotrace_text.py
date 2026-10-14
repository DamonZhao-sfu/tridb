"""EvoTrace text and search-state layer: prompts, LLM calls, change intent, membership.

Everything here reads the raw run directories (`dataset/evotrace/raw/<run_rel>/`) one session
at a time and streams into COPY; nothing holds a whole file in memory except per-session rows.

Join keys (docs/trimem_trajectory_storage_v0.2.0.md §1.3):
  prompt   -> attempt : `prompts.node_uid` (exact); shinka: the program's `metadata.llm_result`.
  llm_call -> attempt : sha1(strip(response)) == sha1(strip(raw llm content)) within the session.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

SEARCH = "<<<<<<< SEARCH"
_DESCRIPTION = re.compile(r"<DESCRIPTION>(.*?)</DESCRIPTION>", re.S)
SHINKA_SUMMARY = "You are an expert programming assistant analyzing"
SHINKA_RECOMMEND = "You are an expert programming assistant generating actionable"


def sha1(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()


def extract_intent(response: str | None) -> str | None:
    """The LLM's own statement of the intended change: prose before the first SEARCH block,
    else the <DESCRIPTION> body, else prose before the first code fence. None if empty."""
    if not response:
        return None
    if SEARCH in response:
        text = response.split(SEARCH, 1)[0]
    elif m := _DESCRIPTION.search(response):
        text = m.group(1)
    elif "```" in response:
        text = response.split("```", 1)[0]
    else:
        return None
    text = text.strip()
    return text or None


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


@dataclass(frozen=True)
class NodeText:
    template: str | None
    system: str | None
    user: str | None
    response: str | None
    intent: str | None
    intent_source: str | None
    source: str  # 'prompt_blob' | 'shinka_llm_result'


def node_text(
    node: dict, run_dir: Path, raw_prog: dict | None, blob_rel: str | None
) -> NodeText | None:
    """Resolve one node's prompt/response. Returns None when the dataset holds neither.
    A referenced prompt blob that is missing on disk raises: the caller decides (from
    prompts.jsonl `blob_present`) whether a blob is expected and passes None otherwise."""
    if blob_rel:
        blob = run_dir / blob_rel
        if not blob.is_file():
            raise FileNotFoundError(f"prompt blob missing: {blob}")
        payload = json.loads(blob.read_text())
        if len(payload) != 1:
            raise ValueError(
                f"{blob}: expected one template key, got {sorted(payload)}"
            )
        ((template, v),) = payload.items()
        response = (v.get("responses") or [None])[0]
        intent = extract_intent(response)
        return NodeText(
            template,
            v.get("system"),
            v.get("user"),
            response,
            intent,
            "response_prose" if intent else None,
            "prompt_blob",
        )
    md = (raw_prog or {}).get("metadata") or {}
    lr = md.get("llm_result")
    desc = (md.get("patch_description") or "").strip()
    if md.get("patch_type") == "init":
        desc = ""  # 'Initial program setup' is not a change
    if isinstance(lr, dict) or desc:
        lr = lr if isinstance(lr, dict) else {}
        return NodeText(
            None,
            lr.get("system_msg"),
            lr.get("msg"),
            lr.get("content"),
            desc or None,
            "shinka_patch_description" if desc else None,
            "shinka_llm_result",
        )
    return None


def classify_call(raw: dict, linked: bool, backend: str) -> str:
    content = raw.get("content") or ""
    if raw.get("error"):
        return "error"
    if not content:
        return "empty"
    if linked:
        return "mutation"
    if SEARCH in content or "<NAME>" in content:
        return (
            "mutation_unretained"  # a proposal whose program the framework did not keep
        )
    if backend == "evox":
        return "search_algo"  # evox co-evolves its search algorithm with separate LLM calls
    system = raw.get("system_msg") or ""
    if system.startswith(SHINKA_SUMMARY):
        return "summary"
    if system.startswith(SHINKA_RECOMMEND):
        return "recommendation"
    return "other"


class Link(NamedTuple):
    attempt_id: int | None  # set ONLY for an unambiguous 1:1 match
    method: str | None  # 'response_sha1' | 'response_sha1_ambiguous' | None
    candidates: tuple[
        int, ...
    ]  # every attempt the response text matches (ambiguous rows)


def pair_links(
    call_sha: list[str | None], node_sha: dict[str, list[int]]
) -> list[Link]:
    """Exact sha1 links for one session's calls (in seq order).

    A response text that occurs in exactly one call AND exactly one node is a provenance fact
    ('response_sha1'). When several calls and/or nodes share the text, which call produced which
    node is not recorded, so no attempt_id is asserted: every call of the group is
    'response_sha1_ambiguous' and lists all matching attempts as candidates."""
    calls_by: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(call_sha):
        if h and h in node_sha:
            calls_by[h].append(i)
    out = [Link(None, None, ())] * len(call_sha)
    for h, idxs in calls_by.items():
        nodes = tuple(node_sha[h])
        for i in idxs:
            out[i] = (
                Link(nodes[0], "response_sha1", nodes)
                if len(idxs) == 1 and len(nodes) == 1
                else Link(None, "response_sha1_ambiguous", nodes)
            )
    return out


class MembershipRLE:
    """Collapse per-checkpoint membership snapshots into intervals.

    Contiguity is measured on the session's checkpoint sequence (not iteration+1): a member
    present at consecutive checkpoints c_i..c_j is one (from_iter=c_i, to_iter=c_j) row. A value
    change (e.g. gepa metric_best score) starts a new interval. For one (session, role, slot,
    node) the intervals are disjoint by construction, so at any checkpoint there is at most one
    row per member (the table's exclusion constraint enforces the same). Two DIFFERENT values for
    the same member at the same checkpoint are a data error and raise; exact repeats are counted
    in `duplicates` and dropped."""

    def __init__(self) -> None:
        self.runs: dict[tuple, dict[int, str | None]] = defaultdict(dict)
        self.checkpoints: dict[str, set[int]] = defaultdict(set)
        self.events = 0
        self.duplicates = 0

    def add(self, e: dict) -> None:
        slot = None if e["slot_key"] is None else str(e["slot_key"])
        value = None if e["value"] is None else json.dumps(e["value"], sort_keys=True)
        key = (e["session_uid"], e["role"], slot, e["node_uid"])
        it = e["iteration"]
        seen = self.runs[key]
        if it in seen:
            if seen[it] != value:
                raise ValueError(
                    f"conflicting membership values at {key} iteration {it}"
                )
            self.duplicates += 1
        seen[it] = value
        self.checkpoints[e["session_uid"]].add(it)
        self.events += 1

    def checkpoint_rows(self) -> Iterator[tuple[str, int]]:
        for s, its in self.checkpoints.items():
            for it in sorted(its):
                yield s, it

    def intervals(self) -> Iterator[tuple]:
        order = {s: sorted(its) for s, its in self.checkpoints.items()}
        pos = {s: {it: k for k, it in enumerate(its)} for s, its in order.items()}
        for (s, role, slot, node), seen in self.runs.items():
            ks = sorted((pos[s][it], v) for it, v in seen.items())
            start, prev, value = ks[0][0], ks[0][0], ks[0][1]
            for k, v in [*ks[1:], (None, None)]:
                if k is not None and k == prev + 1 and v == value:
                    prev = k
                    continue
                yield s, role, slot, node, order[s][start], order[s][prev], value
                if k is not None:
                    start, prev, value = k, k, v
