"""P1 — turn the EvoTrace file tree into the Experience Graph staging corpus.

This is the source of truth for everything downstream: the GEM loader reads these
JSONL files, never the raw tree. Keeping the normalization outside the database means
a count can be re-audited by hand without a live engine.

    python3 tools/evotrace/normalize.py                 # normalize + integrity report
    python3 tools/evotrace/normalize.py --check         # re-print the report only

Objects emitted (arXiv:2606.29823's logical model, EvoTraceDoc.md §3.1):

    tasks.jsonl          problem specification + success metric  (the ANN entry)
    sessions.jsonl       one evolutionary run
    nodes.jsonl          one ATTEMPT (accepted, rejected or failed alike)
    lineage_edges.jsonl  parent -> child causal edits
    context_edges.jsonl  inspiration/context influence, kept SEPARATE from lineage
    prompts.jsonl        prompt-history references, with per-node completeness
    artifacts.jsonl      content-addressed code blobs
    state_events.jsonl   per-iteration search state (frontier + time-travel substrate)
    llm_calls.jsonl      per-session LLM usage
    replay_manifest.json per-run readiness gates
    integrity_report.json every gate, with numerator AND denominator

Boundaries this module enforces rather than documents:

* Rejected and failed nodes are RETAINED. They are the Repair negatives.
* Artifacts dedup by content; nodes never do. Two attempts at identical code are two
  attempts.
* A parent id that does not resolve inside its own session is REJECTED, not guessed.
* All EvoTrace wall-clock timestamps are null. Ordering is by ``iteration`` only, and
  nothing here fabricates a time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tools.evotrace.download import REVISION, discover_runs
from tools.evotrace.ids import (
    RunName,
    RunNameError,
    artifact_uid,
    node_uid,
    parse_run_name,
    prompt_uid,
    session_uid,
)

DEFAULT_ROOT = Path("data/evotrace")

#: Counts the paper (arXiv:2605.20086) reports, checked against what we observe. A
#: mismatch is REPORTED, never silently adopted — see `integrity_report.json`.
PAPER_CLAIMS = {
    "tasks": 16,
    "sessions": 121,
    "nodes": 10_672,
    "accepted": 8_964,
    "rejected": 1_708,
    "lineage_edges": 10_479,
    "llm_calls": 18_400,
}


def dumps(row: Any) -> str:
    """json.dumps that REFUSES non-finite floats.

    Bare NaN/Infinity are not JSON. Emitting them produced a staging file Python could
    read back and PostgreSQL could not, so the failure surfaced only at load time with
    no pointer to the offending record. allow_nan=False turns that into an immediate,
    located error; `json_safe` is what legitimately clears the known cases first.
    """
    return json.dumps(row, allow_nan=False)


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (row offset, object). Row offset is provenance, kept on every record."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            line = line.strip()
            if line:
                yield offset, json.loads(line)


def _blob_rel(sha: str, suffix: str) -> str:
    """EvoTrace blob layout: ``blobs/<first 2 hex>/<sha>.<suffix>``."""
    return f"blobs/{sha[:2]}/{sha}.{suffix}"


#: `judge_result` values that mean "the program ran correctly". Everything else in that
#: field is a real execution failure class.
_JUDGE_OK = {"ACCEPTED", "OK", "AC"}


def error_signature(metrics: dict[str, Any], status: str) -> str | None:
    """A coarse, comparable failure label, prefixed by WHO rejected the attempt.

    The prefix is load-bearing, because two very different things look alike in this
    corpus and conflating them poisons the Repair ground truth:

    ``judge:<CLASS>`` / ``error:<line>``
        The *executor* rejected it — it timed out, failed to compile, crashed. This is
        a repair target: there is a defect to fix.
    ``rejected:not_improved``
        The *search* rejected it. The program compiled and ran fine (``judge_result``
        is an OK value, no error text); the backend discarded it for not beating what
        it already had. **Not** a repair target.
    ``None``
        Accepted and clean.

    Measured on the pinned corpus, 698 of 1,708 rejected nodes fall in the second
    class. Labelling them ``judge_result:ACCEPTED`` — as the first version of this
    function did — reads as "the judge accepted it, so it is a failure of type
    ACCEPTED", which is nonsense and would have put 698 non-defects into the repair
    population.
    """
    judge = metrics.get("judge_result")
    judge = judge.strip() if isinstance(judge, str) else None
    error_text = next(
        (
            str(metrics[key]).strip()
            for key in ("error_type", "error")
            if isinstance(metrics.get(key), str) and str(metrics[key]).strip()
        ),
        None,
    )

    # An execution failure, whichever field the backend put it in.
    if judge and judge.upper() not in _JUDGE_OK:
        return f"judge:{judge[:120]}"
    if error_text:
        # First line only: the tail is a stack trace / stderr dump, not a class.
        return f"error:{error_text.splitlines()[0][:120]}"

    if status == "accepted":
        return None
    # Ran clean, but the search still discarded it.
    return "rejected:not_improved" if judge else f"rejected:{status}"


#: Integer sentinels some evaluators emit for "this did not run" while still calling
#: the field a score. Detected by EXACT equality, never by a magnitude threshold: the
#: nearest real reward in this corpus is -4.25e7, eleven orders of magnitude away, so
#: there is no ambiguity to resolve and no risk of clipping a legitimate value.
_FITNESS_SENTINELS = frozenset({float(-(2**63)), float(2**63), float(2**63 - 1)})


def fitness_defect(value: Any) -> str | None:
    """Why this ``combined_score`` cannot be used as a reward, or None if it can.

    Two defects, both of which look like a number and neither of which is one:

    ``nonfinite``
        NaN/Inf. Poison, because `NaN > x` is False for every x — one NaN silently
        turns every reward comparison in the ground truth into "did not improve"
        while raising nothing. 38 nodes.
    ``sentinel``
        Exactly -2**63. An evaluator's "failed" marker sitting in the score field.
        193 nodes, all in ale:ahc025 and ale:ahc027, and **163 of them are
        status=accepted** with 50 carrying no error signature at all — so they read as
        ordinary healthy nodes with an astronomically bad reward. Left alone they drag
        a task's reward minimum to -9.2e18 and wreck every percentile that the
        selectivity sweep computes.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None if value is None else "not_a_number"
    number = float(value)
    if not math.isfinite(number):
        return "nonfinite"
    if number in _FITNESS_SENTINELS:
        return "sentinel"
    return None


def _fitness(metrics: dict[str, Any]) -> tuple[float | None, str | None]:
    """(usable reward, defect). A defective reward becomes None and is RECORDED."""
    value = metrics.get("combined_score")
    if value is None:
        return None, None
    defect = fitness_defect(value)
    return (None, defect) if defect else (float(value), None)


def strip_nul(text: str) -> str:
    """Remove NUL code points from a string.

    PostgreSQL `text` and `jsonb` cannot represent U+0000 at all — the load aborts with
    "unsupported Unicode escape sequence: \\u0000 cannot be converted to text". The
    corpus contains them inside captured stderr (truncated binary output landing in an
    evaluator message). Stripping is lossless for our purposes: a NUL carries no
    information a failure signature or a metrics blob needs.
    """
    return text.replace("\x00", "")


def json_safe(
    value: Any,
    _path: str = "",
    _nonfinite: list[str] | None = None,
    _nul: list[str] | None = None,
) -> Any:
    """Make a value storable: non-finite floats -> null, NUL code points stripped.

    Both defects come from the source data and both are invisible until three stages
    later, so they are fixed once, here, and RECORDED rather than silently swallowed:

    * bare `NaN`/`Infinity` are not JSON — PostgreSQL rejects them and so does any
      non-Python reader of our staging files;
    * U+0000 cannot exist in PostgreSQL text.
    """
    nonfinite = [] if _nonfinite is None else _nonfinite
    nul = [] if _nul is None else _nul
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(_path or "<root>")
        return None
    if isinstance(value, str):
        if "\x00" in value:
            nul.append(_path or "<root>")
            return strip_nul(value)
        return value
    if isinstance(value, dict):
        return {
            k: json_safe(v, f"{_path}.{k}" if _path else k, nonfinite, nul)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [json_safe(v, f"{_path}[]", nonfinite, nul) for v in value]
    return value


def _opt_strip(text: str | None) -> str | None:
    return None if text is None else strip_nul(text)


def _clean_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    nonfinite: list[str] = []
    nul: list[str] = []
    cleaned = json_safe(metrics, "", nonfinite, nul)
    return cleaned, nonfinite, nul


@dataclass
class SessionAccum:
    """Everything gathered for one run before it is written out."""

    run_rel: str
    parsed: RunName
    backend: str
    group: str | None
    domain: str
    uid: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    state_events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    files_present: dict[str, bool] = field(default_factory=dict)
    #: Problem statement recovered from this run's prompt blobs (None when none readable).
    specification: str | None = None


def task_specification(base: Path, prompts: list[dict[str, Any]]) -> str | None:
    """The problem statement the model actually saw, from a prompt blob's system message.

    The core files ship no canonical problem statement, but every generation prompt
    carries one: each blob is `{<message_kind>: {system, user, responses}}` and the
    `system` field holds the full spec the backend put in front of the model.

    Measured on the pinned revision: for all 18 tasks the system message is
    BYTE-IDENTICAL across every backend that recorded one, so "the first blob we can
    read" is not an arbitrary pick -- there is only one text per task. `normalize()`
    asserts that invariant across runs rather than trusting it.

    Returns None when this run published no readable blob; the caller then leaves the
    task incomplete rather than inventing a spec.
    """
    for prompt in prompts:
        if not prompt.get("blob_present"):
            continue
        blob = base / prompt["blob_rel"]
        try:
            payload = json.loads(blob.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for message in payload.values():
            if isinstance(message, dict) and message.get("system"):
                return str(message["system"])
    return None


def normalize_session(root: Path, run_rel: str, revision: str) -> SessionAccum:
    """Normalize one run directory. Raises RunNameError on an unregistered Task."""
    base = root / run_rel
    parts = run_rel.split("/")
    backend = parts[0]
    group = parts[1] if len(parts) == 3 else None
    name = parts[-1]

    parsed = parse_run_name(name)
    uid = session_uid(revision, run_rel)
    acc = SessionAccum(
        run_rel=run_rel,
        parsed=parsed,
        backend=backend,
        group=group,
        domain=parsed.domain,
        uid=uid,
    )

    meta_path = base / "meta.json"
    acc.meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    acc.files_present = {
        "meta.json": meta_path.is_file(),
        "programs.jsonl": (base / "programs.jsonl").is_file(),
        "iterations.jsonl": (base / "iterations.jsonl").is_file(),
        "iter_scalars.jsonl": (base / "iter_scalars.jsonl").is_file(),
        "logs/llm_calls.jsonl": (base / "logs/llm_calls.jsonl").is_file(),
        "run_info.json": (base / "run_info.json").is_file(),
        "run_config.yaml": (base / "run_config.yaml").is_file(),
        "evaluate.py": (base / "evaluate.py").is_file(),
        "blobs": (base / "blobs").is_dir(),
    }

    _load_programs(base, acc)
    _load_state_events(base, acc)
    _load_llm_calls(base, acc)
    acc.specification = task_specification(base, acc.prompts)
    return acc


def _load_programs(base: Path, acc: SessionAccum) -> None:
    """Programs, lineage, context edges, prompts. Parent resolution is fail-closed."""
    rows: list[tuple[int, dict[str, Any]]] = list(_read_jsonl(base / "programs.jsonl"))
    known: set[str] = {row.get("id") for _, row in rows if row.get("id")}

    for offset, row in rows:
        program_id = row.get("id")
        if not program_id:
            acc.rejects.append({"reason": "program_without_id", "row_offset": offset})
            continue
        raw_metrics = row.get("metrics") or {}
        metrics, nonfinite, nul = _clean_metrics(raw_metrics)
        fitness_value, fitness_flaw = _fitness(raw_metrics)
        status = row.get("status") or "unknown"
        uid = node_uid(acc.uid, program_id)
        solution_sha = row.get("solution_sha256")
        prompts_sha = row.get("prompts_sha256")
        metadata = row.get("metadata") or {}

        acc.nodes.append(
            {
                "node_uid": uid,
                "session_uid": acc.uid,
                "task_uid": acc.parsed.task_uid,
                "program_id": program_id,
                "iteration": row.get("iteration_found"),
                "generation": row.get("generation"),
                "status": status,
                # `accepted` is the backends' own admission decision. It is NOT a
                # correctness claim, and the raw status is kept beside it.
                "is_valid": status == "accepted",
                "language": row.get("language"),
                "fitness": fitness_value,
                "metrics": metrics,
                "nonfinite_metrics": nonfinite,
                "nul_stripped_metrics": nul,
                "fitness_defect": fitness_flaw,
                "error_signature": _opt_strip(error_signature(metrics, status)),
                "changes": _opt_strip(metadata.get("changes")),
                "artifact_uid": artifact_uid(solution_sha) if solution_sha else None,
                "solution_sha256": solution_sha,
                "prompts_sha256": prompts_sha,
                "source": row.get("source"),
                "row_offset": offset,
            }
        )

        parent_id = row.get("parent_id")
        if parent_id:
            if parent_id in known:
                acc.lineage.append(
                    {
                        "src_node_uid": node_uid(acc.uid, parent_id),
                        "dst_node_uid": uid,
                        "session_uid": acc.uid,
                        "relation": "has_child",
                        "provenance": "programs.jsonl:parent_id",
                        "row_offset": offset,
                    }
                )
            else:
                # A parent outside this session's namespace is not evidence of a
                # cross-session edit; it is an unresolved pointer. Reject it.
                acc.rejects.append(
                    {
                        "reason": "unresolved_parent_id",
                        "node_uid": uid,
                        "parent_id": parent_id,
                        "row_offset": offset,
                    }
                )

        # `other_context_ids` can name the same source twice. A graph edge is set
        # membership, so the duplicate is dropped HERE and counted, rather than being
        # eaten silently by the loader's primary key three stages later — where the
        # 13 missing edges would look like data loss with no explanation.
        seen_context: set[str] = set()
        for ctx_id in row.get("other_context_ids") or []:
            if ctx_id in seen_context:
                acc.rejects.append(
                    {"reason": "duplicate_context_id", "node_uid": uid, "context_id": ctx_id}
                )
                continue
            seen_context.add(ctx_id)
            if ctx_id in known and ctx_id != program_id:
                acc.context.append(
                    {
                        "src_node_uid": node_uid(acc.uid, ctx_id),
                        "dst_node_uid": uid,
                        "session_uid": acc.uid,
                        "relation": "context_for",
                        "provenance": "programs.jsonl:other_context_ids",
                        "row_offset": offset,
                    }
                )
            else:
                acc.rejects.append(
                    {
                        "reason": "unresolved_context_id",
                        "node_uid": uid,
                        "context_id": ctx_id,
                        "row_offset": offset,
                    }
                )

        if prompts_sha:
            blob = base / _blob_rel(prompts_sha, "json")
            acc.prompts.append(
                {
                    "prompt_uid": prompt_uid(uid, prompts_sha),
                    "node_uid": uid,
                    "session_uid": acc.uid,
                    "prompts_sha256": prompts_sha,
                    "blob_rel": _blob_rel(prompts_sha, "json"),
                    # `referenced` vs `present`: the reference always exists, the blob
                    # may not have been published. Never conflate the two.
                    "blob_present": blob.is_file(),
                }
            )


def _load_state_events(base: Path, acc: SessionAccum) -> None:
    """Checkpoint membership + scalars -> the logical-step change log.

    These are *search state*, not attempts: an archive membership row is not a new
    Program Node, and modelling it as one would inflate every node count.
    """
    for offset, row in _read_jsonl(base / "iterations.jsonl"):
        program_id = row.get("program_id")
        acc.state_events.append(
            {
                "session_uid": acc.uid,
                "iteration": row.get("iteration"),
                "entity": "membership",
                "role": row.get("role"),
                "slot_key": row.get("slot_key"),
                "program_id": program_id,
                "node_uid": node_uid(acc.uid, program_id) if program_id else None,
                "value": json_safe(row.get("value")),
                "provenance": "iterations.jsonl",
                "row_offset": offset,
            }
        )
    for offset, row in _read_jsonl(base / "iter_scalars.jsonl"):
        key = row.get("key")
        value = row.get("value")
        acc.state_events.append(
            {
                "session_uid": acc.uid,
                "iteration": row.get("iteration"),
                "entity": "scalar",
                "role": key,
                "slot_key": None,
                "program_id": value if key == "best_program_id" else None,
                "node_uid": (
                    node_uid(acc.uid, value)
                    if key == "best_program_id" and isinstance(value, str)
                    else None
                ),
                "value": json_safe(value),
                "provenance": "iter_scalars.jsonl",
                "row_offset": offset,
            }
        )


def _load_llm_calls(base: Path, acc: SessionAccum) -> None:
    """Per-session LLM usage.

    EvoTrace's ``llm_calls.jsonl`` carries no program id, so these aggregate at SESSION
    grain only. Joining them to nodes would be a fabrication; the schema reflects that.
    """
    for offset, row in _read_jsonl(base / "logs/llm_calls.jsonl"):
        acc.llm_calls.append(
            {
                "session_uid": acc.uid,
                "seq": offset,
                "model": row.get("model"),
                "api_base": row.get("api_base"),
                "temperature": row.get("temperature"),
                "finish_reason": row.get("finish_reason"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "reasoning_tokens": row.get("reasoning_tokens"),
                "error": row.get("error"),
                "provenance": "logs/llm_calls.jsonl",
                "row_offset": offset,
            }
        )


def _session_record(acc: SessionAccum) -> dict[str, Any]:
    valid = [n for n in acc.nodes if n["is_valid"] and n["fitness"] is not None]
    best = max(valid, key=lambda n: n["fitness"], default=None)
    prompt_complete = (
        len(acc.prompts) / len(acc.nodes) if acc.nodes else 0.0
    )
    return {
        "session_uid": acc.uid,
        "task_uid": acc.parsed.task_uid,
        "run_rel": acc.run_rel,
        "backend": acc.backend,
        "group": acc.group,
        "domain": acc.domain,
        "model": acc.parsed.model,
        "mode": acc.parsed.mode,
        "temperature": acc.parsed.temperature,
        "configured_iterations": acc.parsed.configured_iterations,
        "run_hash": acc.parsed.run_hash,
        "search_algorithm": acc.backend,
        "node_count": len(acc.nodes),
        "root_count": sum(1 for n in acc.nodes if n["iteration"] == 0)
        or sum(1 for n in acc.nodes if n["generation"] == 0),
        "lineage_edge_count": len(acc.lineage),
        "context_edge_count": len(acc.context),
        "state_event_count": len(acc.state_events),
        "llm_call_count": len(acc.llm_calls),
        "best_node_uid": best["node_uid"] if best else None,
        "best_fitness": best["fitness"] if best else None,
        "status_counts": dict(Counter(n["status"] for n in acc.nodes)),
        "prompt_complete_ratio": round(prompt_complete, 4),
        "meta_counts": acc.meta.get("counts", {}),
        "files_present": acc.files_present,
        # Wall clock is gone from the whole corpus. Say so on every session rather
        # than letting a null column read as "not captured yet".
        "wall_clock_available": False,
    }


def _readiness(acc: SessionAccum, session: dict[str, Any]) -> dict[str, Any]:
    """Per-run replay gates (EvoTraceDoc.md §E1).

    A run that fails a gate still enters the graph — it is only excluded from the
    denominator of the claim that gate protects.
    """
    core = all(
        acc.files_present[name]
        for name in (
            "meta.json",
            "programs.jsonl",
            "iterations.jsonl",
            "iter_scalars.jsonl",
            "logs/llm_calls.jsonl",
        )
    )
    reasons: list[str] = []
    if not core:
        reasons.append("missing_canonical_file")
    if session["prompt_complete_ratio"] <= 0:
        reasons.append("no_prompt_history")
    if not acc.files_present["evaluate.py"]:
        reasons.append("no_run_local_evaluator")
    if not acc.files_present["run_config.yaml"]:
        reasons.append("no_run_config")
    return {
        "session_uid": acc.uid,
        "run_rel": acc.run_rel,
        "core_graph_ready": core and bool(acc.nodes),
        "prompt_complete_ratio": session["prompt_complete_ratio"],
        "llm_log_ready": acc.files_present["logs/llm_calls.jsonl"],
        "evaluator_ref_present": acc.files_present["evaluate.py"],
        "config_present": acc.files_present["run_config.yaml"],
        "blobs_present": acc.files_present["blobs"],
        # R1/R2 need an evaluator and an environment we have NOT pinned yet. Nothing
        # here may be reported as replay-ready on file presence alone.
        "artifact_replay_ready": False,
        "same_prompt_replay_ready": False,
        "full_session_replay_ready": False,
        "failure_reason": reasons,
    }


def normalize(root: Path, revision: str = REVISION) -> dict[str, Any]:
    raw = root / "raw"
    out = root / "normalized"
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(raw)
    if not runs:
        raise SystemExit(f"no runs under {raw}; run tools/evotrace/download.py first")

    sessions: list[dict[str, Any]] = []
    readiness: list[dict[str, Any]] = []
    task_index: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    unparsed: list[dict[str, str]] = []
    all_rejects: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    writers = {
        name: (out / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in (
            "nodes",
            "lineage_edges",
            "context_edges",
            "prompts",
            "state_events",
            "llm_calls",
        )
    }
    try:
        for run in runs:
            try:
                acc = normalize_session(raw, run.rel, revision)
            except RunNameError as exc:
                # Fail closed per run: the corpus is still emitted, and the run is
                # named in the report instead of being bucketed into a fake Task.
                unparsed.append({"run": run.rel, "error": str(exc)})
                continue

            session = _session_record(acc)
            sessions.append(session)
            readiness.append(_readiness(acc, session))
            all_rejects.extend({"run": run.rel, **r} for r in acc.rejects)

            task = task_index.setdefault(
                acc.parsed.task_uid,
                {
                    "task_uid": acc.parsed.task_uid,
                    "domain": acc.parsed.domain,
                    "task_key": acc.parsed.task_key,
                    "task_family": acc.parsed.task_key.rstrip("0123456789-_") or acc.parsed.task_key,
                    "session_count": 0,
                    "node_count": 0,
                    "backends": [],
                    "models": [],
                    "languages": [],
                    # Filled from the first run of this task that published a
                    # readable prompt blob; see task_specification(). Stays incomplete
                    # when no run of the task published one.
                    "specification": None,
                    "specification_source": "run_name_only",
                    "specification_complete": False,
                },
            )
            if acc.specification is not None:
                if task["specification"] is None:
                    task["specification"] = acc.specification
                    task["specification_source"] = "prompt_system_message"
                    task["specification_complete"] = True
                elif task["specification"] != acc.specification:
                    # Measured invariant: on the pinned revision every backend of a
                    # task records a byte-identical system message. A mismatch means
                    # the assumption behind "any blob will do" has broken, so record
                    # it as a reject instead of silently picking one text.
                    all_rejects.append(
                        {
                            "run": run.rel,
                            "reason": "task_specification_divergence",
                            "task_uid": acc.parsed.task_uid,
                            "kept_chars": len(task["specification"]),
                            "seen_chars": len(acc.specification),
                        }
                    )
            task["session_count"] += 1
            task["node_count"] += len(acc.nodes)
            for key, value in (
                ("backends", acc.backend),
                ("models", acc.parsed.model),
            ):
                if value not in task[key]:
                    task[key].append(value)
            for node in acc.nodes:
                lang = node.get("language")
                if lang and lang not in task["languages"]:
                    task["languages"].append(lang)

            for node in acc.nodes:
                writers["nodes"].write(dumps(node) + "\n")
                totals["nodes"] += 1
                totals[f"status:{node['status']}"] += 1
                uid = node["artifact_uid"]
                if uid:
                    entry = artifacts.setdefault(
                        uid,
                        {
                            "artifact_uid": uid,
                            "sha256": node["solution_sha256"],
                            "kind": "solution",
                            "language": node.get("language"),
                            "blob_rel": _blob_rel(node["solution_sha256"], "txt"),
                            "reference_count": 0,
                            "sessions": [],
                        },
                    )
                    entry["reference_count"] += 1
                    if acc.uid not in entry["sessions"]:
                        entry["sessions"].append(acc.uid)
            for name, rows in (
                ("lineage_edges", acc.lineage),
                ("context_edges", acc.context),
                ("prompts", acc.prompts),
                ("state_events", acc.state_events),
                ("llm_calls", acc.llm_calls),
            ):
                for row in rows:
                    writers[name].write(dumps(row) + "\n")
                    totals[name] += 1
    finally:
        for handle in writers.values():
            handle.close()

    (out / "tasks.jsonl").write_text(
        "".join(dumps(t) + "\n" for t in sorted(task_index.values(), key=lambda t: t["task_uid"]))
    )
    (out / "sessions.jsonl").write_text("".join(dumps(s) + "\n" for s in sessions))
    (out / "artifacts.jsonl").write_text(
        "".join(dumps(a) + "\n" for a in artifacts.values())
    )
    (out / "replay_manifest.json").write_text(
        json.dumps({"revision": revision, "runs": readiness}, indent=2)
    )

    report = _integrity_report(
        revision=revision,
        tasks=list(task_index.values()),
        sessions=sessions,
        artifacts=artifacts,
        totals=totals,
        readiness=readiness,
        unparsed=unparsed,
        rejects=all_rejects,
    )
    (out / "integrity_report.json").write_text(json.dumps(report, indent=2))
    return report


def _integrity_report(
    *,
    revision: str,
    tasks: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    totals: Counter[str],
    readiness: list[dict[str, Any]],
    unparsed: list[dict[str, str]],
    rejects: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = {
        "tasks": len(tasks),
        "sessions": len(sessions),
        "nodes": totals["nodes"],
        "accepted": totals["status:accepted"],
        "rejected": totals["status:rejected"],
        "failed": totals["status:failed"],
        "lineage_edges": totals["lineage_edges"],
        "context_edges": totals["context_edges"],
        "prompts": totals["prompts"],
        "state_events": totals["state_events"],
        "llm_calls": totals["llm_calls"],
        "unique_artifacts": len(artifacts),
        "artifact_references": sum(a["reference_count"] for a in artifacts.values()),
    }
    domains = Counter(s["domain"] for s in sessions)
    backends = Counter(s["backend"] for s in sessions)

    comparisons = {
        key: {
            "observed": observed.get(key),
            "paper_claim": claim,
            "agrees": observed.get(key) == claim,
        }
        for key, claim in PAPER_CLAIMS.items()
    }

    base_entities = observed["tasks"] + observed["sessions"] + observed["nodes"]
    cross_session_artifacts = sum(
        1 for a in artifacts.values() if len(a["sessions"]) > 1
    )
    return {
        "revision": revision,
        "observed": observed,
        "base_entities": base_entities,
        "canonical_logical_edges": observed["sessions"]
        + observed["nodes"]
        + observed["lineage_edges"],
        "paper_comparison": comparisons,
        "domain_split": dict(domains),
        "backend_split": dict(backends),
        "core_graph_ready_runs": sum(1 for r in readiness if r["core_graph_ready"]),
        "runs_with_prompts": sum(1 for r in readiness if r["prompt_complete_ratio"] > 0),
        "runs_with_evaluator": sum(1 for r in readiness if r["evaluator_ref_present"]),
        # Leakage-relevant: the same code text appearing in more than one session is
        # exactly what a cross-session split must audit for.
        "artifacts_shared_across_sessions": cross_session_artifacts,
        "unparsed_runs": unparsed,
        "reject_counts": dict(Counter(r["reason"] for r in rejects)),
        "rejects_sample": rejects[:25],
        "notes": [
            "All EvoTrace wall-clock timestamps are null; ordering is by iteration only.",
            "llm_calls carry no program id and aggregate at SESSION grain only.",
            "Nodes are attempts and are never deduplicated; artifacts dedup by sha256.",
            "artifact_replay_ready / same_prompt_replay_ready are false everywhere "
            "until an evaluator and environment are pinned (P5 gate, not this stage).",
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    obs = report["observed"]
    print(f"revision        : {report['revision'][:8]}")
    print(f"base entities   : {report['base_entities']:,}")
    print(f"canonical edges : {report['canonical_logical_edges']:,}")
    print("")
    print(f"{'metric':22} {'observed':>10} {'paper':>10}  agrees")
    for key, row in report["paper_comparison"].items():
        claim = row["paper_claim"]
        mark = "yes" if row["agrees"] else "NO"
        print(f"{key:22} {str(row['observed']):>10} {str(claim):>10}  {mark}")
    print("")
    print(f"domains         : {report['domain_split']}")
    print(f"backends        : {report['backend_split']}")
    print(f"unique artifacts: {obs['unique_artifacts']:,} "
          f"(refs {obs['artifact_references']:,}, "
          f"shared across sessions {report['artifacts_shared_across_sessions']:,})")
    print(f"core-ready runs : {report['core_graph_ready_runs']}/{obs['sessions']}")
    print(f"runs w/ prompts : {report['runs_with_prompts']}/{obs['sessions']}")
    print(f"rejects         : {report['reject_counts'] or 'none'}")
    if report["unparsed_runs"]:
        print(f"UNPARSED RUNS   : {len(report['unparsed_runs'])}")
        for row in report["unparsed_runs"][:5]:
            print(f"  {row['run']}: {row['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true", help="print the stored report only")
    args = parser.parse_args(argv)

    report_path = args.root / "normalized" / "integrity_report.json"
    if args.check:
        if not report_path.is_file():
            print(f"no report at {report_path}; run without --check first", file=sys.stderr)
            return 2
        print_report(json.loads(report_path.read_text()))
        return 0

    report = normalize(args.root)
    print_report(report)
    return 0 if not report["unparsed_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
