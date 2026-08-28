"""Pinned MemoryArena dataset normalization and decision-point construction.

The public Hugging Face rows do not contain minimal evidence-edge annotations.  The
paper protocol does require every subquery to build on *all* prior subqueries, so this
module records all prior sessions as ``protocol_dependencies`` and labels that basis
explicitly.  They are valid Dependency Recall ground truth; they must not be described
as human-annotated minimal evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SUPPORTED_CONFIGS = frozenset(
    {"progressive_search", "formal_reasoning_math", "formal_reasoning_phys"}
)
SCHEMA_VERSION = "memoryarena_normalized_v0.1.0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256_bytes(encoded)


def extract_exact_answer(answer: str) -> str | None:
    """Extract the released Progressive Search answer's final short answer.

    The release is not fully normalized: some rows render Markdown emphasis before or
    after the colon.  The full released answer remains separately preserved.
    """
    values: list[str] = []
    for line in answer.splitlines():
        # The release contains both ``Exact Answer: x`` and
        # ``**Exact Answer:** **x**``.  Removing emphasis markers before parsing
        # avoids treating Markdown presentation as part of the gold answer.
        plain = line.replace("**", "").strip()
        label, separator, value = plain.partition(":")
        if separator and label.strip().casefold() == "exact answer":
            value = value.strip()
            if value:
                values.append(value)
    return values[-1] if values else None


@dataclass(frozen=True)
class MemoryArenaSession:
    task_uid: str
    session_uid: str
    ordinal: int
    question: str
    gold_answer: str
    gold_exact_answer: str | None
    background: str
    protocol_dependencies: tuple[str, ...]
    dependency_basis: str

    @property
    def cutoff_ordinal(self) -> int:
        """The first unavailable ordinal for this session's retrieval."""
        return self.ordinal


@dataclass(frozen=True)
class MemoryArenaTask:
    task_uid: str
    released_id: int
    config: str
    paper_name: str | None
    sessions: tuple[MemoryArenaSession, ...]

    def decision_sessions(self) -> tuple[MemoryArenaSession, ...]:
        """Every non-initial session has historical state it may retrieve."""
        return self.sessions[1:]


@dataclass(frozen=True)
class MemoryArenaCorpus:
    config: str
    source_dataset: str
    source_revision: str
    source_file_sha256: str
    manifest_sha256: str
    tasks: tuple[MemoryArenaTask, ...]
    dependency_annotation: str

    @property
    def session_count(self) -> int:
        return sum(len(task.sessions) for task in self.tasks)

    @property
    def decision_count(self) -> int:
        return sum(max(0, len(task.sessions) - 1) for task in self.tasks)

    def iter_decision_sessions(self) -> Iterator[MemoryArenaSession]:
        for task in self.tasks:
            yield from task.decision_sessions()


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return value


def _backgrounds(row: Mapping[str, Any], count: int) -> list[str]:
    raw = row.get("backgrounds")
    if raw is None:
        return [""] * count
    values = _strings(raw, "backgrounds")
    if len(values) != count:
        raise ValueError(
            f"background count {len(values)} does not match question count {count}"
        )
    return values


def _task_uid(config: str, released_id: int) -> str:
    return f"memoryarena:{config}:task:{released_id}"


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: str,
    source_dataset: str = "ZexueHe/memoryarena",
    source_revision: str = "unknown",
    source_file_sha256: str = "unknown",
    manifest_sha256: str = "unknown",
) -> MemoryArenaCorpus:
    if config not in SUPPORTED_CONFIGS:
        raise ValueError(f"unsupported MemoryArena config: {config}")

    tasks: list[MemoryArenaTask] = []
    seen_ids: set[int] = set()
    for row in rows:
        released_id = row.get("id")
        if not isinstance(released_id, int) or isinstance(released_id, bool):
            raise ValueError("id must be an integer")
        if released_id in seen_ids:
            raise ValueError(f"duplicate released id: {released_id}")
        seen_ids.add(released_id)

        questions = _strings(row.get("questions"), "questions")
        answers = _strings(row.get("answers"), "answers")
        if not questions:
            raise ValueError(f"task {released_id} has no sessions")
        if len(questions) != len(answers):
            raise ValueError(
                f"task {released_id}: {len(questions)} questions != "
                f"{len(answers)} answers"
            )
        backgrounds = _backgrounds(row, len(questions))
        task_uid = _task_uid(config, released_id)
        session_uids = tuple(
            f"{task_uid}:session:{ordinal}" for ordinal in range(len(questions))
        )
        sessions = tuple(
            MemoryArenaSession(
                task_uid=task_uid,
                session_uid=session_uids[ordinal],
                ordinal=ordinal,
                question=question,
                gold_answer=answers[ordinal],
                gold_exact_answer=(
                    extract_exact_answer(answers[ordinal])
                    if config == "progressive_search"
                    else None
                ),
                background=backgrounds[ordinal],
                protocol_dependencies=session_uids[:ordinal],
                dependency_basis="paper_protocol_all_prior_sessions",
            )
            for ordinal, question in enumerate(questions)
        )
        paper_name = row.get("paper_name")
        if paper_name is not None and not isinstance(paper_name, str):
            raise ValueError("paper_name must be a string when present")
        tasks.append(
            MemoryArenaTask(
                task_uid=task_uid,
                released_id=released_id,
                config=config,
                paper_name=paper_name,
                sessions=sessions,
            )
        )

    tasks.sort(key=lambda task: task.released_id)
    return MemoryArenaCorpus(
        config=config,
        source_dataset=source_dataset,
        source_revision=source_revision,
        source_file_sha256=source_file_sha256,
        manifest_sha256=manifest_sha256,
        tasks=tuple(tasks),
        dependency_annotation=(
            "protocol-required all-prior context; not minimal human evidence edges"
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def load_export(path: str | Path) -> MemoryArenaCorpus:
    """Load one fetched config directory and verify its content-addressed manifest."""
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "memoryarena_fetch_v0.1.0":
        raise ValueError("unsupported MemoryArena fetch manifest")
    data_path = directory / str(manifest["data_file"])
    observed = sha256_bytes(data_path.read_bytes())
    if observed != manifest.get("data_sha256"):
        raise ValueError(
            f"MemoryArena data checksum mismatch: {observed} != "
            f"{manifest.get('data_sha256')}"
        )
    manifest_without_digest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest_sha = canonical_json_sha256(manifest_without_digest)
    if manifest_sha != manifest.get("manifest_sha256"):
        raise ValueError("MemoryArena manifest checksum mismatch")
    rows = _read_jsonl(data_path)
    if len(rows) != manifest.get("rows"):
        raise ValueError("MemoryArena row count does not match manifest")
    return normalize_rows(
        rows,
        config=str(manifest["config"]),
        source_dataset=str(manifest["dataset"]),
        source_revision=str(manifest["resolved_revision"]),
        source_file_sha256=observed,
        manifest_sha256=manifest_sha,
    )
