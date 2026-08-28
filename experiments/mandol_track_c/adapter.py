"""Mandol main-branch adapter for the controlled-model Track C comparison.

This adapter deliberately lives outside ``bench/agent_memory`` while the
four-system snapshot sweep is active.  The shared protocol hashes that tree;
editing it during a canonical build would make the remaining receipts use a
different benchmark hash.

Mandol keeps its serving state in memory.  ``ingest_history`` therefore builds
one official ``MemorySystem`` per LoCoMo conversation and persists it with
``MemorySystem.save``.  Search processes reconstruct those systems with
``MemorySystem.load``.  The persisted directory is the canonical artifact and
can be copied/reflinked for any number of QPS points without rebuilding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.table5_track_c.dataset import EventItem, LoCoMoCorpus, QueryItem
from bench.agent_memory.table5_track_c.instrumentation import (
    instrument_method,
    stage_span,
)

_EVIDENCE_ID = re.compile(r"\bD\d+:\d+\b")
_MANIFEST = "snapshot_manifest.json"
_SNAPSHOT_SCHEMA = "mandol_track_c_snapshot_v0.1.0"


@dataclass(frozen=True)
class MandolConfig:
    snapshot_dir: str
    dataset_path: str
    answer_base_url: str = "http://127.0.0.1:8010/v1"
    answer_model: str = "Qwen/Qwen3-32B"
    embedding_base_url: str = "http://127.0.0.1:8011/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    reranker_base_url: str = "http://127.0.0.1:8012"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    request_timeout_seconds: int = 60
    mandol_source_root: str = "/localhome/hza214/Mandol"
    resume_snapshot_source: str | None = None
    resume_expected_samples: tuple[str, ...] = ()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == _MANIFEST:
            continue
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        size = path.stat().st_size
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        files += 1
        total_bytes += size
    return digest.hexdigest(), files, total_bytes


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MandolAdapter:
    """Official Mandol ``MemorySystem`` with a persistent build boundary."""

    name = "mandol"

    def __init__(self, config: MandolConfig) -> None:
        self.config = config
        self.snapshot_dir = Path(config.snapshot_dir).resolve()
        self.dataset_path = Path(config.dataset_path).resolve()
        self._systems: dict[str, Any] = {}
        self._systems_lock = threading.Lock()
        self._build_stats: dict[str, Any] = {}
        self._stage_tracing = False
        self._stage_instrumentation: dict[str, bool] = {}

    def enable_stage_tracing(self) -> None:
        self._stage_tracing = True

    def _adapter_sha256(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _source_identity(self) -> dict[str, Any]:
        root = Path(self.config.mandol_source_root)
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {
            "root": str(root.resolve()),
            "commit": commit,
            "tracked_dirty": tracked_dirty,
            "version": "0.1.0",
        }

    def _providers(self) -> tuple[Any, Any, Any]:
        from mandol.infrastructure.openai_compatible_embedding_provider import (
            OpenAICompatibleEmbeddingConfig,
            OpenAICompatibleEmbeddingProvider,
        )
        from mandol.infrastructure.openai_compatible_llm_provider import (
            OpenAICompatibleLLMProvider,
        )
        from mandol.infrastructure.openai_compatible_reranker import (
            OpenAICompatibleRerankConfig,
            OpenAICompatibleReranker,
        )

        embedder = OpenAICompatibleEmbeddingProvider(
            model=self.config.embedding_model,
            dim=self.config.embedding_dim,
            token="EMPTY",
            config=OpenAICompatibleEmbeddingConfig(
                base_url=self.config.embedding_base_url,
                api_path="/embeddings",
                timeout_s=self.config.request_timeout_seconds,
            ),
        )
        reranker = OpenAICompatibleReranker(
            model=self.config.reranker_model,
            token="EMPTY",
            config=OpenAICompatibleRerankConfig(
                base_url=self.config.reranker_base_url,
                api_path="/v1/rerank",
                timeout_s=self.config.request_timeout_seconds,
            ),
        )
        llm = OpenAICompatibleLLMProvider(
            model=self.config.answer_model,
            base_url=self.config.answer_base_url,
            api_key="EMPTY",
            timeout_s=self.config.request_timeout_seconds,
        )
        if self._stage_tracing:
            targets = (
                (
                    embedder,
                    "embed_text",
                    "embedding",
                    "mandol.embedding.embed_text",
                    "http_client",
                ),
                (
                    reranker,
                    "rerank",
                    "fusion",
                    "mandol.reranker.rerank",
                    "http_client",
                ),
                (
                    llm,
                    "chat",
                    "llm",
                    "mandol.llm.chat",
                    "http_client",
                ),
            )
            for target, method, category, operation, call_kind in targets:
                self._stage_instrumentation[operation] = instrument_method(
                    target,
                    method,
                    category,
                    operation,
                    backend=(
                        self.config.embedding_model
                        if category == "embedding"
                        else self.config.reranker_model
                        if operation == "mandol.reranker.rerank"
                        else self.config.answer_model
                    ),
                    attributes={"observable_call_kind": call_kind},
                )
        return embedder, reranker, llm

    def _memory_config(self) -> Any:
        from mandol.application.memory_system import MemorySystemConfig

        return MemorySystemConfig(
            embedder_model=self.config.embedding_model,
            embedder_dim=self.config.embedding_dim,
            reranker_model=self.config.reranker_model,
            llm_model=self.config.answer_model,
            similarity_top_k=5,
            similarity_threshold=0.7,
            similarity_recent_window=20,
            bfs_expansion_per_seed=3,
            bfs_expansion_hops=1,
            max_context_units=20,
            use_remote_embedder=True,
            use_remote_reranker=True,
            embedder_remote_base_url=self.config.embedding_base_url,
            embedder_remote_api_path="/embeddings",
            embedder_remote_timeout=self.config.request_timeout_seconds,
            reranker_remote_base_url=self.config.reranker_base_url,
            reranker_remote_api_path="/v1/rerank",
            reranker_remote_timeout=self.config.request_timeout_seconds,
        )

    def _new_system(self, sample_id: str) -> Any:
        from mandol import MemorySystem

        embedder, reranker, llm = self._providers()
        return MemorySystem(
            config=self._memory_config(),
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm,
            root=sample_id,
        )

    def _load_system(self, sample_id: str) -> Any:
        from mandol import MemorySystem

        embedder, reranker, llm = self._providers()
        return MemorySystem.load(
            str(self.snapshot_dir / sample_id),
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm,
        )

    def _scope(self, sample_id: str) -> Any:
        """Create one isolated conversation scope, once, under concurrent Add."""
        with self._systems_lock:
            system = self._systems.get(sample_id)
            if system is None:
                system = self._new_system(sample_id)
                self._systems[sample_id] = system
            return system

    def init_schema(self) -> dict[str, Any]:
        expected_dataset = _sha256_file(self.dataset_path)
        source = self._source_identity()
        if source["tracked_dirty"]:
            raise RuntimeError("Mandol tracked source is dirty; refusing benchmark run")
        manifest_path = self.snapshot_dir / _MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != _SNAPSHOT_SCHEMA:
                raise RuntimeError("Mandol snapshot schema mismatch")
            if manifest.get("dataset_sha256") != expected_dataset:
                raise RuntimeError("Mandol snapshot dataset checksum mismatch")
            if manifest.get("adapter_sha256") != self._adapter_sha256():
                raise RuntimeError("Mandol adapter changed after canonical build")
            if manifest.get("mandol_source", {}).get("commit") != source["commit"]:
                raise RuntimeError("Mandol source commit changed after canonical build")
            return {
                "mode": "load_persisted",
                "snapshot_dir": str(self.snapshot_dir),
                "manifest": str(manifest_path),
            }
        resume_source = (
            Path(self.config.resume_snapshot_source).resolve()
            if self.config.resume_snapshot_source
            else None
        )
        if resume_source is not None:
            if resume_source == self.snapshot_dir:
                raise RuntimeError("Mandol resume source and target must be different")
            if not resume_source.is_dir():
                raise RuntimeError(
                    f"Mandol resume source does not exist: {resume_source}"
                )
            if (resume_source / _MANIFEST).exists():
                raise RuntimeError(
                    "Mandol resume source is already a complete canonical snapshot"
                )
            if not self.config.resume_expected_samples:
                raise RuntimeError("Mandol resume expected-sample set is empty")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if any(self.snapshot_dir.iterdir()):
            raise RuntimeError(
                f"Mandol snapshot is non-empty without a manifest: {self.snapshot_dir}"
            )
        if resume_source is not None:
            return {
                "mode": "resume_verified_partial",
                "snapshot_dir": str(self.snapshot_dir),
                "resume_source": str(resume_source),
                "expected_reused_samples": sorted(self.config.resume_expected_samples),
            }
        return {
            "mode": "fresh_build",
            "snapshot_dir": str(self.snapshot_dir),
        }

    @staticmethod
    def _session_number(item: EventItem) -> int:
        value = item.metadata.get("session_number")
        if value is not None:
            return int(value)
        match = re.fullmatch(r"S(\d+)", item.session_id)
        return int(match.group(1)) if match else 0

    def _write_events(self, system: Any, events: Sequence[EventItem]) -> dict[str, int]:
        from mandol.domain.memory_unit import MemoryUnit
        from mandol.domain.types import SpaceName, Uid

        if not events:
            return {"base_units": 0, "temporal_edges": 0, "embedded_units": 0}
        sample_id = events[0].sample_id
        semantic_map = system.graph.semantic_map
        input_space = semantic_map.create_space(SpaceName(sample_id))
        base_space = semantic_map.create_space(SpaceName(f"{sample_id}_base_memory"))
        semantic_map.attach_child_space(
            input_space.name, base_space.name, ensure_exists=True
        )

        grouped: dict[int, list[EventItem]] = defaultdict(list)
        for item in events:
            if item.sample_id != sample_id:
                raise ValueError(
                    "events from different samples reached one Mandol graph"
                )
            grouped[self._session_number(item)].append(item)

        previous_uid: str | None = None
        temporal_edges = 0
        for session_number in sorted(grouped):
            session_space = semantic_map.create_space(
                SpaceName(f"{sample_id}_session_{session_number}")
            )
            semantic_map.attach_child_space(
                base_space.name, session_space.name, ensure_exists=True
            )
            for item in sorted(
                grouped[session_number], key=lambda event: event.ordinal
            ):
                uid = f"{sample_id}_dialogue_{item.event_id}"
                unit = MemoryUnit(
                    uid=Uid(uid),
                    raw_data={
                        "type": "dialogue",
                        "dia_id": item.event_id,
                        "speaker": item.role,
                        "text": item.text,
                        "text_content": item.text,
                        "session_datetime": item.timestamp,
                    },
                    metadata={
                        **item.metadata,
                        "unit_type": "dialogue",
                        "sample_id": sample_id,
                        "session_number": session_number,
                        "dia_id": item.event_id,
                        "speaker": item.role,
                        "timestamp": item.timestamp,
                    },
                    embedding=None,
                )
                system.graph.add_unit(
                    unit,
                    space_names=[session_space.name, base_space.name],
                    ensure_embedding=False,
                )
                if previous_uid is not None:
                    system.graph.add_relationship(previous_uid, uid, "PRECEDES")
                    system.graph.add_relationship(uid, previous_uid, "FOLLOWS")
                    temporal_edges += 2
                previous_uid = uid

        embedded = semantic_map.batch_embed_unembedded(batch_size=64)
        return {
            "base_units": len(events),
            "temporal_edges": temporal_edges,
            "embedded_units": int(embedded),
        }

    @staticmethod
    def _system_counts(system: Any) -> dict[str, int]:
        return {
            "units": len(system.semantic_map.list_units()),
            "spaces": len(system.semantic_map.list_spaces()),
            "edges": len(system.graph.get_graph_store().get_all_edges()),
        }

    def _validate_resume_sample(
        self,
        sample_dir: Path,
        sample_id: str,
        expected_events: Sequence[EventItem],
    ) -> dict[str, Any]:
        required = (
            "manifest.json",
            "config.json",
            "data/units.json",
            "data/spaces.json",
            "data/graph.json",
            "data/sessions.json",
            "state/processed_state.json",
        )
        missing = [name for name in required if not (sample_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Mandol resume sample {sample_id} is missing files: {missing}"
            )

        manifest = json.loads(
            (sample_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("version") != "1.0":
            raise RuntimeError(f"Mandol resume sample {sample_id} version mismatch")
        if manifest.get("build_state") != "complete":
            raise RuntimeError(f"Mandol resume sample {sample_id} is not complete")
        if manifest.get("pending_session_ids") != []:
            raise RuntimeError(f"Mandol resume sample {sample_id} has pending sessions")

        config_document = json.loads(
            (sample_dir / "config.json").read_text(encoding="utf-8")
        )
        saved = config_document.get("config", {})
        memory = saved.get("memory_system_config", {})
        expected_config = {
            "root": sample_id,
            "embedder_model": self.config.embedding_model,
            "embedder_dim": self.config.embedding_dim,
            "reranker_model": self.config.reranker_model,
            "llm_model": self.config.answer_model,
        }
        observed_config = {
            "root": saved.get("root"),
            "embedder_model": memory.get("embedder_model"),
            "embedder_dim": memory.get("embedder_dim"),
            "reranker_model": memory.get("reranker_model"),
            "llm_model": memory.get("llm_model"),
        }
        if observed_config != expected_config:
            raise RuntimeError(
                f"Mandol resume sample {sample_id} config mismatch: "
                f"{observed_config!r} != {expected_config!r}"
            )

        units_document = json.loads(
            (sample_dir / "data/units.json").read_text(encoding="utf-8")
        )
        units = units_document.get("units")
        if not isinstance(units, list):
            raise RuntimeError(f"Mandol resume sample {sample_id} units are invalid")
        dialogue_ids = [
            str(unit.get("raw_data", {}).get("dia_id"))
            for unit in units
            if unit.get("raw_data", {}).get("type") == "dialogue"
        ]
        expected_ids = [event.event_id for event in expected_events]
        if len(dialogue_ids) != len(set(dialogue_ids)):
            raise RuntimeError(
                f"Mandol resume sample {sample_id} has duplicate dialogue ids"
            )
        if set(dialogue_ids) != set(expected_ids):
            raise RuntimeError(
                f"Mandol resume sample {sample_id} does not match LoCoMo events"
            )
        stats = manifest.get("stats", {})
        if stats.get("unit_count") != len(units):
            raise RuntimeError(f"Mandol resume sample {sample_id} unit count mismatch")
        digest, files, total_bytes = _tree_digest(sample_dir)
        return {
            "source_sha256": digest,
            "source_files": files,
            "source_bytes": total_bytes,
            "base_units": len(dialogue_ids),
            "saved_stats": stats,
        }

    def _seed_resume_snapshot(
        self, grouped: dict[str, list[EventItem]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source = Path(str(self.config.resume_snapshot_source)).resolve()
        expected = tuple(sorted(set(self.config.resume_expected_samples)))
        if len(expected) != len(self.config.resume_expected_samples):
            raise RuntimeError("Mandol resume expected samples contain duplicates")
        unknown = sorted(set(expected) - set(grouped))
        if unknown:
            raise RuntimeError(
                f"Mandol resume samples are absent from LoCoMo: {unknown}"
            )
        observed = sorted(path.name for path in source.iterdir() if path.is_dir())
        root_files = sorted(path.name for path in source.iterdir() if path.is_file())
        if observed != list(expected) or root_files:
            raise RuntimeError(
                "Mandol resume source set mismatch: "
                f"directories={observed!r}, files={root_files!r}, "
                f"expected={list(expected)!r}"
            )

        validations = {
            sample_id: self._validate_resume_sample(
                source / sample_id, sample_id, grouped[sample_id]
            )
            for sample_id in expected
        }
        source_sha256, source_files, source_bytes = _tree_digest(source)
        reports: dict[str, Any] = {}
        for sample_id in expected:
            target = self.snapshot_dir / sample_id
            shutil.copytree(source / sample_id, target)
            system = self._load_system(sample_id)
            self._systems[sample_id] = system
            counts = self._system_counts(system)
            saved_stats = validations[sample_id]["saved_stats"]
            expected_counts = {
                "units": saved_stats.get("unit_count"),
                "spaces": saved_stats.get("space_count"),
                "edges": saved_stats.get("edge_count"),
            }
            if counts != expected_counts:
                raise RuntimeError(
                    f"Mandol copied resume sample {sample_id} count mismatch: "
                    f"{counts!r} != {expected_counts!r}"
                )
            reports[sample_id] = {
                "origin": "verified_partial_snapshot",
                "ingest": {
                    "base_units": validations[sample_id]["base_units"],
                    "reused": True,
                },
                "high_level": {"status": "reused_complete"},
                "save": {
                    "source_sha256": validations[sample_id]["source_sha256"],
                    "source_files": validations[sample_id]["source_files"],
                    "source_bytes": validations[sample_id]["source_bytes"],
                },
                "counts": counts,
            }
        provenance = {
            "policy": "copy validated complete samples; build missing samples",
            "source": str(source),
            "source_sha256": source_sha256,
            "source_files": source_files,
            "source_bytes": source_bytes,
            "reused_samples": list(expected),
        }
        return reports, provenance

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        if (self.snapshot_dir / _MANIFEST).exists():
            raise RuntimeError("refusing to rebuild an existing Mandol snapshot")
        started = time.perf_counter()
        grouped: dict[str, list[EventItem]] = defaultdict(list)
        for item in events:
            grouped[item.sample_id].append(item)

        reports: dict[str, Any] = {}
        resume_provenance = None
        if self.config.resume_snapshot_source:
            reports, resume_provenance = self._seed_resume_snapshot(grouped)
        reused_samples = set(reports)
        for sample_id in sorted(grouped):
            if sample_id in reports:
                continue
            system = self._new_system(sample_id)
            self._systems[sample_id] = system
            ingest = self._write_events(system, grouped[sample_id])
            build_report = system.build_high_level(mode="auto")
            if build_report.status not in {"completed", "no_units"}:
                raise RuntimeError(
                    f"Mandol high-level build failed for {sample_id}: "
                    f"{build_report.error_message}"
                )
            save_result = system.save(str(self.snapshot_dir / sample_id))
            reports[sample_id] = {
                "ingest": ingest,
                "high_level": _jsonable(build_report),
                "save": _jsonable(save_result),
                "counts": self._system_counts(system),
            }
        self._build_stats = {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "samples": len(grouped),
            "sample_reports": reports,
            "resume": {
                **resume_provenance,
                "built_samples": sorted(set(grouped) - reused_samples),
            }
            if resume_provenance
            else None,
        }
        return self._build_stats

    def finalize_build(self) -> dict[str, Any]:
        if not self._build_stats:
            manifest = json.loads(
                (self.snapshot_dir / _MANIFEST).read_text(encoding="utf-8")
            )
            return {"reused_manifest": True, **manifest["totals"]}

        content_sha256, files, total_bytes = _tree_digest(self.snapshot_dir)
        totals = {
            "units": sum(
                report["counts"]["units"]
                for report in self._build_stats["sample_reports"].values()
            ),
            "spaces": sum(
                report["counts"]["spaces"]
                for report in self._build_stats["sample_reports"].values()
            ),
            "edges": sum(
                report["counts"]["edges"]
                for report in self._build_stats["sample_reports"].values()
            ),
            "files": files,
            "bytes": total_bytes,
        }
        manifest = {
            "schema": _SNAPSHOT_SCHEMA,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": _sha256_file(self.dataset_path),
            "content_sha256": content_sha256,
            "adapter_sha256": self._adapter_sha256(),
            "mandol_source": self._source_identity(),
            "models": {
                "answer": self.config.answer_model,
                "embedding": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "reranker": self.config.reranker_model,
            },
            "resume": self._build_stats.get("resume"),
            "samples": sorted(self._build_stats["sample_reports"]),
            "totals": totals,
        }
        _atomic_json(self.snapshot_dir / _MANIFEST, manifest)
        return totals

    def prepare_reused_build(self, corpus: LoCoMoCorpus) -> None:
        for sample_id in corpus.sample_ids:
            if sample_id not in self._systems:
                self._systems[sample_id] = self._load_system(sample_id)

    @staticmethod
    def _hit_evidence(hit: Any) -> tuple[list[str], str]:
        unit = hit.unit
        raw = dict(getattr(unit, "raw_data", {}) or {})
        metadata = dict(getattr(unit, "metadata", {}) or {})
        text = str(
            raw.get("text_content")
            or raw.get("content")
            or raw.get("summary")
            or raw.get("text")
            or ""
        )
        ids: list[str] = []
        for value in (
            raw.get("dia_id"),
            metadata.get("dia_id"),
            metadata.get("source_turns"),
            raw.get("source_turns"),
        ):
            if isinstance(value, (list, tuple)):
                ids.extend(str(item) for item in value)
            elif value:
                ids.append(str(value))
        ids.extend(_EVIDENCE_ID.findall(text))
        ids.extend(_EVIDENCE_ID.findall(str(unit.uid)))
        return list(dict.fromkeys(ids)), text

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        system = self._systems.get(item.sample_id)
        if system is None:
            raise RuntimeError(
                f"Mandol sample {item.sample_id} was not loaded before timed search"
            )
        with stage_span(
            "fusion",
            "mandol.holistic_retrieve",
            backend="mandol_main_0.1.0_in_memory_semantic_map_graph",
            attributes={
                "includes": ["embedding", "dense", "bm25", "sparse", "graph", "rerank"],
                "attribution": "compound_native_api",
            },
        ):
            hits = system.holistic_retrieve(
                item.question,
                top_k=top_k,
                use_rerank=True,
                auto_build_if_empty=False,
            )
        hit_ids: list[str] = []
        contexts: list[str] = []
        memory_ids: list[str] = []
        for hit in hits:
            ids, text = self._hit_evidence(hit)
            hit_ids.extend(ids)
            contexts.append(text)
            memory_ids.append(str(hit.unit.uid))
        return {
            "result_count": len(hits),
            "hit_ids": list(dict.fromkeys(hit_ids)),
            "memory_ids": memory_ids,
            "contexts": contexts,
            "context_tokens": None,
            "empty": not hits,
            "mode": "holistic_retrieve",
        }

    def add(self, item: EventItem, *, visibility: bool) -> dict[str, Any]:
        from mandol.domain.memory_unit import MemoryUnit
        from mandol.domain.types import Uid

        system = self._scope(item.sample_id)
        unit = MemoryUnit(
            uid=Uid(item.event_id),
            raw_data={
                "type": "dialogue",
                "dia_id": item.event_id,
                "speaker": item.role,
                "text": item.text,
                "text_content": item.text,
                "session_datetime": item.timestamp,
            },
            metadata={
                **item.metadata,
                "unit_type": "dialogue",
                "sample_id": item.sample_id,
                "session_id": item.session_id,
                "dia_id": item.event_id,
                "speaker": item.role,
                "timestamp": item.timestamp,
            },
            embedding=None,
        )
        with stage_span(
            "framework_other",
            "mandol.memory_system.add",
            backend="mandol_main_0.1.0_in_memory_semantic_map_graph",
            attributes={
                "includes": ["chunking", "embedding", "vector", "graph"],
                "excludes": ["asynchronous_high_level_memory_completion"],
                "attribution": "native_foreground_api",
            },
        ):
            system.add(unit)
        committed_at_ns = time.perf_counter_ns()
        visible = None
        visibility_error = None
        searchable_at_ns = committed_at_ns
        if visibility:
            try:
                probe = QueryItem(
                    sample_id=item.sample_id,
                    question_id=f"probe:{item.event_id}",
                    question=item.text,
                    answer="",
                    category="probe",
                    evidence_ids=(item.event_id,),
                    ordinal=0,
                )
                visible = item.event_id in self.search(probe, top_k=35)["hit_ids"]
            except Exception as exc:  # noqa: BLE001 - measured probe evidence
                visible = False
                visibility_error = f"{type(exc).__name__}: {exc}"
            searchable_at_ns = time.perf_counter_ns() if visible else None
        return {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": searchable_at_ns,
            "visibility_probe": visible,
            "visibility_error": visibility_error,
            "native_add_is_intrinsically_searchable": True,
            "foreground_boundary": (
                "chunking+embedding+vector-index+immediate-similarity-edges"
            ),
            "asynchronous_high_level_memory_completion_included": False,
        }

    def stats(self) -> dict[str, Any]:
        manifest_path = self.snapshot_dir / _MANIFEST
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        runtime_samples = {
            sample_id: {
                **self._system_counts(system),
                "pending_units": len(getattr(system, "_pending_units", [])),
                "processed_sessions": len(
                    getattr(system, "_processed_session_ids", set())
                ),
                "llm_token_usage": system.get_token_usage(),
            }
            for sample_id, system in sorted(self._systems.items())
        }
        return {
            "config": asdict(self.config),
            "adapter_sha256": self._adapter_sha256(),
            "mandol_source": self._source_identity(),
            "loaded_samples": sorted(self._systems),
            "snapshot_manifest": manifest,
            "runtime_samples": runtime_samples,
            "native_shape": "in-memory SemanticMap + SemanticGraph, JSON snapshot",
            "native_add_boundary": (
                "foreground add is searchable on return; asynchronous high-level "
                "memory completion is excluded"
            ),
            "stage_instrumentation": self._stage_instrumentation,
            "query_llm_policy": (
                "observed via mandol.llm.chat span; expected call count is zero"
            ),
        }

    def snapshot_fingerprint(self) -> dict[str, Any]:
        manifest_path = self.snapshot_dir / _MANIFEST
        if not manifest_path.exists():
            raise RuntimeError("Mandol snapshot manifest does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "schema": manifest["schema"],
            "dataset_sha256": manifest["dataset_sha256"],
            "content_sha256": manifest["content_sha256"],
            "adapter_sha256": manifest["adapter_sha256"],
            "mandol_commit": manifest["mandol_source"]["commit"],
            "samples": len(manifest["samples"]),
            **manifest["totals"],
        }

    def close(self) -> None:
        for system in self._systems.values():
            executor = getattr(system, "_executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
        self._systems.clear()
