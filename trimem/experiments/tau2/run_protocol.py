"""Immutable effective protocol for opt-in controlled τ² reruns."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[3]
MARKER = ".run_protocol.json"
TRANSPORT_MARKER = ".transport_protocol.json"
TRANSPORT_FILES = {
    "trimem/experiments/tau2/runner.py",
    "trimem/experiments/tau2/run_protocol.py",
    "trimem/experiments/tau2/controlled_compare.py",
    "trimem/experiments/tau2/semantic_pool.py",
    "trimem/experiments/semantic_rubric.py",
}


def validate_transport_change(base: dict, current: dict, authorization: dict) -> None:
    """Bind reviewed source bytes and reject non-transport setting changes.

    Prompt/algorithm equivalence of the source patch requires the recorded code review.
    """
    for record in (base, current):
        if digest(record["protocol"]) != record["sha256"]:
            raise ValueError("invalid protocol digest")
    if authorization.get("base_sha256") != base["sha256"] or not authorization.get(
        "summary"
    ):
        raise ValueError(
            "transport authorization needs the base digest and review summary"
        )
    before = json.loads(json.dumps(base["protocol"]))
    after = json.loads(json.dumps(current["protocol"]))
    if (
        before["semantic_base"] != "episode-endpoint"
        or after["semantic_base"] != "request-load-v1"
    ):
        raise ValueError("unsupported semantic routing transition")
    changed = {
        name: {
            "old": before["source_hashes"].get(name),
            "new": after["source_hashes"].get(name),
        }
        for name in before["source_hashes"].keys() | after["source_hashes"].keys()
        if before["source_hashes"].get(name) != after["source_hashes"].get(name)
    }
    if (
        changed != authorization.get("changed_sources")
        or not set(changed) <= TRANSPORT_FILES
    ):
        raise ValueError("unreviewed transport source changes")
    normalized = {
        **after,
        "source_hashes": before["source_hashes"],
        "semantic_base": before["semantic_base"],
    }
    if normalized != before:
        raise ValueError("transport change alters evaluation protocol")
    if before.get("memory_config", {}).get("plan") != "P2":
        raise ValueError(
            "transport continuation is restricted to the P2 SemFilter plan"
        )


def verify_base_results(folder: Path, authorization: dict):
    for name, sha in authorization["base_results"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha
        ):
            raise ValueError(f"base result changed: {name}")


def protocol_records(folder: Path) -> dict[str, dict]:
    """Validate the immutable base and optional transport extension before accepting results."""
    base = json.loads((folder / MARKER).read_text())
    if digest(base["protocol"]) != base["sha256"]:
        raise ValueError("invalid base protocol digest")
    records = {base["sha256"]: base}
    path = folder / TRANSPORT_MARKER
    if path.exists():
        extension = json.loads(path.read_text())
        validate_transport_change(base, extension, extension["authorization"])
        verify_base_results(folder, extension["authorization"])
        records[extension["sha256"]] = extension
    return records


def active_protocol(folder: Path) -> dict:
    return list(protocol_records(folder).values())[-1]


def claim_serving_segment(s, path: Path) -> str:
    segment = json.loads(path.read_text())
    if (
        segment["model"] != s.model
        or segment["endpoints"] != list(s.api_bases)
        or not isinstance(segment["max_num_seqs"], int)
        or segment["max_num_seqs"] < 1
        or not isinstance(segment["max_num_batched_tokens"], int)
        or segment["max_num_batched_tokens"] < 1
    ):
        raise ValueError("serving segment disagrees with runtime settings")
    sha = digest(segment)
    folder = Path(s.out_dir) / "serving_segments"
    folder.mkdir(exist_ok=True)
    target = folder / f"{sha}.json"
    if target.exists():
        if json.loads(target.read_text()) != segment:
            raise ValueError("serving segment changed")
    else:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(segment, indent=2))
        tmp.replace(target)  # caller holds the exclusive arm lock
    return sha


@contextmanager
def exclusive_arm(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"{out}: another runner owns this arm") from exc
        yield


def actual_imports() -> dict:
    paths = {}
    for name in (
        "tau2",
        "tau2.runner",
        "tau2.agent.llm_agent",
        "tau2.user.user_simulator",
        "tau2.evaluator.evaluator",
        "tau2.evaluator.evaluator_nl_assertions",
        "tau2.utils.llm_utils",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to((ROOT / "vendor/tau2-bench/src").resolve()):
            raise ValueError(
                f"{name} imported outside the pinned vendor checkout: {path}"
            )
        paths[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    from tau2.utils.utils import DATA_DIR

    if Path(DATA_DIR).resolve() != (ROOT / "vendor/tau2-bench/data").resolve():
        raise ValueError("tau2 DATA_DIR is outside the pinned checkout")
    paths["data_dir"] = str(Path(DATA_DIR).resolve())
    return paths


def corpus_snapshot(dsn: str) -> dict:
    """Hash memory payloads, embeddings and native topology in one read-only snapshot."""
    import psycopg
    from psycopg import sql

    result = {}
    with psycopg.connect(dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for table in ("task", "session", "step", "hub"):
            hashes = [
                hashlib.sha256(row[0].encode()).hexdigest()
                for row in conn.execute(
                    sql.SQL("SELECT to_jsonb(t)::text FROM {} t").format(
                        sql.Identifier(table)
                    )
                )
            ]
            result[table] = {"count": len(hashes), "sha256": digest(sorted(hashes))}
        edges = [
            tuple(row)
            for row in conn.execute(
                "SELECT src,dst,type_id FROM graph_store.gph_dump_edges()"
            )
        ]
        result["native_edges"] = {"count": len(edges), "sha256": digest(sorted(edges))}
        result["edge_types"] = list(
            conn.execute("SELECT id,name FROM graph_store.edge_type ORDER BY id")
        )
        result["vertex_map"] = digest(
            sorted(conn.execute("SELECT ext_id,vid FROM graph_store.gph_vid_map"))
        )
    return json.loads(json.dumps(result))


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_hashes(root: Path = ROOT) -> dict:
    files = set()
    for directory in (
        "trimem/experiments/tau2",
        "trimem/optimizer/query_router",
        "vendor/tau2-bench/src/tau2",
        "vendor/tau2-bench/data/tau2",
    ):
        suffixes = (
            {".py", ".json", ".md", ".yaml", ".txt"}
            if directory.startswith("vendor/")
            else {".py"}
        )
        files.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in suffixes
        )
    for name in (
        "session_retrieval",
        "session_profile_store",
        "session_store_ext",
        "memory_policy",
        "llm_budget",
        "semantic_rubric",
        "colbert_rank",
    ):
        files.add(root / f"trimem/experiments/{name}.py")
    files.add(root / "trimem/optimizer/session_space_agentenv.py")
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def effective_protocol(s, items, embed_url: str) -> dict:
    from trimem.experiments.tau2 import runner

    pool = "|".join(s.api_bases) if s.api_bases else None
    args = runner.llm_args(pool or s.api_base)
    user_args = runner.llm_args(pool or s.user_api_base or s.api_base)
    # Official LLMConfigMixin.set_seed adds the derived seed to both objects.
    trial_seeds = [runner.trial_seed(s.seed, k) for k in range(s.num_trials)]
    args["seed"] = user_args["seed"] = (
        runner.trial_seed(s.seed, s.trial) if s.num_trials == 1 else trial_seeds
    )
    judge_args = runner.llm_args(
        pool or s.judge_api_base or s.api_base, json_mode=s.judge_json_mode
    )
    for kwargs in (args, user_args, judge_args):
        kwargs.pop("api_key", None)
    packages = {}
    for name in ("tau2", "litellm", "openai", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "version": "tau2-controlled-rerun-v2",
        "mode": s.mode,
        "split": s.split,
        "opt_split": s.opt_split,
        "experiment": s.experiment,
        "model": s.model,
        "agent": "trimem_memory_agent" if s.mode == "memory" else "llm_agent",
        "user": "user_simulator",
        "evaluation_type": "ALL",
        "auto_review": False,
        "agent_llm_args": args,
        "user_llm_args": user_args,
        "judge_llm_args": judge_args,
        "seed": s.seed,
        "trial": s.trial if s.num_trials == 1 else None,
        "num_trials": s.num_trials,
        "trial_seeds": trial_seeds if s.num_trials > 1 else None,
        # pool: each episode pinned to the least-loaded endpoint (same served model checked
        # by the supervisor); the endpoint actually used is recorded per episode.
        "endpoint_pool": list(s.api_bases) or None,
        "max_steps": s.max_steps,
        "max_errors": s.max_errors,
        "timeout": s.timeout,
        "memory_config": dataclasses.asdict(s.config) if s.config else None,
        "memory_policy": s.memory_policy if s.mode == "memory" else None,
        "strict_entities": s.strict_entities,
        "semantic_base": "request-load-v1"
        if s.semantic_pool
        else ("episode-endpoint" if s.api_bases else s.semantic_base),
        "rank_base": s.rank_base,
        "embed_url": embed_url,
        "embed_model": runner.EMBED_MODEL,
        "store_dsn": s.store_dsn,
        "opt_dsn": s.opt_dsn,
        "corpus_snapshot": corpus_snapshot(s.store_dsn),
        "actual_imports": actual_imports(),
        "baseline_sha256": digest(s.baseline),
        "protocol_extra": s.protocol_extra,
        "historical_reward_reuse": s.reuse,
        "tasks": {f"{d}:{t.id}": digest(t.model_dump(mode="json")) for d, t in items},
        "source_hashes": source_hashes(),
        "packages": packages,
        "context_note": "Only output max_tokens is set by this runner; total context is server-controlled.",
    }


def claim_protocol(s, items, embed_url: str) -> str:
    protocol = effective_protocol(s, items, embed_url)
    sha = digest(protocol)
    record = {"sha256": sha, "protocol": protocol}
    out = Path(s.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / MARKER
    if marker.exists():
        if (out / TRANSPORT_MARKER).exists() and not s.transport_change:
            raise ValueError(
                "active transport extension requires explicit continuation"
            )
        if json.loads(marker.read_text()) != json.loads(json.dumps(record)):
            if s.transport_change:
                base = json.loads(marker.read_text())
                authorization = json.loads(s.transport_change.read_text())
                validate_transport_change(base, record, authorization)
                verify_base_results(out, authorization)
                extension = {**record, "authorization": authorization}
                path = out / TRANSPORT_MARKER
                if path.exists():
                    if json.loads(path.read_text()) != json.loads(
                        json.dumps(extension)
                    ):
                        raise ValueError(
                            "transport protocol changed after continuation"
                        )
                else:
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(extension, indent=2, ensure_ascii=False))
                    tmp.replace(path)  # protected by exclusive_arm
                s.accepted_protocols = (base["sha256"], sha)
                s.base_results = authorization["base_results"]
                return sha
            raise ValueError(
                f"{out}: changed run protocol; use a fresh output directory"
            )
        return sha
    if s.transport_change:
        raise ValueError("transport continuation requires an existing base protocol")
    if any(p for p in out.glob("*.json") if not p.name.startswith(".")):
        raise ValueError(f"{out}: existing results have no frozen protocol")
    # One process owns each arm directory. Exclusive creation also refuses a duplicate launcher.
    with marker.open("x") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return sha
