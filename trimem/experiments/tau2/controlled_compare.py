"""Sequential, resumable five-arm τ² comparison with immutable provenance.

Run with .venv-tau2. Each arm gets a separate process, output directory and opt
experiment. A train-only smoke must finish before the full test phase starts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from trimem.experiments.tau2 import runner
from trimem.experiments.tau2.run_protocol import (
    ROOT,
    MARKER,
    source_hashes,
    corpus_snapshot,
    protocol_records,
    active_protocol,
    digest,
)

ARMS = ("NoMemory", "VGR", "VGR_SemFilter", "BO_Qbest", "BRANE_frozen")
SEED = 300
DEFAULT_POOL = ("http://127.0.0.1:8030/v1", "http://127.0.0.1:8031/v1")
MAX_WORKERS = 64


def is_successful(reward: float) -> bool:
    """Official τ² success (``tau2.metrics.agent_metrics.is_successful``)."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Official τ² pass^k for one task: C(c, k) / C(n, k) (arXiv:2406.12045)."""
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def pass_at_k(num_trials: int, success_count: int, k: int) -> float:
    """Unbiased pass@k for one task: 1 - C(n - c, k) / C(n, k)."""
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return 1 - math.comb(num_trials - success_count, k) / math.comb(num_trials, k)


def trial_metrics(rows: list[dict]) -> dict:
    """τ²-style multi-trial metrics over episode rows (each row = one (task, trial)).

    pass^1 = mean reward over all episodes (official ``avg_reward``); pass^k / pass@k average the
    per-task estimators over tasks; agreement = share of tasks whose trials all succeed or all fail,
    and the mean pairwise agreement of trial outcomes."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(f"{r['domain']}:{r['task_id']}", []).append(r)
    counts = {len(v) for v in by_task.values()}
    if len(counts) != 1:
        raise ValueError(f"tasks have different trial counts: {sorted(counts)}")
    n = counts.pop()
    succ = {k: sum(is_successful(r["reward"]) for r in v) for k, v in by_task.items()}
    pairs = math.comb(n, 2)
    return {
        "num_trials": n,
        "tasks": len(by_task),
        "pass^1": statistics.mean(r["reward"] for r in rows),
        "pass_hat_k": {
            k: statistics.mean(pass_hat_k(n, c, k) for c in succ.values())
            for k in range(1, n + 1)
        },
        "pass_at_k": {
            k: statistics.mean(pass_at_k(n, c, k) for c in succ.values())
            for k in range(1, n + 1)
        },
        "trial_agreement_all": statistics.mean(c in (0, n) for c in succ.values()),
        "trial_agreement_pairwise": (
            statistics.mean(
                (math.comb(c, 2) + math.comb(n - c, 2)) / pairs for c in succ.values()
            )
            if pairs
            else 1.0
        ),
        "success_count_histogram": {
            c: sum(v == c for v in succ.values()) for c in range(n + 1)
        },
    }


def service_identity() -> dict:
    import httpx

    identity = {}
    for port, route in (
        (8030, "/v1/models"),
        (8031, "/v1/models"),
        (8021, "/v1/models"),
        (8040, "/health"),
    ):
        response = httpx.get(f"http://127.0.0.1:{port}{route}", timeout=10)
        response.raise_for_status()
        body = response.json()
        identity[str(port)] = (
            [
                {
                    k: m.get(k)
                    for k in (
                        "id",
                        "root",
                        "max_model_len",
                        "revision",
                        "embedding_dimension",
                    )
                }
                for m in body["data"]
            ]
            if "data" in body
            else body
        )
    for port in ("8030", "8031"):
        if not any(
            m["id"] == runner.MODEL.removeprefix("openai/") for m in identity[port]
        ):
            raise ValueError(f"unexpected served model on {port}")
    return identity


def accepted_sources(sources: dict, changes: dict | None) -> dict:
    """``sources`` after applying a recorded, reviewed source change (``{path: {old, new}}``)."""
    out = dict(sources)
    for path, change in (changes or {}).items():
        if out.get(path) != change["old"]:
            raise ValueError(
                f"source change record does not start from the run's {path}"
            )
        if change["new"] is None:
            out.pop(path, None)
        else:
            out[path] = change["new"]
    return out


def load_source_change(path: Path | None, recorded: dict, current: dict) -> dict | None:
    """Validate an explicit behaviour-equivalence record against the manifest and the tree.

    Every differing hashed file must be listed with its manifest (old) and current (new) hash, and
    the record must carry a human summary of the equivalence check. Nothing else may differ."""
    if path is None:
        return None
    record = json.loads(Path(path).read_text())
    if not str(record.get("summary", "")).strip():
        raise ValueError("source change record needs an equivalence summary")
    changes = record["changed"]
    if accepted_sources(recorded, changes) != current:
        raise ValueError(
            "source change record does not explain every source difference"
        )
    return record


def assert_common_protocol(
    folder: Path, baseline_folder: Path, source_changes: dict | None = None
) -> None:
    actual = active_protocol(folder)["protocol"]
    baseline = json.loads((baseline_folder / MARKER).read_text())["protocol"]
    if source_changes:
        baseline = {
            **baseline,
            "source_hashes": accepted_sources(
                baseline["source_hashes"], source_changes
            ),
        }
    for field in (
        "model",
        "agent_llm_args",
        "user_llm_args",
        "judge_llm_args",
        "seed",
        "trial",
        "max_steps",
        "max_errors",
        "timeout",
        "tasks",
        "source_hashes",
        "packages",
        "corpus_snapshot",
        "actual_imports",
        "opt_dsn",
    ):
        if actual[field] != baseline[field]:
            raise ValueError(f"{folder}: {field} differs from paired NoMemory protocol")


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def command(
    root: Path,
    phase: str,
    arm: str,
    workers: int,
    *,
    num_trials: int = 1,
    api_bases: tuple[str, ...] = (),
    max_rss_gb: float = 0.0,
    semantic_workers: int = 4,
    retrieval_slots: int = 0,
    transport_change: Path | None = None,
    drain_file: Path | None = None,
    serving_segment: Path | None = None,
) -> list[str]:
    routed = arm == "BRANE_frozen"
    module = "router_runner" if routed else "runner"
    args = [
        sys.executable,
        "-u",
        "-m",
        f"trimem.experiments.tau2.{module}",
        "--mode",
        "nomemory" if arm == "NoMemory" else "memory",
        "--split",
        "train" if phase == "smoke" else "test",
        "--out_dir",
        str(root / phase / arm),
        "--freeze_protocol",
        "--seed",
        str(SEED),
        "--max_steps",
        "200",
        "--max_errors",
        "10",
        "--max_retries",
        "0",
        "--max_concurrency",
        str(workers),
        "--model",
        runner.MODEL,
        "--api_base",
        runner.API_BASE,
        *(
            ["--api_bases", ",".join(api_bases)]
            if api_bases
            else [
                "--user_api_base",
                runner.API_BASE,
                "--judge_api_base",
                runner.API_BASE,
            ]
        ),
        "--num_trials",
        str(num_trials),
        "--semantic_workers",
        str(semantic_workers),
        "--max_retrieval_concurrency",
        str(retrieval_slots),
        "--max_rss_gb",
        str(max_rss_gb),
        "--semantic_base",
        runner.SEMANTIC_BASE,
        "--rank_base",
        "http://127.0.0.1:8040",
        "--embed_url",
        runner.EMBED,
        "--memory_policy",
        "success+failure-v1",
    ]
    if transport_change:
        if arm != "VGR_SemFilter":
            raise ValueError("transport continuation is limited to VGR_SemFilter")
        args += [
            "--semantic_load_balance",
            "--transport_change",
            str(transport_change.resolve()),
        ]
    if drain_file:
        args += ["--drain_file", str(drain_file.resolve())]
    if serving_segment:
        args += ["--serving_segment", str(serving_segment.resolve())]
    if phase == "smoke":
        args += ["--smoke", "--num_tasks", "1"]
    if arm != "NoMemory":
        args += [
            "--baseline_dir",
            str(root / phase / "NoMemory"),
            "--experiment",
            f"tau2:{root.name}:{phase}:{arm}:ranked-session-topk",
        ]
        if routed:
            args += [
                "--router",
                str(root / "router.json"),
                "--allow_frozen_router_transfer",
            ]
        else:
            cfg = {"seed_count": 8, "depth": 1, "fanout": 4, "top_k": 3}
            plan, edges = ("P2" if arm == "VGR_SemFilter" else "VGR"), "similar"
            if arm == "BO_Qbest":
                plan, edges = "VGR+Rank", "similar+similar_up"
                cfg.update(seed_count=4, top_k=5)
            args += [
                "--plan",
                plan,
                "--edges",
                edges,
                "--memory_config",
                json.dumps(cfg),
            ]
    return args


def expected_tasks(phase: str, num_trials: int = 1) -> set[str]:
    """Episode keys an arm must contain: ``domain:task`` or ``domain:task#t<k>`` per trial."""
    keys = set()
    trials = range(num_trials) if num_trials > 1 else (None,)
    for domain in runner.ent.DOMAINS:
        tasks = runner.load_split(domain, "train" if phase == "smoke" else "test")
        for task in tasks[:1] if phase == "smoke" else tasks:
            keys.update(runner.episode_key(domain, str(task.id), k) for k in trials)
    return keys


def validate_arm(folder: Path, expected: set[str]) -> list[dict]:
    protocols = protocol_records(folder)
    base_sha = next(iter(protocols))
    active = list(protocols.values())[-1]
    rows = []
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("."):
            continue
        r = runner.read_result(path)
        if r is None:
            raise ValueError(f"invalid episode file: {path}")
        if r.get("run_protocol_sha256") not in protocols:
            raise ValueError(f"mixed run protocol: {path}")
        protocol = protocols[r["run_protocol_sha256"]]
        if len(protocols) > 1:
            if r["run_protocol_sha256"] == base_sha:
                if path.name not in active["authorization"]["base_results"]:
                    raise ValueError(f"unregistered base result: {path}")
            else:
                sha = r.get("serving_segment_sha256")
                if (
                    not isinstance(sha, str)
                    or len(sha) != 64
                    or any(c not in "0123456789abcdef" for c in sha)
                ):
                    raise ValueError(f"missing serving segment: {path}")
                segment = json.loads(
                    (folder / "serving_segments" / f"{sha}.json").read_text()
                )
                if (
                    digest(segment) != sha
                    or segment["model"] != protocol["protocol"]["model"]
                    or segment["endpoints"] != protocol["protocol"]["endpoint_pool"]
                ):
                    raise ValueError(f"invalid serving segment: {path}")
        if not math.isfinite(r["reward"]) or not 0 <= r["reward"] <= 1:
            raise ValueError(f"invalid official reward: {path}")
        if r.get("reused_from"):
            raise ValueError(f"historical reward reuse: {path}")
        n = protocol.get("protocol", {}).get("num_trials", 1)
        if (r.get("num_trials") or 1) != n or r.get("trial_seed") != runner.trial_seed(
            r.get("seed"), r.get("trial")
        ):
            raise ValueError(f"trial/seed evidence disagrees with the protocol: {path}")
        mem = r.get("memory")
        if mem is not None:
            ids = mem["selected_source_task_ids"]
            keys = mem["selected_source_task_keys"]
            if (
                len(mem["selected"]) != len(set(mem["selected"]))
                or len(ids) != len(mem["selected"])
                or len(keys) != len(ids)
            ):
                raise ValueError(f"session selection evidence is inconsistent: {path}")
            if mem.get("memory_policy") != "success+failure-v1":
                raise ValueError(f"wrong memory policy: {path}")
        elif (
            r["mode"] == "memory" and r.get("memory_status") != "no_first_user_message"
        ):
            raise ValueError(f"missing memory evidence: {path}")
        rows.append(r)
    keys = [runner.result_episode_key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            f"{folder}: expected {len(expected)} distinct episodes, found {len(keys)}"
        )
    # Persisted native-memory retrieval receipts must agree with the injected evidence.
    import psycopg

    pairs = {r["retrieval_id"]: r for r in rows if r.get("retrieval_id")}
    if pairs:
        with psycopg.connect(runner.OPT_DSN) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            stored = conn.execute(
                "SELECT retrieval_id::text,status,quality,receipt,protocol FROM opt.session_retrieval "
                "WHERE retrieval_id=ANY(%s::uuid[])",
                (list(pairs),),
            ).fetchall()
        if len(stored) != len(pairs):
            raise ValueError("missing opt rows")
        for rid, status, quality, receipt, meta in stored:
            r = pairs[rid]
            if (
                status != "evaluated"
                or quality != r["quality"]
                or meta.get("run_protocol_sha256") != r["run_protocol_sha256"]
                or meta.get("serving_segment_sha256") != r.get("serving_segment_sha256")
                or [card["id"] for card in receipt["cards"]] != receipt["selected_ids"]
                or [card["task_id"] for card in receipt["cards"]]
                != r["memory"]["selected_source_task_ids"]
                or [card["key"] for card in receipt["cards"]]
                != r["memory"]["selected_source_task_keys"]
                or receipt["selected_ids"] != r["memory"]["selected"]
                or receipt["selected_ids"]
                != receipt["ranked_ids"][: receipt["config"]["top_k"]]
            ):
                raise ValueError(f"opt receipt disagrees with result: {rid}")
    return rows


def summary(rows: list[dict], baseline: list[dict]) -> dict:
    """Arm summary. Pairing is per episode (same task and trial seed); wins/losses and the sign
    test are per task on the trial-mean reward, so repeated trials are not counted as independent."""
    base = {runner.result_episode_key(r): r["reward"] for r in baseline}
    by_task: dict[str, list[float]] = {}
    for r in rows:
        diff = r["reward"] - base[runner.result_episode_key(r)]
        by_task.setdefault(f"{r['domain']}:{r['task_id']}", []).append(diff)
    differences = [statistics.mean(v) for v in by_task.values()]
    wins, losses = sum(x > 0 for x in differences), sum(x < 0 for x in differences)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
    domain_rates = {
        d: statistics.mean(r["reward"] for r in rows if r["domain"] == d)
        for d in runner.ent.DOMAINS
    }
    return {
        "episodes": len(rows),
        **trial_metrics(rows),
        "reward_mean": statistics.mean(r["reward"] for r in rows),
        "domain_reward_mean": domain_rates,
        "domain_macro_mean": statistics.mean(domain_rates.values()),
        "paired_gain": statistics.mean(differences),
        "wins": wins,
        "losses": losses,
        "paired_binary_exact_p": p,
        "wall_seconds_mean": statistics.mean(r["wall_ms"] for r in rows) / 1000,
        "wall_seconds_median": statistics.median(r["wall_ms"] for r in rows) / 1000,
        "agent_tokens": sum(r["agent_tokens"] for r in rows),
        "user_tokens": sum(r["user_tokens"] for r in rows),
        "semantic_tokens": sum(
            (r.get("memory") or {}).get("semantic_tokens", 0) for r in rows
        ),
        "nl_judge_integrity_mismatches": [
            f"{r['domain']}:{r['task_id']}"
            for r in rows
            if r.get("nl_judge_integrity", {}).get("status") == "mismatch"
        ],
        "routed_plans": dict(
            Counter(
                (r.get("memory") or {}).get("route", {}).get("config", {}).get("plan")
                for r in rows
                if (r.get("memory") or {}).get("route")
            )
        ),
        "selected_failed_sessions": sum(
            (r.get("memory") or {}).get("card_outcomes", {}).get("failed", 0)
            for r in rows
        ),
        "no_first_user_message": sum(
            r.get("memory_status") == "no_first_user_message" for r in rows
        ),
        "episodes_per_endpoint": dict(Counter(r.get("endpoint") for r in rows)),
    }


def write_progress(root: Path, progress: dict) -> None:
    atomic_json(root / "progress.json", progress)
    lines = [
        "# τ² controlled rerun",
        "",
        f"Status: **{progress['status']}**.",
        "",
        f"Full test: 100 tasks per arm (Airline 20, Retail 40, Telecom 40) x {progress.get('num_trials', 1)} trial(s), "
        "official per-trial seeds derived from seed 300.",
        "Memory: all 712 train sessions, success+failure-v1, ranked session top-k; multiple sessions from the same source task are allowed.",
        "BO uses historical fixed parameters. BRANE uses a frozen historical router under the new",
        "memory policy. Neither was retrained for this protocol. No historical reward reuse.",
        "Agent, user and NL judge use local Qwen, with pinned official τ² environment/evaluator code.",
        "This is a local controlled comparison, not a reproduction of the released GPT-4.1 score.",
        "",
        "| Phase/arm | Episodes | pass^1 | pass^k (k=1..n) | pass@n | All-trials agree | Domain macro | Δ vs NoMemory | Task W/L | Paired p | Mean seconds |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in progress["completed"].items():
        hats = list((row.get("pass_hat_k") or {}).values())
        ats = list((row.get("pass_at_k") or {}).values())
        lines.append(
            f"| {name} | {row['episodes']} | {row['reward_mean']:.1%} | "
            f"{' / '.join(f'{v:.1%}' for v in hats) or '-'} | "
            f"{f'{ats[-1]:.1%}' if ats else '-'} | "
            f"{row.get('trial_agreement_all', 1.0):.1%} | "
            f"{row['domain_macro_mean']:.1%} | {row['paired_gain']:+.1%} | "
            f"{row['wins']}/{row['losses']} | {row['paired_binary_exact_p']:.4f} | "
            f"{row['wall_seconds_mean']:.2f} |"
        )
    for name, row in progress["completed"].items():
        if row.get("nl_judge_integrity_mismatches"):
            lines += [
                "",
                f"Judge completeness warning ({name}): "
                + ", ".join(row["nl_judge_integrity_mismatches"])
                + ". Official rewards are preserved, but these scores require inspection.",
            ]
    lines += [
        "",
        "Only completed, validated arms appear above. Smoke uses three train tasks and does not estimate test quality.",
        "pass^1 = mean official reward over all (task, trial) episodes; pass^k and pass@k use the official",
        "per-task estimators C(c,k)/C(n,k) and 1-C(n-c,k)/C(n,k) (success = reward 1) averaged over tasks.",
        "Δ is paired per (task, trial seed); W/L and the exact sign test are per task on the trial-mean reward.",
        "Selected sessions must equal the ranked top-k and match persisted opt retrieval receipts. Source task identity is recorded without limiting repeats.",
        "Wall time includes the full episode and evaluation. Agent, user and semantic token counts are separate in progress.json.",
        "NL judge token usage is not recorded by the official SimulationRun and is not included in those token counts.",
    ]
    for note in progress.get("provenance_notes") or []:
        lines += ["", "Provenance: " + note]
    if progress.get("skipped"):
        lines += [
            "",
            "Not run (user decision): " + ", ".join(progress["skipped"]) + ".",
        ]
    if progress.get("error"):
        lines += ["", "Stopped: " + progress["error"]]
    tmp = root / "report.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(root / "report.md")


def run(
    root: Path,
    router_path: Path,
    workers: int,
    phase: str,
    *,
    num_trials: int = 1,
    api_bases: tuple[str, ...] = (),
    max_rss_gb: float = 0.0,
    semantic_workers: int = 4,
    retrieval_slots: int = 0,
    arms: tuple[str, ...] = ARMS,
    source_change: Path | None = None,
    transport_change: Path | None = None,
    drain_file: Path | None = None,
    serving_segment: Path | None = None,
) -> None:
    """``arms``: arms to run (others are only re-validated from disk when complete, else
    recorded as skipped). ``source_change``: explicit reviewed record that lets a resumed
    comparison continue after hashed sources changed (see ``load_source_change``)."""
    if not arms or set(arms) - set(ARMS):
        raise ValueError(f"arms must be a subset of {ARMS}")
    if transport_change and (
        arms != ("VGR_SemFilter",) or phase != "test" or not serving_segment
    ):
        raise ValueError(
            "transport continuation requires test/VGR_SemFilter and serving segment"
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = source_hashes()
        services = service_identity()
        corpus = corpus_snapshot(runner.STORE_DSN)
        manifest = {
            "version": 3,
            "sources": sources,
            "workers": workers,
            "num_trials": num_trials,
            "api_bases": list(api_bases),
            "max_rss_gb": max_rss_gb,
            "semantic_workers": semantic_workers,
            "retrieval_slots": retrieval_slots,
            # runner raises the official 10-connection litellm pool to max(16, 2 x workers)
            "llm_http_connections": max(16, 2 * workers),
            "router_sha256": hashlib.sha256(router_path.read_bytes()).hexdigest(),
            "memory_policy": "success+failure-v1",
            "session_selection": "ranked-session-topk",
            "arms": list(ARMS),
            "historical_reward_reuse": False,
            "router_mode": "frozen-transfer-no-retraining",
            "services": services,
            "corpus": corpus,
        }
        path = root / "manifest.json"
        change = None
        if path.exists():
            stored = json.loads(path.read_text())
            change = load_source_change(source_change, stored["sources"], sources)
            if {
                **stored,
                "sources": sources if change else stored["sources"],
            } != manifest:
                raise ValueError("comparison provenance changed; use a new root")
            if (
                hashlib.sha256((root / "router.json").read_bytes()).hexdigest()
                != manifest["router_sha256"]
            ):
                raise ValueError("frozen router copy changed")
        else:
            if any((root / p).exists() for p in ("smoke", "test")):
                raise ValueError(
                    "refusing result directories without a comparison manifest"
                )
            if source_change is not None:
                raise ValueError("a source change record needs an existing comparison")
            (root / "router.json").write_bytes(router_path.read_bytes())
            atomic_json(path, manifest)
        changes = (change or {}).get("changed")
        notes = []
        if change:
            log = root / "source_changes.json"
            history = json.loads(log.read_text()) if log.exists() else []
            entry = {**change, "arms_run_after_change": list(arms)}
            if entry not in history:
                atomic_json(log, history + [entry])
            notes.append(
                "sources changed after "
                + ", ".join(change.get("after_arms", []) or ["earlier arms"])
                + "; behaviour-equivalence checked: "
                + change["summary"]
            )
        atomic_json(root / "services_observed.json", services)
        progress = {
            "supervisor_pid": os.getpid(),
            "status": "running",
            "num_trials": num_trials,
            "completed": {},
            "provenance_notes": notes,
            "arms_selected": list(arms),
        }
        phases = ("smoke", "test") if phase == "all" else (phase,)
        try:
            if phase == "test":
                for arm in ARMS:
                    validate_arm(
                        root / "smoke" / arm, expected_tasks("smoke", num_trials)
                    )
            for part in phases:
                baseline = None
                for arm in ARMS:
                    folder = root / part / arm
                    if arm not in arms:
                        if (folder / MARKER).exists():
                            # completed earlier in this comparison: re-validate, do not rerun
                            rows = validate_arm(
                                folder, expected_tasks(part, num_trials)
                            )
                            if arm == "NoMemory":
                                baseline = rows
                            progress["completed"][f"{part}/{arm}"] = summary(
                                rows, baseline
                            )
                        else:
                            progress.setdefault("skipped", []).append(f"{part}/{arm}")
                        continue
                    if source_hashes() != sources:
                        raise ValueError("source/data files changed during comparison")
                    if service_identity() != services:
                        raise ValueError(
                            "served model/context identity changed during comparison"
                        )
                    args = command(
                        root,
                        part,
                        arm,
                        workers,
                        num_trials=num_trials,
                        api_bases=api_bases,
                        max_rss_gb=max_rss_gb,
                        semantic_workers=semantic_workers,
                        retrieval_slots=retrieval_slots,
                        transport_change=transport_change,
                        drain_file=drain_file,
                        serving_segment=serving_segment,
                    )
                    if corpus_snapshot(runner.STORE_DSN) != corpus:
                        raise ValueError(
                            "memory corpus/native graph changed during comparison"
                        )
                    folder.mkdir(parents=True, exist_ok=True)
                    atomic_json(folder / ".command", args)
                    progress.update(phase=part, arm=arm)
                    write_progress(root, progress)
                    print(f"START {part}/{arm}", flush=True)
                    with (folder / "runner.log").open("a") as log:
                        result = subprocess.run(
                            args,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),),
                        )
                    if result.returncode == 75:
                        subset = {
                            runner.result_episode_key(runner.read_result(p))
                            for p in folder.glob("*__t*.json")
                        }
                        validate_arm(folder, subset)
                        if (
                            corpus_snapshot(runner.STORE_DSN) != corpus
                            or service_identity() != services
                        ):
                            raise ValueError(
                                "corpus or services changed before drain completed"
                            )
                        progress.update(
                            status="drained",
                            note="Admission stopped; completed episodes preserved",
                        )
                        raise SystemExit(
                            75
                        )  # process exit acknowledges that all in-flight work ended
                    if result.returncode:
                        raise RuntimeError(
                            f"{part}/{arm} exited {result.returncode}; see {folder}/runner.log"
                        )
                    rows = validate_arm(folder, expected_tasks(part, num_trials))
                    if (
                        corpus_snapshot(runner.STORE_DSN) != corpus
                        or service_identity() != services
                    ):
                        raise ValueError(
                            "corpus or model service changed while arm was running"
                        )
                    assert_common_protocol(folder, root / part / "NoMemory", changes)
                    if arm == "NoMemory":
                        baseline = rows
                    progress["completed"][f"{part}/{arm}"] = summary(rows, baseline)
                    write_progress(root, progress)
                    print(f"DONE {part}/{arm}: {len(rows)} episodes", flush=True)
            progress["status"] = "complete"
        except Exception as exc:
            progress.update(status="failed", error=str(exc))
            raise
        finally:
            write_progress(root, progress)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument(
        "--workers", type=int, default=4, help="concurrent episodes per arm"
    )
    ap.add_argument("--phase", choices=("smoke", "test", "all"), default="all")
    ap.add_argument("--num_trials", type=int, default=1, help="trials per task")
    ap.add_argument(
        "--api_bases",
        default=",".join(DEFAULT_POOL),
        help="endpoint pool (each episode pinned to the least-loaded); '' = 8030 only",
    )
    ap.add_argument(
        "--max_rss_gb", type=float, default=4.0, help="per-arm runner RSS guard"
    )
    ap.add_argument("--semantic_workers", type=int, default=4)
    ap.add_argument("--transport_change", type=Path)
    ap.add_argument("--drain_file", type=Path)
    ap.add_argument("--serving_segment", type=Path)
    ap.add_argument(
        "--retrieval_slots",
        type=int,
        default=12,
        help="concurrent retrievals per arm (PG 55450 max_connections is 32)",
    )
    ap.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated arms to run; completed others are re-validated, missing ones skipped",
    )
    ap.add_argument(
        "--resume_after_source_change",
        type=Path,
        default=None,
        help="JSON record {summary, after_arms, changed: {path: {old, new}}} of a reviewed, "
        "behaviour-equivalent source change; required to resume once hashed sources changed",
    )
    a = ap.parse_args()
    # Arms stay sequential; episodes inside an arm run in parallel (user decision 2026-10-19).
    if not 1 <= a.workers <= MAX_WORKERS:
        ap.error(f"workers must be between 1 and {MAX_WORKERS}")
    if (
        a.num_trials < 1
        or a.semantic_workers < 1
        or a.max_rss_gb < 0
        or a.retrieval_slots < 0
    ):
        ap.error("--num_trials/--semantic_workers must be >= 1, --max_rss_gb >= 0")
    run(
        a.root.resolve(),
        a.router.resolve(),
        a.workers,
        a.phase,
        num_trials=a.num_trials,
        api_bases=tuple(b for b in a.api_bases.split(",") if b),
        max_rss_gb=a.max_rss_gb,
        semantic_workers=a.semantic_workers,
        retrieval_slots=a.retrieval_slots,
        arms=tuple(x for x in a.arms.split(",") if x),
        source_change=a.resume_after_source_change,
        transport_change=a.transport_change,
        drain_file=a.drain_file,
        serving_segment=a.serving_segment,
    )


if __name__ == "__main__":
    main()
