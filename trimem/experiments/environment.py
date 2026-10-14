"""Capture the experiment environment of the math and ALE runs, for strictly reproducible comparisons.

Everything a later comparison must hold fixed: hardware, software and model versions, the vLLM
serving arguments, OpenEvolve's frozen configuration, evaluators, database / experience-graph state,
semantic predicate and prompts, the run protocol, and each cell's resolved settings (from its receipt).

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.environment --out triMEM_optimizer/environment.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

R = Path("results/trimem_v0.1.0")
DSN = f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo"


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def _pyver(python: str, mods: list[str]) -> dict:
    code = ("import importlib, json, sys; out = {'python': sys.version.split()[0]}\n"
            f"for m in {mods!r}:\n"
            "    try: out[m] = getattr(importlib.import_module(m), '__version__', 'installed')\n"
            "    except Exception as e: out[m] = 'missing'\n"
            "print(json.dumps(out))")
    try:
        return json.loads(subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120).stdout)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _snapshot(model_dir: str) -> str:
    snaps = sorted(Path(f"/localhome/hza214/hf-cache/hub/{model_dir}/snapshots").glob("*"))
    return snaps[-1].name if snaps else "<missing>"


def _evaluator(path: Path) -> dict:
    s = path.read_text()
    get = lambda k: (m.group(1) if (m := re.search(rf"^{k} = (.+)$", s, re.M)) else None)  # noqa: E731
    return {"file": str(path), "NUM_CASES": get("NUM_CASES"), "WORKERS": get("WORKERS"), "LANGUAGE": get("LANGUAGE"),
            "PROBLEM": get("PROBLEM")}


def hardware() -> dict:
    return {"cpu_model": _sh("lscpu | sed -n 's/^Model name: *//p'"), "logical_cpus": _sh("nproc"),
            "physical_cores": _sh("lscpu | sed -n 's/^Core(s) per socket: *//p'"),
            "memory": _sh("free -g | awk '/Mem:/{print $2\" GiB\"}'"),
            "gpus": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            "kernel": _sh("uname -r"), "host_os": _sh("lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1")}


def software() -> dict:
    return {
        "venv_main (.venv)": _pyver(".venv/bin/python", ["psycopg", "numpy", "httpx", "tokenizers"]),
        "venv_openevolve (.venv-e0)": _pyver(".venv-e0/bin/python", [
            "openevolve", "ale_bench", "psycopg", "numpy", "sklearn", "xgboost", "lightgbm", "tokenizers", "openai"]),
        "vllm_env (vllm17)": _sh("/localhome/hza214/miniconda3/envs/vllm17/bin/python -c "
                                 "\"import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__,"
                                 " 'cuda', torch.version.cuda)\""),
        "podman": _sh("/localhome/hza214/miniconda3/envs/podmantest/bin/podman --version"),
        "ale_bench_image": _sh("DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock "
                               "/localhome/hza214/miniconda3/envs/podmantest/bin/podman images "
                               "--format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep ale-bench"),
        "git": {"branch": _sh("git rev-parse --abbrev-ref HEAD"), "commit": _sh("git rev-parse HEAD"),
                "dirty_files": _sh("git status --porcelain | wc -l")},
    }


def serving() -> dict:
    procs = [line for line in _sh("ps -eo args").splitlines() if "vllm serve" in line]
    return {"generator_model": "Qwen/Qwen3.8-27B-FP8", "generator_snapshot": _snapshot("models--Qwen--Qwen3.8-27B-FP8"),
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_snapshot": _snapshot("models--Qwen--Qwen3-Embedding-0.6B"),
            "vllm_serve_commands": procs,
            "note": "generator context 32,768 -> 131,072 -> 262,144 on 2026-10-12; earlier math runs used 32,768; math_sem and next ALE round use 262,144 with max_tokens 32,768"}


def database() -> dict:
    import psycopg

    with psycopg.connect(DSN) as c:
        ver = c.execute("SELECT version()").fetchone()[0]
        ext = dict(c.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1").fetchall())
        gucs = dict(c.execute("SELECT name, setting FROM pg_settings WHERE name IN "
                              "('shared_preload_libraries','block_size','max_connections','shared_buffers','work_mem')").fetchall())
        graph = [dict(zip(("type_id", "edge_type", "n_vertices", "n_src", "n_edges", "max_deg"), r))
                 for r in c.execute("SELECT type_id, edge_type, n_vertices, n_src, n_edges, max_deg FROM trimem.graph_stats ORDER BY 1")]
        corpus = [dict(zip(("task", "attempts", "with_code", "with_changes", "sessions", "valid"), r)) for r in c.execute(
            "SELECT task, count(*), count(code), count(changes), count(DISTINCT session), count(*) FILTER (WHERE is_valid) "
            "FROM attempt WHERE cell IS NULL GROUP BY 1 ORDER BY 1")]
    return {"postgres": ver, "extensions": ext, "settings": gucs, "graph_edge_types": graph, "experience_graph": corpus,
            "dsn": "postgresql://<user>@127.0.0.1:55441/trimem_evo",
            "session_gucs": {"jit": "off", "hnsw.ef_search": 100, "hnsw.iterative_scan": "relaxed_order",
                             "trimem.timeout_ms": 900000, "trimem.prompt_version": "oe-<out dir>-<cell> (per cell)"}}


def prompts() -> dict:
    from trimem.llm import SCORE_PROMPT
    from trimem.replay.memory import DERIVE_SYSTEM, SEM_PREDICATE

    return {"semantic_predicate": SEM_PREDICATE, "sem_score_prompt": SCORE_PROMPT,
            "change_summary_system_prompt": DERIVE_SYSTEM,
            "change_summary_input": "unified diff parent->child (math: first 6,000 chars; ALE: first 60,000 chars)"}


def openevolve_config() -> dict:
    from bench.agent_memory.gem_oe.run_arm import FROZEN

    return {"version": "openevolve 0.3.2", "frozen": FROZEN, "thinking": False,
            "llm_timeout_s": 1800, "checkpoint_interval": 10}


def protocol() -> dict:
    ale = sorted((Path("bench/out/ale_evaluators")).glob("ahc0*_n50.py"))
    return {
        "common": {"seed": 42, "iterations": "1-20 in it20/<cell>, resumed from checkpoint_20 to 50 in it50/<cell>",
                   "injection_frequency": 1.0, "injection_policy": "fixed (replace own inspirations)",
                   "inspiration_slots": 10, "memory_items_fixed_plans": 5, "memory_items_adaptive_max": 8,
                   "scheduler": "nice -n 10, taskset -c 0-27, start new cell only while CPU < 80%",
                   "corpus_isolation": "attempt.cell IS NULL (offline EvoTrace corpus only)"},
        "math": {"tasks": ["circle_packing", "heilbronn_convex_13", "second_autocorr_ineq", "third_autocorr_ineq",
                           "heilbronn_triangle"],
                 "evaluator": "corpus evaluate.py (EvoTrace raw run), wrapped with a hard deadline",
                 "eval_timeout_s": 240, "language": "python", "inject_as": "code",
                 "retrieval_floor": ">= 0.8 x parent score", "generator_context": "32768 (msem runs: 262144, max_tokens 32768)"},
        "ale": {"tasks": ["ahc015", "ahc025", "ahc024", "ahc039", "ahc046"],
                "evaluators": [_evaluator(p) for p in ale],
                "public_cases": 50, "private_rescoring": "ALE-Bench Lite Session.private_eval (seed and public-best)",
                "eval_timeout_s": 600, "language": "cpp (cpp20 judge)", "inject_as": "changes (LLM change summaries)",
                "prompt_budget_tokens": 220000, "max_tokens": 32768, "generator_context": 262144,
                "retrieval_floor": ">= parent - 0.2|parent| (signed scores)",
                "env": {"ALE_BENCH_DATA": "/localhome/hza214/ale-bench-data", "ALE_BENCH_NO_CONTAINER_USER": "1",
                        "DOCKER_HOST": "unix:///run/user/<uid>/podman/podman.sock"},
                "order": "one task at a time per chain; 3 chains in parallel (2026-10-12)",
                "change_log": "10-case runs archived to ale/archive_n10 (not comparable)"},
    }


def cells() -> list[dict]:
    rows = []
    for base in (R / "openevolve", R / "objective", R / "ale", R / "math_sem"):
        for phase in ("it20", "it50"):
            for receipt in sorted((base / phase).glob("*/run_receipt.json")):
                r = json.loads(receipt.read_text())
                pol = r.get("policy") or {}
                rows.append({"domain": base.name, "phase": phase, "cell": receipt.parent.name, "task": r.get("task_uid"),
                             "arm": r.get("arm"), "retriever": r.get("retriever"), "status": r.get("status"),
                             "language": r.get("language"), "inject_as": r.get("inject_as"),
                             "injection_slots": r.get("injection_slots"), "max_injected": r.get("max_injected"),
                             "injection_frequency": r.get("injection_frequency"), "resolved_seed": r.get("resolved_seed"),
                             "max_iterations_override": r.get("max_iterations_override"),
                             "resumed_from": r.get("resumed_from"), "evaluator": r.get("evaluator"),
                             "evaluator_wrapped": r.get("evaluator_wrapped"), "llm_model": (r.get("llm") or {}).get("model"),
                             "policy_space": pol.get("space"), "policy_kind": pol.get("kind"), "policy_mode": pol.get("mode"),
                             "policy_prior": pol.get("prior"), "git_commit": (r.get("git") or {}).get("commit"),
                             "git_dirty": (r.get("git") or {}).get("dirty"), "started_at": r.get("started_at"),
                             "finished_at": r.get("finished_at")})
    return rows


def collect() -> dict:
    return {"hardware": hardware(), "software": software(), "serving": serving(), "database": database(),
            "openevolve": openevolve_config(), "prompts": prompts(), "protocol": protocol()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="triMEM_optimizer/environment.json")
    args = ap.parse_args(argv)
    env = collect()
    Path(args.out).write_text(json.dumps(env, indent=1, default=str))
    print(args.out, list(env))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
