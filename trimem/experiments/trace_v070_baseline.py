"""NoMemory protected baseline of a TRACE v0.7.0 run (plan 2026-11-05 section 6.1, Amendment A.4).

NoMemory is NOT a configuration of the search space: it lives in its own ledger
(``<run>/baselines/nomemory/ledger.sqlite``; TEST: ``<run>/holdout/baselines/nomemory``), so it is never in the
main ledger -> never in the GP, the latent encoding, ``staged_batch_size``, ``initial_keys``, ``pruning_log`` or
``prune_many``: outside the BO budget and unprunable by construction. It is a candidate for the TRAIN Pareto set,
TEST selection, OneConfig and the Opt-PerQuery router. It is NOT in the latency-cap front reference
(user 2026-11-05, fix round 2: ``latency_front`` false).

TRAIN sample: its own stratified 100-task sample, drawn like every configuration's with key = its baseline key.
``BaselineSpace`` duck-types ``JointSpace`` for ``SearchLedger.reserve`` / ``check_receipt``.
"""

from __future__ import annotations

import json
from pathlib import Path

from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import digest

NOMEMORY = {"baseline": "NoMemory"}
PROTOCOL = "trace-v070-nomemory-v1"
SPACE_SCHEMA = "trace-v070-baseline-space-v1"
WORKER = "tools/trace_v070_nomemory_worker.py"


class BaselineSpace:
    """One-point 'space' {NoMemory}; identity bound to the run's execution fingerprint."""

    def __init__(self, execution_fingerprint: str):
        self.execution_fingerprint = execution_fingerprint
        self.identity = digest(self.manifest())

    def manifest(self) -> dict:
        return {"schema": SPACE_SCHEMA, "baselines": ["NoMemory"], "protocol": PROTOCOL,
                "execution_fingerprint": self.execution_fingerprint}

    def canonicalize(self, config) -> dict:
        if dict(config) != NOMEMORY:
            raise ValueError(f"BaselineSpace only holds {NOMEMORY}, got {dict(config)!r}")
        return dict(NOMEMORY)

    def key(self, config) -> str:
        return digest([self.identity, self.canonicalize(config)])


def baseline_space(run_manifest) -> BaselineSpace:
    return BaselineSpace(run_manifest["space"]["execution_fingerprint"])


def baseline_key(run_manifest) -> str:
    return baseline_space(run_manifest).key(NOMEMORY)


def baselines_block(run_manifest, *, source: str = "evaluate", imported_from=None) -> dict:
    return {"NoMemory": {"protocol": PROTOCOL, "key": baseline_key(run_manifest), "dir": "baselines/nomemory",
                         "source": source, "imported_from": imported_from, "test": "always",
                         "prune_comparator": False, "latency_front": False, "worker": WORKER}}


def baseline_dir(run_dir, split: str = "TRAIN") -> Path:
    run_dir = Path(run_dir)
    return run_dir / ("holdout/baselines/nomemory" if split == "TEST" else "baselines/nomemory")


def train_sample(run_manifest) -> list[str]:
    from .trace_v070_samples import config_sample

    return config_sample(run_manifest["v070"]["sampling"], baseline_key(run_manifest))


def baseline_manifest(run_manifest, *, split: str, tasks, run_dir=None) -> dict:
    if split not in ("TRAIN", "TEST"):
        raise ValueError("split must be TRAIN or TEST")
    space = baseline_space(run_manifest)
    # Canonical absolute path: prepare and the live launcher may pass relative or symlinked forms (live bug 2026-11-05).
    run_dir = Path(run_dir or run_manifest["live"]["run_dir"]).resolve()
    live = {**run_manifest["live"], "run_dir": str(baseline_dir(run_dir, split))}
    return {
        "baseline": "NoMemory",
        "baseline_protocol": PROTOCOL,
        "run_protocol": run_manifest["protocol"],
        "protocol": run_manifest["protocol"] + ("-nomemory-test" if split == "TEST" else "-nomemory"),
        "evaluation_split": split,
        "tasks": [str(t) for t in tasks],
        "budget": 1,
        "space": space.manifest(),
        "key": space.key(NOMEMORY),
        "fixed": run_manifest["fixed"],
        "live": live,
        "smoke": bool(run_manifest.get("smoke")),
        "task_concurrency": run_manifest.get("task_concurrency", 16),
        "max_attempts": run_manifest.get("max_attempts", 2),
        "failure_breaker": run_manifest.get("failure_breaker", 3),
        "holdout": {k: run_manifest["holdout"][k] for k in ("ids_sha256",) if k in run_manifest["holdout"]},
    }


PARITY_KEYS = ("model", "llm_bases", "environment_bases", "rank_bases", "environment", "max_round",
               "action_format", "agent_max_tokens")


def parity_violations(run_manifest, bm) -> list[str]:
    """Fix round 1 (finding 6): NoMemory runs with the TRACE episodes' model, vLLM bases, env servers, env vars,
    max_round, action format, max_tokens, concurrency and runtime identity."""
    out = [f"live.{k}" for k in PARITY_KEYS if bm["live"].get(k) != run_manifest["live"].get(k)]
    out += [k for k in ("fixed", "task_concurrency", "smoke") if bm.get(k) != run_manifest.get(k, bm.get(k))]
    if bm.get("task_concurrency") != run_manifest.get("task_concurrency", 16):
        out.append("task_concurrency")
    return sorted(set(out))


def _write(path: Path, body) -> None:
    from trimem.experiments.trace_joint_plan_search import atomic_json

    atomic_json(path, body)


def prepare_baseline(run_dir, run_manifest, *, split: str = "TRAIN", tasks=None) -> dict:
    """Create (or verify, when it exists) the NoMemory ledger of one split; returns its manifest."""
    if tasks is None:
        tasks = train_sample(run_manifest) if split == "TRAIN" else run_manifest["holdout"]["tasks"]
    bm = baseline_manifest(run_manifest, split=split, tasks=tasks, run_dir=run_dir)
    d = baseline_dir(run_dir, split)
    if (d / "manifest.json").exists():
        old = json.loads((d / "manifest.json").read_text())
        if digest(old) != digest(bm):
            raise ValueError(f"NoMemory {split} manifest at {d} differs from this run")
        return old
    d.mkdir(parents=True, exist_ok=True)
    with SearchLedger(d / "ledger.sqlite", bm) as led:
        led.reserve(baseline_space(run_manifest), [NOMEMORY])
    _write(d / "manifest.json", bm)
    return bm


def load_baseline(run_dir, split: str = "TRAIN") -> dict:
    return json.loads((baseline_dir(run_dir, split) / "manifest.json").read_text())


def baseline_entry(run_dir, split: str = "TRAIN") -> dict | None:
    """{key, config, receipts, complete, tasks} of the NoMemory ledger, or None if it does not exist."""
    d = baseline_dir(run_dir, split)
    if not (d / "manifest.json").exists():
        return None
    bm = load_baseline(run_dir, split)
    with SearchLedger(d / "ledger.sqlite", bm) as led:
        receipts = led.receipts(bm["key"])
        pending = led.pending()
    done = {str(r["task_id"]) for r in receipts}
    return {"key": bm["key"], "config": dict(NOMEMORY), "baseline": "NoMemory", "receipts": receipts,
            "tasks": list(bm["tasks"]), "complete": not pending and done == set(bm["tasks"]), "pruned": False}


def baseline_rows(run_dir, split: str = "TRAIN") -> list[dict]:
    e = baseline_entry(run_dir, split)
    return [] if e is None else e["receipts"]


def baseline_complete(run_dir, split: str = "TRAIN") -> bool:
    e = baseline_entry(run_dir, split)
    return bool(e and e["complete"])


def run_baseline(run_dir, run_manifest, split: str, evaluator, *, recover=None) -> dict:
    """Evaluate every missing NoMemory episode of one split (resume = only missing ones) via the driver's
    ``evaluate_pending`` (same failure classes, breaker and receipt checks as TRACE episodes)."""
    from trimem.experiments import trace_joint_plan_search as driver

    bm = prepare_baseline(run_dir, run_manifest, split=split)
    bad = parity_violations(run_manifest, bm)
    if bad:
        raise ValueError(f"NoMemory runtime differs from the TRACE episodes: {bad}")
    d = baseline_dir(run_dir, split)
    if recover is not None:
        recover(d, bm)
    with SearchLedger(d / "ledger.sqlite", bm) as led:
        pending = led.pending()
    if pending:
        driver.evaluate_pending(d, bm, baseline_space(run_manifest), pending, evaluator)
    e = baseline_entry(run_dir, split)
    if not e["complete"]:
        raise RuntimeError(f"NoMemory {split} incomplete after evaluation: {d}")
    return summarize(e)


def summarize(entry) -> dict:
    rows = entry["receipts"]
    n = len(rows)
    out = {"key": entry["key"], "baseline": "NoMemory", "episodes": n, "complete": entry["complete"]}
    for f in ("raw_score", "total_tokens", "latency_seconds", "agent_steps"):
        if n and all(r.get(f) is not None for r in rows):
            out[f] = sum(r[f] for r in rows) / n
    return out


def import_baseline(src_run, dst_run, dst_manifest) -> dict:
    """Copy a COMPLETE NoMemory TRAIN ledger from the shared init into an arm (same key, same sample); receipts
    keep every measured value, their run identity is restated with ``imported_from`` provenance."""
    src_run, dst_run = Path(src_run), Path(dst_run)
    src = baseline_entry(src_run, "TRAIN")
    if src is None or not src["complete"]:
        raise ValueError(f"NoMemory TRAIN of {src_run} is missing or incomplete; refusing import")
    if src["key"] != baseline_key(dst_manifest):
        raise ValueError("NoMemory key differs between the shared init and the arm (runtime identity)")
    tasks = train_sample(dst_manifest)
    if sorted(src["tasks"]) != sorted(tasks):
        raise ValueError("NoMemory sample differs between the shared init and the arm")
    # fix round 2: the source NoMemory runtime (model, bases, env vars, max_round, max_tokens, concurrency, fixed
    # runtime identity) must equal the destination run's before any receipt is restated
    bad = parity_violations(dst_manifest, load_baseline(src_run, "TRAIN"))
    runtime = dst_manifest["fixed"].get("runtime")
    if runtime is not None and any(r.get("runtime", runtime) != runtime for r in src["receipts"]):
        bad.append("receipt runtime")
    if bad:
        raise ValueError(f"NoMemory runtime differs between the shared init and the arm: {bad}")
    bm = prepare_baseline(dst_run, dst_manifest, split="TRAIN", tasks=tasks)
    d = baseline_dir(dst_run, "TRAIN")
    with SearchLedger(d / "ledger.sqlite", bm) as led:
        for r in src["receipts"]:
            restated = {**r, "protocol": bm["protocol"],
                        "imported_from": {"run": str(src_run), "protocol": r["protocol"]}}
            led.claim(bm["key"], r["task_id"])
            led.finish(bm["key"], r["task_id"], restated)
    return bm
