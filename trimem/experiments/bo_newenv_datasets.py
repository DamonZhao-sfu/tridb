"""Dataset adapters of the generic session-BO driver (``bo_newenv_driver``) for AppWorld and τ²-bench.

One adapter per dataset says everything the driver must not hard-code: the runner command of one GPU
shard, the profiling panel (ids + hash) and the test ids, the completion rule's result files, the
quality definition, the leak guards and the experiment ids. No botorch / tau2 / appworld import here
(the driver runs in ``.venv-optimizer``; the runners in their own venvs).

GPU layout. Every arm is split into shards, one per GPU in ``gpus`` (8030 on GPU0, 8031 on GPU1);
the shard's LLM, semantic operator and ColBERT (LLM port + 10) all run on its own GPU.

* AppWorld shards by task (``--panel <shard file>`` for profiling, ``--tasks`` for a test half), each
  shard spawning ``per_gpu`` worker processes (the runner's ``--workers``).
* τ² cannot shard arbitrary task sets: ``--task_ids`` is matched within every domain listed in
  ``--domains`` and airline / retail reuse the same numeric ids. A shard therefore takes colliding
  domains whole and splits only collision-free ones (telecom); ``tau2_shards`` verifies that each
  shard's runner selection is exactly its intended task set. Train 178 -> retail 74 + telecom 15 |
  airline 30 + telecom 59 (89 | 89); test 100 -> 50 | 50.

Profiling defaults to both GPUs (one arm at a time, as ALFWorld / ScienceWorld). The test stage
defaults to the GPU and concurrency of the already-running test arms (``results/run_new_test_arms.sh``:
AppWorld GPU0 x 8 workers, τ² GPU1 x concurrency 8), so BO / router test arms share their No-Memory,
VGR and VGR+SemFilter arms' conditions (latency is an objective).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from trimem.experiments import memory_policy as mp

ROOT = Path(__file__).resolve().parents[2]
BO_OUT = (
    ROOT / "results/bo_agentenv_20261016"
)  # same root as ALFWorld / ScienceWorld (export, router)
OPT_DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa"
LLM_PORTS = (8030, 8031)
MAX_RSS_GB = 4  # per runner process (+ children); leak guard, user 2026-10-17
APPWORLD_RECYCLE = 50
RUN_ENV = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "no_proxy": "127.0.0.1,localhost"}


@dataclass(frozen=True)
class Shard:
    gpu: int
    items: tuple[str, ...]


def llm_base(gpu: int) -> str:
    return f"http://127.0.0.1:{LLM_PORTS[gpu]}"


def memory_args(session: dict) -> list[str]:
    """Runner flags of a fixed-configuration memory arm (identical for both runners)."""
    return [
        "--plan",
        session["plan"],
        "--memory_config",
        json.dumps(session["memory_config"]),
        "--edges",
        session["edges"],
        "--baseline_dir",
        str(session["baseline_dir"]),
        "--experiment",
        session["experiment"],
        "--protocol_extra",
        json.dumps(session["protocol_extra"]),
        *(["--reuse_identical_memory"] if session.get("reuse") else []),
    ]


def router_args(session: dict) -> list[str]:
    return [
        "--router",
        session["router"],
        "--baseline_dir",
        str(session["baseline_dir"]),
        "--experiment",
        session["experiment"],
        "--protocol_extra",
        json.dumps(session["protocol_extra"]),
    ]


def interleave(items: list[str], gpus: tuple[int, ...]) -> list[Shard]:
    return [Shard(g, tuple(items[i :: len(gpus)])) for i, g in enumerate(gpus)]


class Dataset:
    """Shared behaviour; subclasses set the class attributes and the dataset-specific methods."""

    name: str
    python: Path
    module: str
    router_module: str
    panel_path: Path
    panel_tag: str
    test_tag: str
    arms_root: Path
    test_root: Path
    test_gpus: tuple[int, ...]
    per_gpu: int = 8
    stall_s: int = 1200
    quality_def: str

    # ------------------------------------------------------------ ids
    def panel(self) -> dict:
        return json.loads(self.panel_path.read_text())

    def panel_ids(self) -> list[str]:
        return [x["item_id"] for x in self.panel()["items"]]

    # Memory policy (``memory_policy``, 2026-10-18): experiment ids, driver state and memory-arm result
    # directories carry it (success-only ids / dirs unchanged); No-Memory directories never do.
    def experiment_id(self) -> str:
        return mp.tag(f"{self.name}:{self.panel_tag}:bo3obj-v1")

    def test_experiment(self, arm: str) -> str:
        return mp.tag(f"{self.name}:{self.test_tag}:controlled-v1:{arm}")

    @property
    def out(self) -> Path:
        """Driver state (state.json, selection.json, logs, shard files): no task plaintext."""
        return mp.tag_dir(BO_OUT / self.name)

    def arm_dir(self, name: str) -> Path:
        """Profiling arm output (``bo-`` prefix keeps AppWorld's per-run experiment names unique)."""
        d = self.arms_root / f"bo-{name}"
        return d if name == "NoMemory" else mp.tag_dir(d)

    def test_dir(self, arm: str) -> Path:
        return (
            self.test_root / arm
            if arm == "NoMemory"
            else mp.tag_dir(self.test_root / arm)
        )

    @property
    def test_nomemory(self) -> Path:
        return self.test_root / "NoMemory"

    # ------------------------------------------------------------ result files
    def result_files(self, out: Path) -> dict[str, Path]:
        raise NotImplementedError

    def orphan_paths(self, out: Path, qid: str, path: Path) -> list[Path]:
        return [path]

    def unretrievable(self, out: Path) -> list[str]:
        """Memory-arm result files that can never get an opt row (rerunning would not help)."""
        return []

    def record(self, path: Path) -> dict:
        """quality / tokens / latency_ms of one result file (No-Memory column, summaries)."""
        raise NotImplementedError

    def move_orphans(
        self, out: Path, files: dict[str, Path], orphan: list[str], stamp: str
    ) -> Path:
        dest = out / "_orphans" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        for q in orphan:
            for p in self.orphan_paths(out, q, files[q]):
                if p.exists():
                    shutil.move(
                        str(p), dest / (p.name if p == files[q] else f"memory_{p.name}")
                    )
        return dest

    # ------------------------------------------------------------ commands
    def env(self) -> dict[str, str]:
        return {
            **RUN_ENV,
            "PYTHONPATH": str(ROOT),
            mp.ENV_VAR: mp.current(),
        }  # pin the policy

    def shards(
        self, items: list[str], gpus: tuple[int, ...], *, stage: str
    ) -> list[Shard]:
        return interleave(items, gpus)

    def selection_args(
        self, shard: Shard, *, stage: str, full: bool, workdir: Path
    ) -> list[str]:
        raise NotImplementedError

    def command(
        self, shard: Shard, out: Path, session: dict | None, sel: list[str]
    ) -> list[str]:
        raise NotImplementedError


class AppWorld(Dataset):
    name = "appworld"
    python = ROOT / "vendor/appworld-venv-0.2/bin/python"
    module = "trimem.experiments.appworld.run"
    router_module = "trimem.experiments.appworld.router_run"
    panel_path = ROOT / "data/external/panels/appworld_traindev_v02.json"
    panel_tag = "traindev_v02"
    test_tag = "test_normal168"
    # AppWorld plaintext must stay under data/ (common.ensure_ignored)
    arms_root = ROOT / "data/external/appworld-0.2/runs/bo3obj"
    test_root = ROOT / "data/external/appworld-0.2/runs/test_normal"
    test_gpus = (0,)
    per_gpu = 8  # --workers per shard
    stall_s = 1200
    quality_def = (
        "AppWorld task goal completion (TGC): 1 iff every official test passes"
    )

    def test_ids(self) -> list[str]:
        from trimem.experiments.appworld import common

        return common.load_split("test_normal")

    def result_files(self, out: Path) -> dict[str, Path]:
        from trimem.experiments.appworld.common import TASK_ID

        found = {}
        for f in Path(out).glob("*.json"):
            if not TASK_ID.match(f.stem):
                continue
            try:
                if json.loads(f.read_text())["task_id"] == f.stem:
                    found[f.stem] = f
            except (ValueError, KeyError, OSError):
                continue
        return found

    def orphan_paths(self, out: Path, qid: str, path: Path) -> list[Path]:
        return [path, Path(out) / "memory" / f"{qid}.json"]

    def record(self, path: Path) -> dict:
        r = json.loads(Path(path).read_text())
        return {
            "quality": float(r["quality"]),
            "tokens": float(r["generation_tokens"] + r["semantic_tokens"]),
            "latency_ms": float(r["episode_ms"] + r["retrieval_ms"]),
        }

    def selection_args(
        self, shard: Shard, *, stage: str, full: bool, workdir: Path
    ) -> list[str]:
        if stage == "profile":
            if full:
                return ["--panel", str(self.panel_path)]
            pnl = self.panel()
            by_id = {x["item_id"]: x for x in pnl["items"]}
            f = Path(workdir) / f"panel_gpu{shard.gpu}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "dataset": self.name,
                "panel_hash": pnl["panel_hash"],
                "shard_of": str(self.panel_path),
                "gpu": shard.gpu,
                "items": [by_id[i] for i in shard.items],
            }
            if not f.exists() or json.loads(f.read_text()) != body:
                f.write_text(json.dumps(body, indent=1))
            return ["--panel", str(f)]
        if full:
            return ["--split", "test_normal"]
        return ["--tasks", ",".join(shard.items)]

    def command(
        self, shard: Shard, out: Path, session: dict | None, sel: list[str]
    ) -> list[str]:
        base = llm_base(shard.gpu)
        module = (
            self.router_module if session and session.get("router") else self.module
        )
        cmd = [
            str(self.python),
            "-m",
            module,
            *sel,
            "--out_dir",
            str(out),
            "--workers",
            str(self.per_gpu),
            "--llm_base",
            f"{base}/v1",
            "--semantic_base",
            base,
            "--max_rss_gb",
            str(MAX_RSS_GB),
            "--recycle_every",
            str(APPWORLD_RECYCLE),
        ]
        if session is None:
            return [*cmd, "--no_memory"]
        return [
            *cmd,
            *(router_args(session) if session.get("router") else memory_args(session)),
        ]


TAU2_DOMAINS = ("airline", "retail", "telecom")
TAU2_DATA = ROOT / "vendor/tau2-bench/data/tau2/domains"


def tau2_split_ids(domain: str, split: str, data: Path = TAU2_DATA) -> list[str]:
    return [
        str(i)
        for i in json.loads((data / domain / "split_tasks.json").read_text())[split]
    ]


def tau2_key(domain: str, task_id: str) -> str:
    return f"{domain}:{task_id}"  # = entities.task_key (the opt query_id)


def tau2_selection(
    domains: list[str], task_ids: set[str] | None, universe: dict[str, list[str]]
) -> list[str]:
    """Keys the τ² runner selects for ``--domains`` / ``--task_ids`` (its own filter, re-stated)."""
    return [
        tau2_key(d, t)
        for d in domains
        for t in universe[d]
        if task_ids is None or t in task_ids
    ]


def tau2_shards(
    keys: list[str], gpus: tuple[int, ...], universe: dict[str, list[str]]
) -> list[Shard]:
    """Balanced shards whose runner selection is exactly the intended keys (see module docstring)."""
    if len(gpus) == 1:
        return [Shard(gpus[0], tuple(keys))]
    by_dom: dict[str, list[str]] = {}
    for k in keys:
        d, _ = k.split(":", 1)
        by_dom.setdefault(d, []).append(k)
    ids = {d: set(universe[d]) for d in universe}
    whole = [d for d in by_dom if any(ids[d] & ids[o] for o in universe if o != d)]
    split = [d for d in by_dom if d not in whole]
    parts: list[list[str]] = [[] for _ in gpus]
    for d in sorted(whole, key=lambda d: -len(by_dom[d])):
        min(parts, key=len).extend(by_dom[d])
    for d in split:
        for k in by_dom[d]:
            min(parts, key=len).append(k)
    out = [Shard(g, tuple(p)) for g, p in zip(gpus, parts, strict=True) if p]
    for s in out:
        doms = sorted({k.split(":", 1)[0] for k in s.items})
        sel = tau2_selection(doms, {k.split(":", 1)[1] for k in s.items}, universe)
        if sorted(sel) != sorted(s.items):
            raise ValueError(
                f"τ² shard on GPU{s.gpu} would select {len(sel)} tasks, intended {len(s.items)}"
            )
    return out


class Tau2(Dataset):
    name = "tau2"
    python = ROOT / ".venv-tau2/bin/python"
    module = "trimem.experiments.tau2.runner"
    router_module = "trimem.experiments.tau2.router_runner"
    panel_path = ROOT / "data/external/panels/tau2_train178.json"
    panel_tag = "train178"
    test_tag = "test100"
    arms_root = BO_OUT / "tau2" / "arms"
    test_root = ROOT / "results/tau2_test"
    test_gpus = (1,)
    per_gpu = 8  # --max_concurrency per shard
    stall_s = (
        1800  # an episode has two LLMs (agent + user simulator) and up to 200 steps
    )
    quality_def = "official τ² reward in [0, 1] (pass^1, one trial)"

    def universe(self, split: str) -> dict[str, list[str]]:
        return {d: tau2_split_ids(d, split) for d in TAU2_DOMAINS}

    def build_panel(self) -> dict:
        items = [
            {"item_id": tau2_key(d, t), "domain": d, "task_id": t, "split": "train"}
            for d in TAU2_DOMAINS
            for t in tau2_split_ids(d, "train")
        ]
        ids = [x["item_id"] for x in items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate τ² train keys")
        test = {tau2_key(d, t) for d in TAU2_DOMAINS for t in tau2_split_ids(d, "test")}
        if test & set(ids):
            raise ValueError("a test task is in the profiling panel")
        return {
            "dataset": "tau2",
            "split": "train",
            "size": len(items),
            "seed": None,
            "source": "vendor/tau2-bench/data/tau2/domains/{airline,retail,telecom}/split_tasks.json (train)",
            "stratified_by": None,
            "counts": {d: sum(x["domain"] == d for x in items) for d in TAU2_DOMAINS},
            "excluded_test_items": [],
            "user_decision": "2026-10-16 train 178",
            "panel_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
            "items": items,
        }

    def write_panel(self) -> dict:
        p = self.build_panel()
        if self.panel_path.exists():
            old = json.loads(self.panel_path.read_text())
            if old["panel_hash"] != p["panel_hash"]:
                raise SystemExit(
                    f"{self.panel_path} exists with another panel_hash; frozen panels are never rewritten"
                )
            return old
        self.panel_path.parent.mkdir(parents=True, exist_ok=True)
        self.panel_path.write_text(json.dumps(p, indent=1))
        return p

    def test_ids(self) -> list[str]:
        return [tau2_key(d, t) for d in TAU2_DOMAINS for t in tau2_split_ids(d, "test")]

    def _read(self, f: Path) -> dict | None:
        from trimem.experiments.tau2 import runner

        r = runner.read_result(f)
        if r is None or runner.result_name(r["domain"], str(r["task_id"])) != f.name:
            return None
        return r

    def result_files(self, out: Path) -> dict[str, Path]:
        found = {}
        for f in Path(out).glob("*.json"):
            r = self._read(f)
            if r is not None:
                found[tau2_key(r["domain"], str(r["task_id"]))] = f
        return found

    def unretrievable(self, out: Path) -> list[str]:
        bad = []
        for q, f in self.result_files(out).items():
            r = json.loads(f.read_text())
            if (
                r.get("mode") == "memory"
                and r.get("memory_status") == "no_first_user_message"
            ):
                bad.append(q)
        return bad

    def record(self, path: Path) -> dict:
        r = json.loads(Path(path).read_text())
        sem = (r.get("memory") or {}).get("semantic_tokens", 0) or 0
        return {
            "quality": float(r["quality"]),
            "tokens": float(r["agent_tokens"] + sem),
            "latency_ms": float(r["wall_ms"]),
        }

    def shards(
        self, items: list[str], gpus: tuple[int, ...], *, stage: str
    ) -> list[Shard]:
        return tau2_shards(
            items, gpus, self.universe("train" if stage == "profile" else "test")
        )

    def selection_args(
        self, shard: Shard, *, stage: str, full: bool, workdir: Path
    ) -> list[str]:
        split = "train" if stage == "profile" else "test"
        sel = ["--split", split]
        if full:
            return sel
        doms = [
            d for d in TAU2_DOMAINS if any(k.startswith(d + ":") for k in shard.items)
        ]
        tids = [k.split(":", 1)[1] for k in shard.items]
        if any("," in t for t in tids):
            raise ValueError("a τ² task id contains ','; --task_ids cannot carry it")
        if sorted(tau2_selection(doms, set(tids), self.universe(split))) != sorted(
            shard.items
        ):
            raise ValueError("τ² shard selection mismatch")
        return [*sel, "--domains", ",".join(doms), "--task_ids", ",".join(tids)]

    def command(
        self, shard: Shard, out: Path, session: dict | None, sel: list[str]
    ) -> list[str]:
        base = llm_base(shard.gpu)
        routed = bool(session and session.get("router"))
        cmd = [
            str(self.python),
            "-m",
            self.router_module if routed else self.module,
            "--mode",
            "nomemory" if session is None else "memory",
            *sel,
            "--out_dir",
            str(out),
            "--api_base",
            f"{base}/v1",
            "--user_api_base",
            f"{base}/v1",
            "--judge_api_base",
            f"{base}/v1",
            "--semantic_base",
            base,
            "--max_concurrency",
            str(self.per_gpu),
            "--max_rss_gb",
            str(MAX_RSS_GB),
        ]
        if session is None:
            return cmd
        return [*cmd, *(router_args(session) if routed else memory_args(session))]


DATASETS: dict[str, Dataset] = {"appworld": AppWorld(), "tau2": Tau2()}


def get(name: str) -> Dataset:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r} (choose from {sorted(DATASETS)})")
    return DATASETS[name]
