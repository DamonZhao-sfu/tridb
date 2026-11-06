"""AppWorld file access shared by the store builder, adapter and runner (no ``appworld`` import).

Everything here reads the released data directory (``$APPWORLD_ROOT/data``) directly, so it runs in
the repo ``.venv`` as well as in ``vendor/appworld-venv-0.2`` / ``vendor/appworld-venv``. Default root:
data 0.2.0 (``data/external/appworld-0.2``, the HEAD agent); data 0.1.0 (``data/external/appworld``,
the v0.1.3.post1 agent) via ``APPWORLD_ROOT``. Same task ids and splits in both versions.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "data/external/appworld-0.2"
SPLITS = ("train", "dev", "test_normal", "test_challenge")
# opt.session_retrieval.split only admits train / test / smoke: train + dev are the profiling split.
OPT_SPLIT = {
    "train": "train",
    "dev": "train",
    "test_normal": "test",
    "test_challenge": "test",
    "smoke": "smoke",
}
TASK_ID = re.compile(r"^([0-9a-f]+)_(\d+)$")


def appworld_root() -> Path:
    return Path(os.environ.get("APPWORLD_ROOT", DEFAULT_ROOT))


def data_dir(root: Path | None = None) -> Path:
    return (root or appworld_root()) / "data"


def scenario_of(task_id: str) -> str:
    """Task ids are ``<scenario>_<variant>``; the 3 variants of a scenario are one family."""
    m = TASK_ID.match(task_id)
    if not m:
        raise ValueError(f"not an AppWorld task id: {task_id!r}")
    return m.group(1)


def variant_of(task_id: str) -> int:
    scenario_of(task_id)
    return int(task_id.rsplit("_", 1)[1])


def load_split(split: str, root: Path | None = None) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown AppWorld split {split!r}")
    return (data_dir(root) / "datasets" / f"{split}.txt").read_text().split()


def task_dir(task_id: str, root: Path | None = None) -> Path:
    scenario_of(task_id)
    return data_dir(root) / "tasks" / task_id


def load_spec(task_id: str, root: Path | None = None) -> dict:
    """Instruction, supervisor and datetime (every split has ``specs.json``)."""
    return json.loads((task_dir(task_id, root) / "specs.json").read_text())


def load_ground_truth(task_id: str, root: Path | None = None) -> dict:
    """Train / dev ground truth: API calls, metadata, required apps. Test ground truth is not released
    in plaintext; asking for it raises."""
    gt = task_dir(task_id, root) / "ground_truth"
    calls = gt / "api_calls.json"
    if not calls.exists():
        raise FileNotFoundError(
            f"{task_id}: no plaintext ground truth (held-out split)"
        )
    apps_file = gt / "required_apps.json"
    return {
        "api_calls": json.loads(calls.read_text()),
        "metadata": json.loads((gt / "metadata.json").read_text()),
        "required_apps": json.loads(apps_file.read_text())
        if apps_file.exists()
        else [],
    }


@lru_cache(maxsize=4)
def api_index(root: str) -> list[tuple[str, re.Pattern, str, list[str]]]:
    """(METHOD, path regex, "app.api_name", path parameter names) for every documented API."""
    out = []
    for f in sorted((Path(root) / "data/api_docs/standard").glob("*.json")):
        for name, doc in json.loads(f.read_text()).items():
            if not isinstance(doc, dict) or "path" not in doc:
                continue
            params = re.findall(r"\{(\w+)\}", doc["path"])
            rx = re.compile(
                "^" + re.sub(r"\\\{\w+\\\}", r"[^/]+", re.escape(doc["path"])) + "$"
            )
            out.append(
                (
                    doc["method"].upper(),
                    rx,
                    f"{doc['app_name']}.{doc['api_name']}",
                    params,
                )
            )
    # Literal paths before templated ones so /x/me never matches /x/{id}.
    out.sort(key=lambda r: (len(r[3]), -len(r[1].pattern)))
    return out


def api_name(method: str, url: str, root: Path | None = None) -> tuple[str, list[str]]:
    path = url.split("?", 1)[0]
    for m, rx, name, params in api_index(str(root or appworld_root())):
        if m == method.upper() and rx.match(path):
            return name, params
    return f"{method.upper()} {path}", []


def gt_steps(task_id: str, root: Path | None = None) -> list[dict]:
    """Ground-truth API calls as session steps: ``apis.<app>.<api>(arg names)``. Argument values
    (other users' emails, passwords, tokens) are not kept; the call order and the argument names are."""
    steps = []
    for call in load_ground_truth(task_id, root)["api_calls"]:
        name, path_params = api_name(call["method"], call["url"], root)
        args = [
            *path_params,
            *(k for k in (call.get("data") or {}) if k not in path_params),
        ]
        action = (
            f"apis.{name}({', '.join(args)})"
            if not name.startswith(call["method"].upper())
            else name
        )
        steps.append(
            {
                "thought": None,
                "action": action,
                "verb": name,
                "observation": None,
                "score": None,
            }
        )
    return steps


API_CALL = re.compile(r"apis\.(\w+)\.(\w+)\s*\(")


def code_verbs(code: str) -> list[str]:
    """``app.api`` names called in a code cell, in order (agent sessions)."""
    return [f"{a}.{b}" for a, b in API_CALL.findall(code or "") if a != "api_docs"]


def ensure_ignored(path: Path) -> Path:
    """AppWorld plaintext (instructions, observations, agent code) may only be written outside the
    repository or under its gitignored ``data/`` / ``vendor/`` trees."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO.resolve())
    except ValueError:
        return p
    if rel.parts[:1] in (("data",), ("vendor",)):
        return p
    raise ValueError(
        f"{p}: AppWorld outputs must live under {REPO}/data/ (gitignored) or outside the repo"
    )


def wall_ms(t0: float) -> float:
    return (now() - t0) * 1000


def now() -> float:
    """Monotonic seconds that AppWorld's freezegun does not freeze (``time.time`` / ``monotonic``
    stop inside an open world)."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def release_shell(world) -> bool:
    """Free the IPython shell of a closed AppWorld world. ``AppWorld.__init__`` builds one
    ``InteractiveShellEmbed`` per world and ``AppWorld.close`` never disposes it, while IPython's
    ``InteractiveShell.__init__`` registers ``atexit.register(self.atexit_operations)``: every shell,
    its namespace (``apis``, ``requester``, the task's executed code and objects) and its magics stay
    reachable until interpreter exit (~1.1 MB and ~7k objects per task). This does what IPython does
    at exit for this one shell (``atexit_operations``: reset namespaces, remove temp files), drops the
    atexit hook and the singleton slot if this shell holds it. Returns whether a shell was released."""
    shell = getattr(world, "shell", None)
    if shell is None:
        return False
    world.shell = None
    atexit.unregister(shell.atexit_operations)
    try:
        shell.atexit_operations()
    finally:
        cls = type(shell)
        if getattr(cls, "_instance", None) is shell:
            cls.clear_instance()
    return True


# FastAPI memoizes "is this callable a coroutine / generator" per callable (lru_cache, 4096 entries
# each). AppWorld builds a fresh FastAPI app (``build_main_app`` / ``setup_app``) per world, so every
# task adds ~21 route closures per cache, each holding its world's app objects, until the caches fill.
ROUTE_CACHES = (
    ("fastapi.dependencies.models", "_is_gen_callable_cached"),
    ("fastapi.dependencies.models", "_is_async_gen_callable_cached"),
    ("fastapi.dependencies.models", "_is_coroutine_callable_cached"),
)


def clear_route_caches() -> int:
    """Clear the FastAPI per-callable caches (pure memoization: safe). Returns entries dropped."""
    import sys

    dropped = 0
    for mod, name in ROUTE_CACHES:
        fn = getattr(sys.modules.get(mod), name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            dropped += fn.cache_info().currsize
            fn.cache_clear()
    return dropped


def release_shells_on_close(appworld_cls) -> None:
    """Wrap ``AppWorld.close`` (idempotent): the official close runs unchanged, then the world's shell
    is released (``release_shell``). The official agents evaluate before closing (0.1.3 answerer) or
    from the saved end state (HEAD evaluator), so nothing reads the shell after close."""
    official = appworld_cls.close
    if getattr(official, "_trimem_releases_shell", False):
        return

    def close(self, *args, **kwargs):
        try:
            return official(self, *args, **kwargs)
        finally:
            release_shell(self)
            clear_route_caches()

    close._trimem_releases_shell = True
    close.__wrapped__ = official
    appworld_cls.close = close
