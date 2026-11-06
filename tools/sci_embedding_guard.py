"""Record and recheck the pinned local embedding service used by an ablation."""

import hashlib
import json
import urllib.request
from pathlib import Path

from tools.scienceworld_ablation_identity import _listener_pid, _process_start

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def observe_embedding_service():
    pid = _listener_pid(8021)
    start = _process_start(pid)
    with urllib.request.urlopen("http://127.0.0.1:8021/health", timeout=5) as response:
        health = json.load(response)
    args = [
        x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x
    ]
    if (
        health
        != {"status": "ok", "model": MODEL, "revision": REVISION, "device": "cpu"}
        or "bench.agent_memory.evomembench.cpu_embedding_server" not in args
        or "--model-path" not in args
        or Path(args[args.index("--model-path") + 1]).name != REVISION
        or "--port" not in args
        or args[args.index("--port") + 1] != "8021"
        or "--threads" not in args
        or args[args.index("--threads") + 1] != "2"
        or _listener_pid(8021) != pid
        or _process_start(pid) != start
    ):
        raise ValueError("embedding service identity differs")
    source = (
        Path(__file__).resolve().parents[1]
        / "bench/agent_memory/evomembench/cpu_embedding_server.py"
    )
    return {
        "pid": pid,
        "start_ticks": start,
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "threads": 2,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limitation": "pinned offline model and process identity, not a direct attestation of loaded weights",
    }


def verify_embedding_service(root):
    expected = json.loads((Path(root) / "embedding_service_identity.json").read_text())
    if observe_embedding_service() != expected:
        raise ValueError("embedding service changed since campaign initialization")
    return True
