"""P0 gate for MemOS: can it express the W1 query family, and at what ingest cost?

Run inside the memos venv, with Neo4j already up:

    MEMOS_NEO4J_PASSWORD=... scripts/table5_track_c_memos_neo4j.sh start w1x_p0
    /localhome/hza214/agent-memory-table5/venv/memos/bin/python \
        experiments/e2/p0_gate_memos.py

Like Cognee, MemOS exposes no per-passage metadata filter on search, so the identifier
travels inside the text and scoping has to be structural. Its structural unit is the
MemCube, so the design is **one cube per session** — the same shape Cognee needs from
its datasets, and for the same reason: `session_uid != target` is the core cross-session
constraint and a cube list is the only place it can be expressed.

The interesting extra question here is G4: MemOS's `tree_text` memory is an explicit
hierarchy on Neo4j. If a user-defined parent/child structure can be imposed on it, the
lineage queries (W1.b / W1.d) are not automatically unsupported.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import uuid
from pathlib import Path

EMBED_DIM = 1024
MARKER = re.compile(r"\[node_uid=([^\]]+)\]")
RUN = uuid.uuid4().hex[:8]
STATE = Path(f"/tmp/w1x_memos_{RUN}")

RESULT: dict[str, object] = {"system": "memos", "run": RUN}

SESSIONS = ["s_t0_a", "s_t0_b", "s_t1_a"]
NODES = [
    {
        "node_uid": f"349117b0:evox/{s}#prog-{i}",
        "session_uid": f"349117b0:evox/{s}",
        "task_uid": "math:task_0" if s.startswith("s_t0") else "math:task_1",
        "text": (
            f"[node_uid=349117b0:evox/{s}#prog-{i}] Language: python. "
            f"Outcome: {'accepted' if i % 3 else 'rejected'}. "
            f"Edit: rewrite variant {i}. Code: def solve_{i}(x): return sum(x) * {i}"
        ),
    }
    for s in SESSIONS
    for i in range(3)
]


def main() -> int:
    try:
        from memos.configs.mem_cube import GeneralMemCubeConfig  # noqa: F401
        from memos.configs.mem_os import MOSConfig  # noqa: F401
        from memos.mem_cube.general import GeneralMemCube
        from memos.mem_os.core import MOSCore
        from memos.mem_user.user_manager import UserManager
        from memos.mem_os.utils.default_config import (
            get_default_config,
            get_default_cube_config,
        )
    except Exception as exc:  # noqa: BLE001
        RESULT["fatal_import"] = f"{type(exc).__name__}: {exc}"[:400]
        print(json.dumps(RESULT, indent=2, default=str))
        return 0

    STATE.mkdir(parents=True, exist_ok=True)
    RESULT["search_signature"] = str(inspect.signature(MOSCore.search))
    RESULT["add_signature"] = str(inspect.signature(MOSCore.add))

    def build_scope(session: str):
        user_id = f"w1x_{RUN}_{session}"
        cube_id = f"cube_{RUN}_{session}"
        common = dict(
            openai_api_key="EMPTY",
            openai_api_base="http://127.0.0.1:8000/v1",
            text_mem_type="tree_text",
            user_id=user_id,
            model_name="Qwen/Qwen3-32B",
            embedder_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_dimension=EMBED_DIM,
            neo4j_uri="bolt://127.0.0.1:17687",
            neo4j_user="neo4j",
            neo4j_password=os.environ.get("MEMOS_NEO4J_PASSWORD", "w1xprobe_neo4j"),
            neo4j_db_name="neo4j",
            neo4j_auto_create=False,
            use_multi_db=False,
            enable_reorganize=False,
            temperature=0.0,
            max_tokens=512,
            top_k=10,
            cube_id=cube_id,
        )
        mos_config = get_default_config(**common)
        cube_config = get_default_cube_config(**common)
        for cfg in (mos_config.mem_reader.config.embedder, cube_config.text_mem.config.embedder):
            cfg.config.base_url = "http://127.0.0.1:8001/v1"
            cfg.config.model_name_or_path = "Qwen/Qwen3-Embedding-0.6B"
            # Qwen3-Embedding is fixed 1024-d and rejects OpenAI's `dimensions` param.
            cfg.config.embedding_dims = None
        manager = UserManager(db_path=str(STATE / f"{session}.sqlite3"), user_id=user_id)
        mos = MOSCore(mos_config, user_manager=manager)
        cube = GeneralMemCube(cube_config)
        mos.register_mem_cube(cube, mem_cube_id=cube_id, user_id=user_id)
        return user_id, cube_id, mos

    try:
        scopes = {s: build_scope(s) for s in SESSIONS}
    except Exception as exc:  # noqa: BLE001
        RESULT["fatal_build"] = f"{type(exc).__name__}: {exc}"[:600]
        print(json.dumps(RESULT, indent=2, default=str))
        return 0

    # -- G2: ingest cost (mem_reader extraction is LLM-driven) -----------
    started = time.perf_counter()
    errors: list[str] = []
    for node in NODES:
        user_id, cube_id, mos = scopes[node["node_uid"].split("/")[1].split("#")[0]]
        try:
            mos.add(
                messages=[{"role": "user", "content": node["text"]}],
                mem_cube_id=cube_id,
                user_id=user_id,
                session_id=node["session_uid"],
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}"[:200])
    seconds = time.perf_counter() - started
    RESULT["G2_ingest_seconds"] = round(seconds, 2)
    RESULT["G2_ms_per_node"] = round(seconds * 1000 / len(NODES), 1)
    RESULT["G2_projected_hours_3200_nodes"] = round(seconds / len(NODES) * 3200 / 3600, 2)
    RESULT["G2_errors"] = errors[:3]

    # -- G1 + G3: identifier recovery and cube-level session exclusion ---
    known = {n["node_uid"] for n in NODES}
    target = "s_t0_a"
    probes: dict[str, object] = {}
    for label, cubes in (
        ("all_task0_sessions", ["s_t0_a", "s_t0_b"]),
        # The W1 scoping: task_0 minus the target session.
        ("task0_excluding_target", ["s_t0_b"]),
    ):
        try:
            user_id, _, mos = scopes[cubes[0]]
            cube_ids = [scopes[c][1] for c in cubes]
            t = time.perf_counter()
            result = mos.search(
                "python solve rewrite variant",
                user_id=user_id,
                install_cube_ids=cube_ids,
                top_k=10,
                mode="fast",
            )
            elapsed = (time.perf_counter() - t) * 1000
            memories: list = []
            for cube_result in result.get("text_mem", []):
                memories.extend(cube_result.get("memories", []))
            blob = "\n".join(str(m) for m in memories)
            found = list(dict.fromkeys(MARKER.findall(blob)))
            probes[label] = {
                "ok": True,
                "ms": round(elapsed, 1),
                "memories": len(memories),
                "node_uids_recovered": len(found),
                "all_known": all(u in known for u in found),
                "target_session_leaked": any(f"/{target}#" in u for u in found),
                "sample": found[:3],
            }
        except Exception as exc:  # noqa: BLE001
            probes[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    RESULT["G1_G3_probes"] = probes

    print(json.dumps(RESULT, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
