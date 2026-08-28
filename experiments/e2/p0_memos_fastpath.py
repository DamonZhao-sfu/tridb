"""Does MemOS have a non-LLM ingest path? An 8-hour question.

    MEMOS_NEO4J_PASSWORD=... /localhome/hza214/agent-memory-table5/venv/memos/bin/python \
        experiments/e2/p0_memos_fastpath.py

`MOSCore.add` takes either `messages=` or `memory_content=`. The `messages` path runs
mem_reader, an LLM extraction step, and was measured at 9,730 ms/node — 8.65 hours for
the 3,200-node corpus, the single largest cost in the cross-system comparison. If
`memory_content` writes the text straight into tree_text memory, that cost disappears
and MemOS can hold the same corpus as everyone else.

Three things have to hold for the fast path to be usable, not just fast:
  * it must be materially faster than the messages path;
  * `node_uid` must still come back out of a search;
  * the stored text must be OURS, not an LLM's paraphrase — otherwise the corpus the
    system holds is not the corpus the ground truth describes.
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
STATE = Path(f"/tmp/w1x_memosfast_{RUN}")
N = 6

RESULT: dict[str, object] = {"system": "memos", "probe": "fastpath", "run": RUN}


def node(i: int) -> dict:
    uid = f"349117b0:evox/fast_s#prog-{i}"
    return {
        "node_uid": uid,
        "text": (
            f"[node_uid={uid}] Language: python. Outcome: accepted. "
            f"Edit: rewrite variant {i}. Code: def solve_{i}(x): return sum(x) * {i}"
        ),
    }


NODES = [node(i) for i in range(N)]


def main() -> int:
    from memos.mem_cube.general import GeneralMemCube
    from memos.mem_os.core import MOSCore
    from memos.mem_os.utils.default_config import (
        get_default_config,
        get_default_cube_config,
    )
    from memos.mem_user.user_manager import UserManager

    STATE.mkdir(parents=True, exist_ok=True)

    def build(tag: str):
        user_id, cube_id = f"fast_{RUN}_{tag}", f"cube_{RUN}_{tag}"
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
            cfg.config.embedding_dims = None
        manager = UserManager(db_path=str(STATE / f"{tag}.sqlite3"), user_id=user_id)
        mos = MOSCore(mos_config, user_manager=manager)
        mos.register_mem_cube(GeneralMemCube(cube_config), mem_cube_id=cube_id, user_id=user_id)
        return user_id, cube_id, mos

    def measure(tag: str, use_memory_content: bool) -> dict:
        user_id, cube_id, mos = build(tag)
        started = time.perf_counter()
        errors: list[str] = []
        for item in NODES:
            try:
                if use_memory_content:
                    mos.add(
                        memory_content=item["text"],
                        mem_cube_id=cube_id,
                        user_id=user_id,
                        session_id="349117b0:evox/fast_s",
                    )
                else:
                    mos.add(
                        messages=[{"role": "user", "content": item["text"]}],
                        mem_cube_id=cube_id,
                        user_id=user_id,
                        session_id="349117b0:evox/fast_s",
                    )
            except Exception as exc:  # noqa: BLE001 - the probe reports rather than raises
                errors.append(f"{type(exc).__name__}: {exc}"[:200])
        seconds = time.perf_counter() - started

        found: list[str] = []
        verbatim = None
        search_ms = None
        try:
            t = time.perf_counter()
            result = mos.search(
                "python solve rewrite variant",
                user_id=user_id,
                install_cube_ids=[cube_id],
                top_k=10,
                mode="fast",
            )
            search_ms = round((time.perf_counter() - t) * 1000, 1)
            memories: list = []
            for cube_result in result.get("text_mem", []):
                memories.extend(cube_result.get("memories", []))
            blob = "\n".join(str(m) for m in memories)
            found = list(dict.fromkeys(MARKER.findall(blob)))
            # Verbatim means the stored text is ours. A paraphrase would mean the
            # corpus MemOS holds is not the corpus the ground truth describes.
            originals = {n["node_uid"]: n["text"] for n in NODES}
            verbatim = any(originals[u] in blob for u in found if u in originals)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"search: {type(exc).__name__}: {exc}"[:200])

        return {
            "nodes": len(NODES),
            "seconds": round(seconds, 2),
            "ms_per_node": round(seconds * 1000 / len(NODES), 1),
            "projected_hours_3200": round(seconds / len(NODES) * 3200 / 3600, 2),
            "errors": errors[:3],
            "node_uids_recovered": len(found),
            "all_known": all(u in {n["node_uid"] for n in NODES} for u in found),
            "text_verbatim": verbatim,
            "search_ms": search_ms,
        }

    RESULT["add_signature"] = str(inspect.signature(MOSCore.add))
    RESULT["fast_memory_content"] = measure("fast", True)
    RESULT["baseline_messages"] = measure("base", False)

    fast = RESULT["fast_memory_content"]["ms_per_node"]
    base = RESULT["baseline_messages"]["ms_per_node"]
    RESULT["speedup"] = round(base / fast, 2) if fast else None
    RESULT["hours_saved_on_3200"] = round(
        (RESULT["baseline_messages"]["projected_hours_3200"]
         - RESULT["fast_memory_content"]["projected_hours_3200"]),
        2,
    )
    print(json.dumps(RESULT, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
