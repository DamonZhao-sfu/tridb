"""Render a real OpenEvolve prompt from current retrieval, without running generation.

Reuses a recorded OpenEvolve parent state, retrieves against the rebuilt corpus,
and uses OpenEvolve's actual sampler and memory registration. Semantic judging may
call the LLM; assembling the inspiration cards never does.
"""

import argparse
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from openevolve.config import DatabaseConfig, PromptConfig

from bench.agent_memory.gem_oe.memory_database import GemMemoryDatabase
from bench.agent_memory.gem_oe.trajectory_prompt import TrajectoryPromptSampler
from bench.agent_memory.gem_oe.trajectory_retriever import TrajectoryRetriever
from trimem.llm import OpenAIJudge
from trimem.objective.store import count_tokens
from trimem.worker import Worker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", default="P2+Rank")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dsn = "host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
    source = Path(
        "results/trimem_v0.1.0/math_sem/it20/heilbronn_triangle__msem-p2__s42/evolution_trace.jsonl"
    )
    history = [json.loads(line) for line in source.open()]
    trace = next(row for row in history if row["iteration"] == 6)
    own_programs = recorded_programs(history, 6)
    parent = SimpleNamespace(
        id=trace["parent_id"],
        code=trace["parent_code"],
        metrics=trace["parent_metrics"],
        metadata={},
    )
    worker = Worker(
        dsn, OpenAIJudge("http://127.0.0.1:8000/v1", concurrency=2), max_take=2
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    retriever = TrajectoryRetriever(dsn, plan=args.plan, out=args.out)
    try:
        records = retriever.retrieve(
            task_uid="math:heilbronn_triangle", parent=parent, k=4, iteration=6
        )
        db = GemMemoryDatabase(
            DatabaseConfig(),
            retriever=retriever,
            task_uid="math:heilbronn_triangle",
            render_mode="code",
        )
        programs = [db._register(r).to_dict() for r in records]
        spec = retriever.conn.execute(
            "select specification from task where task=%s", ("math:heilbronn_triangle",)
        ).fetchone()["specification"]
        cfg = PromptConfig()
        cfg.num_top_programs = 1
        cfg.num_diverse_programs = 4
        cfg.template_dir = str(
            Path("bench/agent_memory/gem_oe/prompts/trajectory").resolve()
        )
        cfg.suggest_simplification_after_chars = None
        cfg.code_length_threshold = None
        cfg.system_message = spec
        sampler = TrajectoryPromptSampler(
            cfg,
            own_programs=own_programs,
            parent_id=parent.id,
            iteration=6,
            artifacts_by_program={
                p["id"]: p["artifacts"] for p in own_programs if p.get("artifacts")
            },
        )
        prompt = sampler.build_prompt(
            current_program=parent.code,
            program_metrics=parent.metrics,
            inspirations=programs,
            language="python",
            evolution_round=6,
            diff_based_evolution=True,
            feature_dimensions=["complexity", "diversity"],
        )
        messages = [
            {"role": role, "content": prompt[role]} for role in ("system", "user")
        ]
        (args.out / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2)
        )
        (args.out / "full_prompt.txt").write_text(
            "[SYSTEM MESSAGE]\n"
            + prompt["system"]
            + "\n\n[USER MESSAGE]\n"
            + prompt["user"]
        )
        (args.out / "inspirations.json").write_text(
            json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2)
        )
        assert parent.code in prompt["user"]
        assert all(r.changes_description in prompt["user"] for r in records)
        assert all(r.code in prompt["user"] for r in records)
        sha = hashlib.sha256(
            json.dumps(
                messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        report = {
            "example_scope": "New retrieval + actual OpenEvolve renderer using recorded iteration-6 parent. Generation not executed.",
            "historical_parent_source": str(source),
            "plan": args.plan,
            "inspirations": len(records),
            "messages_sha256": sha,
            "prompt_body_tokens_excluding_chat_template": sum(
                count_tokens(m["content"]) for m in messages
            ),
            "full_parent_code_verified": True,
            "all_cards_verified_in_prompt": True,
            "full_inspiration_code_verified": True,
            "system_matches_task_original": prompt["system"] == spec,
            "own_context": sampler.context_receipt,
            "post_retrieval_summary_llm_calls": 0,
            "semantic_worker": asdict(worker.stats),
            "retrieval": retriever.telemetry[-1],
        }
        (args.out / "receipt.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        retriever.close()
        worker.stop()
        thread.join(timeout=5)


def recorded_programs(history, before_iteration):
    """Reconstruct only programs known before the selected generation request."""
    programs = {}
    for row in history:
        if row["iteration"] >= before_iteration:
            break
        programs.setdefault(
            row["parent_id"],
            {
                "id": row["parent_id"],
                "code": row["parent_code"],
                "metrics": row["parent_metrics"],
                "language": "python",
                "iteration_found": 0,
            },
        )
        programs[row["child_id"]] = {
            "id": row["child_id"],
            "code": row["child_code"],
            "language": "python",
            "metrics": row["child_metrics"],
            "parent_id": row["parent_id"],
            "iteration_found": row["iteration"],
            "metadata": {
                **row.get("metadata", {}),
                "parent_metrics": row["parent_metrics"],
            },
            "artifacts": row.get("artifacts", {}),
        }
    return list(programs.values())


if __name__ == "__main__":
    main()
