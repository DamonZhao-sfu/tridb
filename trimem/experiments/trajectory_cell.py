"""Experiment-local instrumentation. Core retrieval and prompt semantics are unchanged."""

import hashlib
import json
import os
import runpy
import time
from pathlib import Path


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def install():
    from bench.agent_memory.gem_oe.trajectory_prompt import TrajectoryPromptSampler
    from bench.agent_memory.gem_oe.trajectory_retriever import TrajectoryRetriever
    from trimem.code_embedding import FullCodeEmbedder

    if getattr(TrajectoryRetriever, "_experiment_instrumented", False):
        return
    original_retrieve = TrajectoryRetriever.retrieve
    original_embed = FullCodeEmbedder.embed
    original_embed_init = FullCodeEmbedder.__init__
    original_build = TrajectoryPromptSampler.build_prompt

    def embed_init(self, *args, **kwargs):
        # Device placement changes capacity only, not the full-code embedding contract.
        if not args and "device" not in kwargs:
            kwargs["device"] = os.environ.get("TRIMEM_EMBEDDING_DEVICE", "cuda:0")
        original_embed_init(self, *args, **kwargs)
        self.torch.set_num_threads(int(os.environ.get("TRIMEM_EMBEDDING_THREADS", "1")))

    def embed(self, code):
        started = time.perf_counter()
        try:
            return original_embed(self, code)
        finally:
            self.experiment_last_embed_ms = (time.perf_counter() - started) * 1000

    def retrieve(self, **kwargs):
        started = time.perf_counter()
        row = {
            "iteration": kwargs["iteration"],
            "parent_id": kwargs["parent"].id,
            "plan": self.plan,
            "started_unix": time.time(),
            "status": "failed",
        }
        try:
            selected = original_retrieve(self, **kwargs)
            row.update(
                status="success",
                selected=[{"uid": p.uid, "provenance": p.provenance} for p in selected],
            )
            return selected
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            row["retrieval_total_ms"] = (time.perf_counter() - started) * 1000
            row["embed_ms"] = getattr(self.embedder, "experiment_last_embed_ms", None)
            row["timing_scope"] = (
                "retrieval inclusive of embedding load/inference, SQL and semantic wait; not additive with semantic API time"
            )
            if self.out:
                append(self.out / "retrieval_audit.jsonl", row)

    def build(self, *args, **kwargs):
        prompt = original_build(self, *args, **kwargs)
        root = os.environ.get("TRIMEM_EXPERIMENT_CELL")
        if root:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
            digest = hashlib.sha256(
                json.dumps(
                    messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            target = Path(root) / "generation_prompts" / f"{digest}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
            append(
                Path(root) / "generation_prompt_index.jsonl",
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "sha256": digest,
                    "path": str(target),
                    "iteration": getattr(self, "iteration", None),
                },
            )
        return prompt

    FullCodeEmbedder.embed = embed
    FullCodeEmbedder.__init__ = embed_init
    TrajectoryRetriever.retrieve = retrieve
    TrajectoryRetriever._experiment_instrumented = True
    TrajectoryPromptSampler.build_prompt = build


# Spawned OpenEvolve workers import this module as __mp_main__, installing logging there too.
install()
if __name__ == "__main__":
    runpy.run_module("bench.agent_memory.gem_oe.run_arm", run_name="__main__")
