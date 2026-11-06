"""One AgentGym episode (ALFWorld / ScienceWorld / WebShop) through the native retrieval runner.

Process-local adaptations only: fail-fast API calls and raw-score opt receipts.
The frozen memory graph is read-only; new opt receipts use a unique experiment.
"""

import json
import math
import os
import time

ORIGINAL_PROCESS = None


def strict_process(index, args):
    return ORIGINAL_PROCESS(index, {**args, "episode_fail_fast": True})


def strict_generate(self, conversation):
    from trimem.experiments import agentgym_official_eval as base

    began = time.time()
    call = {"item": base.CURRENT_ITEM, "pid": os.getpid(), "t": began}
    try:
        r = self.client.with_options(
            max_retries=0, timeout=120
        ).chat.completions.create(
            model=self.model,
            messages=[
                {"role": c["role"], "content": c["content"]} for c in conversation
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=base.NO_THINKING,
        )
        if (
            r.usage is None
            or r.usage.prompt_tokens is None
            or r.usage.completion_tokens is None
        ):
            raise ValueError("Agent token usage unavailable")
        call.update(
            prompt_tokens=r.usage.prompt_tokens,
            completion_tokens=r.usage.completion_tokens,
            finish_reason=r.choices[0].finish_reason,
            status="ok",
        )
        return r.choices[0].message.content, getattr(
            r.choices[0].message, "reasoning_content", None
        )
    except Exception as exc:
        call.update(status="error", error=repr(exc))
        raise
    finally:
        call["seconds"] = time.time() - began
        if base.CALL_LOG is not None:
            with base.CALL_LOG.open("a") as f:
                f.write(json.dumps(call) + "\n")


def main():
    global ORIGINAL_PROCESS
    import psycopg
    from psycopg.types.json import Jsonb

    from trimem.experiments import agentgym_official_eval as base
    from trimem.experiments import agentgym_session_eval as evaluator
    from trimem.experiments.session_store_ext import ScalarSessionStore

    class RawScoreStore(ScalarSessionStore):
        def evaluate_scalar(self, rid, quality, native, baseline_quality=None):
            reward = native.get("reward")
            if (
                reward is None
                or not math.isfinite(float(reward))
                or float(quality) != float(reward)
            ):
                raise ValueError("Missing or inconsistent environment raw score")
            with psycopg.connect(self.dsn) as conn:
                row = conn.execute(
                    "UPDATE opt.session_retrieval SET status='evaluated',judge=%s,quality=%s,"
                    "paired_gain=NULL,updated_at=now() WHERE retrieval_id=%s AND generation IS NOT NULL "
                    "RETURNING retrieval_id",
                    (
                        Jsonb(
                            {
                                "kind": "simulator",
                                "quality_source": "environment_raw_score",
                                **native,
                            }
                        ),
                        float(reward),
                        rid,
                    ),
                ).fetchone()
                if not row:
                    raise ValueError("Missing generation receipt")

    def semantic_tokens(calls):
        total = 0
        for call in calls:
            usage = call.get("usage") or {}
            value = usage.get("total_tokens")
            if type(value) is not int or value < 0:
                raise ValueError("Semantic token usage unavailable")
            total += value
        return total

    ORIGINAL_PROCESS = evaluator.official.process
    evaluator.official.process = strict_process
    base.NoThinkingAPIAgent.generate = strict_generate
    evaluator.ScalarSessionStore = RawScoreStore
    # The search objective is the unmodified environment reward: ALFWorld won
    # (0/1), ScienceWorld raw score (negative kept), WebShop reward in [0, 1].
    evaluator.quality_of = lambda env, reward: (
        float(reward)
        if env in ("alfworld", "sciworld", "webshop")
        else (_ for _ in ()).throw(ValueError(f"Unsupported environment {env}"))
    )
    for env in ("alfworld", "sciworld", "webshop"):
        evaluator.METRIC[env] = "raw_environment_score"
    evaluator.semantic_tokens = semantic_tokens
    evaluator.main()


if __name__ == "__main__":
    main()
