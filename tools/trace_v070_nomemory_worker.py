"""One NoMemory AgentGym episode through the official evaluator, under the TRACE episodes' strict patches.

Run with ``.venv-agentgym`` (as ``tools.trace_joint_episode_worker``). The hashed TRACE worker module is imported
read-only and only its two strict functions are reused: fail-fast episodes (``strict_process``: the official
``process`` with ``episode_fail_fast``) and fail-fast, token-logged API calls (``strict_generate``: no client
retries, 120 s timeout, every call appended to ``llm_calls.jsonl`` with its item). Everything else is
``agentgym_official_eval.main``: the official process pool, Evaluator loop, ``max_round`` and result files, plus
``process_logged`` (``episodes.jsonl``: episode_ms, generation_tokens, agent steps). No memory, no retrieval.
"""

from __future__ import annotations


def main() -> None:
    import tools.trace_joint_episode_worker as w
    from trimem.experiments import agentgym_official_eval as base

    official = base.official
    w.ORIGINAL_PROCESS = official.process
    official.process = w.strict_process
    base.NoThinkingAPIAgent.generate = w.strict_generate
    base.main()


if __name__ == "__main__":
    main()
