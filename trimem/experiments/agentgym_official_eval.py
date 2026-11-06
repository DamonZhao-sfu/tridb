"""AgentGym's official multiprocessing evaluator, with Qwen thinking disabled and calls logged.

Everything except the agent's HTTP request is AgentGym's own code
(``vendor/AgentGym/agentenv/examples/basic/openai_eval_multiprocessing.py``: process pool, the
Evaluator loop, ``max_round``, reward/success accounting, per-item output files). The only change
is ``APIAgent.generate``: it sends ``chat_template_kwargs.enable_thinking = false`` (the setting
of the SWE-QA-Pro and Math runs; the served Qwen3.8 thinks by default) and appends one JSON line
per call (tokens, latency) to ``<output_dir>/llm_calls.jsonl``. Sampling (temperature, top_p,
max_tokens) is passed through the official CLI arguments.

Run with ``.venv-agentgym``; the official env server (``alfworld`` / ``sciworld`` / ``webshop``) must be up:

    .venv-agentgym/bin/python -m trimem.experiments.agentgym_official_eval --api_key EMPTY \\
        --base_url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.8-27B-FP8 --temperature 0 \\
        --inference_file data/external/agentgym/alfworld_test.json --output_dir OUT \\
        --task_name alfworld --max_round 30 --env_server_base http://127.0.0.1:36001 --processes 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing.util
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "vendor/AgentGym/agentenv/examples/basic"
sys.path.insert(0, str(OFFICIAL))

import openai_eval_multiprocessing as official  # noqa: E402
import transformers  # noqa: E402
from agentenv.controller import APIAgent  # noqa: E402

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
CALL_LOG: Path | None = None
CURRENT_ITEM: int | None = (
    None  # set per episode by memory runners; logged with every call
)


class NoThinkingAPIAgent(APIAgent):
    """APIAgent.generate with thinking disabled; retry behaviour is the official one."""

    def generate(self, conversation):
        retry_ledger = None
        if any(os.environ.get(k) for k in ("TRIMEM_ABLATION_RESOURCE_LEDGER", "TRIMEM_ABLATION_MANIFEST_HASH", "TRIMEM_ABLATION_RESOURCE_LIMITS")):
            from trimem.experiments import scienceworld_trace_ablation as ab

            retry_ledger = ab.retry_budget_from_environment()
            self.client.max_retries = 0  # enforce the explicit shared retry ledger
        while True:
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": c["role"], "content": c["content"]}
                        for c in conversation
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body=NO_THINKING,
                )
                msg = r.choices[0].message
                u = getattr(r, "usage", None)
                if retry_ledger is not None and (
                    u is None or u.prompt_tokens is None or u.completion_tokens is None
                ):
                    raise SystemExit("agent API token usage unavailable; ablation paused")
                if CALL_LOG is not None:
                    with open(CALL_LOG, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "t": t0,
                                    "seconds": round(time.time() - t0, 3),
                                    "pid": os.getpid(),
                                    "item": CURRENT_ITEM,
                                    "prompt_tokens": u.prompt_tokens if u else None,
                                    "completion_tokens": u.completion_tokens
                                    if u
                                    else None,
                                    "finish_reason": r.choices[0].finish_reason,
                                }
                            )
                            + "\n"
                        )
                return msg.content, getattr(msg, "reasoning_content", None)
            except Exception as e:  # official behaviour: print and retry
                print(e)
                if retry_ledger is not None:
                    ab.reserve_api_retry(retry_ledger)
                time.sleep(1)


INNER_PROCESS = None
INNER_INIT = None


def close_env_clients() -> None:
    """Close this worker's env clients (frees the ScienceWorld JVM). Idempotent, never raises."""
    evaluator = getattr(official, "evaluator", None)
    for task in getattr(evaluator, "tasks", []) or []:
        for client in getattr(task, "clients", []) or []:
            if getattr(client, "_trimem_closed", False):
                continue
            try:
                client.close()
            except Exception as exc:  # server gone or env already reaped
                print(f"[env-close] {exc!r}", flush=True)
            client._trimem_closed = True


def init_with_env_cleanup(args):
    """Official pool initializer + cleanup. The official evaluator never closes its env clients and
    the pool ends its workers with SIGTERM, so each launch leaked one server-side env (one
    ScienceWorld JVM) per worker; on 2026-10-16 153 leaked JVMs took the host down. Workers now close
    their clients at exit and on SIGTERM (pool terminate, watchdog kill)."""
    INNER_INIT(args)
    # Pool workers leave via os._exit(), which skips atexit; multiprocessing's own exit finalizers
    # do run (normal exit), and SIGTERM (pool terminate, watchdog kill) is handled below.
    multiprocessing.util.Finalize(None, close_env_clients, exitpriority=10)

    def on_term(signum, frame):
        close_env_clients()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)


EPISODE_LOG = "episodes.jsonl"
# Drain (bo_pool_driver rebalancing, user 2026-10-19): when the driver sets this variable for a runner and creates
# the file, episodes not yet started are skipped (no result file, rerun on the next launch) while in-flight ones
# finish, so the runner exits soon without losing work. Unset (every other caller): no effect.
DRAIN_ENV = "TRIMEM_DRAIN_FILE"


def drained(out: Path) -> bool:
    """True when this runner is being drained and the item has no result yet (skip it, do not run it)."""
    p = os.environ.get(DRAIN_ENV)
    return bool(p) and Path(p).exists() and not out.exists()


def episode_cost(call_log: Path, data_idx: int, pid: int) -> tuple[int, int]:
    """(LLM calls, prompt + completion tokens) of one episode: this worker's calls for the item (the
    definition of ``agentgym_session_eval``'s ``generation_tokens``)."""
    if not call_log.exists():
        return 0, 0
    mine = [
        c
        for c in map(json.loads, open(call_log))
        if c.get("item") == data_idx and c.get("pid") == pid
    ]
    return len(mine), sum(
        (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in mine
    )


def process_logged(data_idx, args):
    """Module-level (picklable) wrapper: every LLM call of the episode is logged with its item id; a newly run
    episode appends {item, pid, episode_ms, llm_calls, generation_tokens} to ``episodes.jsonl`` (the
    episode wall time measured like ``agentgym_session_eval``'s ``episode_ms``; NoMemory arms' cost)."""
    global CURRENT_ITEM
    out = Path(args["output_dir"]) / f"{args['task_name']}_{data_idx}.json"
    if drained(
        out
    ):  # not started: the relaunched runner picks it up (official resume semantics)
        return 0.0, 0
    fresh = (
        not out.exists()
    )  # official resume: an existing result is not re-run (and not re-timed)
    CURRENT_ITEM = data_idx
    t0 = time.monotonic()
    try:
        return INNER_PROCESS(data_idx, args)
    finally:
        CURRENT_ITEM = None
        if fresh and out.exists() and CALL_LOG is not None:
            try:
                from trimem.experiments.agent_step_metrics import (
                    annotate_agentgym_result,
                )

                steps = annotate_agentgym_result(out)
                n, tok = episode_cost(CALL_LOG, data_idx, os.getpid())
                with open(Path(args["output_dir"]) / EPISODE_LOG, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "item": data_idx,
                                "pid": os.getpid(),
                                "episode_ms": round((time.monotonic() - t0) * 1000, 1),
                                "llm_calls": n,
                                "generation_tokens": tok,
                                **steps,
                            }
                        )
                        + "\n"
                    )
            except (
                OSError,
                ValueError,
            ) as exc:  # cost logging must never fail the episode
                print(f"[episode-log] {exc!r}", flush=True)


def with_action_format(task_cls: type, action_format: str) -> type:
    """Official Task with the official env client's ``action_format`` (react / function_calling /
    code_as_action); the official CLI does not expose it, the client constructor accepts it."""

    class Task(task_cls):
        def __init__(self, client_args, *args, **kwargs):
            super().__init__(
                {**client_args, "action_format": action_format}, *args, **kwargs
            )

    Task.__name__ = task_cls.__name__
    return Task


def main() -> None:
    global CALL_LOG
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--action_format",
        default="react",
        choices=("react", "function_calling", "code_as_action"),
    )
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    official.APIAgent = (
        NoThinkingAPIAgent  # pool workers fork after this, inheriting the patch
    )
    global INNER_PROCESS, INNER_INIT
    INNER_PROCESS = official.process
    official.process = process_logged
    INNER_INIT = official.init
    official.init = init_with_env_cleanup
    official.AlfWorldTask = with_action_format(
        official.AlfWorldTask, known.action_format
    )
    official.SciworldTask = with_action_format(
        official.SciworldTask, known.action_format
    )
    official.WebshopTask = with_action_format(official.WebshopTask, known.action_format)
    (args,) = transformers.HfArgumentParser(
        official.EvalArguments
    ).parse_args_into_dataclasses()
    args = vars(args)
    os.makedirs(args["output_dir"], exist_ok=True)
    CALL_LOG = Path(args["output_dir"]) / "llm_calls.jsonl"
    (Path(args["output_dir"]) / "run_args.json").write_text(
        json.dumps(
            args
            | {
                "thinking": "disabled (chat_template_kwargs.enable_thinking=false)",
                "action_format": known.action_format,
                "official_script": str(OFFICIAL / "openai_eval_multiprocessing.py"),
            },
            indent=2,
        )
    )
    official.main(args)


if __name__ == "__main__":
    main()
