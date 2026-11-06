"""τ² pool-profiling DESIGN selector (user decision 2026-10-23: re-profile τ² in a NEW isolated run, ``pool-v6``).

``TAU2_POOL_DESIGN`` (unset / ``v5`` = the -v5 run's design, unchanged; ``explore-v6`` / ``v6`` = the re-profile)
is read at import by ``tau2.pool_profile`` AND ``tau2.runner_pool`` (the runner subprocesses inherit it), so both use
the same session space. Stdlib + space modules only (imported by the .venv-tau2 runner too).

explore-v6 (``EXPLORE_V6``; recorded once in state.json ``design``, part of the run identity check):

* space ``session-space-tau2-v6`` (= -v5 + plan P1; six plans proposable from round 0).
* rounds: round 0 = 10 Sobol configurations (quasi-random breadth draws over the proposal domain, sim_gate none,
  NO plan defaults); round 1 = 10 sobol-only-v1 picks (max posterior-SD product over the Sobol breadth rows, kriging
  believer); from round 2 channel-slots-v6 batches of 8 = the best of each channel (llm = analyst rows + dls-agent
  endpoints, dls, xover, sobol; per-channel rules of channel-slots-v5) + 4 hv picks = the top remaining qLogNEHVI
  candidates over the union of every channel's candidates; never stall-forced; no qfirst slot; the last round keeps
  the first k of the layout. Total 50 configurations (10 + 10 + 8 + 8 + 8 + 6), NoMemory extra.
* semfilter-v2 from round 0 (state.json ``semfilter``, the knobs of ``SEMFILTER_V2_ENV``, opt-in TAU2_SEMFILTER=v2),
  hv-ref-anchored-v1 and gp-fix-v1 from round 0, quality reuse / concurrency / async staging from round 0 (env of the
  launch script), staged pruning tau2-pool-staged-v2 (50 -> 100 -> 178, z 1.96, corner rule, plan guard).
"""

from __future__ import annotations

import os

DESIGN_ENV = "TAU2_POOL_DESIGN"
EXPLORE_V6 = "explore-v6"
ALIASES = {"": None, "v5": None, "v6": EXPLORE_V6, EXPLORE_V6: EXPLORE_V6}

ROUND0_SOBOL = 10
RANDOM_UNTIL_ROUND = 1  # sobol-only-v1 rounds 1..1
FIRST_BATCHES = {0: 10, 1: 10}
BATCH = 8
TOTAL_CONFIGS = 50
SEMFILTER_V2_ENV = {  # = the τ² sfv2 supersede's recorded env = ALF / Sci v7 (see tau2.fixed_plans_test.SEMFILTER_V2)
    "TRIMEM_SEMFILTER_V2": "1",
    "TRIMEM_SEMFILTER_TOPN": "32",
    "TRIMEM_SEMFILTER_BATCH": "8",
    "TRIMEM_SEMFILTER_MAX_TOKENS": "96",
    "TRIMEM_SEMFILTER_JUDGE_INPUT": "raw",
    "TRIMEM_SEMFILTER_RUBRIC": "semantic-rubric-v3",
    "TRIMEM_SEMFILTER_GUIDED": "1",
}


def design(env=None) -> str | None:
    env = os.environ if env is None else env
    raw = env.get(DESIGN_ENV, "").strip()
    if raw not in ALIASES:
        raise ValueError(
            f"{DESIGN_ENV}={raw!r}: expected one of {sorted(k for k in ALIASES if k)}"
        )
    return ALIASES[raw]


def space_module(env=None):
    """The τ² session space of the selected design (-v5 module unless explore-v6)."""
    if design(env) == EXPLORE_V6:
        from trimem.optimizer import session_space_tau2_v6 as space
    else:
        from trimem.optimizer import session_space_tau2_v2 as space
    return space


def batch_for(r: int, batch: int = BATCH) -> int:
    """Requested batch of round ``r`` (before the total-configuration cap). Pure."""
    return FIRST_BATCHES.get(r, batch)


def schedule(total: int = TOTAL_CONFIGS, batch: int = BATCH) -> list[int]:
    """Round sizes until ``total`` configurations (10, 10, 8, 8, 8, 6 for 50). Pure."""
    out: list[int] = []
    while sum(out) < total:
        out.append(min(batch_for(len(out), batch), total - sum(out)))
    return out


def block(space_version: str, now: str) -> dict:
    """state.json ``design`` (recorded at creation; compared on every start)."""
    return {
        "protocol": EXPLORE_V6,
        "space_version": space_version,
        "round0": f"{ROUND0_SOBOL} Sobol breadth draws (session_pool.sobol_breadth over the proposal domain; "
        "NoMemory / sim_gate != none / duplicates skipped), no plan defaults",
        "random_rounds": f"sobol-only-v1 rounds 1..{RANDOM_UNTIL_ROUND} ({FIRST_BATCHES[1]} picks each)",
        "channel_slots": "channel-slots-v6 from round 2 (llm, dls, xover, sobol, hv x 4)",
        "batches": {str(k): v for k, v in FIRST_BATCHES.items()} | {"rest": BATCH},
        "total_configs": TOTAL_CONFIGS,
        "schedule": schedule(),
        "llm_accounting": "llm slot = LLM analyst rows (agent / agent-gap) + dls-agent endpoints",
        "recorded_at": now,
        "note": "user 2026-10-23: τ² re-profiled with the latest exploration rule in a new isolated run",
    }
