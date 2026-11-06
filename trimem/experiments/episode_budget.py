"""Episode turn budget max-round-v1 (user decision 2026-10-23): WebShop uses AgentGym's own WebShop evaluation turn
limit, ``max_round = 10`` (``vendor/AgentGym/agentenv/examples/basic/base_eval.sh:21``: task_name webshop,
max_round 10), from the next profiling round to be proposed (round 2 of webshop_pool-strat200-v7) and for everything
after profiling (deployed test, NoMemory / reference arms, fixed-plan arms, PerQuery sweep). ALFWorld /
ScienceWorld / tau2 / Exp2 keep 30 (``bo_agentenv_driver.COMMON``): nothing changes for them (opt-in, WebShop only).

Opt-in ``POOL_EPISODE_MAX_ROUND=10`` + ``POOL_EPISODE_MAX_ROUND_REASON``; recorded ONCE in state.json
``episode_budget`` (``record_block``) with ``from_round`` = the next round to be proposed, never retroactive.

Episodes measured at the native 30 turns (rounds < from_round, the NoMemory baseline, the noise repeat, the
controlled-v1 references) enter every comparison through the DERIVED view (``derived-truncation-<n>``), never as
measured: AgentGym's loop (``agentenv/controller/task.py`` ``_generate_experience_one``) is generate -> env step ->
``rounds += 1`` -> break at ``max_rounds``; the returned reward is the LAST step's reward, and WebShop's env returns
reward 0 / done False on every step except "click[Buy Now]" (``web_agent_site/envs/web_agent_text_env.py``: status
defaults to reward 0, done False; only the buy branch sets reward and done). So at temperature 0 the first n turns
of a 30-turn episode ARE the n-turn episode:

* environment_steps <= n: unchanged (the episode ended by itself within n turns);
* environment_steps > n: reward 0 / success 0 (not bought by turn n), steps n, generation tokens = prompt +
  completion of the first n LLM calls (``llm_calls.jsonl``; the definition of ``generation_tokens``, verified
  equal on every logged WebShop episode), latency = the measured episode wall MINUS the removed tail = (end of
  the last LLM call - start of LLM call n+1) + one final env step (this episode's median gap between calls).
  The measured head (env reset, retrieval, the first n turns) is kept as measured; only the final env step is
  estimated (``latency_check`` reports the error of that estimate against the result-file mtime).

Caveats (also in the block): vLLM batching is not bitwise deterministic at temperature 0 (the first n turns are the
n-turn run up to that), and a native 10-turn run has a different load mix (shorter episodes) than the measured
30-turn regime, which no derivation can reproduce. Native n-turn episodes pass through the view unchanged
(steps <= n), so the view is idempotent.
"""

from __future__ import annotations

import json
import os
import statistics
from functools import lru_cache
from pathlib import Path

PROTOCOL = "max-round-v1"
ENV = "POOL_EPISODE_MAX_ROUND"
REASON_ENV = "POOL_EPISODE_MAX_ROUND_REASON"
NATIVE_MAX_ROUND = (
    30  # bo_agentenv_driver.COMMON (every row without protocol.max_round ran at 30)
)
SOURCES = {
    "webshop": "vendor/AgentGym/agentenv/examples/basic/base_eval.sh:21 (task_name webshop, max_round 10)"
}
DERIVATION = "derived-truncation-v1"
REWARD_RULE = (
    "AgentGym returns the last step's reward; WebShop gives reward > 0 (and done) only on click[Buy Now] "
    "(web_agent_text_env.py: status reward 0 / done False otherwise) -> an episode not done by turn n has reward 0 "
    "at turn n"
)
LATENCY_RULE = (
    "derived wall = measured wall - [(end of the last LLM call - start of LLM call n+1) + final env step], the final "
    "env step = the episode's median gap between consecutive LLM calls; the measured head (env reset, retrieval, "
    "turns 1..n) is kept"
)


def tag(n: int) -> str:
    return f"derived-truncation-{n}"


# ------------------------------------------------------------------ state block


def requested(ds: str, env=None) -> int | None:
    """``$POOL_EPISODE_MAX_ROUND`` (None when unset); only WebShop, only below the native 30."""
    env = os.environ if env is None else env
    v = env.get(ENV, "").strip()
    if not v:
        return None
    if ds not in SOURCES:
        raise SystemExit(f"{ENV}: WebShop only (dataset {ds} keeps max_round 30)")
    n = int(v)
    if not 1 <= n < NATIVE_MAX_ROUND:
        raise SystemExit(f"{ENV}={n}: must be in [1, {NATIVE_MAX_ROUND})")
    return n


def record_block(state: dict, ds: str, now: str, env=None) -> dict | None:
    """Record (once) or check state.json ``episode_budget``. ``from_round`` = the next round to be proposed
    (``len(state['rounds'])``): rounds already in state.json -- including one still running -- finish at 30.
    A recorded block is kept on resume (a different request is refused); without block and request: None."""
    env = os.environ if env is None else env
    want = requested(ds, env)
    blk = state.get("episode_budget")
    if blk is not None:
        if blk.get("protocol") != PROTOCOL:
            raise SystemExit(
                f"state.json episode_budget {blk.get('protocol')!r} unknown"
            )
        if want is not None and want != blk["max_round"]:
            raise SystemExit(
                f"{ENV}={want} but state.json episode_budget records {blk['max_round']}; refusing to mix"
            )
        return blk
    if want is None:
        return None
    reason = env.get(REASON_ENV, "").strip()
    if not reason:
        raise SystemExit(f"{ENV} needs {REASON_ENV}")
    rounds = state.get("rounds") or []
    state["episode_budget"] = blk = {
        "protocol": PROTOCOL,
        "max_round": want,
        "native_max_round": NATIVE_MAX_ROUND,
        "from_round": len(rounds),
        "applies_to": "every episode run from round from_round on (profiling) and every test / NoMemory / "
        "reference / fixed-plan / PerQuery-sweep arm after profiling (runner --max_round, opt row "
        "protocol.max_round)",
        "derived_view": {
            "protocol": DERIVATION,
            "tag": tag(want),
            "applies_to": "every 30-turn episode compared from round from_round on and after profiling: rounds < "
            "from_round, NoMemory baseline, noise repeat, controlled-v1 references (GP observations, HV reference "
            "/ NoMemory anchor + SE, prune comparators, profile table / router catalog, test references)",
            "reward_rule": REWARD_RULE,
            "tokens_rule": "prompt + completion of the first n LLM calls (+ the row's own SemFilter tokens)",
            "latency_rule": LATENCY_RULE,
            "rounds_before_decided_at_30": "stage decisions of rounds < from_round use the measured 30-turn values",
        },
        "quality_reuse": "a runner at max_round n reuses only sources recorded with the same max_round "
        "(protocol.max_round, absent = 30)",
        "aux_observations": "off from from_round (30-turn train200 panel of another protocol; not re-derived)",
        "runtime_label": None,  # filled by runtime_label()
        "caveats": [
            "vLLM batching is not bitwise deterministic at temperature 0: the first n turns of a 30-turn episode "
            "equal the n-turn episode up to that",
            "a native n-turn run has another load mix (shorter episodes); derived latencies keep the 30-turn "
            "regime's per-turn latency",
        ],
        "reason": reason,
        "source": SOURCES[ds],
        "recorded_at": now,
    }
    blk["runtime_label"] = runtime_label(state)
    return blk


def block(state: dict | None) -> dict | None:
    return (state or {}).get("episode_budget")


def max_round_for(state: dict | None, r: int) -> int | None:
    """Runner max_round of round ``r``'s episodes: the recorded n from ``from_round`` on, else None (= COMMON's 30)."""
    blk = block(state)
    return blk["max_round"] if blk and r >= blk["from_round"] else None


def view_max_round(state: dict | None, r: int | None = None) -> int | None:
    """Turn budget of the values the optimizer compares in round ``r`` (None = after profiling: the recorded n):
    the recorded n from ``from_round`` on, else None (measured values, unchanged)."""
    blk = block(state)
    if not blk:
        return None
    return blk["max_round"] if r is None or r >= blk["from_round"] else None


def runtime_label(state: dict | None, base: str | None = None) -> str | None:
    """``<runtime profile>+max-round-<n>`` (e.g. controlled-c16-v1+max-round-10) once the block exists."""
    blk = block(state)
    name = base or ((state or {}).get("runtime_profile") or {}).get("name") or "default"
    return f"{name}+max-round-{blk['max_round']}" if blk else None


def with_max_round(cmd: list[str], n: int | None) -> list[str]:
    """Runner argv with ``--max_round n`` (None: unchanged). Refuses an argv without ``--max_round``."""
    if n is None:
        return cmd
    cmd = list(cmd)
    i = cmd.index("--max_round")
    cmd[i + 1] = str(n)
    return cmd


def protocol_extra(n: int | None) -> dict:
    """Opt-row protocol keys of a budgeted arm ({} without a budget: rows stay byte-identical)."""
    return {} if n is None else {"max_round": n, "episode_budget": PROTOCOL}


def row_max_round(protocol: dict | None) -> int:
    return int((protocol or {}).get("max_round") or NATIVE_MAX_ROUND)


# ------------------------------------------------------------------ truncation (pure)


def _start(c: dict) -> float:
    return float(c["t"])


def _end(c: dict) -> float:
    return float(c["t"]) + float(c["seconds"])


def _tok(c: dict) -> int:
    return int(c.get("prompt_tokens") or 0) + int(c.get("completion_tokens") or 0)


def final_step_ms(calls: list[dict]) -> float:
    """The episode's median gap between consecutive LLM calls (ms; env step + prompt assembly), >= 0."""
    gaps = [
        max(0.0, _start(b) - _end(a)) * 1000
        for a, b in zip(calls, calls[1:], strict=False)
    ]
    return statistics.median(gaps) if gaps else 0.0


def truncate(
    *, reward: float, steps: int, calls: list[dict], episode_ms: float, max_round: int
) -> dict:
    """The max_round-n view of one episode. ``calls``: its agent LLM calls (any order; t = start, seconds),
    ``episode_ms``: its measured wall. Returns {reward, success, steps, generation_tokens (None = unchanged),
    episode_ms, truncated, removed_ms}. Pure."""
    if steps <= max_round:
        return {
            "reward": float(reward),
            "success": float(reward == 1),
            "steps": steps,
            "generation_tokens": None,
            "episode_ms": float(episode_ms),
            "truncated": False,
            "removed_ms": 0.0,
        }
    cs = sorted(calls, key=_start)
    if len(cs) <= max_round:
        raise ValueError(
            f"{steps}-step episode with {len(cs)} logged LLM calls: cannot truncate at {max_round}"
        )
    removed = (max(map(_end, cs)) - _start(cs[max_round])) * 1000 + final_step_ms(cs)
    return {
        "reward": 0.0,
        "success": 0.0,
        "steps": max_round,
        "generation_tokens": sum(_tok(c) for c in cs[:max_round]),
        "episode_ms": max(0.0, float(episode_ms) - removed),
        "truncated": True,
        "removed_ms": removed,
    }


# ------------------------------------------------------------------ episode sources (result dirs)


@lru_cache(maxsize=256)
def _calls(path: str, mtime: float, size: int) -> dict[tuple[int, int], list[dict]]:
    out: dict[tuple[int, int], list[dict]] = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if c.get("item") is not None:
                out.setdefault((int(c["item"]), int(c["pid"])), []).append(c)
    return out


def dir_calls(d: Path) -> dict[tuple[int, int], list[dict]]:
    p = Path(d) / "llm_calls.jsonl"
    if not p.exists():
        return {}
    st = p.stat()
    return _calls(str(p), st.st_mtime, st.st_size)


@lru_cache(maxsize=256)
def _episodes(path: str, mtime: float, size: int) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get(
                "llm_calls"
            ):  # a reused episode logs 0 calls here: its source holds them
                out[int(e["item"])] = e
    return out


def dir_episodes(d: Path) -> dict[int, dict]:
    p = Path(d) / "episodes.jsonl"
    if not p.exists():
        return {}
    st = p.stat()
    return _episodes(str(p), st.st_mtime, st.st_size)


def episode_calls(d: Path, item: int) -> list[dict]:
    """The agent LLM calls of ``item``'s recorded episode in result dir ``d``: the pid of its episodes.jsonl record,
    else (older dirs) the pid of the item's last call (``bo_pool_driver.nomemory_rows``' rule)."""
    calls = dir_calls(d)
    ep = dir_episodes(d).get(item)
    if ep is not None and (item, int(ep["pid"])) in calls:
        return calls[(item, int(ep["pid"]))]
    mine = [(k, v) for k, v in calls.items() if k[0] == item]
    if not mine:
        return []
    last = max(mine, key=lambda kv: max(map(_start, kv[1])))
    return last[1]


def result_steps(res: dict) -> int | None:
    from trimem.experiments import agent_step_metrics as sm

    return sm.agentgym_steps(res)["environment_steps"]


def derive_result(
    result_file: Path, episode_ms: float, max_round: int, *, steps: int | None = None
) -> dict:
    """``truncate`` of the episode stored in ``result_file`` (its dir's llm_calls.jsonl), wall ``episode_ms``."""
    result_file = Path(result_file)
    res = json.loads(result_file.read_text())
    steps = result_steps(res) if steps is None else steps
    if steps is None:
        raise ValueError(f"{result_file}: environment steps unparseable")
    item = int(result_file.stem.rsplit("_", 1)[1])
    calls = episode_calls(result_file.parent, item) if steps > max_round else []
    return truncate(
        reward=float(res["reward"]),
        steps=steps,
        calls=calls,
        episode_ms=episode_ms,
        max_round=max_round,
    )


def derive_opt_row(row: dict, max_round: int | None) -> dict:
    """An opt row (``quality`` / ``total_tokens`` / ``e2e_ms`` + ``generation``) in the max_round view (None or a
    row recorded at <= max_round: unchanged). The episode = ``generation.reused_from_file`` (a reused row: its
    source episode) else ``generation.result_file``; the row's own SemFilter tokens / retrieval ms are kept."""
    if max_round is None or row_max_round(row.get("protocol")) <= max_round:
        return row
    g = row.get("generation") or {}
    src = g.get("reused_from_file") or g.get("result_file")
    if not src or not Path(src).exists():
        raise ValueError(
            f"opt row {row.get('query_id')}: no result file to derive from"
        )
    t = derive_result(Path(src), float(g["episode_ms"]), max_round)
    if not t["truncated"]:
        return {**row, "derived": tag(max_round), "truncated": False}
    gen_tok = t["generation_tokens"]
    sem = float(g.get("semantic_tokens") or 0)
    ret = float(g.get("retrieval_ms") or 0)
    return {
        **row,
        "quality": 0.0,
        "total_tokens": float(gen_tok + sem),
        "e2e_ms": t["episode_ms"] + ret,
        "generation": {
            **g,
            "reward": 0.0,
            "llm_calls": max_round,
            "environment_steps": max_round,
            "agent_steps": max_round,
            "generation_tokens": gen_tok,
            "episode_ms": t["episode_ms"],
        },
        "derived": tag(max_round),
        "truncated": True,
        "removed_ms": t["removed_ms"],
    }


# ------------------------------------------------------------------ opt rows


ROW_SQL = (
    "SELECT experiment_id, dataset, split, query_id, status, config, protocol, paired_gain, total_tokens, e2e_ms, "
    "quality, generation FROM opt.session_retrieval WHERE experiment_id = %s AND split = 'train' "
    "AND status = 'evaluated'"
)


def train_rows(experiment: str, max_round: int | None) -> list[dict]:
    """``bo_agentenv_driver.train_rows`` (same rows / keys) in the max_round view (None: the plain rows). Read-only."""
    from trimem.experiments import bo_agentenv_driver as drv

    if max_round is None:
        return drv.train_rows(experiment.split(":", 1)[0], experiment)
    import psycopg
    from psycopg.rows import dict_row

    dsn = drv.OPT_DSN + " options='-c default_transaction_read_only=on'"
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        rows = c.execute(ROW_SQL, (experiment,)).fetchall()
    out = []
    for r in rows:
        keys = [k for k in r if k != "generation"]  # the keys of drv.train_rows' rows
        d = derive_opt_row(r, max_round)
        d = {k: d[k] for k in keys}
        d["total_tokens"] = (
            float(d["total_tokens"]) if d["total_tokens"] is not None else None
        )
        out.append(d)
    return out


def derived_summary(experiment: str, max_round: int) -> list[dict]:
    """Per opt row of ``experiment``: the measured and the derived (quality, tokens, e2e_ms, steps) -- the audit
    snapshot of the view (written next to state.json, never over an original file). Read-only."""
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import bo_agentenv_driver as drv

    dsn = drv.OPT_DSN + " options='-c default_transaction_read_only=on'"
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        rows = c.execute(ROW_SQL, (experiment,)).fetchall()
    out = []
    for r in rows:
        d = derive_opt_row(r, max_round)
        g = r.get("generation") or {}
        out.append(
            {
                "query_id": r["query_id"],
                "config": (r.get("protocol") or {}).get("space_config"),
                "row_max_round": row_max_round(r.get("protocol")),
                "measured": {
                    "quality": float(r["quality"]),
                    "total_tokens": float(r["total_tokens"]),
                    "e2e_ms": float(r["e2e_ms"]),
                    "steps": g.get("environment_steps"),
                },
                "derived": {
                    "quality": float(d["quality"]),
                    "total_tokens": float(d["total_tokens"]),
                    "e2e_ms": float(d["e2e_ms"]),
                    "steps": (d.get("generation") or g).get("environment_steps"),
                },
                "truncated": bool(d.get("truncated")),
                "reused_from_file": g.get("reused_from_file"),
            }
        )
    return out


# ------------------------------------------------------------------ checks / summaries


def latency_check(dirs: list[Path], ds: str = "webshop") -> dict:
    """Error of the estimated episode end (end of the last LLM call + the median inter-call gap) against the
    result-file mtime (written at the episode's end), on every original episode with calls -- the only estimated
    term of the derived latency. Also the error of reproducing the whole episode_ms from the calls alone (no env
    reset): why the derivation keeps the measured head instead. ms quantiles."""
    tail_err, head_err, rel_head = [], [], []
    for d in dirs:
        eps = dir_episodes(d)
        for item, e in eps.items():
            f = Path(d) / f"{ds}_{item}.json"
            calls = episode_calls(Path(d), item)
            if not f.exists() or not calls:
                continue
            cs = sorted(calls, key=_start)
            end = f.stat().st_mtime
            est_end = max(map(_end, cs)) + final_step_ms(cs) / 1000
            tail_err.append((est_end - end) * 1000)
            span = (est_end - _start(cs[0])) * 1000
            head_err.append(span - e["episode_ms"])
            rel_head.append((span - e["episode_ms"]) / e["episode_ms"])

    def q(v):
        if len(v) < 2:
            return None
        s = sorted(v)
        return {
            "n": len(s),
            "p05": s[int(0.05 * (len(s) - 1))],
            "p50": statistics.median(s),
            "p95": s[int(0.95 * (len(s) - 1))],
            "mean_abs": statistics.mean(abs(x) for x in s),
        }

    return {
        "removed_tail_end_error_ms": q(tail_err),
        "calls_only_episode_ms_error_ms": q(head_err),
        "calls_only_episode_ms_rel_error": q(rel_head),
    }


def truncated_spans(d: Path, ds: str, max_round: int) -> dict[str, float]:
    """qid -> LLM-call span ms (first .. ``max_round``-th call of the item's last worker; the
    ``alf_test_c16.call_spans`` definition) of every episode in result dir ``d`` longer than ``max_round`` turns."""
    out = {}
    calls = dir_calls(Path(d))
    for f in Path(d).glob(f"{ds}_*.json"):
        if not f.stem.rsplit("_", 1)[1].isdigit():
            continue
        steps = result_steps(json.loads(f.read_text()))
        if steps is None or steps <= max_round:
            continue
        item = int(f.stem.rsplit("_", 1)[1])
        mine = [(k, v) for k, v in calls.items() if k[0] == item]
        if not mine:
            continue
        cs = sorted(max(mine, key=lambda kv: max(map(_start, kv[1])))[1], key=_start)
        cs = cs[:max_round]
        out[f.stem] = (max(map(_end, cs)) - _start(cs[0])) * 1000
    return out


def view_quality(ds: str, res: dict, max_round: int | None) -> float:
    """Quality of an AgentGym result in the max_round view (``agentgym_session_eval.quality_of``; 0 for an episode
    longer than ``max_round`` turns; None = as measured)."""
    if max_round is not None and (result_steps(res) or 0) > max_round:
        return 0.0
    r = float(res["reward"])
    return max(r, 0) / 100 if ds == "sciworld" else r
