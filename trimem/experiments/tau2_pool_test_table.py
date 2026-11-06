"""τ² pool -v5 TEST table shared by the qa-trajectory.html renderers (user 2026-10-21).

One place that turns the pool -v5 test stage (results/tau2_pool_v5_20261020/test: frozen routing.json + the
executed (configuration, task) pair files) and the three reused reference arms (NoMemory / VGR / VGR_SemFilter of
results/tau2_uncapped_4trials_20261019/test, their trial-0 episodes = the same official seed as the single Opt
trial) into per-task rows, so every τ² number on the page comes from the same files:

* protocol: 100 official test tasks x 1 trial (trial 0, seed 300), quality = pass^1 (task success), latency =
  episode wall time (+ the Opt-PerQuery router overhead, counted as retrieval), tokens = agent + SemFilter;
* Opt arms: ``pool_profile.test_rows`` (the driver's own validated rows) per routed pair; the raw result file of
  that pair gives the cost split (agent input / output tokens from the simulation messages, SemFilter tokens and
  retrieval ms from the memory receipt), cross-checked against the row;
* references: ``pool_profile.ref_episodes`` (trial 0), cost split from the same raw file;
* every arm's mean quality / tokens / latency (and the Opt arms' paired gain) is cross-checked against
  test_report.json when it exists (SystemExit on any disagreement).

Paired statistics vs NoMemory (same tasks, same trial): Δ pass^1 ± paired SE, wins / losses (discordant tasks),
exact two-sided sign test p (= exact McNemar for 0/1 outcomes) and the two-sided normal p of the paired mean.
Read-only; no episodes are run.
"""

from __future__ import annotations

import json
import math
import statistics as st
from functools import lru_cache
from pathlib import Path

REFS = (
    ("NoMemory", "No-Memory"),
    ("VGR", "VGR"),
    ("VGR_SemFilter", "VGR+SemFilter"),
)
OPT_COLORS = {
    "Opt-OneConfig-SLO-1": "#c26a1b",
    "Opt-OneConfig-SLO-2": "#a0521a",
    "Opt-OneConfig-SLO-3": "#e09a4f",
    "Opt-PerQuery-default": "#2f8f6b",
    "Opt-PerQuery-quality": "#16624a",
    "Opt-PerQuery-tokens": "#3b6fb6",
    "Opt-PerQuery-latency": "#8a4fb0",
}
REF_COLORS = {"NoMemory": "#8a8f87", "VGR": "#2a6fb3", "VGR_SemFilter": "#8f5bb5"}
TOL = 1e-6


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test on the discordant pairs (= exact McNemar for 0/1 outcomes)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _se(xs: list[float]) -> float | None:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def paired(a: dict[str, dict], b: dict[str, dict], key: str = "q") -> dict:
    """Paired a - b over the common tasks: mean ± SE, wins / losses / ties, sign-test p and normal p."""
    ks = sorted(set(a) & set(b))
    d = [a[k][key] - b[k][key] for k in ks]
    se = _se(d) or 0.0
    mean = st.mean(d) if d else 0.0
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {
        "n": len(ks),
        "delta": mean,
        "se": se,
        "wins": wins,
        "losses": losses,
        "ties": len(d) - wins - losses,
        "p_sign": sign_test_p(wins, losses),
        "p_z": math.erfc(abs(mean / se) / math.sqrt(2)) if se > 0 else 1.0,
    }


def episode_costs(r: dict, overhead_ms: float = 0.0) -> dict:
    """One τ² result dict -> {q, tin, tout, tsem, ret, gen, tok, lat} (seconds), the render_atlas_results split:
    agent input / output tokens from the assistant messages' usage, SemFilter tokens and retrieval ms from the
    memory receipt, generation = wall - retrieval; router overhead (Opt-PerQuery) is added to retrieval."""
    msgs = (r.get("simulation") or {}).get("messages") or []
    use = [(m.get("usage") or {}) for m in msgs if m.get("role") == "assistant"]
    tin = sum(u.get("prompt_tokens", 0) for u in use)
    tout = sum(u.get("completion_tokens", 0) for u in use)
    if tin + tout != r["agent_tokens"]:
        raise ValueError(f"{r['domain']}:{r['task_id']}: agent usage mismatch")
    mem = r.get("memory") or {}
    tsem = float(mem.get("semantic_tokens") or 0)
    ret = float(mem.get("retrieval_ms") or 0) / 1000
    wall = float(r["wall_ms"]) / 1000
    if not 0 <= ret <= wall:
        raise ValueError(f"{r['domain']}:{r['task_id']}: invalid retrieval time")
    oh = overhead_ms / 1000
    from trimem.experiments.tau2 import controlled_compare as cc

    return {
        "q": float(cc.is_successful(float(r["reward"]))),
        "tin": float(tin),
        "tout": float(tout),
        "tsem": tsem,
        "ret": ret + oh,
        "gen": wall - ret,
        "tok": float(tin + tout) + tsem,
        "lat": wall + oh,
        "reuse": bool(r.get("reuse")),
    }


def summarize(items: dict[str, dict], base: dict[str, dict] | None) -> dict:
    """Mean ± task-level SE of quality / tokens / latency (+ cost split) and the paired comparison vs ``base``."""
    ids = sorted(items)
    col = {
        k: [items[i][k] for i in ids]
        for k in ("q", "tok", "lat", "tin", "tout", "tsem", "ret", "gen")
    }
    out = {
        "n": len(ids),
        **{k: st.mean(v) for k, v in col.items()},
        **{f"{k}_se": _se(col[k]) for k in ("q", "tok", "lat")},
        "reused": sum(bool(items[i].get("reuse")) for i in ids),
    }
    if base is not None:
        out["vs_nomemory"] = paired(items, base)
    return out


def _check(name: str, what: str, mine: float, theirs: float | None) -> None:
    if theirs is None or not math.isclose(mine, theirs, rel_tol=TOL, abs_tol=TOL):
        raise SystemExit(
            f"{name}: {what} {mine} disagrees with test_report.json ({theirs})"
        )


@lru_cache(maxsize=1)
def table() -> dict:
    """Per-task rows + summaries of the 7 Opt arms and the 3 references (status != complete: no arms)."""
    from trimem.experiments.tau2 import pool_profile as pp
    from trimem.experiments.tau2 import runner as R

    out: dict = {
        "dir": str(pp.OUT.relative_to(pp.ROOT)),
        "reuse_dir": str(pp.REUSE.relative_to(pp.ROOT)),
        "trials": pp.TEST_TRIALS,
        "quality": f"pass^{pp.TEST_K}",
        "arms": [],
    }
    rp_path, rep_path = pp.OUT / "test" / "routing.json", pp.OUT / "test_report.json"
    if not rp_path.exists() or not rep_path.exists():
        out["status"] = "test stage not finished (no test_report.json)"
        return out
    tab = json.loads(rp_path.read_text())
    rep = json.loads(rep_path.read_text())
    if (
        tab.get("test_trials") != pp.TEST_TRIALS
        or rep.get("test_trials") != pp.TEST_TRIALS
    ):
        raise SystemExit(
            "routing.json / test_report.json were frozen for another test-trial protocol"
        )
    keys = pp.test_keys()
    rows = pp.test_rows(tab)

    def raw(path: Path, overhead: float, check: dict | None) -> dict:
        r = R.read_result(path)
        if r is None:
            raise SystemExit(f"missing result file {path}")
        e = episode_costs(r, overhead)
        if check is not None:  # the driver's validated row of this pair
            for k, v in (
                ("q", check["quality"]),
                ("tok", check["tokens"]),
                ("lat", (check["e2e_ms"] + overhead) / 1000),
            ):
                if not math.isclose(e[k], v, rel_tol=TOL, abs_tol=TOL):
                    raise SystemExit(f"{path}: {k} {e[k]} != driver row {v}")
        return e

    refs: dict[str, dict[str, dict]] = {}
    for name, _ in REFS:
        # NoMemory: the driver's validated reader; VGR / VGR_SemFilter (memory arms of the v1 config) are read
        # raw, as pool_profile.reference_arms does, with the trial / split / protocol checked here.
        if name == "NoMemory":
            eps = pp.ref_episodes(pp.REUSE / name, keys)
            if any(len(eps.get(k, {})) < pp.TEST_TRIALS for k in keys):
                raise SystemExit(f"{pp.REUSE / name} lacks trial 0 of some test task")
        for k in keys:
            r = R.read_result(R.result_path(pp.REUSE / name, *k.split(":", 1), 0))
            if (
                r is None
                or f"{r['domain']}:{r['task_id']}" != k
                or r.get("trial") != 0
                or r.get("split") != "test"
                or r.get("num_trials") != pp.REF_TRIALS
            ):
                raise SystemExit(
                    f"{pp.REUSE / name}: trial 0 of {k} missing or of another protocol"
                )
        refs[name] = {
            k: raw(R.result_path(pp.REUSE / name, *k.split(":", 1), 0), 0.0, None)
            for k in keys
        }
    base = refs["NoMemory"]
    nm_key = pp.space.key(pp.space.NOMEM)
    for name, arm in tab["arms"].items():
        items = {}
        for q, route in arm["routes"].items():
            cfg, k = route["config"], pp.space.key(route["config"])
            row = rows.get((k, q))
            if row is None:
                raise SystemExit(f"{name}: test pair {q} missing")
            d, t = q.split(":", 1)
            if k == nm_key and (pp.REUSE / "NoMemory").exists() and q in base:
                path = R.result_path(pp.REUSE / "NoMemory", d, t, 0)
            else:
                path = R.result_path(
                    pp.OUT / "test" / "pairs" / pp.cfg_name(cfg), d, t, None
                )
            items[q] = {
                **raw(path, float(route["overhead_ms"]), row),
                "plan": cfg["plan"],
                "cfg": k,
            }
        s = summarize(items, base)
        r = rep["arms"][name]
        _check(name, "pass^1", s["q"], r.get("quality"))
        _check(name, "tokens", s["tok"], r.get("tokens"))
        _check(name, "latency", s["lat"], r.get("latency_s"))
        _check(
            name,
            "paired gain",
            s["vs_nomemory"]["delta"],
            r.get("paired_gain_vs_nomemory"),
        )
        kind = arm["kind"]
        out["arms"].append(
            {
                "id": name,
                "label": name.replace("Opt-OneConfig-", "Opt-OneConfig ").replace(
                    "Opt-PerQuery-", "Opt-PerQuery "
                ),
                "kind": kind,
                "slo": arm.get("slo"),
                "weights": arm.get("weights"),
                "color": OPT_COLORS.get(name, "#2f8f6b"),
                "distinct_configs": r.get("distinct_configs"),
                "items": items,
                **s,
            }
        )
    ref_rep = rep.get("references") or {}
    ref_rows = []
    for name, label in REFS:
        s = summarize(refs[name], None if name == "NoMemory" else base)
        r = ref_rep.get(name) or {}
        _check(name, "pass^1", s["q"], r.get("q"))
        _check(name, "tokens", s["tok"], r.get("tok"))
        _check(name, "latency", s["lat"], r.get("lat"))
        a4 = r.get("all_trials") or {}
        ref_rows.append(
            {
                "id": name,
                "label": label,
                "kind": "ref",
                "color": REF_COLORS[name],
                "items": refs[name],
                "all4": {
                    "trials": a4.get("trials"),
                    "pass1": (a4.get("pass_hat") or {}).get("1"),
                    "q": a4.get("q"),
                    "tok": a4.get("tok"),
                    "lat": a4.get("lat"),
                },
                **s,
            }
        )
    _check(
        "NoMemory",
        "baseline pass^1",
        ref_rows[0]["q"],
        (rep.get("baseline") or {}).get("quality"),
    )
    out["arms"] = ref_rows + out["arms"]
    out.update(
        status="complete",
        experiment=rep.get("experiment"),
        test_version=rep.get("test_version"),
        concurrency=rep.get("test_concurrency"),
        pairs=rep.get("pairs"),
        generated=rep.get("generated"),
        new_pairs=sum(len(v) for k, v in pp.pairs_of(tab).items() if k != nm_key),
        reuse_hits=_test_reuse_hits(pp, tab),
    )
    return out


def _test_reuse_hits(pp, tab: dict) -> int:
    """Executed test pairs whose result is a quality-reuse record (identical injected memory, same task / trial)."""
    from trimem.experiments.tau2 import runner as R

    n = 0
    for k, qs in pp.pairs_of(tab).items():
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            continue
        for q in qs:
            r = R.read_result(
                R.result_path(
                    pp.OUT / "test" / "pairs" / pp.cfg_name(cfg), *q.split(":", 1), None
                )
            )
            n += bool(r and r.get("reuse"))
    return n


def strip(t: dict) -> dict:
    """Page payload: the table without per-task rows."""
    return {
        **t,
        "arms": [{k: v for k, v in a.items() if k != "items"} for a in t["arms"]],
    }
