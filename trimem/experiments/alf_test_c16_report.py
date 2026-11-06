"""Report of the ALFWorld v7 c16 test rerun (``alf_test_c16``): ``<run>/test_c16_report.json`` + ``report_c16.md``.

Same tables as ``test_report.json`` (outer 5-fold CV: mean ± std across folds + SE_tasks; deployed arms) over the
c16 pairs (LimeQO excluded, user 2026-10-21: LimeQO is shown only from the c128 run, labelled), with TWO latency
definitions per arm:

* ``latency_s`` (wall) = official episode wall time + retrieval ms + router overhead (opt-row ``e2e_ms``; the
  definition of the v7 arms and of the controlled-v1 VGR reference);
* ``latency_span_s`` (span) = LLM-call span of the episode (first call start .. last call end, ``llm_calls.jsonl``)
  + retrieval ms + router overhead (the definition of the old controlled-v1 NoMemory reference, whose run predates
  ``episodes.jsonl``).

References: the OLD controlled-v1 runs (2026-10-15) and their RERUNS in this c16 session (``test_c16/ref_*``),
both definitions where available. User 2026-10-21: the PRIMARY references are the old controlled-v1 runs (VGR latency =
wall, NoMemory latency = LLM-call span: compare it with the arms' span column); the reruns were stopped (NoMemory
200/200 kept as a secondary record, VGR drained after 30 tasks, not used). Paired gain is vs the old controlled-v1
NoMemory (``gain_vs_nomemory_rerun`` / ``paired_gain_vs_nomemory_rerun`` = vs the NoMemory rerun, secondary).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import alf_test_c16 as c16

DS = c16.DS


def _rows_of_opt(experiment: str, cfg_filter: bool = True) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import bo_pool_driver as d

    with psycopg.connect(d.drv.OPT_DSN, row_factory=dict_row) as c:
        return c.execute(
            "SELECT query_id, protocol->'space_config' AS cfg, quality, total_tokens, e2e_ms, generation FROM "
            "opt.session_retrieval WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'",
            (experiment,),
        ).fetchall()


def memory_row(r: dict, spans: dict[str, float]) -> tuple[dict, dict | None]:
    """(wall row, span row) of one opt row. Span row None when the item has no LLM call record. Pure."""
    from trimem.experiments import bo_pool_driver as d

    g = r.get("generation") or {}
    wall = d.with_steps(
        DS,
        {
            "quality": float(r["quality"]),
            "tokens": float(r["total_tokens"]),
            "e2e_ms": float(r["e2e_ms"]),
        },
        d.episode_steps(DS, g, None),
    )
    sp = spans.get(r["query_id"])
    return wall, c16.with_span(wall, sp, float(g.get("retrieval_ms") or 0.0))


def nomemory_dir_rows(
    pair_dir: Path, ids: list[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Official-evaluator dir -> (wall rows from episodes.jsonl, span rows from llm_calls.jsonl)."""
    from trimem.experiments import bo_pool_driver as d

    rows = d.nomemory_rows(DS, pair_dir, ids)
    spans = c16.call_spans(pair_dir)
    wall = {
        q: {k: v for k, v in r.items() if k != "cost_source"}
        for q, r in rows.items()
        if r.get("cost_source") == "episodes"
    }
    span = {
        q: c16.with_span(
            {k: v for k, v in r.items() if k != "cost_source"}, spans[q], 0.0
        )
        for q, r in rows.items()
        if q in spans
    }
    return wall, span


def pair_rows(base: Path, tab: dict, cv: dict) -> tuple[dict, dict]:
    """(config key, query) -> row, in the wall and span definitions, of the c16 pairs."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    pdir = base / c16.SUBDIR / "pairs"
    spans: dict[str, dict[str, float]] = {}
    wall, span = {}, {}
    for r in _rows_of_opt(d.test_experiment(DS, c16.ARM_TAG)):
        k = space.key(r["cfg"])
        name = d.cfg_name(r["cfg"])
        if name not in spans:
            spans[name] = c16.call_spans(pdir / name)
        w, s = memory_row(r, spans[name])
        wall[(k, r["query_id"])] = w
        if s is not None:
            span[(k, r["query_id"])] = s
    nm_key = space.key(space.NOMEM)
    nm_pairs = c16.pair_set(tab, cv).get(nm_key, [])
    if nm_pairs:
        w, s = nomemory_dir_rows(pdir / d.cfg_name(space.NOMEM), nm_pairs)
        wall |= {(nm_key, q): r for q, r in w.items()}
        span |= {(nm_key, q): r for q, r in s.items()}
    return wall, span


def summarize_rows(rows: list[dict], nm: dict[str, float], ids: list[str]) -> dict:
    """Reference-arm metrics (``bo_pool_driver.reference_arms`` fields) of qid-keyed rows. Pure."""
    from trimem.experiments import bo_pool_driver as d

    have = [r for r in rows if r is not None]
    d_ = [r["quality"] - nm[r["qid"]] for r in have if r["qid"] in nm]
    return {
        "n": len(ids),
        "covered": len(have),
        "quality": sum(r["quality"] for r in have) / len(have) if have else None,
        "quality_se": d._se([r["quality"] for r in have]) if have else None,
        "tokens": sum(r["tokens"] for r in have) / len(have) if have else None,
        "latency_s": sum(r["e2e_ms"] for r in have) / len(have) / 1000
        if have
        else None,
        "paired_gain_vs_nomemory": sum(d_) / len(d_) if d_ else None,
        "paired_gain_se": d._se(d_) if d_ else None,
        **d.step_metrics(have),
    }


def ref_block(
    wall: dict[str, dict], span: dict[str, dict], ids: list[str], baselines: dict
) -> dict:
    """One reference arm: wall / span metrics + paired gains vs each NoMemory baseline. Pure."""
    out: dict = {}
    for label, rows in (("wall", wall), ("span", span)):
        if not rows:
            out[label] = None
            continue
        got = [{**rows[q], "qid": q} if q in rows else None for q in ids]
        m = summarize_rows(got, baselines["old"], ids)
        if baselines.get("rerun"):
            m2 = summarize_rows(got, baselines["rerun"], ids)
            m["paired_gain_vs_nomemory_rerun"] = m2["paired_gain_vs_nomemory"]
            m["paired_gain_se_rerun"] = m2["paired_gain_se"]
        out[label] = m
    return out


def mean_inflight(intervals: list[tuple[float, float]]) -> float | None:
    """Time-weighted mean number of in-flight episodes while ANY episode is in flight = sum(durations) / length
    of the union of the intervals (idle gaps, e.g. a drained and resumed run or env restarts, excluded). Pure."""
    if not intervals:
        return None
    busy, end = 0.0, None
    for s_, e_ in sorted(intervals):
        if end is None or s_ > end:
            busy += e_ - s_
            end = e_
        elif e_ > end:
            busy += e_ - end
            end = e_
    return sum(e_ - s_ for s_, e_ in intervals) / busy if busy > 0 else None


def call_intervals(pair_dir: Path) -> list[tuple[float, float]]:
    """Per item (first LLM call start, last call end) of a run directory (LLM-span proxy of the episode)."""
    f = pair_dir / "llm_calls.jsonl"
    if not f.exists():
        return []
    by: dict[int, list] = {}
    for line in open(f):
        c = json.loads(line)
        if c.get("item") is not None:
            by.setdefault(c["item"], []).append(c)
    return [
        (min(c["t"] for c in cs), max(c["t"] + c["seconds"] for c in cs))
        for cs in by.values()
    ]


def memory_drift(old_dir: Path, new_dir: Path) -> dict:
    """Injected-memory identity of the VGR rerun vs the old controlled-v1 VGR (memory/<idx>.json)."""

    def shas(d: Path) -> dict[str, str]:
        return {
            f.stem: hashlib.sha256(json.load(open(f))["memory"].encode()).hexdigest()
            for f in (d / "memory").glob("*.json")
        }

    a, b = shas(old_dir), shas(new_dir)
    common = set(a) & set(b)
    same = sum(a[k] == b[k] for k in common)
    return {"compared": len(common), "identical": same, "differs": len(common) - same}


def compare(c16_m: dict | None, c128_m: dict | None, kind: str) -> dict | None:
    """c16 vs c128 of one arm (``kind`` cv: mean/std dicts; deployed: plain values with *_se). Pure."""
    if not c16_m or not c128_m:
        return None
    if kind == "cv":
        q16, q128 = c16_m["quality"]["mean"], c128_m["quality"]["mean"]
        se16 = c16_m["se_tasks_mean"]["quality"]
        se128 = c128_m["se_tasks_mean"]["quality"]
        l16, l128 = c16_m["latency_s"]["mean"], c128_m["latency_s"]["mean"]
        s16, s128 = c16_m["steps"]["mean"], c128_m["steps"]["mean"]
        t16, t128 = c16_m["tokens"]["mean"], c128_m["tokens"]["mean"]
    else:
        q16, q128 = c16_m["quality"], c128_m["quality"]
        se16, se128 = c16_m["quality_se"], c128_m["quality_se"]
        l16, l128 = c16_m["latency_s"], c128_m["latency_s"]
        s16, s128 = c16_m.get("steps"), c128_m.get("steps")
        t16, t128 = c16_m["tokens"], c128_m["tokens"]
    return {
        "quality_c16": q16,
        "quality_c128": q128,
        "quality_diff": q16 - q128,
        "quality_diff_z": c16.diff_z(q16, se16, q128, se128),
        "latency_c16_s": l16,
        "latency_c128_s": l128,
        "latency_ratio_c16_over_c128": c16.ratio(l16, l128),
        "tokens_ratio_c16_over_c128": c16.ratio(t16, t128),
        "steps_c16": s16,
        "steps_c128": s128,
    }


def report() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import router_outer_cv as rcv

    base = c16.run_base()
    tab, cv = c16.routing(base)
    ids = d.test_ids(DS)
    old = json.load(open(base / "test_report.json"))
    ctrl_nm = d.drv.CTRL / f"{DS}_NoMemory"
    nm_old = {
        i: d.quality_of(DS, json.load(open(f))["reward"])
        for i, f in d.drv.item_files(ctrl_nm, DS).items()
    }
    # references: old (controlled-v1, 2026-10-15) and rerun (this c16 session)
    r_nm_dir, r_vgr_dir = (
        base / c16.SUBDIR / "ref_NoMemory",
        base / c16.SUBDIR / "ref_VGR",
    )
    nm_rerun_w, nm_rerun_s = nomemory_dir_rows(r_nm_dir, ids)
    nm_rerun = {
        i: d.quality_of(DS, json.load(open(f))["reward"])
        for i, f in d.drv.item_files(r_nm_dir, DS).items()
    }
    _, nm_old_s = nomemory_dir_rows(ctrl_nm, ids)
    baselines = {"old": nm_old, "rerun": nm_rerun}
    # paired baseline of the c16 arms: the NoMemory RERUN (same runtime) once complete, else the old run
    rerun_ok = set(ids) <= set(nm_rerun)
    nm = nm_old  # primary paired baseline (user 2026-10-21: old controlled-v1 references)
    nm_alt = nm_rerun if rerun_ok else None
    wall, span = pair_rows(base, tab, cv)

    def fold_metrics(rows, base_q=None):
        def f(t: dict) -> dict:
            got = d.assemble(t, rows, base_q or nm)
            return {k: v for k, v in got.items() if v.get("covered") == v.get("n")}

        return f

    arms_wall, arms_span = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    arms_alt = d.assemble(tab, wall, nm_alt) if nm_alt else {}
    arms = {
        k: {
            **v,
            "latency_span_s": arms_span[k].get("latency_s"),
            "paired_gain_vs_nomemory_rerun": (arms_alt.get(k) or {}).get(
                "paired_gain_vs_nomemory"
            ),
        }
        for k, v in arms_wall.items()
    }
    ts_wall = rcv.test_summary(cv, fold_metrics(wall))
    ts_span = rcv.test_summary(cv, fold_metrics(span))
    ts_alt = rcv.test_summary(cv, fold_metrics(wall, nm_alt)) if nm_alt else None
    for k, v in ts_wall["summary"].items():
        v["latency_span_s"] = ts_span["summary"].get(k, {}).get("latency_s")
        v["gain_vs_nomemory_rerun"] = (
            ts_alt["summary"].get(k, {}).get("gain") if ts_alt else None
        )
    incomplete = sorted(
        n for n, v in d.assemble(cv, wall, nm).items() if v.get("covered") != v.get("n")
    )

    def vgr_rows(experiment: str, run_dir: Path) -> tuple[dict, dict]:
        spans = c16.call_spans(run_dir)
        w, s = {}, {}
        for r in _rows_of_opt(experiment):
            if r["query_id"] not in set(ids):
                continue
            if r.get("generation") and "environment_steps" not in r["generation"]:
                f = d.drv.item_files(run_dir, DS).get(r["query_id"])
                r = {
                    **r,
                    "generation": {
                        **r["generation"],
                        "result_file": str(f) if f else None,
                    },
                }
            a, b = memory_row(r, spans)
            w[r["query_id"]] = a
            if b is not None:
                s[r["query_id"]] = b
        return w, s

    vgr_old_w, vgr_old_s = vgr_rows(
        f"{DS}:agenteval200:controlled-v1:VGR", d.drv.CTRL / f"{DS}_VGR"
    )
    vgr_new_w, vgr_new_s = vgr_rows(c16.REF_EXPERIMENT, r_vgr_dir)
    refs = {
        "NoMemory": {
            "old": ref_block({}, nm_old_s, ids, baselines)
            | {
                "note": "controlled-v1 2026-10-15; no episodes.jsonl: wall time unavailable, span only"
            },
            "rerun": ref_block(nm_rerun_w, nm_rerun_s, ids, baselines),
        },
        "VGR": {
            "old": ref_block(vgr_old_w, vgr_old_s, ids, baselines),
            "rerun": ref_block(vgr_new_w, vgr_new_s, ids, baselines),
            "memory_drift": memory_drift(d.drv.CTRL / f"{DS}_VGR", r_vgr_dir),
        },
    }

    # c16 vs c128
    old_cv = old.get("outer_cv", {}).get("test", {}).get("summary", {})
    cmp_cv = {k: compare(v, old_cv.get(k), "cv") for k, v in ts_wall["summary"].items()}
    cmp_dep = {k: compare(v, old["arms"].get(k), "deployed") for k, v in arms.items()}

    # load actually seen by vLLM: mean in-flight episodes per run directory (LLM-span intervals)
    pdir = base / c16.SUBDIR / "pairs"
    inflight = {
        p.name: mean_inflight(call_intervals(p))
        for p in sorted([r_nm_dir, r_vgr_dir, *pdir.glob("cfg_*")])
    } | {
        "old controlled-v1 NoMemory": mean_inflight(call_intervals(ctrl_nm)),
        "old controlled-v1 VGR": mean_inflight(
            call_intervals(d.drv.CTRL / f"{DS}_VGR")
        ),
    }
    plan = json.load(open(base / c16.SUBDIR / "plan.json"))
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "dataset": DS,
        "experiment": d.test_experiment(DS, c16.ARM_TAG),
        "plan": plan,
        "concurrency": c16.CONCURRENCY,
        "quality_reuse": False,
        "latency_definitions": {
            "latency_s": "wall: official episode wall time + retrieval ms + router overhead (opt e2e_ms; = v7 c128 "
            "arms and controlled-v1 VGR reference)",
            "latency_span_s": "span: first LLM call start .. last LLM call end of the episode (llm_calls.jsonl) + "
            "retrieval ms + router overhead (= the old controlled-v1 NoMemory reference, which has no "
            "episodes.jsonl)",
        },
        "limeqo": "LimeQO arms excluded from the c16 rerun (user 2026-10-21); LimeQO numbers exist only in test_report.json "
        "(c128, quality reuse on)",
        "baseline": {
            "dir": str(ctrl_nm),
            "label": "controlled-v1 NoMemory (old run; paired baseline of gain, as c128)",
            "quality": sum(nm.values()) / max(len(nm), 1),
            "rerun_quality": sum(nm_rerun.values()) / max(len(nm_rerun), 1)
            if rerun_ok
            else None,
        },
        "references_note": "primary references = the old controlled-v1 NoMemory / VGR runs (user 2026-10-21); VGR "
        "latency = wall, NoMemory latency = LLM-call span (compare with latency_span_s). NoMemory rerun (200/200, "
        "this session) = secondary record; VGR rerun stopped after 30 tasks (not used).",
        "pairs": len(wall),
        "pairs_span": len(span),
        "arms": arms,
        "outer_cv": {
            "protocol": cv["protocol"],
            "folds": cv["folds"],
            "test": {**ts_wall, "per_fold_span": ts_span["per_fold"]},
            "incomplete_test_arms": incomplete,
            "labels": old.get("outer_cv", {}).get("labels"),
        },
        "reference_arms": refs,
        "c16_vs_c128": {"cv": cmp_cv, "deployed": cmp_dep},
        "mean_inflight_episodes": inflight,
        "batches": {
            p.name: batch_stats(
                p,
                {
                    q: r
                    for (k, q), r in wall.items()
                    if d.cfg_name(json.loads(k)) == p.name
                },
            )
            for p in sorted(pdir.glob("cfg_*"))
        },
        "vllm_pre_post": vllm_compare(base),
        "runtime_profile": "controlled-c16-v1",
        "runtime_notes": [
            "both halves (8030 + env 36001, 8031 + env 36011) x 8 processes per batch, one configuration at a time; "
            "a per-batch monitor logs per-GPU running requests and relaunches both halves if one is starved",
            "cfg_6b876cca9ff6: its first 142 pairs ran with one half dead (memory_policy.claim_dir marker race; "
            "8 concurrent) -> opt rows status superseded-load8, directory test_c16/_superseded_load8, rerun at c16; "
            "race fixed (marker written atomically)",
            "infra-fix pause 23:08:32Z (user): fix 2 = PG 55450 pinned to cores 32-39 + orphaned bun processes "
            "killed; TRIMEM_RETRIEVAL_SLOTS 16 -> 24. Fixes 3+4 (vLLM --mamba-block-size 784 + MTP speculative "
            "decoding) ran 23:19-23:22Z, made per-call LLM time worse (2.81 -> 4.48 s) and were REVERTED at 23:24Z "
            "(vLLM back to the reference flags). cfg_321be1a0a950 (91/110 done before the pause) superseded "
            "(status superseded-prefix-pause, test_c16/_superseded_prefix_pause); its 46 MTP-window episodes "
            "superseded too (status superseded-mtp, test_c16/_superseded_mtp); every post-fix pair kept here ran "
            "after the revert (23:41Z+) with fix 2 only",
        ],
        "calibration": calibration_summary(base),
    }
    d.save(base / "test_c16_report.json", rep)
    (base / "report_c16.md").write_text("\n".join(markdown(rep)) + "\n")
    return rep


PAUSE_AT = datetime(
    2026, 10, 21, 23, 8, 32, tzinfo=UTC
).timestamp()  # infra fixes 2-4 applied after this


def phase_of(t: float, pause_at: float = PAUSE_AT) -> str:
    return "pre-fix" if t < pause_at else "post-fix"


def batch_stats(
    pair_dir: Path, wall: dict[str, dict], pause_at: float = PAUSE_AT
) -> dict[str, dict]:
    """Per phase (pre / post the infra-fix pause) of one configuration directory: episodes, busy time (union of
    the episodes' LLM-call intervals), episodes / busy minute, mean wall s, mean per-call LLM s, mean non-LLM s
    inside the LLM span, mean wall - span s. ``wall`` = qid -> wall row (e2e_ms). Reads llm_calls.jsonl."""
    f = pair_dir / "llm_calls.jsonl"
    if not f.exists():
        return {}
    by: dict[str, list[dict]] = {}
    for line in open(f):
        c = json.loads(line)
        if c.get("item") is not None:
            by.setdefault(f"{DS}_{c['item']}", []).append(c)
    out: dict[str, dict] = {}
    for q, cs in by.items():
        if q not in wall:
            continue
        last = max(cs, key=lambda c: c["t"])["pid"]
        mine = [c for c in cs if c["pid"] == last]
        t0, t1 = min(c["t"] for c in mine), max(c["t"] + c["seconds"] for c in mine)
        g = out.setdefault(
            phase_of(t0, pause_at),
            {"iv": [], "wall": [], "call": [], "nonllm": [], "gap": []},
        )
        g["iv"].append((t0, t1))
        g["wall"].append(wall[q]["e2e_ms"] / 1000)
        g["call"] += [c["seconds"] for c in mine]
        llm = sum(c["seconds"] for c in mine)
        g["nonllm"].append(t1 - t0 - llm)
        g["gap"].append(wall[q]["e2e_ms"] / 1000 - (t1 - t0))
    res = {}
    for ph, g in out.items():
        n = len(g["wall"])
        busy = (
            n / mean_inflight(g["iv"]) * (sum(e - s_ for s_, e in g["iv"]) / n)
            if g["iv"]
            else None
        )
        res[ph] = {
            "episodes": n,
            "busy_min": busy / 60 if busy else None,
            "episodes_per_min": n / (busy / 60) if busy else None,
            "mean_inflight": mean_inflight(g["iv"]),
            "wall_mean": sum(g["wall"]) / n,
            "call_mean": sum(g["call"]) / len(g["call"]),
            "nonllm_mean": sum(g["nonllm"]) / n,
            "wall_minus_span": sum(g["gap"]) / n,
        }
    return res


VLLM_LOGS = (
    Path("results/bo_agentenv_20261016/pool_strat200/vllm_gpu0.log"),
    Path("results/bo_agentenv_20261016/pool_strat200/vllm_gpu1.log"),
)
PRE_WINDOWS = (
    ("15:05", "15:41"),
    ("15:51", "16:08"),
)  # local (PDT) pre-fix c16 batches, calibration excluded
LOGGER_RE = re.compile(
    r"10-21 (\d\d:\d\d):\d\d .*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) "
    r"tokens/s, Running: (\d+) reqs.*Prefix cache hit rate: ([\d.]+)%"
)


def logger_summary(lines, windows=PRE_WINDOWS) -> dict | None:
    """Mean vLLM logger stats (prompt / generation tok/s, running, prefix-cache hit rate) over busy lines inside the
    ``windows`` (HH:MM local, inclusive start, exclusive end). Pure."""
    rows = []
    for line in lines:
        m = LOGGER_RE.search(line)
        if not m or int(m.group(4)) == 0:
            continue
        if any(a <= m.group(1) < b for a, b in windows):
            rows.append([float(m.group(i)) for i in (2, 3, 4, 5)])
    if not rows:
        return None
    n = len(rows)
    return {
        "samples": n,
        "prompt_tok_s": sum(r[0] for r in rows) / n,
        "gen_tok_s": sum(r[1] for r in rows) / n,
        "running": sum(r[2] for r in rows) / n,
        "prefix_hit_rate_pct": sum(r[3] for r in rows) / n,
    }


def counter_summary(samples: list[dict]) -> dict | None:
    """One port's sampled counters (first -> last): prefix-cache hit rate, cached share of prompt tokens, MTP
    draft acceptance, generation tok/s. Pure."""
    if len(samples) < 2:
        return None
    a, b = samples[0], samples[-1]
    dd = lambda k: b.get(k, 0.0) - a.get(k, 0.0)  # noqa: E731
    dt = b["t"] - a["t"]
    q, p = dd("vllm:prefix_cache_queries_total"), dd("vllm:prompt_tokens_total")
    drafts = dd("vllm:spec_decode_num_draft_tokens_total")
    return {
        "window_s": dt,
        "prefix_hit_rate_pct": 100 * dd("vllm:prefix_cache_hits_total") / q
        if q
        else None,
        "prompt_cached_pct": 100 * dd("vllm:prompt_tokens_cached_total") / p
        if p
        else None,
        "prompt_tok_s": p / dt if dt else None,
        "gen_tok_s": dd("vllm:generation_tokens_total") / dt if dt else None,
        "mtp_acceptance_pct": 100
        * dd("vllm:spec_decode_num_accepted_tokens_total")
        / drafts
        if drafts
        else None,
        "running_mean": sum(x.get("vllm:num_requests_running", 0) for x in samples)
        / len(samples),
    }


def vllm_compare(base: Path) -> dict:
    pre = {}
    for i, f in enumerate(VLLM_LOGS):
        if f.exists():
            pre[f"gpu{i}"] = logger_summary(open(f, errors="replace"))
    post = {}
    f = base / c16.SUBDIR / "vllm_metrics_postfix.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in open(f) if x.strip()]
        for i, port in enumerate((8030, 8031)):
            post[f"gpu{i}"] = counter_summary([r for r in rows if r["port"] == port])
    return {
        "pre_fix": pre,
        "pre_fix_source": "vLLM logger lines (interval stats) during the pre-fix c16 batches",
        "post_fix": post,
        "post_fix_source": "vllm_metrics_postfix.jsonl (counters sampled every 60 s)",
        "post_fix_vllm": "MAX_NUM_SEQS 128, --mamba-block-size 784 (fix 3 attempted; vLLM still set the attention "
        "block size to 1584), --speculative-config mtp num_speculative_tokens 1 (fix 4)",
    }


def calibration_summary(base: Path) -> dict | None:
    """The concurrency calibration (``alf_c16_calibration``; user 2026-10-21: stopped, c16 kept) as evidence."""
    p = base / c16.SUBDIR / "calib" / "calibration.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    return {
        "decision": "stay at controlled-c16-v1 (user 2026-10-21): latency rose at c32 / c48 with no throughput gain",
        "levels": {
            c: {
                "tasks": v["n"],
                "mean_inflight": v["mean_inflight"],
                "wall_mean": v["compare"]["wall_s"]["mean"],
                "wall_mean_c16": v["compare"]["wall_s"]["mean_c16"],
                "wall_diff": v["compare"]["wall_s"]["diff"],
                "wall_p90": v["compare"]["wall_s"]["p90"],
                "wall_p90_c16": v["compare"]["wall_s"]["p90_c16"],
                "success": v["compare"]["quality"]["mean"],
                "success_c16": v["compare"]["quality"]["mean_c16"],
                "load": v["load"],
            }
            for c, v in r["levels"].items()
        },
        "diagnosis": "per-call LLM time 2.71 -> 3.64 -> 4.48 s (c16 / c32 / c48; prompts ~3.9k tokens, generation "
        "tok/s flat ~146 per GPU: prefill-bound GPUs) and non-LLM time inside the episode 7.1 -> 13.3 -> 19.8 s, "
        "wall - span (env reset + retrieval) 3.4 -> 8.3 -> 14.0 s: the two ALFWorld env servers are single Python "
        "processes (80-85% CPU each at c64, GIL-bound); client taskset 16-31 and 16 retrieval slots are secondary "
        "suspects",
        "file": str(p),
    }


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def _ms(m: dict | None, fmt: str) -> str:
    if not m or m.get("mean") is None:
        return "–"
    s = format(m["mean"], fmt)
    return s + (
        f" ± {format(m['std'], fmt.lstrip('+'))}" if m.get("std") is not None else ""
    )


def markdown(rep: dict) -> list[str]:
    L = [
        "# ALFWorld v7 test — concurrency-16 rerun",
        "",
        f"Generated {rep['generated']}. {rep['pairs']} (config, task) pairs at concurrency {rep['concurrency']}, "
        "quality reuse OFF, reference runtime layout (one configuration at a time, 2 halves × 8 processes). "
        f"{rep['limeqo']}.",
        "",
        "Latency: **wall** = "
        + rep["latency_definitions"]["latency_s"]
        + "; **span** = "
        + rep["latency_definitions"]["latency_span_s"]
        + ".",
        "",
        "## Outer 5-fold CV — test set (mean ± std across folds)",
        "",
        "| arm | quality | tokens | latency wall s | latency span s | steps | steps (succ.) | gain vs NoMemory | "
        "gain vs NoMemory rerun | SE_tasks quality | SE_tasks steps | SE_tasks gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, x in rep["outer_cv"]["test"]["summary"].items():
        se = x.get("se_tasks_mean", {})
        L.append(
            f"| {n} | {_ms(x['quality'], '.3f')} | {_ms(x['tokens'], ',.0f')} | {_ms(x['latency_s'], '.1f')} | "
            f"{_ms(x.get('latency_span_s'), '.1f')} | {_ms(x['steps'], '.2f')} | {_ms(x['steps_success'], '.2f')} | "
            f"{_ms(x['gain'], '+.3f')} | {_f(x.get('gain_vs_nomemory_rerun') and x['gain_vs_nomemory_rerun']['mean'], '+.3f')} | {_f(se.get('quality'), '.3f')} | {_f(se.get('steps'), '.2f')} | "
            f"{_f(se.get('gain'), '.3f')} |"
        )
    L += [
        "",
        "## References (single runs, 200 tasks)",
        "",
        "| arm | run | quality ± SE | tokens | latency wall s | latency span s | steps | steps (succ.) | "
        "gain vs NoMemory (old) | gain vs NoMemory rerun |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in rep["reference_arms"].items():
        for run in ("old", "rerun"):
            w, s = r[run].get("wall"), r[run].get("span")
            m = w or s
            if not m:
                continue
            L.append(
                f"| {n} | {'old controlled-v1 (PRIMARY)' if run == 'old' else 'c16 rerun (secondary)'} "
                f"{m['covered']}/{m['n']} | {_f(m['quality'], '.3f')} ± {_f(m['quality_se'], '.3f')} | {_f(m['tokens'], ',.0f')} | "
                f"{_f(w and w['latency_s'], '.1f')} | {_f(s and s['latency_s'], '.1f')} | {_f(m['steps'], '.2f')} | "
                f"{_f(m['steps_success'], '.2f')} | {_f(m['paired_gain_vs_nomemory'], '+.3f')} | "
                f"{_f(m.get('paired_gain_vs_nomemory_rerun'), '+.3f')} |"
            )
    dr = rep["reference_arms"]["VGR"]["memory_drift"]
    L += [
        "",
        f"VGR rerun (stopped, not used) injected memory identical to the old run on {dr['identical']}/"
        f"{dr['compared']} tasks. {rep.get('references_note', '')}",
        "",
        "## Deployed arms (single routing, 200 tasks)",
        "",
        "| arm | quality ± SE | tokens | latency wall s | latency span s | steps | steps (succ.) | gain vs NoMemory | "
        "gain vs NoMemory rerun |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, x in rep["arms"].items():
        if x.get("quality") is None:
            continue
        L.append(
            f"| {n} | {x['quality']:.3f} ± {x['quality_se']:.3f} | {x['tokens']:,.0f} | {x['latency_s']:.1f} | "
            f"{_f(x.get('latency_span_s'), '.1f')} | {_f(x.get('steps'), '.2f')} | {_f(x.get('steps_success'), '.2f')} | "
            f"{_f(x.get('paired_gain_vs_nomemory'), '+.3f')} | {_f(x.get('paired_gain_vs_nomemory_rerun'), '+.3f')} |"
        )
    L += [
        "",
        "## c16 vs c128 (wall latency; c128 = test_report.json, quality reuse on)",
        "",
        "| arm | quality c16 | quality c128 | Δ quality (z) | latency c16 s | latency c128 s | ratio c16/c128 | "
        "tokens ratio | steps c16 | steps c128 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ("cv", "deployed"):
        for n, c in rep["c16_vs_c128"][kind].items():
            if not c:
                continue
            L.append(
                f"| {'CV ' if kind == 'cv' else ''}{n} | {c['quality_c16']:.3f} | {c['quality_c128']:.3f} | "
                f"{c['quality_diff']:+.3f} ({_f(c['quality_diff_z'], '+.1f')}) | {c['latency_c16_s']:.1f} | "
                f"{c['latency_c128_s']:.1f} | {_f(c['latency_ratio_c16_over_c128'], '.2f')} | "
                f"{_f(c['tokens_ratio_c16_over_c128'], '.2f')} | {_f(c['steps_c16'], '.2f')} | {_f(c['steps_c128'], '.2f')} |"
            )
    L += [
        "",
        "Δ quality z = (c16 − c128) / sqrt(SE16² + SE128²) with task SEs; the two runs share tasks, so this "
        "independent-run z is conservative. CV rows use the mean within-fold SE_tasks.",
        "",
        "## Runtime profile controlled-c16-v1",
        "",
        *[f"- {x}" for x in rep.get("runtime_notes", [])],
        *(
            [
                "",
                "### Concurrency calibration (evidence only; stopped by the user, c16 kept)",
                "",
                "| c | tasks | mean in-flight | wall mean (c16) | Δ [95% CI] | wall p90 (c16) | success (c16) | "
                "running GPU0 / GPU1 | gen tok/s GPU0 / GPU1 |",
                "|---:|---:|---:|---|---|---|---|---|---|",
                *[
                    f"| {c} | {v['tasks']} | {_f(v['mean_inflight'], '.1f')} | {v['wall_mean']:.1f} "
                    f"({v['wall_mean_c16']:.1f}) | {v['wall_diff']['mean']:+.1f} [{v['wall_diff']['lo']:+.1f}, "
                    f"{v['wall_diff']['hi']:+.1f}] | {v['wall_p90']:.1f} ({v['wall_p90_c16']:.1f}) | "
                    f"{v['success']:.3f} ({v['success_c16']:.3f}) | "
                    f"{_f(v['load']['per_gpu'][0]['running_mean'], '.1f')} / "
                    f"{_f(v['load']['per_gpu'][1]['running_mean'], '.1f')} | "
                    f"{_f(v['load']['per_gpu'][0]['gen_tok_s'], '.0f')} / {_f(v['load']['per_gpu'][1]['gen_tok_s'], '.0f')} |"
                    for c, v in rep["calibration"]["levels"].items()
                ],
                "",
                f"Diagnosis: {rep['calibration']['diagnosis']}.",
            ]
            if rep.get("calibration")
            else []
        ),
        "",
        "## Per-batch throughput and latency, before vs after the infra-fix pause (23:08:32Z)",
        "",
        "Post-fix runtime: PG 55450 pinned to cores 32-39, orphaned bun killed, TRIMEM_RETRIEVAL_SLOTS=24, vLLM with the reference flags (fixes 3+4 reverted) "
        "(plan_history.jsonl). Different configurations ran in each phase, so compare per-call LLM time and the "
        "non-LLM components as well as wall time.",
        "",
        "| configuration | phase | episodes | episodes / busy min | mean in-flight | wall mean s | LLM call s | "
        "non-LLM in span s | wall − span s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {ph} | {v['episodes']} | {_f(v['episodes_per_min'], '.1f')} | "
            f"{_f(v['mean_inflight'], '.1f')} | {v['wall_mean']:.1f} | {v['call_mean']:.2f} | "
            f"{v['nonllm_mean']:.1f} | {v['wall_minus_span']:.1f} |"
            for name, b in rep.get("batches", {}).items()
            for ph, v in sorted(b.items(), reverse=True)
        ],
        "",
        "### vLLM before the fixes vs the MTP window (23:19-23:22Z, fixes 3+4, reverted; no kept pair ran in it)",
        "",
        "| GPU | phase | prompt tok/s | gen tok/s | running | prefix-cache hit % | cached prompt % | MTP acceptance % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {g} | pre-fix | {_f(v and v['prompt_tok_s'], '.0f')} | {_f(v and v['gen_tok_s'], '.0f')} | "
            f"{_f(v and v['running'], '.1f')} | {_f(v and v['prefix_hit_rate_pct'], '.1f')} | – | – |"
            for g, v in rep["vllm_pre_post"]["pre_fix"].items()
        ],
        *[
            f"| {g} | post-fix | {_f(v and v['prompt_tok_s'], '.0f')} | {_f(v and v['gen_tok_s'], '.0f')} | "
            f"{_f(v and v['running_mean'], '.1f')} | {_f(v and v['prefix_hit_rate_pct'], '.1f')} | "
            f"{_f(v and v['prompt_cached_pct'], '.1f')} | {_f(v and v['mtp_acceptance_pct'], '.1f')} |"
            for g, v in rep["vllm_pre_post"]["post_fix"].items()
        ],
        "",
        "## Load check: mean in-flight episodes per run directory (LLM-span intervals)",
        "",
        "| directory | mean in-flight |",
        "|---|---:|",
    ]
    for k, v in rep["mean_inflight_episodes"].items():
        L.append(f"| {k} | {_f(v, '.1f')} |")
    return L


# ------------------------------------------------------------------ smoke test (deployed BRANE default arm)


def quantile(xs: list[float], p: float) -> float | None:
    """Linear-interpolation quantile (numpy default). Pure."""
    if not xs:
        return None
    s = sorted(xs)
    h = (len(s) - 1) * p
    lo = int(h)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (h - lo)


def dist(xs: list[float]) -> dict:
    """mean ± SE, p50, p90, n of a latency list (s). Pure."""
    from trimem.experiments import bo_pool_driver as d

    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs) if xs else None,
        "se": d._se(xs) if xs else None,
        "p50": quantile(xs, 0.5),
        "p90": quantile(xs, 0.9),
    }


def arm_rows(
    routes: dict[str, dict], rows: dict, key
) -> list[tuple[dict, float, dict]]:
    """(route, overhead ms, row) of every routed task with a row. Pure."""
    out = []
    for q, r in routes.items():
        x = rows.get((key(r["config"]), q))
        if x is not None:
            out.append((r, float(r.get("overhead_ms") or 0.0), x))
    return out


def arm_dist(routes: dict[str, dict], rows: dict, key) -> dict:
    got = arm_rows(routes, rows, key)
    lat = [(x["e2e_ms"] + oh) / 1000 for _, oh, x in got]
    st = [x["steps"] for _, _, x in got if x.get("steps") is not None]
    ss = [x["steps_success"] for _, _, x in got if x.get("steps_success") is not None]
    q = [x["quality"] for _, _, x in got]
    from trimem.experiments import bo_pool_driver as d

    return {
        "covered": len(got),
        "n": len(routes),
        "latency_s": dist(lat),
        "quality": sum(q) / len(q) if q else None,
        "quality_se": d._se(q) if q else None,
        "tokens": sum(x["tokens"] for _, _, x in got) / len(got) if got else None,
        "steps": sum(st) / len(st) if st else None,
        "steps_success": sum(ss) / len(ss) if ss else None,
    }


def smoke_summary() -> dict:
    """c16 vs c128 of the deployed ``alf_test_c16.SMOKE_ARM`` (wall and span latency definitions)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    base = c16.run_base()
    tab, cv = c16.routing(base)
    routes = tab["arms"][c16.SMOKE_ARM]["routes"]
    wall, span = pair_rows(base, tab, cv)
    # c128: the original test pairs (opt rows of the c128 experiment + its NoMemory pair files)
    full = json.load(open(base / "test" / "routing.json"))
    w128, s128 = {}, {}
    sp128: dict[str, dict] = {}
    for r in _rows_of_opt(full["experiment"]):
        name = d.cfg_name(r["cfg"])
        if name not in sp128:
            sp128[name] = c16.call_spans(base / "test" / "pairs" / name)
        a, b = memory_row(r, sp128[name])
        w128[(space.key(r["cfg"]), r["query_id"])] = a
        if b is not None:
            s128[(space.key(r["cfg"]), r["query_id"])] = b
    nm_key = space.key(space.NOMEM)
    nm_q = [q for q, r in routes.items() if space.key(r["config"]) == nm_key]
    if nm_q:
        a, b = nomemory_dir_rows(
            base / "test" / "pairs" / d.cfg_name(space.NOMEM), nm_q
        )
        w128 |= {(nm_key, q): r for q, r in a.items()}
        s128 |= {(nm_key, q): r for q, r in b.items()}
    rep = json.load(open(base / "test_report.json"))
    old_refs = rep.get("reference_arms", {})
    return {
        "arm": c16.SMOKE_ARM,
        "configs": len({space.key(r["config"]) for r in routes.values()}),
        "c16": {
            "wall": arm_dist(routes, wall, space.key),
            "span": arm_dist(routes, span, space.key),
        },
        "c128": {
            "wall": arm_dist(routes, w128, space.key),
            "span": arm_dist(routes, s128, space.key),
        },
        "c128_report_latency_s": (rep["arms"].get(c16.SMOKE_ARM) or {}).get(
            "latency_s"
        ),
        "old_reference_latency_s": {k: v.get("latency_s") for k, v in old_refs.items()},
        "old_reference_definition": {
            "VGR": "wall (episode + retrieval)",
            "NoMemory": "LLM-call span (no episodes.jsonl)",
        },
    }
