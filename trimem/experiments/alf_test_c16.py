"""ALFWorld pool-strat200-v7 TEST rerun at concurrency 16 (user decision 2026-10-21).

The v7 test stage (``bo_pool_driver test``) ran its 1742 distinct (configuration, query) pairs at concurrency 128
with quality reuse ON. Its latencies are therefore not comparable to the reused controlled-v1 reference arms
(NoMemory / VGR), which ran at concurrency 16. This module reruns the pairs at concurrency 16 into a SEPARATE
directory (``<run>/test_c16/pairs``) and opt experiment (``...:pairs-c16:...``), leaving the c128 results intact:

* pair set: the FROZEN routing tables (``test/routing.json`` + ``test/routing_cv.json``), WITHOUT the LimeQO arms
  (user 2026-10-21: Opt-OneConfig SLO-1/2/3 + BRANE Opt-PerQuery x 4 weights, deployed + 5-fold CV, plus the
  NoMemory pairs those arms route to). Each distinct pair runs once (pair reuse within the rerun).
* quality reuse OFF (a reused episode copies its source's latency).
* total concurrency 16 (``bo_pool_driver.run_capped``: <= 8 arms at a time, 2 halves x 1+ processes each).

Latency definitions (see ``report``): ``wall`` = official episode wall time + retrieval ms (+ router overhead) =
the opt-row ``e2e_ms`` (definition of the v7 arms and of the VGR reference); ``span`` = first LLM call start to
last LLM call end of the item in ``llm_calls.jsonl`` + retrieval ms (+ router overhead) (definition of the
NoMemory reference, whose official-evaluator run predates ``episodes.jsonl``).

Nothing here changes ``bo_pool_driver`` defaults.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 .venv-optimizer/bin/python -m \
        trimem.experiments.alf_test_c16 {plan|run|report}
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DS = "alfworld"
CONCURRENCY = 16
SUBDIR = "test_c16"
ARM_TAG = "pairs-c16"  # opt experiment = bo_pool_driver.test_experiment(DS, ARM_TAG)
EXCLUDED = "LimeQO"  # arms whose name contains this (or router == limeqo) are not rerun


def run_base() -> Path:
    from trimem.experiments import bo_pool_driver as d

    return d.run_dir(DS)


def is_excluded(name: str, arm: dict) -> bool:
    return EXCLUDED in name or arm.get("router") == "limeqo"


# alf_test_v7 (user 2026-10-22) archives this run's rounds 0-5 routing tables / reports under *_v7r0-5_<ts> and
# records the renames here; this module then reads the archived tables (its own pairs never move).
MANIFEST = "archive_v7r0-5.json"


def archived(base: Path, rel: str) -> Path:
    """``base / rel``, or -- after the v7 re-test archive -- the file the published rounds 0-5 reports were built
    from (manifest ``report_sources``: the *_minmaxraw_20261022 routing / selection / router set, since the
    in-place zscore-log re-route of 2026-10-21 20:44 was never tested), else its archived name (``moved``)."""
    m = base / MANIFEST
    if m.exists():
        man = json.loads(m.read_text())
        for blk in ("report_sources", "moved"):
            if rel in man.get(blk, {}):
                return base / man[blk][rel]
    return base / rel


def routing(base: Path) -> tuple[dict, dict]:
    """(deployed routing table, CV routing table), both without the excluded (LimeQO) arms. Pure on the files.
    After the v7 re-test archive: the archived rounds 0-5 tables."""
    tab = json.load(open(archived(base, "test/routing.json")))
    cv = json.load(open(archived(base, "test/routing_cv.json")))

    def keep(t: dict) -> dict:
        return {
            **t,
            "arms": {n: a for n, a in t["arms"].items() if not is_excluded(n, a)},
        }

    return keep(tab), keep(cv)


def pair_set(tab: dict, cv: dict) -> dict[str, list[str]]:
    from trimem.experiments import bo_pool_driver as d

    return d.pairs_of({"arms": {**tab["arms"], **cv["arms"]}})


def build_arms(base: Path, pairs: dict[str, list[str]]) -> list:
    """One subset arm per configuration (largest first, so ``run_capped`` batches hold similar sizes)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(DS, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(DS, ARM_TAG)
    out = base / SUBDIR / "pairs"
    arms = []
    for k, qs in sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            arms.append(d.Arm("c16 NoMemory pairs", out / d.cfg_name(cfg), qs, None))
            continue
        session = {
            **space.runner_args(cfg),
            "runner_extra": space.runner_cli(cfg),
            "split": "test",
            "experiment": exp,
            "baseline_dir": baseline,
            "panel": None,
            "reuse": False,  # quality reuse OFF: latency must be measured, never copied
            "reuse_from": [],
            "reuse_from_dirs": [],
            "protocol_extra": {
                "panel_hash": f"test:{d.TEST_VERSION}",
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": "test-pairs-c16",
                "round": d.PANEL,
                "test_version": d.TEST_VERSION,
                "test_concurrency": CONCURRENCY,
                "rerun_of": "test-pairs (c128, quality reuse on)",
            },
        }
        arms.append(d.Arm(f"c16 {d.cfg_name(cfg)}", out / d.cfg_name(cfg), qs, session))
    return arms


def plan() -> dict:
    base = run_base()
    tab, cv = routing(base)
    pairs = pair_set(tab, cv)
    full = (
        json.load(open(archived(base, "test/routing.json"))),
        json.load(open(archived(base, "test/routing_cv.json"))),
    )
    from trimem.experiments import bo_pool_driver as d

    n_full = sum(
        len(v)
        for v in d.pairs_of({"arms": {**full[0]["arms"], **full[1]["arms"]}}).values()
    )
    return {
        "arms": len(tab["arms"]) + len(cv["arms"]),
        "pairs": sum(len(v) for v in pairs.values()),
        "configs": len(pairs),
        "pairs_with_limeqo": n_full,
        "per_config": {
            d.cfg_name(json.loads(k)): len(v)
            for k, v in sorted(pairs.items(), key=lambda kv: -len(kv[1]))
        },
    }


# Reference-faithful runtime (user 2026-10-21): the controlled-v1 NoMemory / VGR test arms
# (results/eval_controlled_20261015/run_controlled.sh) ran ONE arm at a time, two interleaved halves (8030 / env
# 36001, 8031 / env 36011) x 8 processes, taskset 16-31, nice 10, temperature 0, max_round 30, max_tokens 4096,
# timeout 2400, thinking off, function calling, semantic calls on the half's own GPU (no request-load routing).
PER_HALF = 8
SMALL = (
    2 * PER_HALF
)  # configurations with fewer pairs than one full layout share one batch (total <= 16)
REF_VGR_CFG = {
    "plan": "VGR",
    "depth": 1,
    "edges": "similar",
    "top_k": 3,
    "fanout": 4,
    "seed_count": 8,
}
REF_EXPERIMENT = (
    "alfworld:agenteval200:controlled-v1-c16rerun:VGR"  # success-only-v0: untagged
)


def reference_arms(base: Path, ids: list[str]) -> list:
    """The controlled-v1 NoMemory and VGR test arms, rerun under this session's runtime (own dirs / experiment):
    NoMemory = official evaluator; VGR = its original configuration (seed 8, depth 1, fanout 4, top-k 3, edges
    similar; legacy retrieval path) under memory policy success-only-v0."""
    from trimem.experiments import bo_pool_driver as d

    out = base / SUBDIR
    vgr = {
        "plan": "VGR",
        "edges": REF_VGR_CFG["edges"],
        "memory_config": {
            k: REF_VGR_CFG[k] for k in ("seed_count", "depth", "fanout", "top_k")
        },
        "runner_extra": [],
        "split": "test",
        "experiment": REF_EXPERIMENT,
        "baseline_dir": d.drv.CTRL / f"{DS}_NoMemory",
        "panel": None,
        "reuse": False,
        "reuse_from": [],
        "reuse_from_dirs": [],
        "protocol_extra": {
            "space_config": REF_VGR_CFG,
            "bo_label": "reference-rerun-c16",
            "rerun_of": "alfworld:agenteval200:controlled-v1:VGR",
            "test_concurrency": CONCURRENCY,
        },
    }
    return [
        d.Arm("c16 ref NoMemory", out / "ref_NoMemory", ids, None),
        d.Arm("c16 ref VGR", out / "ref_VGR", ids, vgr),
    ]


def interleave(a: list[str], b: list[str]) -> list[str]:
    """[a0, b0, a1, b1, ...]: ``drv.shards`` (items[0::2] / items[1::2]) then reproduces halves a / b. Pure."""
    if len(a) not in (len(b), len(b) + 1):
        raise ValueError("halves must differ by at most one item")
    out = [x for pair in zip(a, b, strict=False) for x in pair]
    return out + a[len(b) :]


def reference_order() -> list[str]:
    """The 200 test ids ordered so the reruns get the references' exact halves (shards/alfworld_a/b.json)."""
    from trimem.experiments import bo_pool_driver as d

    a, b = (
        [
            x["item_id"]
            for x in json.load(open(d.drv.CTRL / "shards" / f"{DS}_{h}.json"))
        ]
        for h in ("a", "b")
    )
    return interleave(a, b)


def schedule(arms: list) -> list[list]:
    """Batches run one after another: every arm with >= SMALL pairs alone (2 x 8 processes, the reference
    layout); the smaller ones together in one final batch (total concurrency <= 16). Pure."""
    big = [[a] for a in arms if len(a.ids) >= SMALL]
    small = [a for a in arms if len(a.ids) < SMALL]
    return big + ([small] if small else [])


SMOKE_ARM = "Opt-PerQuery-default"  # deployed BRANE default arm (user 2026-10-21: smoke test first)


def smoke_arms(base: Path) -> list:
    """The configuration subset arms of the deployed ``SMOKE_ARM`` only (its 200 routed pairs), same dirs /
    experiment as the full rerun (the full run later reuses these pairs)."""
    from trimem.experiments import bo_pool_driver as d

    tab, _ = routing(base)
    return build_arms(base, d.pairs_of({"arms": {SMOKE_ARM: tab["arms"][SMOKE_ARM]}}))


def smoke() -> None:
    """Smoke test: the deployed BRANE default arm's pairs, ONE configuration at a time (2 x 8 processes, the
    reference layout), quality reuse off; then ``smoke_summary`` is logged and written."""
    from trimem.experiments import bo_pool_driver as d

    base = run_base()
    arms = smoke_arms(base)
    (base / SUBDIR).mkdir(parents=True, exist_ok=True)
    d.log(
        f"c16 smoke: {SMOKE_ARM} -> {sum(len(a.ids) for a in arms)} pairs over {len(arms)} configurations, "
        f"one at a time at concurrency {CONCURRENCY}"
    )
    res: dict[str, int] = {}
    for a in arms:
        res |= d.run_parallel(DS, [a], CONCURRENCY)
    bad = [a.name for a in arms if res.get(a.name, 0) < len(a.ids)]
    if bad:
        raise SystemExit(f"c16 smoke pairs incomplete: {bad}")
    from trimem.experiments import alf_test_c16_report as rep

    s = rep.smoke_summary()
    d.save(base / SUBDIR / "smoke_summary.json", s)
    d.log("C16 SMOKE DONE " + json.dumps(s))


def run(refs_first: bool = True, refs: bool = True) -> None:
    import os

    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    base = run_base()
    tab, cv = routing(base)
    pairs = pair_set(tab, cv)
    arms = build_arms(base, pairs)
    ref_arms = reference_arms(base, reference_order())
    (base / SUBDIR).mkdir(parents=True, exist_ok=True)
    d.save(
        base / SUBDIR / "plan.json",
        {
            "started": datetime.now(UTC).isoformat(),
            "concurrency": CONCURRENCY,
            "layout": "one configuration at a time, 2 halves x 8 processes (reference layout); configurations "
            f"with < {SMALL} pairs share one final batch (total <= {CONCURRENCY})",
            "quality_reuse": False,
            "retrieval_slots": os.environ.get("TRIMEM_RETRIEVAL_SLOTS")
            or "default (16)",
            "experiment": d.test_experiment(DS, ARM_TAG),
            "reference_experiment": REF_EXPERIMENT,
            "excluded_arms": EXCLUDED,
            **plan(),
        },
    )
    with open(
        base / SUBDIR / "plan_history.jsonl", "a"
    ) as f:  # every (re)launch and its runtime knobs
        f.write(json.dumps(json.load(open(base / SUBDIR / "plan.json"))) + "\n")
    batches = (
        schedule(
            arms
        )  # user 2026-10-21: no reference reruns (old controlled-v1 arms are the references)
        if not refs
        else [[ref_arms[0]], [ref_arms[1]], *schedule(arms)]
        if refs_first
        else [*schedule(arms), [ref_arms[0]], [ref_arms[1]]]
    )
    d.log(
        f"c16 rerun: {'references (2 x 200) + ' if refs else 'no reference reruns, '}{sum(len(a.ids) for a in arms)} pairs over {len(arms)} configurations "
        f"in {len(batches)} batches at concurrency {CONCURRENCY}, quality reuse off -> {base / SUBDIR}"
    )
    res: dict[str, int] = {}
    for b in batches:
        ref_vgr = b[0].name == "c16 ref VGR"
        old = os.environ.get(mp.ENV_VAR)
        if (
            ref_vgr
        ):  # the reference VGR ran under success-only-v0 (the runners inherit the env)
            os.environ[mp.ENV_VAR] = mp.SUCCESS_ONLY
        mon = HalfMonitor(b)
        mon.start()
        try:
            res |= d.run_parallel(DS, b, CONCURRENCY)
        finally:
            mon.stop.set()
            mon.join(timeout=5)
            if old is None:
                os.environ.pop(mp.ENV_VAR, None)
            else:
                os.environ[mp.ENV_VAR] = old
    bad = [
        a.name
        for a in (ref_arms if refs else []) + arms
        if res.get(a.name, 0) < len(a.ids)
    ]
    if bad:
        raise SystemExit(f"c16 pairs incomplete: {bad}")
    d.log("C16 TEST DONE alfworld")


# ------------------------------------------------------------------ both-halves guard (user 2026-10-21)

BASES = ("http://127.0.0.1:8030", "http://127.0.0.1:8031")


def half_remaining(a, half: int) -> int:
    """Items of ``a``'s current shard ``half`` (0 = a / 8030, 1 = b / 8031) without a result file."""
    from trimem.experiments import bo_pool_driver as d

    f = a.out.parent / "_shards" / a.out.name / f"{'ab'[half]}.json"
    if not f.exists():
        return 0
    want = {x["item_id"] for x in json.loads(f.read_text())}
    return len(want - set(d.drv.item_files(a.out, DS)))


def starved_halves(running: list[float], remaining: list[int]) -> list[int]:
    """Halves whose GPU runs nothing while the half still has items and the other GPU is busy. Pure."""
    return [
        h for h in (0, 1) if running[h] == 0 and remaining[h] > 0 and running[1 - h] > 0
    ]


class HalfMonitor(threading.Thread):
    """Every ``dt`` s (first sample after 60 s): log per-GPU running requests and per-half remaining items of the
    batch; when one half is starved on two consecutive samples (its process died, e.g. the 2026-10-21 claim_dir
    race), drain the batch's arms so ``run_parallel`` relaunches BOTH halves."""

    def __init__(self, arms: list, dt: float = 60.0):
        super().__init__(daemon=True)
        self.arms, self.dt, self.stop, self.strikes = arms, dt, threading.Event(), 0

    def running(self) -> list[float]:
        out = []
        for b in BASES:
            with urllib.request.urlopen(b + "/metrics", timeout=5) as r:
                v = [
                    float(x.rsplit(" ", 1)[1])
                    for x in r.read().decode().splitlines()
                    if x.startswith("vllm:num_requests_running{")
                ]
            out.append(sum(v))
        return out

    def run(self) -> None:
        from trimem.experiments import bo_pool_driver as d

        names = ", ".join(a.name for a in self.arms)
        while not self.stop.wait(self.dt):
            try:
                run_ = self.running()
                rem = [sum(half_remaining(a, h) for a in self.arms) for h in (0, 1)]
            except Exception as exc:  # noqa: BLE001 -- the guard must never stop the run
                d.log(f"half monitor [{names}]: sample failed {exc!r}")
                continue
            d.log(
                f"half monitor [{names}]: running GPU0 {run_[0]:.0f} / GPU1 {run_[1]:.0f}; remaining half a "
                f"{rem[0]} / half b {rem[1]}"
            )
            bad = starved_halves(run_, rem)
            self.strikes = self.strikes + 1 if bad else 0
            if self.strikes >= 2:
                d.log(
                    f"half monitor [{names}]: half {'ab'[bad[0]]} starved twice -> draining the batch so both "
                    "halves relaunch"
                )
                for a in self.arms:
                    d.drain_path(a).touch()
                self.strikes = 0


# ------------------------------------------------------------------ latency definitions


def call_spans(pair_dir: Path, ds: str = DS) -> dict[str, float]:
    """qid -> LLM-call span ms (first call start .. last call end) of the item's LAST worker in
    ``llm_calls.jsonl`` (the definition ``bo_pool_driver.nomemory_rows`` uses for the reference NoMemory). Pure."""
    f = pair_dir / "llm_calls.jsonl"
    if not f.exists():
        return {}
    calls: dict[str, list[dict]] = {}
    for line in open(f):
        c = json.loads(line)
        if c.get("item") is not None:
            calls.setdefault(f"{ds}_{c['item']}", []).append(c)
    return {q: span_ms(cs) for q, cs in calls.items()}


def span_ms(calls: list[dict]) -> float:
    last = max(calls, key=lambda c: c["t"])["pid"]
    mine = [c for c in calls if c["pid"] == last]
    return 1000 * (max(c["t"] + c["seconds"] for c in mine) - min(c["t"] for c in mine))


def with_span(row: dict, span: float | None, retrieval_ms: float) -> dict:
    """Row in the ``span`` definition: e2e_ms := LLM-call span + retrieval ms (None span -> row dropped)."""
    return None if span is None else {**row, "e2e_ms": span + retrieval_ms}


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def diff_z(a: float, a_se: float, b: float, b_se: float) -> float | None:
    """(a - b) / sqrt(se_a^2 + se_b^2) (independent-run approximation)."""
    s = math.hypot(a_se, b_se)
    return (a - b) / s if s > 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "smoke", "run", "report"))
    ap.add_argument(
        "--no_refs",
        action="store_true",
        help="run: only the Opt pairs (no NoMemory / VGR reference reruns)",
    )
    ap.add_argument(
        "--refs_last",
        action="store_true",
        help="run: the NoMemory / VGR reference reruns after the Opt pairs",
    )
    a = ap.parse_args(argv)
    if a.stage == "plan":
        print(json.dumps(plan(), indent=1))
    elif a.stage == "smoke":
        smoke()
    elif a.stage == "run":
        run(refs_first=not a.refs_last, refs=not a.no_refs)
    else:
        from trimem.experiments import alf_test_c16_report as rep

        if (run_base() / MANIFEST).exists():
            raise SystemExit(
                f"rounds 0-5 artifacts archived ({MANIFEST}): the c16 report is kept as "
                "test_c16_report_v7r0-5_<ts>.json; the re-test report is alf_test_v7 report"
            )
        rep.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
